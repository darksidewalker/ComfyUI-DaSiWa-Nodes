import math
import torch
import torch.nn.functional as F
import contextlib
from typing import Tuple, Optional

# --- Constants ---

QUALITY_LEVELS = ["Low", "Medium", "High", "Ultra"]
UPSCALE_MODES = ["Off", "VSR", "High Bitrate"]
RESIZE_TYPES = ["Keep Ratio", "Manual", "Preset Ratio", "Scale", "Same Size"]
DIVISIBLE_BY_VALUES = ["8", "16", "32", "64", "128"]
COMMON_RATIOS = ["1:1", "4:3", "3:2", "16:9", "21:9"]
RESIZE_METHODS = ["Center Crop (Fill)", "Letterbox (Fit)"]

# --- Helpers ---

def round_up(value: float, alignment: int) -> int:
    return int(math.ceil(float(value) / alignment) * alignment)

def compute_aligned_ratio_dims(ratio_preset: str, megapixels: float, alignment: int) -> Tuple[int, int]:
    try:
        w_part, h_part = map(float, ratio_preset.split(':'))
    except Exception:
        w_part, h_part = 16.0, 9.0
    
    target_area = max(0.01, float(megapixels)) * 1_000_000.0
    aspect = w_part / h_part
    
    h = math.sqrt(target_area / aspect)
    w = h * aspect
    
    return round_up(w, alignment), round_up(h, alignment)

def _quality_attr(mode: str, quality: str) -> str:
    """Maps UI strings to nvvfx QualityLevel attributes."""
    q = quality.upper()
    if mode == "High Bitrate":
        return f"HIGHBITRATE_{q}"
    elif mode in ["Denoise", "Deblur"]:
        return f"{mode.upper()}_{q}"
    return q  # Default VSR

def _aligned_megapixel_size(source_width: int, source_height: int, megapixels: float, alignment: int) -> Tuple[int, int]:
    target_area = max(0.01, float(megapixels)) * 1_000_000.0
    source_aspect = float(source_width) / float(source_height)
    source_area = max(1.0, float(source_width * source_height))
    scale = math.sqrt(target_area / source_area)
    
    base_width = max(float(alignment), float(source_width) * scale)
    base_height = max(float(alignment), float(source_height) * scale)

    def round_down(value: float) -> int:
        return max(alignment, int(math.floor(float(value) / alignment) * alignment))

    def round_nearest(value: float) -> int:
        return max(alignment, int(math.floor((float(value) / alignment) + 0.5) * alignment))

    def round_up_aligned(value: float) -> int:
        return round_up(value, alignment)

    candidates = set()
    for width_rounder in (round_down, round_nearest, round_up_aligned):
        width_candidate = width_rounder(base_width)
        exact_height = width_candidate / source_aspect
        for height_rounder in (round_down, round_nearest, round_up_aligned):
            candidates.add((width_candidate, height_rounder(exact_height)))

    for height_rounder in (round_down, round_nearest, round_up_aligned):
        height_candidate = height_rounder(base_height)
        exact_width = height_candidate * source_aspect
        for width_rounder in (round_down, round_nearest, round_up_aligned):
            candidates.add((width_rounder(exact_width), height_candidate))

    def candidate_score(dims: Tuple[int, int]) -> Tuple[float, float, float]:
        w, h = dims
        area_error = abs((w * h) - target_area) / target_area
        ratio_error = abs((w / h) - source_aspect) / source_aspect
        distance_error = (abs(w - base_width) / base_width + abs(h - base_height) / base_height)
        return (ratio_error, area_error, distance_error)

    return min(candidates, key=candidate_score)

def _target_size(
    source_width: int,
    source_height: int,
    resize_type: str,
    scale: float,
    megapixels: float,
    width: int,
    height: int,
    alignment: int,
    ratio_preset: str,
) -> Tuple[int, int]:
    if resize_type == "Same Size":
        return source_width, source_height
    if resize_type == "Scale":
        return (
            round_up(float(source_width) * float(scale), alignment),
            round_up(float(source_height) * float(scale), alignment),
        )
    if resize_type == "Keep Ratio":
        return _aligned_megapixel_size(source_width, source_height, megapixels, alignment)
    if resize_type == "Preset Ratio":
        return compute_aligned_ratio_dims(ratio_preset, megapixels, alignment)
    return (
        round_up(int(width), alignment),
        round_up(int(height), alignment),
    )

def _fit_frame_to_target_aspect(frame, target_width: int, target_height: int, resize_method: str):
    _, source_height, source_width = frame.shape
    source_aspect = float(source_width) / float(source_height)
    target_aspect = float(target_width) / float(target_height)

    if abs(source_aspect - target_aspect) < 0.0001:
        return frame.contiguous()

    if resize_method == "Center Crop (Fill)":
        if source_aspect > target_aspect:
            crop_width = max(1, min(int(source_width), int(round(float(source_height) * target_aspect))))
            crop_x = max(0, (int(source_width) - crop_width) // 2)
            return frame[:, :, crop_x:crop_x + crop_width].contiguous()

        crop_height = max(1, min(int(source_height), int(round(float(source_width) / target_aspect))))
        crop_y = max(0, (int(source_height) - crop_height) // 2)
        return frame[:, crop_y:crop_y + crop_height, :].contiguous()

    # Letterbox (Fit)
    if source_aspect > target_aspect:
        padded_height = max(int(source_height), int(math.ceil(float(source_width) / target_aspect)))
        pad_total = padded_height - int(source_height)
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        return F.pad(frame, (0, 0, pad_top, pad_bottom), mode="constant", value=0.0).contiguous()

    padded_width = max(int(source_width), int(math.ceil(float(source_height) * target_aspect)))
    pad_total = padded_width - int(source_width)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return F.pad(frame, (pad_left, pad_right, 0, 0), mode="constant", value=0.0).contiguous()

def _import_vfx():
    try:
        from nvvfx import VideoSuperRes
        return VideoSuperRes
    except ImportError:
        raise RuntimeError(
            "NVIDIA RTX VFX (nvvfx) module not found. "
            "Please ensure NVIDIA RTX Video SDK / Broadcast SDK is installed and the 'nvvfx' package is in your python path."
        )

@contextlib.contextmanager
def _maybe_vfx_effect(VideoSuperRes, enabled, mode, quality, device_index, out_width, out_height):
    if not enabled:
        yield None
        return
    
    attr_name = _quality_attr(mode, quality)
    try:
        q_level = getattr(VideoSuperRes.QualityLevel, attr_name)
    except AttributeError:
        raise ValueError(f"Invalid quality level mapping: {attr_name}")

    try:
        effect = VideoSuperRes(quality=q_level, device=device_index)
        effect.output_width = int(out_width)
        effect.output_height = int(out_height)
        effect.load()
        yield effect
    except Exception as exc:
        raise RuntimeError(f"Failed to create NVIDIA RTX VFX effect ({mode} {quality}): {exc}")
    finally:
        # nvvfx effect objects should be destroyed/cleaned up if possible, 
        # though the 'with' block usually handles the lifecycle if the lib supports it.
        pass

class DaSiWa_RTX_UpscalerRefiner:
    DESCRIPTION = (
        "DaSiWa RTX Upscaler & Refiner: A high-performance 2-pass processing node.\n"
        "1. Refine Pass: Runs at source resolution to clean up noise or blur using NVIDIA RTX VFX.\n"
        "2. Upscale Pass: Scales the image up to 4x using VSR or High Bitrate modes.\n"
        "Processes frame-by-frame to maintain low VRAM usage for video batches."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"description": "Input image or video batch to process."}),
                "denoise": ("BOOLEAN", {"default": False, "description": "Enable NVIDIA RTX Denoise pass to clean up image grain and noise."}),
                "denoise_quality": (QUALITY_LEVELS, {"default": "Ultra", "description": "Quality tier for the Denoise pass. Higher tiers use more GPU compute."}),
                "deblur": ("BOOLEAN", {"default": False, "description": "Enable NVIDIA RTX Deblur pass to reduce motion or focus blur."}),
                "deblur_quality": (QUALITY_LEVELS, {"default": "Ultra", "description": "Quality tier for the Deblur pass."}),
                "upscale": (UPSCALE_MODES, {"default": "VSR", "description": "Upscale Mode:\n- VSR: Standard AI upscaling.\n- High Bitrate: Optimized for heavily compressed/noisy sources.\n- Off: Skip scaling."}),
                "upscale_quality": (QUALITY_LEVELS, {"default": "Ultra", "description": "Quality tier for the Upscale pass."}),
                "resize_type": (RESIZE_TYPES, {"default": "Scale", "description": "Target Size Logic:\n- Same Size: Keep original dimensions.\n- Scale: Multiply resolution by 'scale'.\n- Keep Ratio: Match 'megapixels' while keeping original shape.\n- Preset Ratio: Match 'megapixels' but force to 'ratio_preset'.\n- Manual: Exact pixels."}),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05, "description": "Multiplier for resolution (used by 'Scale' type). 2.0 = 4x the total pixels."}),
                "megapixels": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 64.0, "step": 0.01, "description": "Target total area in millions of pixels (used by Ratio types)."}),
                "width": ("INT", {"default": 1920, "min": 64, "max": 8192, "step": 8, "description": "Target width (used by 'Manual')."}),
                "height": ("INT", {"default": 1080, "min": 64, "max": 8192, "step": 8, "description": "Target height (used by 'Manual')."}),
                "divisible_by": (DIVISIBLE_BY_VALUES, {"default": "32", "description": "Snaps final resolution to a multiple of this value. Use 32 for Video VAEs."}),
                "ratio_preset": (COMMON_RATIOS, {"default": "16:9", "description": "Forced aspect ratio (used by 'Preset Ratio')."}),
                "resize_method": (RESIZE_METHODS, {"default": "Center Crop (Fill)", "description": "Mismatch handling: 'Crop' fills the target area, 'Letterbox' fits inside with black bars."}),
                "device_id": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1, "description": "GPU Index for RTX VFX computation."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = "DaSiWa/Video"

    def execute(
        self,
        images,
        denoise,
        denoise_quality,
        deblur,
        deblur_quality,
        upscale,
        upscale_quality,
        resize_type,
        scale,
        megapixels,
        width,
        height,
        divisible_by,
        ratio_preset,
        resize_method,
        device_id,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("NVIDIA RTX VFX requires CUDA. No CUDA devices found.")

        batch_size, source_height, source_width, channels = images.shape
        if channels < 3:
            raise ValueError("NVIDIA RTX VFX requires RGB images with at least 3 channels.")

        alignment = int(divisible_by)
        upscale_enabled = upscale != "Off"

        # Calculate target dimensions
        if upscale_enabled:
            target_width, target_height = _target_size(
                source_width, source_height, resize_type, scale, megapixels, 
                width, height, alignment, ratio_preset
            )
        else:
            # If upscaling is off, we still respect the alignment for the source 
            # if refining is on, or just return source. 
            # For simplicity, if upscale is off, we use source dimensions.
            target_width, target_height = source_width, source_height

        VideoSuperRes = _import_vfx()
        cuda_device = torch.device(f"cuda:{device_id}")

        # Preallocate output tensor on the same device as input (usually CPU in ComfyUI)
        out_device = images.device
        out_dtype = images.dtype
        out = torch.empty(
            (batch_size, target_height, target_width, 3),
            device=out_device,
            dtype=out_dtype,
        )

        with torch.inference_mode():
            # Pass 1: Denoise
            with _maybe_vfx_effect(
                VideoSuperRes, denoise, "Denoise", denoise_quality, 
                device_id, source_width, source_height
            ) as denoise_effect:

                # Pass 2: Deblur
                with _maybe_vfx_effect(
                    VideoSuperRes, deblur, "Deblur", deblur_quality, 
                    device_id, source_width, source_height
                ) as deblur_effect:

                    # Pass 3: Upscale
                    with _maybe_vfx_effect(
                        VideoSuperRes, upscale_enabled, upscale, upscale_quality, 
                        device_id, target_width, target_height
                    ) as upscale_effect:
                        
                        for i in range(batch_size):
                            # Prep frame for CUDA
                            frame = (
                                images[i, :, :, :3]
                                .to(device=cuda_device, dtype=torch.float32)
                                .permute(2, 0, 1)
                                .contiguous()
                            )

                            # Apply Denoise
                            if denoise_effect:
                                res = denoise_effect.run(frame)
                                frame = torch.from_dlpack(res.image).clone()
                            
                            # Apply Deblur
                            if deblur_effect:
                                res = deblur_effect.run(frame)
                                frame = torch.from_dlpack(res.image).clone()
                            
                            # Apply Upscale
                            if upscale_enabled:
                                frame = _fit_frame_to_target_aspect(
                                    frame, target_width, target_height, resize_method
                                )
                                if upscale_effect:
                                    res = upscale_effect.run(frame)
                                    frame = torch.from_dlpack(res.image).clone()

                            # Convert back to HWC and store
                            enhanced = frame.permute(1, 2, 0).contiguous()
                            out[i].copy_(
                                enhanced.clamp(0.0, 1.0).to(device=out_device, dtype=out_dtype)
                            )
                            
                            del frame, enhanced

        return (out,)

NODE_CLASS_MAPPINGS = {
    "DaSiWa_RTX_UpscalerRefiner": DaSiWa_RTX_UpscalerRefiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DaSiWa_RTX_UpscalerRefiner": "DaSiWa RTX Upscaler & Refiner",
}