import math
import torch
import torch.nn.functional as F
import contextlib
from typing import Tuple

# --- Constants ---

QUALITY_LEVELS = ["Low", "Medium", "High", "Ultra"]
UPSCALE_MODES = ["Off", "VSR", "High Bitrate"]
RESIZE_TYPES = ["Keep Ratio", "Manual", "Preset Ratio", "Scale", "Same Size"]
DIVISIBLE_BY_VALUES = ["8", "16", "32", "64", "128"]
COMMON_RATIOS = ["1:1", "4:3", "3:2", "16:9", "21:9"]
RESIZE_METHODS = ["Center Crop (Fill)", "Letterbox (Fit)"]
MAX_CHUNK_OUTPUT_PIXELS = 1024 * 1024 * 16

# --- Helpers ---

def round_up(value: float, alignment: int) -> int:
    return int(math.ceil(float(value) / alignment) * alignment)

def round_nearest(value: float, alignment: int) -> int:
    return max(alignment, int(math.floor((float(value) / alignment) + 0.5) * alignment))

def _aligned_aspect_size(
    target_width: float,
    target_height: float,
    aspect: float,
    alignment: int,
) -> Tuple[int, int]:
    target_area = max(1.0, float(target_width) * float(target_height))
    base_width = max(float(alignment), float(target_width))
    base_height = max(float(alignment), float(target_height))

    def round_down(value: float) -> int:
        return max(alignment, int(math.floor(float(value) / alignment) * alignment))

    def round_nearest_aligned(value: float) -> int:
        return round_nearest(value, alignment)

    def round_up_aligned(value: float) -> int:
        return round_up(value, alignment)

    candidates = set()
    for width_rounder in (round_down, round_nearest_aligned, round_up_aligned):
        width_candidate = width_rounder(base_width)
        exact_height = width_candidate / aspect
        for height_rounder in (round_down, round_nearest_aligned, round_up_aligned):
            candidates.add((width_candidate, height_rounder(exact_height)))

    for height_rounder in (round_down, round_nearest_aligned, round_up_aligned):
        height_candidate = height_rounder(base_height)
        exact_width = height_candidate * aspect
        for width_rounder in (round_down, round_nearest_aligned, round_up_aligned):
            candidates.add((width_rounder(exact_width), height_candidate))

    def candidate_score(dims: Tuple[int, int]) -> Tuple[float, float, float]:
        w, h = dims
        area_error = abs((w * h) - target_area) / target_area
        ratio_error = abs((w / h) - aspect) / aspect
        distance_error = (abs(w - base_width) / base_width + abs(h - base_height) / base_height)
        return (ratio_error, area_error, distance_error)

    return min(candidates, key=candidate_score)

def compute_aligned_ratio_dims(ratio_preset: str, megapixels: float, alignment: int) -> Tuple[int, int]:
    try:
        w_part, h_part = map(float, ratio_preset.split(':'))
    except Exception:
        w_part, h_part = 16.0, 9.0
    
    target_area = max(0.01, float(megapixels)) * 1_000_000.0
    aspect = w_part / h_part
    
    h = math.sqrt(target_area / aspect)
    w = h * aspect
    
    return _aligned_aspect_size(w, h, aspect, alignment)

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

    def round_nearest_aligned(value: float) -> int:
        return round_nearest(value, alignment)

    def round_up_aligned(value: float) -> int:
        return round_up(value, alignment)

    candidates = set()
    for width_rounder in (round_down, round_nearest_aligned, round_up_aligned):
        width_candidate = width_rounder(base_width)
        exact_height = width_candidate / source_aspect
        for height_rounder in (round_down, round_nearest_aligned, round_up_aligned):
            candidates.add((width_candidate, height_rounder(exact_height)))

    for height_rounder in (round_down, round_nearest_aligned, round_up_aligned):
        height_candidate = height_rounder(base_height)
        exact_width = height_candidate * source_aspect
        for width_rounder in (round_down, round_nearest_aligned, round_up_aligned):
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
        return _aligned_aspect_size(
            float(source_width) * float(scale),
            float(source_height) * float(scale),
            float(source_width) / float(source_height),
            alignment,
        )
    if resize_type == "Keep Ratio":
        return _aligned_megapixel_size(source_width, source_height, megapixels, alignment)
    if resize_type == "Preset Ratio":
        return compute_aligned_ratio_dims(ratio_preset, megapixels, alignment)
    return (
        round_nearest(int(width), alignment),
        round_nearest(int(height), alignment),
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
        import nvvfx
    except ImportError:
        raise RuntimeError(
            "NVIDIA RTX VFX (nvvfx) module not found. "
            "Please ensure NVIDIA RTX Video SDK / Broadcast SDK is installed and the 'nvvfx' package is in your python path."
        )

    VideoSuperRes = getattr(nvvfx, "VideoSuperRes", None)
    if VideoSuperRes is None:
        try:
            from nvvfx import VideoSuperRes
        except ImportError as exc:
            raise RuntimeError("NVIDIA RTX VFX is installed, but VideoSuperRes is unavailable.") from exc

    effects = getattr(nvvfx, "effects", None)
    QualityLevel = getattr(effects, "QualityLevel", None)
    if QualityLevel is None:
        QualityLevel = getattr(VideoSuperRes, "QualityLevel", None)
    if QualityLevel is None:
        raise RuntimeError("NVIDIA RTX VFX VideoSuperRes quality levels are unavailable.")

    return VideoSuperRes, QualityLevel

def _resolve_quality_level(QualityLevel, mode: str, quality: str):
    attr_name = _quality_attr(mode, quality)
    q = quality.upper()
    candidates = [attr_name]
    if mode == "High Bitrate":
        candidates.extend([f"HIGH_BITRATE_{q}", f"HIGHBITRATE{q}"])
    elif mode == "VSR":
        candidates.append(q)

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if hasattr(QualityLevel, candidate):
            return getattr(QualityLevel, candidate), candidate

    available = ", ".join(name for name in dir(QualityLevel) if name.isupper()) or "none"
    raise ValueError(
        f"Invalid or unsupported NVIDIA RTX VFX quality level '{attr_name}'. "
        f"Available levels: {available}"
    )

def _create_vfx_effect(VideoSuperRes, q_level, device_index: int):
    attempts = (
        ((), {"quality": q_level, "device": device_index}),
        ((q_level,), {"device": device_index}),
        ((), {"quality": q_level}),
        ((q_level,), {}),
    )
    last_type_error = None
    for args, kwargs in attempts:
        try:
            return VideoSuperRes(*args, **kwargs)
        except TypeError as exc:
            last_type_error = exc

    raise last_type_error

def _close_vfx_effect(effect):
    for method_name in ("close", "destroy", "unload"):
        method = getattr(effect, method_name, None)
        if callable(method):
            method()
            return

def _run_vfx_effect(effect, frame, cuda_device):
    if not frame.is_contiguous():
        frame = frame.contiguous()

    # nvvfx owns its execution stream and may reuse output buffers. Synchronize
    # before and after each handoff, then clone the DLPack tensor so the next
    # effect/frame cannot overwrite data that has not been assembled yet.
    torch.cuda.current_stream(cuda_device).synchronize()
    res = effect.run(frame)
    torch.cuda.synchronize(cuda_device)
    return torch.from_dlpack(res.image).clone().contiguous()

@contextlib.contextmanager
def _maybe_vfx_effect(vfx_api, enabled, mode, quality, device_index, out_width, out_height):
    if not enabled:
        yield None
        return

    VideoSuperRes, QualityLevel = vfx_api
    try:
        q_level, attr_name = _resolve_quality_level(QualityLevel, mode, quality)
    except AttributeError:
        raise ValueError(f"Invalid quality level mapping: {_quality_attr(mode, quality)}")

    effect_cm = None
    effect = None
    try:
        effect = _create_vfx_effect(VideoSuperRes, q_level, device_index)
        if hasattr(effect, "__enter__") and hasattr(effect, "__exit__"):
            effect_cm = effect
            effect = effect_cm.__enter__()
        effect.output_width = int(out_width)
        effect.output_height = int(out_height)
        if hasattr(effect, "load"):
            effect.load()
    except Exception as exc:
        if effect_cm is not None:
            effect_cm.__exit__(None, None, None)
        elif effect is not None:
            _close_vfx_effect(effect)
        raise RuntimeError(f"Failed to create NVIDIA RTX VFX effect ({mode} {attr_name}): {exc}")

    try:
        yield effect
    finally:
        if effect_cm is not None:
            effect_cm.__exit__(None, None, None)
        elif effect is not None:
            _close_vfx_effect(effect)

class DaSiWa_RTX_UpscalerRefiner:
    DESCRIPTION = (
        "DaSiWa RTX Upscaler & Refiner: A high-performance 3-pass processing node.\n"
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
                "divisible_by": (DIVISIBLE_BY_VALUES, {"default": "8", "description": "Snaps final resolution to a multiple of this value. Use 8 to match RTX VSR; choose 32 only when a downstream video model requires it."}),
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
        has_effects = denoise or deblur or upscale_enabled

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

        if not has_effects:
            return (images[:, :, :, :3],)

        if upscale_enabled and target_width * target_height < source_width * source_height:
            print(
                "[DaSiWa RTX Upscaler & Refiner] Warning: target resolution "
                f"{target_width}x{target_height} is smaller than input "
                f"{source_width}x{source_height}. This will downscale the source and can look softer."
            )

        vfx_api = _import_vfx()
        cuda_device = torch.device(f"cuda:{device_id}")
        frames_per_chunk = max(1, MAX_CHUNK_OUTPUT_PIXELS // max(1, target_width * target_height))
        fit_to_target_aspect = upscale_enabled and resize_type in ("Manual", "Preset Ratio")

        # Preallocate output tensor on the same device as input (usually CPU in ComfyUI)
        out_device = images.device
        out_dtype = images.dtype
        out = torch.zeros(
            (batch_size, target_height, target_width, 3),
            device=out_device,
            dtype=out_dtype,
        )

        with torch.cuda.device(cuda_device), torch.inference_mode():
            # Pass 1: Denoise
            with _maybe_vfx_effect(
                vfx_api, denoise, "Denoise", denoise_quality,
                device_id, source_width, source_height
            ) as denoise_effect:

                # Pass 2: Deblur
                with _maybe_vfx_effect(
                    vfx_api, deblur, "Deblur", deblur_quality,
                    device_id, source_width, source_height
                ) as deblur_effect:

                    # Pass 3: Upscale
                    with _maybe_vfx_effect(
                        vfx_api, upscale_enabled, upscale, upscale_quality,
                        device_id, target_width, target_height
                    ) as upscale_effect:

                        for start in range(0, batch_size, frames_per_chunk):
                            end = min(start + frames_per_chunk, batch_size)
                            chunk = (
                                images[start:end, :, :, :3]
                                .to(device=cuda_device, dtype=torch.float32, non_blocking=True)
                                .permute(0, 3, 1, 2)
                                .contiguous()
                            )
                            for local_index in range(end - start):
                                global_index = start + local_index
                                frame = chunk[local_index]

                                if denoise_effect:
                                    frame = _run_vfx_effect(denoise_effect, frame, cuda_device)

                                if deblur_effect:
                                    frame = _run_vfx_effect(deblur_effect, frame, cuda_device)

                                if upscale_enabled:
                                    if fit_to_target_aspect:
                                        frame = _fit_frame_to_target_aspect(
                                            frame, target_width, target_height, resize_method
                                        )
                                    if upscale_effect:
                                        frame = _run_vfx_effect(upscale_effect, frame, cuda_device)

                                output_frame = (
                                    frame.permute(1, 2, 0)
                                    .contiguous()
                                    .clamp(0.0, 1.0)
                                    .to(device=out_device, dtype=out_dtype, non_blocking=False)
                                )
                                out[global_index].copy_(output_frame, non_blocking=False)

                            torch.cuda.synchronize(cuda_device)
                            if out_device.type == "cuda":
                                torch.cuda.synchronize(out_device)
                            del chunk

        return (out,)

NODE_CLASS_MAPPINGS = {
    "DaSiWa_RTX_UpscalerRefiner": DaSiWa_RTX_UpscalerRefiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DaSiWa_RTX_UpscalerRefiner": "DaSiWa RTX Upscaler & Refiner",
}
