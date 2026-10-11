import gc
import math
import os
import torch
import torch.nn.functional as F
import contextlib
import inspect
from importlib import metadata
from typing import Any, Optional, Tuple

import folder_paths
try:
    from .helper_batch_output import (
        allocate_cpu_output, can_allocate_in_ram, tensor_nbytes, available_ram_bytes,
    )
    from .helper_logging import log_dasiwa
except ImportError:
    from helper_batch_output import (
        allocate_cpu_output, can_allocate_in_ram, tensor_nbytes, available_ram_bytes,
    )
    from helper_logging import log_dasiwa

# Optional: ComfyUI model_management for soft-empty-cache
_model_management = None
try:
    from comfy import model_management as _model_management
except Exception:
    pass

_memory_management = None
try:
    from comfy import memory_management as _memory_management
except ImportError:
    pass

# --- Constants ---

QUALITY_LEVELS = ["Low", "Medium", "High", "Ultra"]
UPSCALE_MODES = ["Off", "VSR", "High Bitrate"]
RESIZE_TYPES = ["Keep Ratio", "Manual", "Preset Ratio", "Scale", "Same Size"]
DIVISIBLE_BY_VALUES = ["8", "16", "32", "64", "128"]
COMMON_RATIOS = ["1:1", "4:3", "3:2", "16:9", "21:9"]
RESIZE_METHODS = ["Center Crop (Fill)", "Letterbox (Fit)"]
MAX_CHUNK_OUTPUT_PIXELS = 1024 * 1024 * 16
# Ceiling for output kept on the GPU. Sized for a 32 GB card: a large batch can
# stay in VRAM when output plus pipeline reserve fits with 15% headroom,
# instead of being bounced to RAM just because the old 8 GiB cap was hit.
MAX_GPU_OUTPUT_BYTES = 24 * 1024 * 1024 * 1024

# --- Helpers ---

def _dasiwa_rtx_parameters(cls):
    try:
        return inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return {}


def _dasiwa_rtx_capabilities():
    """API-only probe: never instantiate effects or initialize CUDA."""
    capabilities: dict[str, Any] = dict(version="unavailable", strength=False, encoding=False,
                        hdr=False, streaming=[], image_encoding=None, true_hdr=None)
    try:
        import nvvfx
        try:
            capabilities["version"] = metadata.version("nvidia-vfx")
        except metadata.PackageNotFoundError:
            capabilities["version"] = "unknown"
        vsr = getattr(nvvfx, "VideoSuperRes", None)
        params = _dasiwa_rtx_parameters(vsr)
        encoding = getattr(nvvfx, "ImageEncoding", None)
        quality = getattr(getattr(nvvfx, "effects", None), "QualityLevel", None)
        if quality is None:
            quality = getattr(vsr, "QualityLevel", None)
        hdr = getattr(nvvfx, "TrueHDR", None)
        hdr_params = _dasiwa_rtx_parameters(hdr)
        capabilities.update(
            strength="strength" in params,
            encoding=("image_encoding" in params and encoding is not None
                      and hasattr(encoding, "RGB8") and hasattr(encoding, "RGB10A2")),
            image_encoding=encoding,
            hdr=(callable(hdr) and all(name in hdr_params for name in
                 ("contrast", "saturation", "middle_gray", "luminance", "debanding_off", "device"))),
            true_hdr=hdr,
            streaming=[label for label, attr in
                       (("Streaming Medium", "STREAMING_MEDIUM"), ("Streaming Ultra", "STREAMING_ULTRA"))
                       if quality is not None and hasattr(quality, attr)],
        )
    except Exception:
        # Broken/missing optional SDK must not break pack import or schema.
        pass
    return capabilities


def _dasiwa_rtx_pack_rgb10a2(rgb):
    """Normalized CHW RGB -> contiguous HxW uint32 R10G10B10A2."""
    values = (rgb[:3].clamp(0, 1) * 1023).round().to(torch.int32)
    return (values[0] | (values[1] << 10) | (values[2] << 20) | -1073741824).view(torch.uint32).contiguous()


def _dasiwa_rtx_unpack_rgb10a2(packed):
    """Packed words -> CHW float32; retain PQ signal values for HDR."""
    words = packed.view(torch.int32)
    return torch.stack([(words >> shift) & 1023 for shift in (0, 10, 20)]).float().div_(1023).contiguous()


def _dasiwa_rtx_integrity(input_frame, output_frame, stage):
    """Bounded diagnostic heuristics, not a proof of image corruption."""
    def sample(frame):
        h, w = frame.shape[-2:]
        return frame[:, ::max(1, math.ceil(h / 64)), ::max(1, math.ceil(w / 64))]
    source, result = sample(input_frame), sample(output_frame)
    if not torch.isfinite(result).all():
        raise RuntimeError(f"RTX channel integrity diagnosis ({stage}): sampled NaN/Inf output.")
    before = source.abs().mean(dim=(1, 2))
    after = result.abs().mean(dim=(1, 2))
    # Only flag a strong formerly populated channel disappearing while another survives.
    collapse = (before > 0.05) & (after < before * 0.001)
    if bool(collapse.any()) and bool((after > 0.05).any()):
        raise RuntimeError(f"RTX channel integrity diagnosis ({stage}): suspected channel collapse; "
                           "this heuristic may flag legitimate color changes. Disable the opt-in check to compare.")


def _dasiwa_rtx_run_pass(effect, frame, cuda_device, packed=False, check=False, stage="VSR", hdr=False, terminal=False):
    sdk_input = _dasiwa_rtx_pack_rgb10a2(frame) if packed else frame
    owned = (_dasiwa_rtx_borrow_output(effect, sdk_input, cuda_device) if terminal
             else _run_vfx_effect(effect, sdk_input, cuda_device))
    result = _dasiwa_rtx_unpack_rgb10a2(owned) if packed or hdr else owned
    if check:
        if hdr:
            # Tone mapping is not SDR channel collapse; check finite values only.
            sample = result[:, ::max(1, math.ceil(result.shape[1] / 64)),
                            ::max(1, math.ceil(result.shape[2] / 64))]
            if not torch.isfinite(sample).all():
                raise RuntimeError("RTX channel integrity diagnosis (HDR): sampled NaN/Inf output.")
        else:
            _dasiwa_rtx_integrity(frame, result, stage)
    return result


@contextlib.contextmanager
def _dasiwa_rtx_execution_stream(vfx_api, capabilities, cuda_device):
    """Isolate the verified 0.2 native-wrapper non-default-stream limitation.

    Binding before load avoids SDK error -12 but did not preserve distinct
    frames reliably in multipass HDR. Use its tested default-stream path, with
    stream-local handoff, rather than retrying corrupt output or globally syncing.
    Other SDKs retain the caller's current stream.
    """
    cls = vfx_api[0]
    if (capabilities["version"] == "0.2.0.0" and
            getattr(cls, "__module__", "") == "nvvfx.effects.video_super_res" and
            "stream_ptr" in _dasiwa_rtx_parameters(cls.run)):
        caller = torch.cuda.current_stream(cuda_device)
        default = torch.cuda.default_stream(cuda_device)
        if caller != default:
            caller.synchronize()
            log_dasiwa("RTX Upscaler & Refiner", "SDK 0.2 native effects use the verified default CUDA stream; "
                       "performing stream-local input/output handoff.")
            with torch.cuda.stream(default):
                try:
                    yield
                finally:
                    default.synchronize()
            return
    yield


@contextlib.contextmanager
def _dasiwa_rtx_hdr_effect(capabilities, enabled, device_index, contrast, saturation,
                           middle_gray, luminance, debanding):
    if not enabled:
        yield None
        return
    effect = capabilities["true_hdr"](contrast=contrast, saturation=saturation,
        middle_gray=middle_gray, luminance=luminance, debanding_off=int(not debanding), device=device_index)
    try:
        # TrueHDR pins dimensions on run and loads lazily. No VSR resize fields.
        yield effect
    finally:
        _close_vfx_effect(effect)


def round_up(value: float, alignment: int) -> int:
    return int(math.ceil(float(value) / alignment) * alignment)

def round_nearest(value: float, alignment: int) -> int:
    return max(alignment, int(math.floor((float(value) / alignment) + 0.5) * alignment))


def _projected_output_bytes(batch_size: int, width: int, height: int, dtype: torch.dtype, channels: int = 3) -> int:
    return tensor_nbytes((batch_size, height, width, channels), dtype)


def _temporary_output_directory() -> str:
    directory = folder_paths.get_temp_directory()
    os.makedirs(directory, exist_ok=True)
    return directory


def _cleanup_cuda(soft_empty_cache: bool, device=None) -> None:
    """Explicit/pressure cleanup on the selected CUDA device only."""
    gc.collect()
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            if soft_empty_cache and _model_management and hasattr(_model_management, "soft_empty_cache"):
                _model_management.soft_empty_cache()
            else:
                torch.cuda.empty_cache()


def _can_fit_in_vram(required_bytes: int, device: torch.device) -> bool:
    if not torch.cuda.is_available():
        return False
    free_bytes, _ = torch.cuda.mem_get_info(device)
    return required_bytes <= max(0, int(free_bytes * 0.85))


def _dasiwa_rtx_working_reserve(source_width, source_height, target_width, target_height,
                               resize_method, effect_count, packed=False, hdr=False):
    """Conservative heuristic, not measured/guaranteed opaque SDK workspace."""
    sw, sh = source_width, source_height
    if resize_method == "Letterbox (Fit)":
        rw, rh = _aspect_ratio_parts(target_width, target_height)
        scale = max(math.ceil(sw / rw), math.ceil(sh / rh))
        sw, sh = max(sw, scale * rw), max(sh, scale * rh)
    largest = max(sw * sh, target_width * target_height)
    staging = source_width * source_height * 3 * 4
    intermediates = largest * (64 if packed or hdr else 48)
    sdk_estimate = effect_count * (512 * 1024 ** 2 + largest * 16)
    return staging + intermediates + sdk_estimate


def _dasiwa_rtx_free_memory(required_bytes, device):
    """Request only enough space on this device; never globally unload models."""
    if _model_management is None or not hasattr(_model_management, "free_memory"):
        return False
    with torch.cuda.device(device):
        _model_management.free_memory(math.ceil(required_bytes / 0.85), device)
    return True


def _dasiwa_rtx_ram_budget(shape, dtype):
    """Reserve host headroom plus bounded frame staging, not a fraction of total RAM.

    Existing input/model residency is already excluded by MemAvailable. This
    protects the OS and transient host copies without reserving another 8 GiB
    simply because the machine has a large installed RAM capacity.
    """
    available = available_ram_bytes()
    system_reserve = max(1024 ** 3, min(2 * 1024 ** 3, available // 10))
    frame_staging = 2 * max(tensor_nbytes(shape[1:], torch.float32), tensor_nbytes(shape[1:], dtype))
    reserve = system_reserve + frame_staging
    return available, reserve, max(0, available - reserve)


def _dasiwa_rtx_reclaim_ram(shape, dtype, required_bytes):
    """Use native pressure eviction, never reset the executing prompt or offload all GPUs."""
    staging = 2 * max(tensor_nbytes(shape[1:], torch.float32), tensor_nbytes(shape[1:], dtype))
    base = required_bytes + staging
    # Solve the 10%-of-available reserve at the requested capacity, not at
    # today's low availability; otherwise successful eviction can still undershoot.
    target = base + max(1024 ** 3, min(2 * 1024 ** 3, math.ceil(base / 9)))
    release_cache = getattr(_memory_management, "extra_ram_release", None)
    free_pins = getattr(_model_management, "free_pins", None)
    cleanup_models = getattr(_model_management, "cleanup_models", None)
    gc.collect()
    available, reserve, usable = _dasiwa_rtx_ram_budget(shape, dtype)
    for stage in ("inactive", "pins", "active"):
        if required_bytes <= usable:
            break
        if stage == "pins":
            if free_pins is not None:
                pin_options = {"loaded": True} if "loaded" in inspect.signature(free_pins).parameters else {}
                free_pins(max(0, target - available), **pin_options)
        elif release_cache is not None:
            release_cache(target, free_active=stage == "active")
        gc.collect()
        if cleanup_models is not None:
            cleanup_models()
        available, reserve, usable = _dasiwa_rtx_ram_budget(shape, dtype)
    return available, reserve, usable


def _allocate_output_tensor(shape, dtype, device, soft_empty_cache=False, allow_mmap=False,
                            auto_unload_models=True, working_reserve_bytes=0, compute_device=None):
    """Full output allocation: GPU (24 GiB cap + working reserve), RAM, opt-in mmap.

    CUDA empty reserves its full size immediately. CPU empty avoids zero-fill,
    but does not make writing an oversized batch safe.
    """
    required_bytes = tensor_nbytes(shape, dtype)
    compute_device = compute_device or device
    if soft_empty_cache:
        _cleanup_cuda(True, compute_device)
    gpu_eligible = device.type == "cuda" and required_bytes <= MAX_GPU_OUTPUT_BYTES
    # An input/output GPU can differ from the SDK compute GPU. Budget each
    # separately instead of charging workspace on the wrong card.
    if compute_device != device and working_reserve_bytes and compute_device.type == "cuda":
        if not _can_fit_in_vram(working_reserve_bytes, compute_device):
            if auto_unload_models and (device.type != "cuda" or gpu_eligible):
                _dasiwa_rtx_free_memory(working_reserve_bytes, compute_device)
            if not _can_fit_in_vram(working_reserve_bytes, compute_device):
                raise RuntimeError("Insufficient CUDA memory for estimated RTX working reserve. "
                                   "Free GPU memory or reduce source/target resolution or enabled passes.")
    budget = required_bytes + (working_reserve_bytes if device == compute_device else 0)
    if gpu_eligible:
        if _can_fit_in_vram(budget, device):
            return torch.empty(shape, device=device, dtype=dtype), None
        # Do not evict models for an output budget that cannot fit even on
        # an otherwise empty card. Workspace-only reclamation remains separate.
        _, total_bytes = torch.cuda.mem_get_info(device)
        if (auto_unload_models and budget <= int(total_bytes * 0.85)
                and _dasiwa_rtx_free_memory(budget, device)):
            if _can_fit_in_vram(budget, device):
                return torch.empty(shape, device=device, dtype=dtype), None
    # Preserve input-device output policy. Above-cap GPU output must not
    # trigger eviction, even if its compute workspace will need memory.
    if working_reserve_bytes and compute_device == device and compute_device.type == "cuda":
        if not _can_fit_in_vram(working_reserve_bytes, compute_device):
            if auto_unload_models and (device.type != "cuda" or gpu_eligible):
                _dasiwa_rtx_free_memory(working_reserve_bytes, compute_device)
            if not _can_fit_in_vram(working_reserve_bytes, compute_device):
                raise RuntimeError("Insufficient CUDA memory for estimated RTX working reserve. "
                                   "Free GPU memory or reduce source/target resolution or enabled passes.")
    available, reserve, usable = _dasiwa_rtx_ram_budget(shape, dtype)
    if required_bytes > usable and auto_unload_models:
        available, reserve, usable = _dasiwa_rtx_reclaim_ram(shape, dtype, required_bytes)
    log_dasiwa("RTX Upscaler & Refiner",
               f"RAM output budget: required={required_bytes / 1024 ** 3:.2f} GiB, "
               f"available={available / 1024 ** 3:.2f} GiB, "
               f"reserve={reserve / 1024 ** 3:.2f} GiB, usable={usable / 1024 ** 3:.2f} GiB.")
    if required_bytes <= usable:
        return torch.empty(shape, dtype=dtype), None
    if not allow_mmap:
        raise RuntimeError(
            f"RTX output requires {required_bytes / 1024 ** 3:.2f} GiB; "
            f"available RAM={available / 1024 ** 3:.2f} GiB, "
            f"reserve={reserve / 1024 ** 3:.2f} GiB, usable={usable / 1024 ** 3:.2f} GiB. "
            "Reduce frame count/target resolution "
            "or explicitly enable use_mmap with sufficient temp disk space.")
    return allocate_cpu_output(shape, dtype, _temporary_output_directory(), force_mmap=True)

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
    if mode in ("Streaming Medium", "Streaming Ultra"):
        return mode.upper().replace(" ", "_")
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
    ratio_width, ratio_height = _aspect_ratio_parts(target_width, target_height)

    if _same_aspect(source_width, source_height, target_width, target_height):
        return frame.contiguous()

    if resize_method == "Center Crop (Fill)":
        ratio_scale = min(int(source_width) // ratio_width, int(source_height) // ratio_height)
        if ratio_scale > 0:
            crop_width = ratio_scale * ratio_width
            crop_height = ratio_scale * ratio_height
        elif source_aspect > target_aspect:
            crop_width = max(1, min(int(source_width), int(round(float(source_height) * target_aspect))))
            crop_height = int(source_height)
        else:
            crop_width = int(source_width)
            crop_height = max(1, min(int(source_height), int(round(float(source_width) / target_aspect))))
        crop_x = max(0, (int(source_width) - crop_width) // 2)
        crop_y = max(0, (int(source_height) - crop_height) // 2)
        return frame[:, crop_y:crop_y + crop_height, crop_x:crop_x + crop_width].contiguous()

    # Letterbox (Fit)
    ratio_scale = max(
        math.ceil(int(source_width) / ratio_width),
        math.ceil(int(source_height) / ratio_height),
    )
    padded_width = max(int(source_width), ratio_scale * ratio_width)
    padded_height = max(int(source_height), ratio_scale * ratio_height)
    pad_width = padded_width - int(source_width)
    pad_height = padded_height - int(source_height)
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    return F.pad(frame, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0).contiguous()

def _aspect_ratio_parts(width: int, height: int) -> Tuple[int, int]:
    divisor = math.gcd(int(width), int(height))
    if divisor <= 0:
        return max(1, int(width)), max(1, int(height))
    return max(1, int(width) // divisor), max(1, int(height) // divisor)


def _same_aspect(source_width: int, source_height: int, target_width: int, target_height: int) -> bool:
    return int(source_width) * int(target_height) == int(target_width) * int(source_height)


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

def _create_vfx_effect(VideoSuperRes, q_level, device_index: int, strength=1.0, image_encoding="8-bit RGB"):
    params = _dasiwa_rtx_parameters(VideoSuperRes)
    extras = {}
    if "strength" in params:
        extras["strength"] = strength
    elif strength != 1.0:
        raise RuntimeError("RTX SDK strength is unavailable.")
    if "image_encoding" in params:
        capabilities = _dasiwa_rtx_capabilities()
        enum = capabilities["image_encoding"]
        if enum is not None:
            extras["image_encoding"] = getattr(enum, "RGB10A2" if image_encoding != "8-bit RGB" else "RGB8")
    if extras:
        # Concrete supported signatures: never silently retry without requested settings/device.
        return VideoSuperRes(quality=q_level, device=device_index, **extras)
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

def _lossless_fp16_image_batch(images, chunk_frames):
    """Check exactness in bounded windows before making a full-size FP16 copy."""
    if images.dtype != torch.float32:
        return images
    half_bytes = images.numel() * 2
    # Capacity BEFORE exactness temporaries, not just before the full copy.
    check_bytes = min(images.numel(), 1024 * 1024 // 8) * 8
    needed = half_bytes + check_bytes
    if images.device.type == "cuda":
        if not _can_fit_in_vram(needed, images.device):
            return images
    elif not can_allocate_in_ram(needed):
        return images
    # Bounded spatial slices also work for noncontiguous RGB views of RGBA;
    # flattening the full batch could allocate a complete extra tensor.
    pixels = max(1, 1024 * 1024 // (8 * images.shape[-1]))
    columns = min(images.shape[2], pixels)
    rows = max(1, pixels // max(1, columns))
    for frame in images:
        for y in range(0, frame.shape[0], rows):
            for x in range(0, frame.shape[1], max(1, columns)):
                part = frame[y:y + rows, x:x + columns]
                if not torch.equal(part.to(torch.float16).to(torch.float32), part):
                    return images
    return images.to(torch.float16)


def _close_vfx_effect(effect):
    for method_name in ("close", "destroy", "unload"):
        method = getattr(effect, method_name, None)
        if callable(method):
            method()
            return

def _dasiwa_rtx_borrow_output(effect, frame, cuda_device):
    frame = frame.contiguous()
    stream = torch.cuda.current_stream(cuda_device)
    if "stream_ptr" in _dasiwa_rtx_parameters(effect.run):
        # Keep SDK blocking default; Torch copies use the same stream.
        result = effect.run(frame, stream_ptr=stream.cuda_stream)
    else:
        stream.synchronize()
        result = effect.run(frame)
        torch.cuda.synchronize(cuda_device)
    return torch.from_dlpack(result.image)


def _run_vfx_effect(effect, frame, cuda_device):
    borrowed = _dasiwa_rtx_borrow_output(effect, frame, cuda_device)
    owned = borrowed.clone().contiguous()
    # Complete ownership before another SDK context/run/close.
    torch.cuda.current_stream(cuda_device).synchronize()
    return owned


def _run_vfx_effect_into(effect, frame, row, cuda_device):
    borrowed = _dasiwa_rtx_borrow_output(effect, frame, cuda_device)
    row.copy_(borrowed.permute(1, 2, 0))
    row.clamp_(0.0, 1.0)
    torch.cuda.current_stream(cuda_device).synchronize()
    if row.device.type == "cuda" and row.device != cuda_device:
        torch.cuda.current_stream(row.device).synchronize()


@contextlib.contextmanager
def _maybe_vfx_effect(vfx_api, enabled, mode, quality, device_index, out_width, out_height, strength=1.0, image_encoding="8-bit RGB"):
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
        effect = _create_vfx_effect(VideoSuperRes, q_level, device_index, strength, image_encoding)
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
        "DaSiWa RTX Upscaler & Refiner: SDK-aware image/video enhancement.\n"
        "Optional Denoise and Deblur at source resolution, then VSR/High Bitrate/Streaming upscale.\n"
        "Optional final TrueHDR returns BT.2020/PQ values; IMAGE carries no HDR metadata.\n"
        "Frame-by-frame processing with one complete IMAGE output batch."
    )

    @classmethod
    def INPUT_TYPES(cls):
        capabilities = _dasiwa_rtx_capabilities()
        schema = {
            "required": {
                "images": ("IMAGE", {"description": "IMAGE batch of still images or decoded video frames, not a VIDEO object. No audio, FPS or timestamps are passed through. Frame order/count stay unchanged; RGB output unless preserve_alpha is enabled."}),
                "denoise": ("BOOLEAN", {"default": False, "description": "Enable NVIDIA RTX Denoise pass to clean up image grain and noise."}),
                "denoise_quality": (QUALITY_LEVELS, {"default": "Ultra", "description": "Denoise SDK tier: Low favors speed, Medium is the middle tier, High favors quality, Ultra is the highest tier. Active only when denoise is on; actual cost/results depend on source, GPU and SDK."}),
                "deblur": ("BOOLEAN", {"default": False, "description": "Enable NVIDIA RTX Deblur pass to reduce motion or focus blur."}),
                "deblur_quality": (QUALITY_LEVELS, {"default": "Ultra", "description": "Deblur SDK tier: Low favors speed, Medium is the middle tier, High favors quality, Ultra is the highest tier. Active only when deblur is on; no guaranteed detail recovery."}),
                "upscale": (UPSCALE_MODES + capabilities["streaming"], {"default": "VSR", "description": "Upscale mode:\n- Off: Skip upscale and keep source dimensions; resize controls are ignored.\n- VSR: Standard reconstruction for typical compressed sources.\n- High Bitrate: SDK high-bitrate reconstruction for clean/high-bitrate sources.\n- Streaming Medium / Streaming Ultra: SDK streaming quality presets when available; ignore upscale_quality. These do not stream VIDEO input/output or reduce full IMAGE batch storage."}),
                "upscale_quality": (QUALITY_LEVELS, {"default": "Ultra", "description": "VSR / High Bitrate SDK tier: Low favors speed, Medium is the middle tier, High favors quality, Ultra is the highest tier. Ignored for Off and Streaming Medium / Streaming Ultra (preset defines the tier)."}),
                "resize_type": (RESIZE_TYPES, {"default": "Scale", "description": "Active only when upscale is not Off. Same Size: exact source dimensions (ignores alignment). Scale: multiply width/height by scale. Keep Ratio: target megapixels and source aspect. Preset Ratio: target megapixels and ratio_preset. Manual: width/height. Other resize modes align by divisible_by; megapixels/aspect can be approximate."}),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05, "description": "Multiplier for resolution (used by 'Scale' type). 2.0 = 4x the total pixels."}),
                "megapixels": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 64.0, "step": 0.01, "description": "Target total area in millions of pixels (used by Ratio types)."}),
                "width": ("INT", {"default": 1920, "min": 64, "max": 8192, "step": 8, "description": "Target width in pixels, active only for Manual with upscale enabled; divisible_by may adjust it."}),
                "height": ("INT", {"default": 1080, "min": 64, "max": 8192, "step": 8, "description": "Target height in pixels, active only for Manual with upscale enabled; divisible_by may adjust it."}),
                "divisible_by": (DIVISIBLE_BY_VALUES, {"default": "8", "description": "For Scale, Keep Ratio, Preset Ratio and Manual with upscale enabled, align dimensions to the selected pixel multiple. 8 matches typical RTX sizes; 16, 32, 64 and 128 select progressively coarser pixel grids and may change requested dimensions. Use larger multiples only if a downstream model requires them. Ignored for Same Size or when upscale is Off."}),
                "ratio_preset": (COMMON_RATIOS, {"default": "16:9", "description": "Width:height shape, active only for Preset Ratio with upscale enabled: 1:1 square, 4:3 classic, 3:2 photo, 16:9 widescreen, 21:9 ultrawide. Alignment can slightly approximate the ratio."}),
                "resize_method": (RESIZE_METHODS, {"default": "Center Crop (Fill)", "description": "When upscale is enabled and source/target aspect differs: Center Crop (Fill) removes centered edges to fill the frame; Letterbox (Fit) retains the whole image and adds black padding. No effect for matching aspects or upscale Off."}),
                "device_id": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1, "description": "CUDA device index for NVIDIA RTX computation (0 is the first GPU). Must identify a supported NVIDIA device; output placement follows the allocator/input device, not necessarily this compute device."}),
            },
            "optional": {
                "empty_cache": ("BOOLEAN", {"default": False, "description": "Run a lightweight GC + empty_cache before allocation; also calls model_management.soft_empty_cache() on the selected device (does not unload models)."}),
                "use_mmap": ("BOOLEAN", {"default": False, "description": "OFF (default): never use disk; fail clearly if neither budgeted VRAM nor available RAM can hold the complete output. **WARN:** enable this ONLY for very long video batches that genuinely exceed available RAM; when on, disk is used as the last tier of the VRAM -> RAM -> disk chain and a multi-giB .mmap temp file is written to your temp drive for the whole run."}),
                "auto_unload_models": ("BOOLEAN", {"default": True, "description": "Reclaim targeted GPU memory under pressure. Before RAM output allocation, reclaim native execution caches and model pins, then re-check. Never reset the running executor or globally offload GPU models into RAM. Native cache eviction requires ComfyUI RAM-pressure cache support."}),
                "chunking": ("BOOLEAN", {"default": True, "description": "Process frames in bounded internal chunks; output remains one standard IMAGE batch."}),
                "lossless_fp16": ("BOOLEAN", {"default": True, "description": "Use FP16 for output storage only when every FP32 pixel round-trips exactly; otherwise keep FP32. May provide no saving for typical VSR output."}),
                "chunk_frames": ("INT", {"default": 16, "min": 1, "max": 1024, "step": 1, "description": "Maximum frames per internal processing chunk when chunking is on; also used by the optional FP16 storage check. Does not limit complete input/output batch allocation."}),
            },
        }

        optional = schema["optional"]
        optional["image_encoding"] = (["8-bit RGB", "10-bit RGB (RGB10A2)"] if capabilities["encoding"] else ["8-bit RGB"], {
            "default": "8-bit RGB", "dasiwa_unavailable": not capabilities["encoding"],
            "description": "8-bit RGB: default SDK encoding. 10-bit RGB (RGB10A2): packed 10-bit SDR SDK processing when supported, unpacked to FP32. Not HDR, restored source detail, or smaller IMAGE storage."})
        for stage in ("denoise", "deblur", "upscale"):
            optional[f"{stage}_strength"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                "step": 0.01, "dasiwa_unavailable": not capabilities["strength"],
                "description": f"{stage.capitalize()} SDK strength from 0 to 1; 1.0 retains the SDK default. Active only when the {stage} stage is enabled; changing it does not enable a skipped stage."})
        optional["preserve_alpha"] = ("BOOLEAN", {"default": False, "description": "Keep input alpha separately, applying the same crop/letterbox/resize; no SDK filtering."})
        optional["channel_integrity_check"] = ("BOOLEAN", {"default": False, "description": "Sampled NaN/Inf and channel-collapse diagnosis before clamp. Heuristic: legitimate color changes can trigger it."})
        optional["hdr_enabled"] = ("BOOLEAN", {"default": False, "dasiwa_unavailable": not capabilities["hdr"],
            "description": "Final TrueHDR pass: normalized BT.2020/PQ HDR10 values, NOT SDR. IMAGE carries no HDR metadata; use a verified HDR encoder."})
        for name, default, lo, hi in (("contrast", 100, 0, 200), ("saturation", 100, 0, 200),
                ("middle_gray", 50, 10, 100), ("luminance", 650, 400, 2000)):
            optional[f"hdr_{name}"] = ("INT", {"default": default, "min": lo, "max": hi, "step": 1,
                "dasiwa_unavailable": not capabilities["hdr"], "description": {
                    "contrast": "TrueHDR contrast control (0-200, default 100); affects light/dark separation. Active only with hdr_enabled.",
                    "saturation": "TrueHDR color saturation control (0-200, default 100); affects color intensity. Active only with hdr_enabled.",
                    "middle_gray": "TrueHDR middle-gray control (10-100, default 50); sets the midtone brightness reference, not peak luminance. Active only with hdr_enabled.",
                    "luminance": "TrueHDR target peak luminance in nits (400-2000, default 650); choose for the HDR display/export target. Active only with hdr_enabled.",
                }[name]})
        optional["hdr_debanding"] = ("BOOLEAN", {"default": True, "dasiwa_unavailable": not capabilities["hdr"],
            "description": "On: enable TrueHDR DL-Debander to reduce visible banding; Off: disable it. Active only with hdr_enabled; on is the NVIDIA default."})
        # Native ComfyUI hover help, also retain descriptions for extension consumers.
        for inputs in schema.values():
            for _, options in inputs.values():
                options["tooltip"] = options["description"]
        return schema

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = "DaSiWa/Video"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # ComfyUI may call this legacy validation hook before execution.
        # The RTX node validates tensor shape, CUDA availability, and VFX
        # support in execute where the real input values are available.
        return True

    def validate_inputs(self, *args, **kwargs):
        # Newer ComfyUI builds can pass positional metadata through the
        # internal validation path. Accept all signatures and defer concrete
        # checks to execute for compatibility across core versions.
        return True

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
        empty_cache=False,
        use_mmap=False,
        auto_unload_models=True,
        chunking=True,
        lossless_fp16=True,
        chunk_frames=16,
        image_encoding="8-bit RGB",
        denoise_strength=1.0,
        deblur_strength=1.0,
        upscale_strength=1.0,
        preserve_alpha=False,
        channel_integrity_check=False,
        hdr_enabled=False,
        hdr_contrast=100,
        hdr_saturation=100,
        hdr_middle_gray=50,
        hdr_luminance=650,
        hdr_debanding=True,
    ):
        capabilities = _dasiwa_rtx_capabilities()
        if image_encoding not in ("8-bit RGB", "10-bit RGB (RGB10A2)"):
            raise ValueError(f"Unsupported RTX image_encoding: {image_encoding}")
        packed = image_encoding != "8-bit RGB"
        if packed and not capabilities["encoding"]:
            raise RuntimeError("RTX SDK RGB10A2 processing is unavailable; restart after installing a supported SDK.")
        if hdr_enabled and not capabilities["hdr"]:
            raise RuntimeError("RTX SDK TrueHDR is unavailable; restart after installing a supported SDK.")
        if upscale.startswith("Streaming") and upscale not in capabilities["streaming"]:
            raise RuntimeError(f"RTX SDK {upscale} is unavailable.")
        for strength in (denoise_strength, deblur_strength, upscale_strength):
            if not math.isfinite(strength) or not 0 <= strength <= 1:
                raise ValueError("RTX strengths must be finite and between 0 and 1.")
            if strength != 1.0 and not capabilities["strength"]:
                raise RuntimeError("RTX SDK strength control is unavailable.")
        if not torch.cuda.is_available():
            raise RuntimeError("NVIDIA RTX VFX requires CUDA. No CUDA devices found.")

        batch_size, source_height, source_width, channels = images.shape
        if channels < 3:
            raise ValueError("NVIDIA RTX VFX requires RGB images with at least 3 channels.")

        alignment = int(divisible_by)
        upscale_enabled = upscale != "Off"
        has_effects = denoise or deblur or upscale_enabled or hdr_enabled
        output_channels = 4 if preserve_alpha and channels >= 4 else 3
        enhanced = packed or hdr_enabled or preserve_alpha or channel_integrity_check

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

        if not isinstance(chunk_frames, int) or chunk_frames < 1:
            raise ValueError("chunk_frames must be at least 1")
        if not has_effects:
            frames = images[:, :, :, :output_channels]
            return (_lossless_fp16_image_batch(frames, chunk_frames) if lossless_fp16 else frames,)

        if upscale_enabled and target_width * target_height < source_width * source_height:
            log_dasiwa(
                "RTX Upscaler & Refiner",
                "Warning: target resolution "
                f"{target_width}x{target_height} is smaller than input "
                f"{source_width}x{source_height}. This will downscale the source and can look softer.",
            )

        vfx_api = _import_vfx()
        cuda_device = torch.device(f"cuda:{device_id}")
        fit_to_target_aspect = upscale_enabled and not _same_aspect(
            source_width, source_height, target_width, target_height
        )

        out_device = images.device
        out_dtype = torch.float32 if packed or hdr_enabled else images.dtype
        projected_bytes = _projected_output_bytes(batch_size, target_width, target_height, out_dtype, output_channels)
        log_dasiwa(
            "RTX Upscaler & Refiner",
            f"Input={source_width}x{source_height}, target={target_width}x{target_height}, "
            f"frames={batch_size}, output={projected_bytes / 1024 ** 3:.2f} GiB.",
        )
        out, mmap_path = _allocate_output_tensor(
            (batch_size, target_height, target_width, output_channels), out_dtype, out_device,
            soft_empty_cache=empty_cache,
            allow_mmap=use_mmap,
            auto_unload_models=auto_unload_models,
            compute_device=cuda_device,
            working_reserve_bytes=_dasiwa_rtx_working_reserve(
                source_width, source_height, target_width, target_height, resize_method,
                sum(map(bool, (denoise, deblur, upscale_enabled, hdr_enabled))), packed, hdr_enabled),
        )
        if mmap_path:
            log_dasiwa("RTX Upscaler & Refiner", f"Disk-backed output: {mmap_path}")

        with torch.cuda.device(cuda_device), torch.inference_mode(), _dasiwa_rtx_execution_stream(
            vfx_api, capabilities, cuda_device
        ), _dasiwa_rtx_hdr_effect(
            capabilities, hdr_enabled, device_id, hdr_contrast, hdr_saturation,
            hdr_middle_gray, hdr_luminance, hdr_debanding
        ) as hdr_effect:
            # Pass 1: Denoise
            with _maybe_vfx_effect(
                vfx_api, denoise, "Denoise", denoise_quality,
                device_id, source_width, source_height, denoise_strength, image_encoding
            ) as denoise_effect:

                # Pass 2: Deblur
                with _maybe_vfx_effect(
                    vfx_api, deblur, "Deblur", deblur_quality,
                    device_id, source_width, source_height, deblur_strength, image_encoding
                ) as deblur_effect:

                    # Pass 3: Upscale
                    with _maybe_vfx_effect(
                        vfx_api, upscale_enabled, upscale, upscale_quality,
                        device_id, target_width, target_height, upscale_strength, image_encoding
                    ) as upscale_effect:

                        step = chunk_frames if chunking else (batch_size or 1)
                        for start in range(0, batch_size, step):
                            stop = min(start + step, batch_size)
                            for global_index in range(start, stop):
                                # Convert host layout before H2D and stage one frame,
                                # avoiding two chunk-sized CUDA layout buffers.
                                frame = images[global_index, :, :, :3].permute(2, 0, 1).contiguous().to(
                                    device=cuda_device, dtype=torch.float32)
                                try:
                                    assert out is not None
                                    row = out[global_index]
                                    if _model_management and hasattr(_model_management, "throw_exception_if_processing_interrupted"):
                                        _model_management.throw_exception_if_processing_interrupted()

                                    if enhanced:
                                        for effect, stage in ((denoise_effect, "Denoise"), (deblur_effect, "Deblur")):
                                            if effect is not None:
                                                frame = _dasiwa_rtx_run_pass(effect, frame, cuda_device,
                                                    packed, channel_integrity_check, stage,
                                                    terminal=(not upscale_enabled and not hdr_enabled and
                                                              (stage == "Deblur" or deblur_effect is None)))
                                        if fit_to_target_aspect:
                                            frame = _fit_frame_to_target_aspect(frame, target_width, target_height, resize_method)
                                        if upscale_effect is not None:
                                            frame = _dasiwa_rtx_run_pass(upscale_effect, frame, cuda_device,
                                                packed, channel_integrity_check, "Upscale", terminal=not hdr_enabled)
                                        if hdr_effect is not None:
                                            frame = _dasiwa_rtx_run_pass(hdr_effect, frame, cuda_device,
                                                check=channel_integrity_check, stage="HDR", hdr=True, terminal=True)
                                        row[:, :, :3].copy_(frame.permute(1, 2, 0).clamp(0, 1))
                                        if output_channels == 4:
                                            alpha = images[global_index, :, :, 3:4].to(
                                                device=cuda_device, dtype=torch.float32).permute(2, 0, 1)
                                            if fit_to_target_aspect:
                                                alpha = _fit_frame_to_target_aspect(alpha, target_width, target_height, resize_method)
                                            if alpha.shape[-2:] != (target_height, target_width):
                                                alpha = F.interpolate(alpha.unsqueeze(0), size=(target_height, target_width),
                                                    mode="bilinear", align_corners=False)[0]
                                            row[:, :, 3:4].copy_(alpha.permute(1, 2, 0).clamp(0, 1))
                                        continue

                                    if denoise_effect:
                                        if not deblur and not upscale_enabled:
                                            _run_vfx_effect_into(denoise_effect, frame, row, cuda_device)
                                            continue
                                        frame = _run_vfx_effect(denoise_effect, frame, cuda_device)

                                    if deblur_effect:
                                        if not upscale_enabled:
                                            _run_vfx_effect_into(deblur_effect, frame, row, cuda_device)
                                            continue
                                        frame = _run_vfx_effect(deblur_effect, frame, cuda_device)

                                    if upscale_enabled:
                                        if fit_to_target_aspect:
                                            frame = _fit_frame_to_target_aspect(
                                                frame, target_width, target_height, resize_method
                                            )
                                        if upscale_effect:
                                            _run_vfx_effect_into(upscale_effect, frame, row, cuda_device)
                                            continue

                                    if out.device == cuda_device and out_dtype == frame.dtype:
                                        row.copy_(frame.permute(1, 2, 0), non_blocking=True)
                                        row.clamp_(0.0, 1.0)
                                    else:
                                        row.copy_(
                                            frame.permute(1, 2, 0).clamp(0.0, 1.0),
                                            non_blocking=(out.device.type == "cuda"),
                                        )

                                finally:
                                    torch.cuda.current_stream(cuda_device).synchronize()
                                    if out is not None and out.device.type == "cuda" and out.device != cuda_device:
                                        torch.cuda.current_stream(out.device).synchronize()
                                    frame = row = alpha = None

        # Cleanup only after the effect has finished; outputs remain owned by ComfyUI.
        if empty_cache:
            _cleanup_cuda(True, cuda_device)

        if lossless_fp16:
            out = _lossless_fp16_image_batch(out, chunk_frames)
        return (out,)

NODE_CLASS_MAPPINGS = {
    "DaSiWa_RTX_UpscalerRefiner": DaSiWa_RTX_UpscalerRefiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DaSiWa_RTX_UpscalerRefiner": "RTX Upscaler & Refiner",
}
