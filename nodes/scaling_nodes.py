import math
import threading
from collections import OrderedDict

import torch
import torch.nn.functional as F

class DaSiWa_ResolutionScaleCalculator:
    # --- DATA ARRAYS ---
    # Resolution presets: value = real_world_pixel_count / (1024*1024)
    # Matches ComfyUI native MP convention (1 MP = 1,048,576 pixels)
    RESOLUTION_PRESETS = {
        "144p": 0.0352,
        "240p": 0.0977,
        "360p": 0.22,
        "480p": 0.391,
        "540p": 0.494,
        "576p": 0.396,
        "720p": 0.879,
        "900p": 1.373,
        "1080p": 1.978,
        "1152p": 2.25,
        "1440p": 3.516,
        "2160p": 7.91,
        "2K": 3.906,
        "4K": 7.91,
    }

    PRECISION_PRESETS = {
        "0.26 MP - Preview": 0.26,
        "0.36 MP - Small": 0.36,
        "0.52 MP - SD": 0.52,
        "0.65 MP - Balanced": 0.65,
        "0.83 MP - HD": 0.83,
        "1.05 MP - HD+": 1.05,
        "1.20 MP - HD++": 1.20,
        "1.35 MP - 2K lite": 1.35,
        "1.55 MP - 2K": 1.55,
        "1.65 MP - 2K+": 1.65,
        "1.75 MP - QHD": 1.75,
        "2.10 MP - FHD": 2.10,
        "3.30 MP - QHD+": 3.30,
        "4.75 MP - 2K Pro": 4.75,
        "6.50 MP - Production": 6.50,
        "8.30 MP - UHD": 8.30,
    }

    PRESETS = {**RESOLUTION_PRESETS, **PRECISION_PRESETS}

    ASPECT_PRESETS = {
        "1:1 - Square": (1, 1),
        "2:3 - Classic": (2, 3),
        "3:4 - Photo": (3, 4),
        "5:8 - Tall": (5, 8),
        "9:16 - Social": (9, 16),
        "9:21 - Cinema": (9, 21),
        "CUSTOM": (0, 0),
    }

    DESCRIPTION = """
    DaSiWa Resolution Scale Calculator
    
    Calculates mathematically precise resolutions based on a target Megapixel area.
    
    - Standard Mode: Pure mathematical scaling.

    - WAN/LTX Mode: Snaps to 32-pixel boundaries (mandatory for WAN/LTX VAEs).

    - No Scale: Overrides all math and outputs the source image dimensions directly.
    
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution_preset": (list(cls.PRESETS.keys()), {"default": "0.52 MP - SD", "description": "Target visual resolution / pixel budget. Pick either a standard resolution label or an optimized megapixel tier."}),
                
                "no_scale": ("BOOLEAN", {"default": False, "label_on": "ON (Source Dims)", "label_off": "OFF (Calculated)", "description": "Bypass all calculations and output the source dimensions exactly."}),
                
                "scale_from_image": ("BOOLEAN", {"default": True, "label_on": "IMAGE ASPECT", "label_off": "USE ASPECT BELOW", "description": "IMAGE ASPECT ignores the aspect controls below and uses the connected image shape. USE ASPECT BELOW uses the visible aspect controls."}),
                "aspect_preset_when_not_image": (list(cls.ASPECT_PRESETS.keys()), {"default": "9:16 - Social", "description": "Used only when scale_from_image is USE ASPECT BELOW. Ignored while IMAGE ASPECT is selected."}),
                "swap_aspect_when_not_image": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no", "description": "Used only when scale_from_image is USE ASPECT BELOW. Flip width and height."}),
                "custom_aspect_width": ("INT", {"default": 16, "min": 1, "max": 8192, "description": "Used only when scale_from_image is USE ASPECT BELOW and aspect preset is CUSTOM. Ratio width, not final pixels."}),
                "custom_aspect_height": ("INT", {"default": 9, "min": 1, "max": 8192, "description": "Used only when scale_from_image is USE ASPECT BELOW and aspect preset is CUSTOM. Ratio height, not final pixels."}),
                "mode": (["Standard", "WAN/LTX (Div32)", "LTX 2-Stage (Div64)", "CUSTOM"], {"default": "WAN/LTX (Div32)", "description": "Snapping engine. Use WAN/LTX (Div32) for modern video models."}),
                "custom_divisor": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1, "description": "Custom pixel boundary snapping."}),
            },
            "optional": {
                "image": ("IMAGE", {"description": "The source image used to calculate the target aspect ratio."}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("width_int", "height_int", "width_float", "height_float")
    FUNCTION = "calculate"
    CATEGORY = "DaSiWa/Scaling"

    def calculate(
        self,
        resolution_preset=None,
        no_scale=False,
        scale_from_image=True,
        aspect_preset_when_not_image="9:16 - Social",
        swap_aspect_when_not_image=False,
        custom_aspect_width=16,
        custom_aspect_height=9,
        mode="WAN/LTX (Div32)",
        custom_divisor=8,
        image=None,
        method=None,
        preset=None,
        precision_presets=None,
        resolution_presets=None,
        aspect_preset=None,
        swap_aspect=None,
        manual_aspect_width=None,
        manual_aspect_height=None,
    ):
        if resolution_preset is None:
            resolution_preset = preset
        aspect_preset = aspect_preset if aspect_preset is not None else aspect_preset_when_not_image
        swap_aspect = swap_aspect if swap_aspect is not None else swap_aspect_when_not_image
        manual_aspect_width = manual_aspect_width if manual_aspect_width is not None else custom_aspect_width
        manual_aspect_height = manual_aspect_height if manual_aspect_height is not None else custom_aspect_height

        # 1. GET SOURCE DIMENSIONS (From Image or Manual)
        if scale_from_image:
            if image is None:
                raise ValueError("DaSiWa Scaler: 'scale_from_image' is set to YES, but no image is connected.")
            try:
                # Get shape from first frame
                _, h, w, _ = image.shape
                source_w, source_h = float(w), float(h)
            except Exception:
                raise ValueError("DaSiWa Scaler: Invalid image input format.")
        else:
            if aspect_preset == "CUSTOM":
                source_w, source_h = float(manual_aspect_width), float(manual_aspect_height)
            else:
                source_w, source_h = self.ASPECT_PRESETS.get(aspect_preset, (1, 1))
            
            if swap_aspect:
                source_w, source_h = source_h, source_w

        # 2. HANDLE NO-SCALE TOGGLE (PASS-THROUGH)
        if no_scale:
            final_w, final_h = int(source_w), int(source_h)
            return (final_w, final_h, float(final_w), float(final_h))

        # 3. GET MP TARGET
        aspect_ratio = source_w / source_h
        if resolution_preset is None:
            if method == "Use Resolution Presets":
                resolution_preset = resolution_presets
            else:
                resolution_preset = precision_presets
        a = self.PRESETS.get(resolution_preset, 0.52)

        # 4. CALCULATE
        # ComfyUI-native MP convention: 1 MP = 1024*1024 = 1,048,576 pixels
        # (matches Scale Image to Total Pixels node)
        target_total_pixels = a * 1024 * 1024
        calc_w = math.sqrt(target_total_pixels * aspect_ratio)
        calc_h = math.sqrt(target_total_pixels / aspect_ratio)
        
        # 5. MODE HANDLING
        if mode == "CUSTOM":
            d = max(1, int(custom_divisor))
            final_w = int(round(calc_w / d) * d)
            final_h = int(round(calc_h / d) * d)
            floor = d
        elif mode == "LTX 2-Stage (Div64)":
            final_w = int(round(calc_w / 64.0) * 64)
            final_h = int(round(calc_h / 64.0) * 64)
            floor = 64
        elif mode == "WAN/LTX (Div32)":
            final_w = int(round(calc_w / 32.0) * 32)
            final_h = int(round(calc_h / 32.0) * 32)
            floor = 32
        else:
            final_w = int(round(calc_w))
            final_h = int(round(calc_h))
            floor = 1

        return (max(final_w, floor), max(final_h, floor), float(final_w), float(final_h))


class _LanczosWeightCache:
    def __init__(self):
        self._entries = OrderedDict()
        self._hits = 0
        self._capacity = 64
        self._lock = threading.Lock()

    def configure(self, capacity):
        with self._lock:
            self._capacity = max(1, capacity)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)

    def get(self, key):
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
                self._hits += 1
            return value

    def put(self, key, value):
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)

    def stats(self):
        with self._lock:
            return {"entries": len(self._entries), "hits": self._hits}


_LANCZOS_CACHE = _LanczosWeightCache()


def _srgb_to_linear(image):
    return torch.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055).pow(2.4),
    )


def _linear_to_srgb(image):
    return torch.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * image.clamp_min(0).pow(1.0 / 2.4) - 0.055,
    )


def _lanczos_weights(in_size, out_size, device, dtype):
    key = (in_size, out_size, str(device), str(dtype))
    cached = _LANCZOS_CACHE.get(key)
    if cached is not None:
        return cached

    scale = in_size / out_size
    kernel_scale = max(scale, 1.0)
    support = 3.0 * kernel_scale
    window = int(math.ceil(support * 2))
    coordinates = (torch.arange(out_size, device=device, dtype=torch.float32) + 0.5) * scale - 0.5
    starts = torch.floor(coordinates - support + 0.5).to(torch.int64)
    indices = starts[:, None] + torch.arange(window, device=device, dtype=torch.int64)[None, :]
    distance = (coordinates[:, None] - indices.to(torch.float32)) / kernel_scale
    weights = torch.sinc(distance) * torch.sinc(distance / 3.0)
    weights = weights * (distance.abs() < 3.0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
    cached = (weights.to(dtype), indices.clamp(0, in_size - 1))
    _LANCZOS_CACHE.put(key, cached)
    return cached


def _lanczos_resample_1d(image, out_size, dim):
    in_size = image.shape[dim]
    if in_size == out_size:
        return image
    weights, indices = _lanczos_weights(in_size, out_size, image.device, image.dtype)
    moved = image.movedim(dim, -1)
    original_shape = moved.shape
    flattened = moved.reshape(-1, in_size)
    result = (flattened[:, indices] * weights.unsqueeze(0)).sum(dim=-1)
    return result.reshape(*original_shape[:-1], out_size).movedim(-1, dim)


def _lanczos_resize(image, size):
    return _lanczos_resample_1d(_lanczos_resample_1d(image, size[1], 3), size[0], 2)


def _interpolate(image, size, method):
    mode = method.lower()
    if mode == "nearest":
        try:
            return F.interpolate(image, size=size, mode="nearest-exact")
        except RuntimeError:
            return F.interpolate(image, size=size, mode="nearest")
    if mode == "area":
        return F.interpolate(image, size=size, mode="area")
    try:
        return F.interpolate(image, size=size, mode=mode, align_corners=False, antialias=True)
    except (RuntimeError, TypeError):
        return F.interpolate(image, size=size, mode=mode, align_corners=False)


def _floor_multiple(value, divisor):
    return value if divisor == 1 else value - (value % divisor)


def _target_box(width, height, divisor):
    target_width = _floor_multiple(int(width), divisor)
    target_height = _floor_multiple(int(height), divisor)
    if target_width < 1 or target_height < 1:
        raise ValueError("Target width and height must be at least divisible_by.")
    return target_width, target_height


def _auto_batch_size(frame_count, source_height, source_width, target_height, target_width, max_megapixels):
    pixels_per_frame = max(source_height * source_width, target_height * target_width)
    pixel_budget = max(1, int(float(max_megapixels) * 1024 * 1024))
    return min(frame_count, max(1, pixel_budget // max(1, pixels_per_frame)))


def _fit_aspect(source_width, source_height, target_width, target_height):
    scale = min(target_width / source_width, target_height / source_height)
    return max(1, round(source_width * scale)), max(1, round(source_height * scale))


def _cover_aspect(source_width, source_height, target_width, target_height):
    scale = max(target_width / source_width, target_height / source_height)
    return max(1, math.ceil(source_width * scale)), max(1, math.ceil(source_height * scale))


def _fit_aspect_divisible(source_width, source_height, target_width, target_height, divisor):
    if divisor == 1:
        return _fit_aspect(source_width, source_height, target_width, target_height)
    divisor = max(1, divisor)
    gcd = math.gcd(source_width, source_height)
    base_width = divisor * (source_width // gcd)
    base_height = divisor * (source_height // gcd)
    fit_width, fit_height = _fit_aspect(source_width, source_height, target_width, target_height)
    multiplier = min(fit_width // base_width, fit_height // base_height)
    if multiplier < 1:
        raise ValueError("Target box is too small to preserve aspect ratio at divisible_by.")
    return base_width * multiplier, base_height * multiplier


def _ar_scale_divisible_crop(source_width, source_height, target_width, target_height, divisor):
    target_width, target_height = _target_box(target_width, target_height, divisor)
    if source_width >= source_height:
        resized_width = target_width
        resized_height = max(1, round(source_height * (resized_width / source_width)))
        output_height = _floor_multiple(min(resized_height, target_height), divisor)
        if output_height < 1:
            raise ValueError("Target height is too small for divisible AR scaling.")
        return resized_width, resized_height, resized_width, output_height
    resized_height = target_height
    resized_width = max(1, round(source_width * (resized_height / source_height)))
    output_width = _floor_multiple(min(resized_width, target_width), divisor)
    if output_width < 1:
        raise ValueError("Target width is too small for divisible AR scaling.")
    return resized_width, resized_height, output_width, resized_height


def _crop_offset(position, resized_width, resized_height, output_width, output_height):
    horizontal = "left" if position.endswith("left") or position == "left" else "right" if position.endswith("right") or position == "right" else "center"
    vertical = "top" if position.startswith("top") or position == "top" else "bottom" if position.startswith("bottom") or position == "bottom" else "center"
    delta_width = resized_width - output_width
    delta_height = resized_height - output_height
    x = 0 if horizontal == "left" else delta_width if horizontal == "right" else delta_width // 2
    y = 0 if vertical == "top" else delta_height if vertical == "bottom" else delta_height // 2
    return x, y


def _pad_sides(position, padding_width, padding_height):
    left = 0 if position.endswith("left") or position == "left" else padding_width if position.endswith("right") or position == "right" else padding_width // 2
    top = 0 if position.startswith("top") or position == "top" else padding_height if position.startswith("bottom") or position == "bottom" else padding_height // 2
    return left, padding_width - left, top, padding_height - top


def _pad_color(value, channels, device, dtype):
    try:
        values = [max(0, min(255, int(component.strip()))) / 255 for component in value.split(",")]
    except (AttributeError, ValueError):
        values = [0.0, 0.0, 0.0]
    values = (values + [0.0, 0.0, 0.0])[:3]
    if channels == 1:
        values = values[:1]
    elif channels == 4:
        values.append(1.0)
    return torch.tensor(values, device=device, dtype=dtype).view(1, channels, 1, 1)


class DaSiWa_TorchResize:
    DESCRIPTION = (
        "Dependency-free PyTorch batch resize for ComfyUI images. Supports pixel-perfect "
        "nearest, bilinear, bicubic, area, and cached separable Lanczos resampling."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "size_mode": (["Multiplier", "Target resolution"], {"default": "Multiplier", "description": "Choose whether to scale by a factor or resize into a target width and height."}),
                "aspect_mode": (["Stretch", "Fit", "Fill and crop", "Fit and pad", "Long side with divisible crop"], {"default": "Fit", "description": "How a target resolution handles the source aspect ratio. Multiplier always preserves it."}),
                "target_width": ("INT", {"default": 1920, "min": 1, "max": 16384, "step": 1, "description": "Target box width. Used in Target resolution mode."}),
                "target_height": ("INT", {"default": 1080, "min": 1, "max": 16384, "step": 1, "description": "Target box height. Used in Target resolution mode."}),
                "scale_multiplier": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 16.0, "step": 0.01, "description": "Scale factor. Used in Multiplier mode."}),
                "interpolation": (["Nearest", "Bilinear", "Bicubic", "Area", "Lanczos"], {"default": "Lanczos"}),
                "gamma_correct": ("BOOLEAN", {"default": True}),
                "divisible_by": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1, "description": "Floor output dimensions to this multiple."}),
                "pad_color": ("STRING", {"default": "0, 0, 0"}),
                "crop_position": (["center", "top-left", "top", "top-right", "left", "right", "bottom-left", "bottom", "bottom-right"], {"default": "center"}),
                "batch_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "max_batch_megapixels": ("FLOAT", {"default": 16.0, "min": 0.25, "max": 512.0, "step": 0.25}),
                "cache_size": ("INT", {"default": 64, "min": 1, "max": 512, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "resize"
    CATEGORY = "DaSiWa/Scaling"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def validate_inputs(self, *args, **kwargs):
        return True

    def resize(self, image, size_mode, aspect_mode, target_width, target_height, scale_multiplier, interpolation, gamma_correct, divisible_by, pad_color, crop_position, batch_size, max_batch_megapixels, cache_size):
        if image.ndim != 4 or image.shape[-1] not in (1, 3, 4):
            raise ValueError("DaSiWa Torch Resize expects IMAGE data shaped [batch, height, width, 1/3/4 channels].")
        source_height, source_width = image.shape[1:3]
        divisor = max(1, int(divisible_by))
        if size_mode == "Multiplier":
            mode = "Multiplier"
            resized_width, resized_height = _target_box(round(source_width * scale_multiplier), round(source_height * scale_multiplier), divisor)
            output_width, output_height = resized_width, resized_height
        elif size_mode != "Target resolution":
            raise ValueError("Unknown size_mode.")
        elif aspect_mode == "Fit":
            mode = "Fit"
            resized_width, resized_height = _fit_aspect_divisible(source_width, source_height, target_width, target_height, divisor)
            output_width, output_height = resized_width, resized_height
        elif aspect_mode == "Stretch":
            mode = "Stretch"
            output_width, output_height = _target_box(target_width, target_height, divisor)
            resized_width, resized_height = output_width, output_height
        elif aspect_mode == "Fill and crop":
            mode = "Fill and crop"
            output_width, output_height = _target_box(target_width, target_height, divisor)
            resized_width, resized_height = _cover_aspect(source_width, source_height, output_width, output_height)
        elif aspect_mode == "Fit and pad":
            mode = "Fit and pad"
            output_width, output_height = _target_box(target_width, target_height, divisor)
            resized_width, resized_height = _fit_aspect(source_width, source_height, output_width, output_height)
        elif aspect_mode == "Long side with divisible crop":
            mode = "Long side with divisible crop"
            resized_width, resized_height, output_width, output_height = _ar_scale_divisible_crop(source_width, source_height, target_width, target_height, divisor)
        else:
            raise ValueError("Unknown aspect_mode.")
        _LANCZOS_CACHE.configure(cache_size)
        chunk_size = int(batch_size) if batch_size > 0 else _auto_batch_size(
            image.shape[0], source_height, source_width, output_height, output_width, max_batch_megapixels,
        )
        output = None
        for start in range(0, image.shape[0], chunk_size):
            chunk = image[start:start + chunk_size].movedim(-1, 1)
            resized = self._resize_chunk(chunk, (resized_height, resized_width), interpolation, gamma_correct)
            if mode in ("Fill and crop", "Long side with divisible crop"):
                offset_x, offset_y = _crop_offset(crop_position, resized_width, resized_height, output_width, output_height)
                resized = resized[:, :, offset_y:offset_y + output_height, offset_x:offset_x + output_width]
            elif mode == "Fit and pad":
                left, _, top, _ = _pad_sides(crop_position, output_width - resized_width, output_height - resized_height)
                canvas = _pad_color(pad_color, resized.shape[1], resized.device, resized.dtype).expand(resized.shape[0], -1, output_height, output_width).clone()
                canvas[:, :, top:top + resized_height, left:left + resized_width] = resized
                resized = canvas
            if output is None:
                output = torch.empty(
                    (image.shape[0], resized.shape[1], output_height, output_width),
                    device=resized.device,
                    dtype=resized.dtype,
                )
            output[start:start + resized.shape[0]].copy_(resized)
        assert output is not None
        return (output.movedim(1, -1).clamp(0.0, 1.0),)

    @staticmethod
    def _resize_chunk(image, size, method, gamma_correct):
        rgb, alpha = (image[:, :3], image[:, 3:]) if image.shape[1] == 4 else (image, None)
        use_gamma = gamma_correct and method != "Nearest" and rgb.shape[1] in (1, 3)
        if use_gamma:
            rgb = _srgb_to_linear(rgb.to(torch.float32))
        resized_rgb = _lanczos_resize(rgb, size) if method == "Lanczos" else _interpolate(rgb, size, method)
        if use_gamma:
            resized_rgb = _linear_to_srgb(resized_rgb)
        if alpha is None:
            return resized_rgb
        resized_alpha = _lanczos_resize(alpha, size) if method == "Lanczos" else _interpolate(alpha, size, method)
        return torch.cat((resized_rgb, resized_alpha), dim=1)
