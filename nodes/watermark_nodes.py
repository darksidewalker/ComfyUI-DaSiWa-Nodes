import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Tuple
import folder_paths
import comfy.utils

# --- Helpers ---

def _bhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    return x.movedim(-1, -3)

def _nchw_to_bhwc(x: torch.Tensor) -> torch.Tensor:
    return x.movedim(-3, -1)

def _rotate_bicubic_expand(x: torch.Tensor, degrees: float) -> torch.Tensor:
    """Rotate around center with bicubic sampling and expand canvas."""
    deg = float(degrees) % 360.0
    if deg == 0.0:
        return x

    N, C, H, W = x.shape
    rad = deg * 3.141592653589793 / 180.0
    cosr = float(torch.cos(torch.tensor(rad)))
    sinr = float(torch.sin(torch.tensor(rad)))

    new_w = max(1, int((abs(W * cosr) + abs(H * sinr)) + 0.9999))
    new_h = max(1, int((abs(H * cosr) + abs(W * sinr)) + 0.9999))

    cx_in, cy_in = (W - 1) * 0.5, (H - 1) * 0.5
    cx_out, cy_out = (new_w - 1) * 0.5, (new_h - 1) * 0.5

    ys = torch.linspace(0, new_h - 1, new_h, device=x.device, dtype=x.dtype)
    xs = torch.linspace(0, new_w - 1, new_w, device=x.device, dtype=x.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    rx, ry = gx - cx_out, gy - cy_out
    x_in = cosr * rx + sinr * ry + cx_in
    y_in = -sinr * rx + cosr * ry + cy_in

    x_norm = (x_in + 0.5) / W * 2.0 - 1.0
    y_norm = (y_in + 0.5) / H * 2.0 - 1.0
    grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0).repeat(N, 1, 1, 1)

    return F.grid_sample(x, grid, mode="bicubic", padding_mode="zeros", align_corners=False)

def _position_xy(position: str, base_w: int, base_h: int, wm_w: int, wm_h: int, pad_x: int, pad_y: int) -> Tuple[int, int]:
    pos = (position or "bottom-right").strip().lower()
    if pos == "center":
        return (base_w - wm_w) // 2, (base_h - wm_h) // 2

    x = 0 if "left" in pos else (base_w - wm_w if "right" in pos else (base_w - wm_w) // 2)
    y = 0 if "top" in pos else (base_h - wm_h if "bottom" in pos else (base_h - wm_h) // 2)

    if "left" in pos: x += pad_x
    if "right" in pos: x -= pad_x
    if "top" in pos: y += pad_y
    if "bottom" in pos: y -= pad_y
    return x, y

def _corner_sequence(start_position: str, switches: int, seed: int) -> Tuple[str, ...]:
    corners = ["top-left", "top-right", "bottom-left", "bottom-right"]
    start = start_position if start_position in corners or start_position == "center" else "bottom-right"
    remaining = [pos for pos in corners if pos != start]

    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)

    sequence = [start]
    while len(sequence) < switches + 1:
        if len(remaining) > 1 and sequence[-1] == remaining[0]:
            remaining[0], remaining[-1] = remaining[-1], remaining[0]
        sequence.extend(remaining)
        rng.shuffle(remaining)
    return tuple(sequence[:switches + 1])

def _interpolate_watermark(x: torch.Tensor, size: Tuple[int, int], mode: str) -> torch.Tensor:
    kwargs = {"size": size, "mode": mode}
    if mode in {"bicubic", "bilinear"}:
        kwargs["align_corners"] = False
    return F.interpolate(x, **kwargs)

def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)

def _sanitize_premultiplied_rgba(x: torch.Tensor) -> torch.Tensor:
    alpha = x[:, 3:4, :, :].clamp(0.0, 1.0)
    rgb = x[:, :3, :, :].clamp(0.0, 1.0).minimum(alpha)
    return torch.cat((rgb, alpha), dim=1)

def _tensor_nbytes(shape: Tuple[int, ...], dtype: torch.dtype) -> int:
    return int(np.prod(shape)) * torch.empty((), dtype=dtype).element_size()

def _safe_output_device(images: torch.Tensor, out_bytes: int) -> torch.device:
    # Watermark compositing is intentionally deterministic. Keeping the output
    # buffer in RAM avoids frame-to-frame differences from opportunistic CUDA
    # blending and avoids duplicating long video batches in VRAM.
    return torch.device("cpu")

class DaSiWa_Watermark:
    DESCRIPTION = (
        "DaSiWa Watermark: A professional-grade watermark overlay node with stable CPU compositing.\n"
        "Stores output batches in RAM and blends from initialized source frames to avoid flicker,\n"
        "Bicubic/Lanczos scaling, rotation, and 'Optical Padding' for perfect visual alignment."
    )

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])

        return {
            "required": {
                "images": ("IMAGE", {"description": "Input image or video batch."}),
                "watermark_path": (sorted(files), {"image_upload": True, "description": "Select the watermark file (PNG recommended for transparency)."}),
                "position": (["bottom-right", "bottom-left", "top-right", "top-left", "center"], {"default": "bottom-right", "description": "Static placement, or starting placement when randomize_position is enabled."}),
                "scale": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01, "description": "Size relative to image width (0.2 = 20% width)."}),
                "resampling": (["bicubic", "bilinear", "nearest", "area", "nearest-exact"], {"default": "bicubic"}),
                "transparency": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "description": "1.0 = Opaque, 0.0 = Invisible."}),
                "rotation": ("INT", {"default": 0, "min": 0, "max": 359, "step": 1}),
                "padding_x": ("INT", {"default": 20, "min": 0, "max": 4096}),
                "padding_y": ("INT", {"default": 20, "min": 0, "max": 4096}),
                "optical_padding": ("BOOLEAN", {"default": False, "description": "Adjust placement by the watermark's visual center (useful for wide/rotated marks)."}),
                "optical_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
                "random_switches": ("INT", {"default": 3, "min": 1, "max": 64, "description": "Number of times to switch position when randomize_position is enabled."}),
                "fade": ("BOOLEAN", {"default": False, "description": "Enable fade in at start and end."}),
                "fade_margin": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01, "description": "Fade duration as a percentage of total frames (0.1 = 10%)."}),
                "randomize_position": ("BOOLEAN", {"default": False, "description": "Cycle through corner positions, starting at the selected position."}),
                "random_seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "description": "0 derives a stable seed from the watermark file name."}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_watermark"
    CATEGORY = "DaSiWa/Video"

    def apply_watermark(
        self,
        images,
        watermark_path,
        position,
        scale,
        resampling,
        transparency,
        rotation,
        padding_x,
        padding_y,
        optical_padding,
        optical_strength,
        random_switches=3,
        fade=False,
        fade_margin=0.1,
        randomize_position=False,
        random_seed=0,
        compute_device=None,
    ):
        fade = _as_bool(fade)
        randomize_position = _as_bool(randomize_position)
        
        B, H, W, C = images.shape
        output_dtype = images.dtype if images.is_floating_point() else torch.float32
        out_bytes = _tensor_nbytes((B, H, W, 3), output_dtype)
        output_device = _safe_output_device(images, out_bytes)
        full_path = folder_paths.get_annotated_filepath(watermark_path)

        with torch.no_grad():
            # 1. Load and prepare the watermark on CPU. This avoids large rotation grids in VRAM.
            with Image.open(full_path) as im:
                im = im.convert("RGBA")
                wm_np = np.asarray(im, dtype=np.float32) / 255.0
                wm_rgba = torch.from_numpy(wm_np).permute(2, 0, 1).contiguous()

            wm_rgba[:3, :, :] *= wm_rgba[3:4, :, :]
            wm_h0, wm_w0 = wm_rgba.shape[1], wm_rgba.shape[2]
            
            # 2. Rescale Watermark (Maintain Aspect Ratio)
            target_w = max(1, int(round(W * scale)))
            target_h = max(1, int(round(wm_h0 * (target_w / wm_w0))))
            
            wm_resampled = _interpolate_watermark(
                wm_rgba.unsqueeze(0),
                size=(target_h, target_w),
                mode=resampling,
            )
            
            # Apply transparency to Alpha channel
            if transparency < 1.0:
                wm_resampled *= transparency

            wm_resampled = _sanitize_premultiplied_rgba(wm_resampled)
            
            # 3. Apply Rotation (Expand Canvas)
            wm_final = _sanitize_premultiplied_rgba(_rotate_bicubic_expand(wm_resampled, rotation)).squeeze(0)
            pm_final, a_final = wm_final[:3, :, :], wm_final[3:4, :, :]
            wm_h, wm_w = pm_final.shape[1], pm_final.shape[2]

            # 4. Precompute Optical Shift (Center of Mass)
            opt_dx, opt_dy = 0.0, 0.0
            if optical_padding:
                alpha = a_final[0]
                denom = alpha.sum()
                if denom > 1e-8:
                    ys = torch.linspace(0, wm_h - 1, wm_h)
                    xs = torch.linspace(0, wm_w - 1, wm_w)
                    cy = (alpha.sum(dim=1) * ys).sum() / denom
                    cx = (alpha.sum(dim=0) * xs).sum() / denom
                    opt_dx = float(((wm_w - 1) * 0.5 - cx) * optical_strength)
                    opt_dy = float(((wm_h - 1) * 0.5 - cy) * optical_strength)

            # 5. Batch Process with a stable compositor.
            # Start from a full copy so every frame is initialized before overlay.
            out = images[:, :, :, :3].to(device=output_device, dtype=output_dtype).clone()

            pbar = comfy.utils.ProgressBar(B)

            if position == "random":
                position = "bottom-right"

            if random_seed == 0:
                random_seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(watermark_path)))
            random_positions = _corner_sequence(position, random_switches, int(random_seed))

            last_pos = None
            x0, y0, x1, y1 = 0, 0, 0, 0
            pm_crop, a_crop = None, None

            num_fade_frames = max(1, int(B * fade_margin)) if fade else 1
            print(
                f"[DaSiWa Watermark] stable compositor: output={output_device.type}, "
                f"dtype={output_dtype}, frames={B}"
            )

            for i in range(B):
                # Determine current position. Random mode starts at the selected position.
                if randomize_position:
                    num_chunks = random_switches + 1
                    chunk_size = B / num_chunks
                    interval_idx = min(int(i // chunk_size), num_chunks - 1)
                    current_pos = random_positions[interval_idx]
                else:
                    current_pos = position

                if current_pos != last_pos:
                    x, y = _position_xy(current_pos, W, H, wm_w, wm_h, padding_x, padding_y)
                    if optical_padding and current_pos != "center":
                        if "right" in current_pos: x += int(round(opt_dx))
                        if "left" in current_pos: x -= int(round(opt_dx))
                        if "bottom" in current_pos: y += int(round(opt_dy))
                        if "top" in current_pos: y -= int(round(opt_dy))
                    x0, y0 = max(0, x), max(0, y)
                    x1, y1 = min(W, x + wm_w), min(H, y + wm_h)
                    if x1 > x0 and y1 > y0:
                        wx0, wy0 = x0 - x, y0 - y
                        w_w, w_h = x1 - x0, y1 - y0
                        pm_crop = pm_final[:, wy0:wy0 + w_h, wx0:wx0 + w_w].to(dtype=torch.float32).contiguous()
                        a_crop = a_final[:, wy0:wy0 + w_h, wx0:wx0 + w_w].to(dtype=torch.float32).contiguous()
                    else:
                        pm_crop, a_crop = None, None
                    last_pos = current_pos

                if fade:
                    if i < num_fade_frames:
                        fade_mult = i / float(num_fade_frames)
                    elif i >= (B - num_fade_frames):
                        fade_mult = (i - (B - num_fade_frames)) / float(num_fade_frames)
                    else:
                        fade_mult = 0.0
                else:
                    fade_mult = 1.0

                # --- Stable CPU Processing ---
                out_i = out[i]
                if pm_crop is not None and fade_mult > 0:
                    roi = out_i[y0:y1, x0:x1, :3].to(device="cpu", dtype=torch.float32).permute(2, 0, 1)
                    alpha = (a_crop * fade_mult).clamp(0.0, 1.0)
                    roi = roi * (1.0 - alpha) + (pm_crop * fade_mult)
                    roi = roi.clamp(0, 1)
                    out_i[y0:y1, x0:x1] = roi.permute(1, 2, 0).to(device=output_device, dtype=output_dtype)

                pbar.update(1)

            return (out,)

NODE_CLASS_MAPPINGS = {
    "DaSiWa_Watermark": DaSiWa_Watermark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DaSiWa_Watermark": "DaSiWa Watermark Overlay",
}
