"""Generic per-step tiled diffusion for 4-D image latents and MiniMax H3 (MODEL patch).

MiniMax H3 models (video and MiniMax-H3-Image) are routed to the pack's H3TiledDiffusion,
which crops the H3 payload (keyframes, optional references) per tile and rebuilds its layout.
Both paths also size ComfyUI's pre-sampling memory reservation for one tile (TileMemoryEstimate).

Written from this pack's own H3TiledDiffusion pattern (nodes/h3_tiled_sampling.py)
and ComfyUI's model_function_wrapper contract: every sampler step the denoiser
runs once per overlapping latent tile and the x0 predictions are blended with
positive weights. Nothing is retained on the wrapper between forwards.
"""
import importlib.util
import sys
from pathlib import Path

import torch
import comfy.model_management


def _h3_tiling():
    """nodes/h3_tiled_sampling.py, also when this file is loaded standalone (tests)."""
    if __package__:
        from . import h3_tiled_sampling
        return h3_tiled_sampling
    name = "dasiwa_h3_tiled_sampling"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("h3_tiled_sampling.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


def tile_starts(total, tile, overlap):
    """Tile origins; the last tile is shifted back so it ends at the edge."""
    if tile >= total:
        return [0]
    starts = list(range(0, total - tile + 1, tile - overlap))
    if starts[-1] + tile < total:
        starts.append(total - tile)
    return starts


def tile_weight(th, tw, device):
    # Separable fp32 tent (pyramid), strictly > 0 so the summed weight buffer
    # never hits zero at edges/corners and fp16 never underflows the merge.
    yy = 1 - (torch.arange(th, device=device).float() - (th - 1) / 2).abs() / ((th + 1) / 2)
    xx = 1 - (torch.arange(tw, device=device).float() - (tw - 1) / 2).abs() / ((tw + 1) / 2)
    return (yy[:, None] * xx[None, :])[None, None]


class TiledDiffusion:
    """model_function_wrapper: tiles the denoiser call, blends x0 predictions."""

    def __init__(self, tile_width_px, tile_height_px, overlap_px, downscale):
        if min(tile_width_px, tile_height_px) <= 0 or overlap_px < 0 or downscale <= 0:
            raise ValueError("Tile sizes must be positive and overlap nonnegative")
        if any(v % 8 or v % downscale for v in (tile_width_px, tile_height_px, overlap_px)):
            raise ValueError(f"Tile sizes and overlap must be multiples of 8 and of the model's latent scale ({downscale}px)")
        if overlap_px >= min(tile_width_px, tile_height_px):
            raise ValueError("Overlap must be smaller than both tile sizes")
        self.tile_w, self.tile_h, self.overlap = (v // downscale for v in (tile_width_px, tile_height_px, overlap_px))

    def __call__(self, model_function, args):
        x, c = args["input"], args["c"]
        if x.ndim != 4:
            raise ValueError("Tiled Diffusion (Image) supports 4-D image latents (B,C,H,W) and MiniMax H3 models only")
        if c.get("control") is not None:
            raise ValueError("Tiled Diffusion (Image) does not support ControlNet; remove the tiled MODEL patch or the ControlNet")
        h, w = x.shape[-2:]
        if h <= self.tile_h and w <= self.tile_w:
            return model_function(x, args["timestep"], **c)
        # shortcut: only denoise_mask / c_concat are spatial in core conds, when a new spatial cond appears add it here.
        spatial = {k: c[k] for k in ("denoise_mask", "c_concat")
                   if isinstance(c.get(k), torch.Tensor) and tuple(c[k].shape[-2:]) == (h, w)}
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        weights = torch.zeros((1, 1, h, w), dtype=torch.float32, device=x.device)
        for y in tile_starts(h, self.tile_h, self.overlap):
            for xs in tile_starts(w, self.tile_w, self.overlap):
                comfy.model_management.throw_exception_if_processing_interrupted()
                y1, x1 = min(h, y + self.tile_h), min(w, xs + self.tile_w)
                ct = dict(c)
                ct["transformer_options"] = dict(c.get("transformer_options", {}))
                for k, v in spatial.items():
                    ct[k] = v[..., y:y1, xs:x1]
                pred = model_function(x[..., y:y1, xs:x1], args["timestep"], **ct)
                weight = tile_weight(y1 - y, x1 - xs, x.device)
                out[..., y:y1, xs:x1] += pred.float() * weight
                weights[..., y:y1, xs:x1] += weight
        return (out / weights).to(x.dtype)


class DaSiWaTiledDiffusion:
    DESCRIPTION = ("Patches a MODEL so every sampling step denoises overlapping image tiles and blends them, letting any image model "
                   "sample canvases far larger than its native size. MiniMax H3 (video and image) uses the H3 tiler; tile sizes and "
                   "overlap must then be multiples of 32 px.")

    @classmethod
    def INPUT_TYPES(cls):
        size = {"default": 1024, "min": 64, "max": 8192, "step": 8}
        return {"required": {
            "model": ("MODEL", {"tooltip": "Image diffusion model to patch; use the native checkpoint resolution as tile size."}),
            "tile_width": ("INT", {**size, "tooltip": "Tile width in pixels (multiple of the model's latent scale, usually 8)."}),
            "tile_height": ("INT", {**size, "tooltip": "Tile height in pixels (multiple of the model's latent scale, usually 8)."}),
            "overlap": ("INT", {"default": 128, "min": 0, "max": 1024, "step": 8, "tooltip": "Overlap between neighbouring tiles in pixels; 64 or more hides seams."}),
        }, "optional": {
            "tile_memory_estimate": ("BOOLEAN", {"default": True, "tooltip": "Size ComfyUI's pre-sampling memory reservation for one tile instead of the whole canvas, so the model is not offloaded for memory a tiled step never uses. Changes model residency only, never the result."}),
            "tile_references": (["full", "crop_video", "crop_all"], {"default": "full", "tooltip": "MiniMax H3 only: what each tile sees of Reference to Video blocks. full: every reference uncropped (character/style). crop_video: reference videos cropped to the tile (video edits). crop_all: reference images too (a picture that is a frame of this shot). Ignored for other models."}),
            "weight_residency": (["auto", "keep", "stream"], {"default": "auto", "tooltip": "With the tile memory estimate: auto keeps the whole model loaded when it fits beside one tile's working memory, otherwise streams it from RAM with almost nothing resident; keep loads as much as fits; stream keeps as little as possible loaded. Never changes the result."}),
            "max_resident_gb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1024.0, "step": 0.5, "tooltip": "With the tile memory estimate, auto/keep: cap the model weights kept on the GPU. 0 = no cap."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "DaSiWa/Image"

    def patch(self, model, tile_width, tile_height, overlap, tile_memory_estimate=True, tile_references="full",
              weight_residency="auto", max_resident_gb=0.0):
        import comfy.model_base
        h3 = _h3_tiling()
        if tile_references not in h3.REFERENCE_MODES:
            raise ValueError(f"tile_references must be one of {', '.join(h3.REFERENCE_MODES)}")
        patched = model.clone()
        if isinstance(model.model, comfy.model_base.MiniMaxH3):
            wrapper = h3.H3TiledDiffusion(tile_width, tile_height, overlap, coordinates="local", references=tile_references)
        else:
            wrapper = TiledDiffusion(tile_width, tile_height, overlap, model.model.latent_format.spacial_downscale_ratio)
        patched.set_model_unet_function_wrapper(wrapper)
        if tile_memory_estimate:
            h3.install_tile_memory_estimate(patched, wrapper.tile_h, wrapper.tile_w, weight_residency, max_resident_gb)
        return (patched,)
