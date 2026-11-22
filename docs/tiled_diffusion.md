# Tiled Diffusion (Image)

**Tiled Diffusion (Image)** is a MODEL patch for ordinary image checkpoints (SD 1.x/SDXL/Flux-family and other 4-D latent models) and, since 0.5.62, for MiniMax H3 (video models and MiniMax-H3-Image). Every sampler step, the patched model denoises the latent in overlapping tiles and blends the per-tile predictions with positive pyramid weights, so a model can sample a canvas far larger than its native resolution (hires-fix / upscale passes) without tile seams and without the full-canvas attention cost.

Registered node type `DaSiWaTiledDiffusion`, category **DaSiWa/Image**. Written from this pack's own MiniMaxH3 per-step tiled sampler; no code from other tiled-diffusion packs.

## Wiring

`Load Checkpoint → Tiled Diffusion (Image) → KSampler (model)`. Everything else (conditioning, latent, VAE) stays as it is. Any sampler and scheduler works, because tiling happens inside the model call.

## Inputs

| Input | Default | Meaning |
|---|---|---|
| `model` | — | Image diffusion model to patch. The clone is patched; the original stays untouched. |
| `tile_width` / `tile_height` | 1024 | Tile size in pixels. Use the model's native resolution (SD 1.5: 512, SDXL/Flux: 1024). Must be a multiple of 8 and of the model's latent scale. |
| `overlap` | 128 | Overlap of neighbouring tiles in pixels; **64 px or more** blends seams away. Must be smaller than both tile sizes. |
| `tile_memory_estimate` | on | Size ComfyUI's pre-sampling memory reservation for one tile instead of the whole canvas, so the model is not offloaded for memory a tiled step never uses. Changes how much of the model stays loaded, never the result. |
| `weight_residency` | `auto` | With the estimate on: `auto` keeps the whole model loaded when it fits beside one tile, otherwise streams it from RAM; `keep` loads as much as fits; `stream` keeps as little as possible loaded. |
| `max_resident_gb` | 0 | Cap on the model weights kept on the GPU in `auto`/`keep`; 0 = no cap. |
| `tile_references` | `full` | MiniMax H3 only: `full` (every Reference to Video block uncropped: character/style), `crop_video` (reference videos cropped to each tile: video edits) or `crop_all` (reference images too: a picture that is a frame of this shot). Ignored for other models. |

Latents no larger than one tile are passed through with a single model call, so the patch costs nothing on small images.

## Limits

- **ControlNet is not supported**: the wrapper raises a clear error. Remove either the ControlNet or the tiled MODEL patch.
- `denoise_mask` (inpaint/mask sampling) and inpaint-model `c_concat` latents are sliced per tile; all other conditioning passes through unchanged.
- Image latents (B,C,H,W) and MiniMax H3. Other video/5-D models raise an error.
- **MiniMax H3:** the model is routed to the pack's H3 tiler (the one inside MiniMaxH3 Enhanced Upscale): each tile gets its crop of the keyframe guides (and, with `tile_references`, of the references) and a native layout for its own size; packed audio is averaged over the tiles. Tile sizes and overlap must be **multiples of 32 px**. For upscaling a finished H3 video use **MiniMaxH3 Enhanced Upscale**, which also plans tiles and temporal windows.
- Per-step tiling multiplies the number of model forwards by the tile count; VRAM stays near one tile's cost.
