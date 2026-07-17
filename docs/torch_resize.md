# DaSiWa Torch Resize

![DaSiWa-Torch-Resize.png](../assets/DaSiWa-Torch-Resize.png)

A batch-aware ComfyUI `IMAGE` resizer implemented only with PyTorch already required by this node pack. It does not install or import `torchlanc`, Triton, Pillow, torchvision, or a vendor-specific GPU SDK.

## Controls

- **Size mode**: The first menu answers how to set the scale. `Multiplier` applies `scale_multiplier` to both source dimensions. `Target resolution` uses `target_width × target_height` as the target box.
- **Aspect mode**: The second menu answers how a target resolution handles the source shape. It is ignored by `Multiplier`, which always preserves aspect ratio.
  - `Stretch`: forces the output to the target box and can distort the source.
  - `Fit`: preserves aspect ratio inside the target box; the output can be smaller on one axis.
  - `Fill and crop`: fills the target box, then crops excess content.
  - `Fit and pad`: preserves aspect ratio inside the target box, then adds a colour canvas.
  - `Long side with divisible crop`: locks the source long side to the requested aligned size, then crops only the short side to its aligned size.
- **Divisible by**: Floors final dimensions to a multiple of this value. Use `1` to disable alignment; use values such as `8`, `16`, `32`, or `64` for downstream model constraints.
- **Crop position**: Selects the preserved edge for crop modes and the placement of the resized image for pad mode.
- **Pad color**: RGB canvas colour for pad mode, written as `r, g, b` in 0–255 values, for example `255, 0, 0`.
- **Method**:
  - `Nearest`: pixel-perfect nearest-neighbour (uses `nearest-exact` when supported).
  - `Bilinear` / `Bicubic`: PyTorch interpolation with anti-aliasing when the current backend supports it.
  - `Area`: PyTorch area resampling, useful for downscaling.
  - `Lanczos`: native, separable 3-lobe Lanczos implemented with PyTorch tensor operations.
- **Gamma correct**: Converts colour (and grayscale) from sRGB to linear light before bilinear, bicubic, area, or Lanczos resampling, then converts it back. Nearest bypasses it to preserve exact source values. Alpha is always resampled separately without gamma conversion.
- **Batch size**: `0` automatically selects a bounded frame chunk from source/output pixel cost and `max_batch_megapixels`; use a positive value to force an exact frame count per chunk.
- **Max batch megapixels**: Auto-chunk memory/throughput budget, default `16 MP`. At a 4K target this normally processes two frames at once; at a 512² target it can process about 64. Lower it for less VRAM, raise it after testing available headroom.
- **Large video batches**: The node preallocates the final output once and writes each frame chunk directly into it. This avoids retaining a Python list of intermediate chunks before concatenation, while keeping output frame order unchanged.
- **Cache size**: Maximum number of in-memory Lanczos weight/indices entries. Repeated source/target dimensions on the same device and dtype reuse their cached weights. The cache is process-local and never writes to disk.

## Acceleration and compatibility

The node preserves the tensor's current device. Therefore it uses normal PyTorch acceleration on the active backend, including CUDA, ROCm, MPS, DirectML-compatible PyTorch paths, or CPU. It has no NVIDIA-only code path and no Triton dependency. Actual speed and availability are determined by the installed PyTorch build and its backend support.

Use `Multiplier` for simple 2×/4× image or video scaling. Use `Target resolution` plus `Fit` when content must fit a known downstream resolution without distortion.
