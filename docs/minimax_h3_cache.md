# MiniMax H3 Cache

`MiniMax H3 Cache` is an approximate, model-scoped acceleration node for ComfyUI's native MiniMax H3 diffusion model. It caches the residual produced by the complete H3 transformer block stack, and reuses that residual when the sampled audio/video-token feature signature changes little enough.

## Wiring

Place it on the `MODEL` path before the guider/sampler:

```mermaid
flowchart TB
    LOADER[MiniMax H3 Model Loader] --> CACHE[MiniMax H3 Cache]
    CACHE --> PKA["Patch Comfy Kitchen Attention (optional)"]
    PKA --> GUIDER[Guider / Sampler]
    CACHE -.->|skip optional patch| GUIDER
```

The Cache and **Patch Comfy Kitchen Attention** are model-clone patches and may be chained in either order. Cache-hit steps bypass the complete H3 block stack, so no attention backend runs on those steps. On cache-miss steps, Comfy Kitchen's selected attention override remains present in `transformer_options` and is used normally.

## Controls

- `reuse_threshold` (default `0.05`): maximum accumulated relative L1 change in the sampled token signature before the cache must refresh. Larger values tend to skip more block-stack evaluations and can reduce output fidelity.
- `start_percent` / `end_percent` (defaults `0.15` / `0.90`): inclusive portion of the sigma schedule where reuse is allowed.
- `max_steps` (default `2`): upper bound for consecutive cache hits.
- `device`: `auto` keeps cached residuals with the active model, `cpu` offloads residuals to system RAM, and `cuda` requires a CUDA-resident H3 model.
- `verbose`: logs individual decisions and a final, theoretical block-stack speedup summary.

This is an approximation. Compare it with Cache disabled using the same checkpoint, prompt, seed, sampler, step count, and resolution. In particular, check audio continuity, motion stability, and reference adherence. A displayed speedup is only a theoretical block-stack ratio; it is not an end-to-end generation benchmark.

## Current-ComfyUI behavior

The node clones the incoming `MODEL`, installs a reversible object patch for only that clone's `diffusion_model._forward`, adds a complete H3 block-loop replacement point, and attaches an outer-sampling lifecycle wrapper. It does not mutate the global `MiniMaxH3Model` class. The patched block runner preserves current ComfyUI `double_block` replacements, prefetching, and `transformer_options`, including optimized-attention overrides.

## Provenance and licensing

This node is GPL-3.0-compatible and is based on the cache algorithm from:

- lihaoyun6, [ComfyUI-MiniMaxH3-Cache](https://github.com/lihaoyun6/ComfyUI-MiniMaxH3-Cache), GPL-3.0-or-later, inspected at `8a45e096a2a05c140dd4d909eb74e4279b673819` (2026-08-04).

The current-ComfyUI model-scoped lifecycle and feature set were independently implemented after studying the design of:

- silveroxides, [ComfyUI-UtilsCollection](https://github.com/silveroxides/ComfyUI-UtilsCollection), AGPL-3.0, inspected at `1de9e5ce174374c1637cbe81d7e2545c4cfae528` (2026-08-16).

No AGPL-licensed source code from UtilsCollection was copied into this GPL-3.0 collection. The silveroxides project itself credits the lihaoyun6 cache as its algorithmic upstream.

## Compatibility

- Requires ComfyUI's native `MiniMaxH3Model`; it rejects other model architectures.
- Requires a ComfyUI version exposing `ModelPatcher.add_object_patch`, `set_model_patch_replace`, and `add_wrapper`.
- Works with ComfyUI's PDD LoRA head bank (0.34+): the node detects the live `FinalLayer.forward` signature at patch time and passes the PDD sigma-schedule arguments automatically, so cache and PDD coexist with no extra setup.
- Honors per-token denoise masks (video and audio): masked rows run at their own strength exactly like Core, so cached and region-masked generations match Core quality. Unmasked and all-generate masks are unaffected.
- Is compatible with this collection's **Patch Comfy Kitchen Attention** model patch.
- Do not characterize this cache as superior to another cache without matched benchmarks and quality comparisons.
