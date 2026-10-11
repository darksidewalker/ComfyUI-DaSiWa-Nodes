# RTX Frame Generation

NVIDIA Video Effects pair interpolation in a separate V1 node. Visible name **RTX Frame Generation**, saved/API ID `DaSiWa_RTX_FrameGeneration`, execution method `dasiwa_rtx_generate_frames`. It does not replace official NVIDIA nodes or the existing RTX Upscaler & Refiner.

## Wiring and timing

`IMAGE batch → RTX Frame Generation → Enhanced Video Combine (or another encoder)`

The input is an ordered batch of decoded frames, **not a native VIDEO object**. Extract images with Get Video Components (or a compatible loader). Route source audio separately to the encoder; the node has no audio/FPS sockets. A native VIDEO input/output mode is a proposal only, not implemented here. Hover each input/control for its meaning and active-mode requirements.

- **frame rate multiplier:** multiplier 2–16. For N source frames, emits `(N-1)*M+1` frames, preserving every original frame and both endpoints exactly. Encode at **source FPS × M** for the original endpoint-to-endpoint playback interval; retaining the last original rather than adding an extra trailing duplicate means container duration can differ by a final frame interval. No FPS or audio is inferred or modified by this IMAGE node.
- **specific timestep:** inserts one frame between each pair, at 0.01–0.99 of the interval. Emits `2*N-1` frames. Spacing is nonuniform unless timestep is 0.5; IMAGE carries no timestamps, so ordinary constant-FPS encoders cannot represent the original unequal intervals precisely. The node logs a warning for non-midpoint requests. Multiplier is ignored in this mode. At timestep 0.5, encode at **source FPS × 2** to preserve the endpoint-to-endpoint interval, with the same final-frame container-duration caveat.
- One source frame passes through unchanged (RGB only), without loading the SDK or requiring CUDA.

## Controls

- **mode:** LOW favors speed, MEDIUM is the middle interpolation tier, HIGH favors quality (default). These select the SDK's actual frame-generation presets, not VSR quality levels. Runtime and visual benefit vary with SDK, hardware and source motion; no fixed throughput or artifact-free quality is promised.
- **automatic_shot_change_detection:** enabled by default; let the SDK detect cuts.
- **shot_change:** off by default; explicitly marks **every submitted pair** as a cut. Most useful with a single pair; this is not a list of cut indices.
- **image_encoding:** default RGB8, or capability-gated packed RGB10A2. 10-bit is SDR precision, not HDR. RGB10A2 output is always float32; RGB8 retains the source floating dtype. Original source frames are copied directly rather than quantized through the SDK.
- **device_id:** CUDA device index (default 0, matching the existing RTX node). Requires supported NVIDIA hardware/driver at execution.
- **auto_unload_models:** targeted GPU reclamation under pressure. Before RAM output allocation, the shared allocator requests native inactive-cache eviction, releasable model pins and, if necessary, native active-cache eviction, rechecking capacity after each step. Cache eviction requires ComfyUI's RAM-pressure cache callback; older versions/other cache modes retain GC and supported pin cleanup. No running-executor reset or global GPU offload into RAM; referenced input/model storage can remain resident.
- **use_mmap:** off by default. Opt-in disk-backed last tier if budgeted VRAM and RAM cannot hold the full output; temp disk must have space. Failed/cancelled execution closes the effect and drops/unlinks failed mmap output.

## RGB, memory and compatibility

RGB only: RGBA alpha is discarded, matching the official frame-generation node. There is no resizing, alpha interpolation, HDR conversion, or audio output. Complete input and output batches remain in memory and may be cached by ComfyUI. Only previous/current input frames and one SDK output are staged; this is not a constant-memory video streaming interface.

Output placement reuses the sibling allocator: input-device CUDA output when output plus reserve fits (24 GiB output cap), otherwise RAM, optionally mmap. A frame-generation-specific estimate reserves 1 GiB plus bounded pair/conversion/output staging. The opaque SDK workspace estimate is conservative heuristic headroom, **not an OOM guarantee**. Memory pressure can still require shorter batches or smaller source resolution. Cancellation is checked for every pair and generated frame; borrowed SDK pixels are copied and synchronized before the next call or effect close.

SDK imports are lazy; pack import never constructs an effect or initializes CUDA. Old installations without `VideoFrameGeneration` retain their existing upscaler functionality and receive a clear execution error when this node is used. No dependency minimum is raised or automatic install attempted. RGB10A2 availability is detected from the actual constructor and enum; saved/API unsupported encoding requests fail rather than silently downgrade.

Pair interpolation is tested separately from VSR. On installed `nvidia-vfx 0.2.0.0`, native nondefault-stream pair calls reproduced SDK error **-12** in RGB8 and RGB10A2. This node uses blocking **default-stream** calls with explicit caller/SDK synchronization, including custom-stream callers; it does not blindly invoke the VSR single-input helper. This conservative implementation prioritizes correctness over asynchronous throughput.

## Validation

Contract tests exercise output counts, original-frame preservation, reusable SDK buffer ownership, dtype, controls, cut forwarding, missing/old SDK behavior, memory-option forwarding, cancellation/mmap cleanup, and stable registrations. Small real RTX 5090 inference was exercised with three distinct 128×128 textured/moving-square frames: RGB8/RGB10A2, multiplier 3 and timestep 0.3, CPU/CUDA input-output placement, default/custom caller streams. All node cases retained exact source frames, finite values, and distinct generated frames across pairs. Native direct custom-stream tests failed -12; node handoff cases succeeded. This establishes small-input correctness, not long-video capacity or interpolation quality across all content.
