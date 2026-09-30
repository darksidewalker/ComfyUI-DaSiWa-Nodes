# Seamless Loop

**Seamless Loop**, under **DaSiWa / video**, takes one RGB `IMAGE` batch and returns one RGB `IMAGE` batch. Connect a decoded video batch, select a native interpolation checkpoint, and send the result to Enhanced Video Combine or another image/video saver. There is no FPS conversion, audio processing, model download, or video encode inside this node.

## Model selector

Put a ComfyUI-compatible RIFE `.safetensors` checkpoint in `models/frame_interpolation/`, or a configured equivalent model path. The optional `model_name` combo uses ComfyUI's normal filename discovery and native Frame Interpolation Model Loader. The `model_name` dropdown supports ComfyUI's standard conversion to an input for an externally supplied checkpoint filename. There is no separate loaded-model socket; existing widget order is preserved. This input takes a filename, not an `INTERP_MODEL` object. There is no `.pth` loader or dependency on WhiteRabbit. The installed RIFE 4.26 safetensors checkpoint was exercised in inference. Native FILM safetensors checkpoints use the same Core loader/API, but FILM inference has not been verified with a local checkpoint.

Requires a ComfyUI build containing `comfy_extras.nodes_frame_interpolation.FrameInterpolationModelLoader`; verified against ComfyUI 0.38.0, commit `fb2315f1`, Python 3.12.13 and PyTorch 2.11.0+cu130. Loading remains lazy, so a missing native interpolation module does not prevent the rest of the pack from importing. This node adds no dependencies.

## Automatic processing

The node analyzes aspect-preserving, reduced-resolution previews. It compares candidate trims of up to 10% at each end and several overlap lengths, using color-compensated MSE/PSNR, local SSIM, spatial edge differences, temporal pixel-difference agreement, scene discontinuity and a penalty for shortening the clip. These are heuristic measures, not semantic motion tracking or a guarantee of perceptual perfection.

It keeps the original middle frames unchanged and morphs overlapping tail/head sequences with bidirectional native interpolation. Smoothstep timing eases the transition; bounded per-channel exposure correction fades to zero at both joins. The playback wrap is placed in the untouched original sequence rather than at a synthesized endpoint. Output duration and starting frame can change. Very short batches use an interpolated endpoint bridge; static and already pixel-closed batches are copied without loading a model.

Only one frame pair is interpolated at a time. Native ComfyUI model management owns device loading/offloading. Processed output uses ComfyUI's configured intermediate device rather than duplicating a GPU-resident input batch in VRAM, while preserving input resolution and dtype; static/already-closed pass-through copies retain their input device. Full input/output batches still consume memory proportional to duration. Cancellation is checked during analysis and interpolation. Replacing the selected checkpoint invalidates this node's cache.

## Exact endpoints versus smooth playback

`exact_endpoint` defaults to **on**: the first output frame is copied to the end, so the output tensor's first and last frames are bit-identical. This creates one duplicated frame when a player repeats the entire batch. Turn it **off** for continuous cyclic playback without that extra held frame. A single duplicate does not make motion direction or velocity continuous.

The log reports selected trim/overlap, preview comparison PSNR and the largest circular frame-change energy relative to the median. A large temporal spike emits a warning. PSNR describes the compared, exposure-compensated previews; it is not a video-wide quality score. Identical endpoints alone do not prove a good loop. Cuts, occlusion, irreversible motion, camera travel and strongly different head/tail content may still morph visibly.

Pixel equality applies to the returned images, not to a later lossy video encoding. H.264/HEVC/AV1 compression and chroma subsampling can change endpoint pixels differently. Use PNG frames or an appropriate lossless RGB export when decoded pixel equality matters. PyAV is useful for encoding/muxing, not for solving temporal or motion seams; the pack's existing video saver already owns that work. Cropping/shortening does not retime an upstream audio track automatically.

## Research and design choice

- [WhiteRabbit](https://github.com/Artificial-Sweetener/WhiteRabbit) is the requested reference for seam preparation, timing analysis and loop assembly. It is AGPL-3.0; no WhiteRabbit source was copied into this implementation.
- [RIFE](https://github.com/hzwer/ECCV2022-RIFE) estimates intermediate flow at arbitrary times. Reusing the installed Core implementation keeps safetensors handling and model lifecycle native.
- [FILM](https://github.com/google-research/frame-interpolation), ECCV 2022, targets large-motion interpolation. It is an alternative already implemented by this ComfyUI version, not a newer method than RIFE's original paper.
- [AMT](https://github.com/MCG-NKU/AMT), CVPR 2023, adds all-pairs, multi-field transforms. Released variants differ in fixed/arbitrary timing support. Its non-commercial license and separate model implementation make it unsuitable as an invisible bundled replacement.
- [GIMM-VFI](https://github.com/GSeanCDAT/GIMM-VFI), NeurIPS 2024, models continuous motion with RAFT/FlowFormer-based variants and perceptually trained variants. It has a [separate ComfyUI implementation](https://github.com/kijai/ComfyUI-GIMM-VFI), different checkpoints and additional backend requirements. It cannot consume a RIFE checkpoint and is not a guaranteed loop solution. Its [S-Lab License 1.0](https://github.com/GSeanCDAT/GIMM-VFI/blob/main/LICENSE) permits noncommercial use; commercial use requires contacting the contributors.

- [Mobius](https://github.com/YisuiTT/Mobius), SIGGRAPH 2025, uses latent shifting in CogVideoX/VideoCrafter2 to generate loops from text. This is a generation-time approach, not a lossless postprocessor for an arbitrary decoded image batch.
- [Loopy](https://github.com/WeChatCV/Loopy), released August 2026, anchors looping positional-embedding shifts in a Wan2.2 generation pipeline with separate loop-adapted weights. It addresses loop generation directly, but requires diffusion models/LoRAs and cannot substitute for a RIFE safetensors checkpoint or preserve arbitrary input footage pixel-for-pixel.
- [Automated Video Looping with Progressive Dynamism](https://hhoppe.com/videoloops.pdf) optimizes different loop periods for independently moving image regions. That is useful for cinemagraphs, but can freeze non-loopable regions and is not equivalent to preserving a coherent moving subject through a global temporal seam.

Newer interpolation research can improve particular frame pairs; it does not establish a universal, artifact-free loop for arbitrary footage. The practical implementation therefore combines automatic temporal-overlap analysis with native interpolation rather than adding unrelated models or claiming a pixel equality check proves seamless motion.

## Verification

Local focused tests (17 passed) exercise native RIFE inference on cyclic and noncyclic textured motion, odd resolutions, static/short/already-closed batches, invalid pixels, schema and registration. They verify original-frame adjacency at the overlap entry/wrap, measure transition-step improvement against the hard seam on the noncyclic fixture, and check CPU intermediate output for CUDA input. An adapter regression test checks model-patcher lifetime and FILM's missing alignment attribute. The browser-instantiated node was checked for exactly one image input/output and a socketless native model combo. An isolated ComfyUI API workflow, `LoadImage → Seamless Loop → SaveImage`, processed a 36-frame synthetic motion clip into 31 frames; PNG readback confirmed identical first/last pixels. This is an integration test, not a real-world perceptual benchmark.
