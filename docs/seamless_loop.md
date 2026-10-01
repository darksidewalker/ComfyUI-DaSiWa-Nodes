# Seamless Loop

**Seamless Loop**, under **DaSiWa / video**, turns a decoded RGB `IMAGE` batch into a looping `IMAGE` batch. Select an interpolation checkpoint and connect the output to Enhanced Video Combine or another saver. Align audio downstream if the frame count changes.

## Model selector

Place a ComfyUI-compatible RIFE or FILM `.safetensors` checkpoint in `models/frame_interpolation/` (or a configured equivalent path), then select it in `model_name`. The dropdown can be converted to an input for an external checkpoint filename.

Requires ComfyUI's native Frame Interpolation Model Loader; no additional dependencies. RIFE 4.26 inference was verified locally. FILM uses the same native loader but has not been inference-tested with a local checkpoint.

## Automatic processing

- **Seam selection:** compares reduced-resolution previews using exposure-compensated MSE/PSNR, local SSIM, edges, temporal differences and scene discontinuities. Tests trims of up to 10% at each end and several overlap lengths.
- **Transition:** morphs overlapping tail/head sequences with bidirectional interpolation, eased timing and bounded exposure correction. Middle frames stay unchanged; the playback wrap lies in the original sequence. Frame count and starting frame can change.
- **Short/static clips:** very short batches use an interpolated endpoint bridge. Static batches pass through unchanged; already-closed batches skip interpolation and drop the duplicated endpoint when `exact_endpoint` is off.
- **Memory:** interpolates one frame pair at a time using native ComfyUI model management. Resolution and dtype are preserved; processed output uses the configured intermediate device. Full batches still require memory proportional to duration.

## Exact endpoints versus smooth playback

**`exact_endpoint` is off by default.** Leave it off for continuous cyclic playback. Enable it to append a copy of the first output frame, making the returned first/last frames pixel-identical but adding a duplicated frame on repeat. Pixel equality is not guaranteed after lossy encoding.

Arbitrary footage may still show visible morphing, especially with cuts, occlusion, camera travel or incompatible head/tail motion. Logs report trim, overlap, preview PSNR and temporal spikes; strong discontinuities trigger a warning. Preview PSNR is not a video-wide quality score.

## References

- [RIFE](https://github.com/hzwer/ECCV2022-RIFE): intermediate-frame flow estimation.
- [FILM](https://github.com/google-research/frame-interpolation): large-motion frame interpolation.
- [WhiteRabbit](https://github.com/Artificial-Sweetener/WhiteRabbit): inspiration for seam preparation and loop assembly.
