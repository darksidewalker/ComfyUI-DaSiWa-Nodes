# Seamless Loop

**Seamless Loop**, under **DaSiWa / video**, turns a decoded RGB `IMAGE` batch into a looping `IMAGE` batch. Choose an interpolation backend and connect the output to Enhanced Video Combine or another saver. Keep source FPS for loop processing. Connect the clip's audio (with `frame_rate`) to get a matching looped soundtrack on the `audio` output.

## Interpolation backend

`interpolation_backend` defaults to **RIFE / FILM**, preserving existing workflows. **RTX Frame Generation** replaces only the transition interpolation with NVIDIA VFX; it does not increase the whole video's frame rate.

- **RIFE / FILM:** `model_name` selects the native checkpoint described below. RTX controls are ignored.
- **RTX Frame Generation:** requires a supported NVIDIA CUDA GPU and `nvidia-vfx` exposing `VideoFrameGeneration.run_at_timestep`. No RIFE/FILM checkpoint is needed; `model_name` is ignored, including the `(no native models)` choice on installations without checkpoints. `rtx_mode` offers LOW/MEDIUM/HIGH (HIGH default); `rtx_device_id` selects the CUDA index (0 default). Uses SDK RGB8 processing and preserves the IMAGE dtype, not HDR/10-bit processing.

Both backends run forward/reverse pair interpolation with complementary timesteps. RTX loads one effect per execution, copies SDK-owned results before reuse, and closes it on completion or failure. Automatic/forced cut detection is disabled for deliberate tail/head morphing. It uses the frame-generation node's synchronized default-stream helpers, not the upscaler's single-input runner. Small real RTX 5090 tests cover overlap/short-clip bridges, CPU/CUDA inputs and default/nondefault caller streams; this is not a guarantee of seamless visual quality on arbitrary footage.

## Native model selector

Place a ComfyUI-compatible RIFE or FILM `.safetensors` checkpoint in `models/frame_interpolation/` (or a configured equivalent path), then select it in `model_name`. The dropdown can be converted to an input for an external checkpoint filename.

Requires ComfyUI's native Frame Interpolation Model Loader; no additional dependencies. RIFE 4.26 inference was verified locally. FILM uses the same native loader but has not been inference-tested with a local checkpoint.

## Automatic processing

- **Seam selection:** compares reduced-resolution previews using exposure-compensated MSE/PSNR, local SSIM, edges, temporal differences and scene discontinuities. Tests trims of up to 10% at each end and several overlap lengths.
- **Transition:** morphs overlapping tail/head sequences with bidirectional interpolation, eased timing and bounded exposure correction. Middle frames stay unchanged; the playback wrap lies in the original sequence. Frame count and starting frame can change.
- **Short/static clips:** very short batches use an interpolated endpoint bridge. Static batches pass through unchanged; already-closed batches skip interpolation and drop the duplicated endpoint when `exact_endpoint` is off.
- **Memory:** interpolates one frame pair at a time. RIFE/FILM uses native ComfyUI model management; RTX retains one SDK effect for the transition. Resolution and dtype are preserved; processed output uses the configured intermediate device. Full batches still require memory proportional to duration; the loop does not inherit the separate frame-generation node's mmap/output-budget controls.

## Audio

Optional. Connect `audio` (ComfyUI AUDIO) and set `frame_rate` to the clip's FPS; without `frame_rate` the node stops with an error. Without audio the `audio` output is empty and the images are exactly as in earlier versions.

- **Same mapping as the video:** frame *f* starts at sample `round(f × sample_rate / frame_rate)`, always from the absolute frame index, so rounding never accumulates. The output audio is exactly `round(output_frames × sample_rate / frame_rate)` samples long. Batch, channels, dtype and device are kept.
- **Per case:** static clips return the audio unchanged; already-closed clips drop the last frame's samples (unless `exact_endpoint`); short-clip bridges append an equal-power crossfade of the last frames' audio into the first frames' audio; normal loops keep the untouched middle and crossfade the tail into the head with equal power (cos/sin). The output end continues straight into the audio that starts the output, so the repeat is sample-continuous. `exact_endpoint` appends one frame of the audio that follows the wrap. The tail start may differ by one sample from its exact frame boundary (rounding).
- **Shorter audio:** if the audio is shorter than the video, missing samples are silence and this is logged.
- **`audio_loop`:** *Crossfade* (default) keeps the visual loop choice. *Beat-aware* estimates the tempo (spectral-flux onsets, autocorrelation over 60–200 BPM). If the tempo is clear, it also tests every end trim within the 10% range and picks the loop closest to a whole number of beats whose visual cost is at most 1.25× the visually best choice. The crossfade then blends beat-aligned material; there is no time-stretching, so the remaining error is frame quantization (at most half a frame). With an unclear tempo, or no whole-beat loop within the tolerance, it uses the Crossfade choice and logs why.
- **Logs:** e.g. `[DaSiWa Loop] Beat-aware: 120.0 BPM, loop 7.00 beats (visual cost 1.07x best)` and `[DaSiWa Loop] audio 96 → 84 frames = 192000 → 168000 samples @ 48000 Hz / 24.000 fps; source frames 0..91 → sample 0; tempo 120.0 BPM (confidence 0.91)`.

## Exact endpoints versus smooth playback

**`exact_endpoint` is off by default.** Leave it off for continuous cyclic playback. Enable it to append a copy of the first output frame, making the returned first/last frames pixel-identical but adding a duplicated frame on repeat. Pixel equality is not guaranteed after lossy encoding.

Arbitrary footage may still show visible morphing, especially with cuts, occlusion, camera travel or incompatible head/tail motion. Logs report trim, overlap, preview PSNR and temporal spikes; strong discontinuities trigger a warning. Preview PSNR is not a video-wide quality score.

## References

- [RIFE](https://github.com/hzwer/ECCV2022-RIFE): intermediate-frame flow estimation.
- [FILM](https://github.com/google-research/frame-interpolation): large-motion frame interpolation.
- [WhiteRabbit](https://github.com/Artificial-Sweetener/WhiteRabbit): inspiration for seam preparation and loop assembly.
