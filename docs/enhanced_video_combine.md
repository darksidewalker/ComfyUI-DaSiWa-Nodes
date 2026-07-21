# DaSiWa Enhanced Video Combine

DaSiWa Enhanced Video Combine converts a ComfyUI `IMAGE` batch into a video, optionally muxes a connected `AUDIO` value, and provides an in-node playback preview after execution. Its enhanced automations select a usable codec, container, encoder, precision, browser-preview path, and generated-asset payload at runtime.

Click the small `?` at the right of the node title for the same concise in-app reference.

## Quick start

1. Connect an `IMAGE` batch and set `frame_rate`.
2. Leave `codec` and `container` on `Auto` for host-aware selection.
3. Optionally connect `AUDIO`.
4. Optionally enable **Save first frame** and/or **Save last frame**.
5. Run the workflow. The node returns the generated file path and, when enabled, the input frame batch for downstream nodes.

## Enhanced automations

The node is deliberately designed to avoid fixed, host-specific video settings:

- **Codec and encoder selection:** `Auto` runtime-tests AV1, H.265/HEVC, VP9, then H.264. For every candidate it prefers NVIDIA NVENC, then Intel QSV, AMD AMF, VAAPI, and finally an appropriate software encoder. A listed FFmpeg encoder is not trusted until a real encode succeeds.
- **Container safety:** Auto tries compatible container sequences for each codec and prevents an incompatible preference from being the only path. If all selected candidates fail, it attempts the mandatory H.264/MP4 fallback.
- **Source precision:** `bit_depth=Auto` detects 8-bit versus 10-bit image-batch quantization before encoding.
- **Browser preview:** A requested H.265 final output remains H.265. When needed, an H.264 MP4 preview sidecar is encoded automatically so the in-node player can work in browsers without HEVC support.
- **Output freshness:** The output node deliberately executes again on every Queue Prompt, including when upstream IMAGE inputs were cached. This ensures a new video, preview, and any selected frame exports are produced for every queue.
- **Asset publication:** The final video and every selected first/last-frame PNG are published to ComfyUI Assets together.

## Video encoding

### Codec

`Auto` tests codecs in this order and uses the first combination FFmpeg can actually encode:

1. AV1
2. H.265 / HEVC
3. VP9
4. H.264 / AVC

Within a codec, the node tries hardware encoders first (NVIDIA NVENC, Intel QSV, AMD AMF, VAAPI) followed by the supported software encoder. This is a runtime test, not just an encoder-list check, so an advertised but unusable host encoder falls back safely.

Selecting a specific codec keeps that codec preference, but still uses its hardware-to-software encoder fallback.

### Container

With `container=Auto`:

- AV1 and VP9: WebM, then MKV, then MP4.
- H.264 and H.265: MP4, then MKV.

If every requested combination fails, the node attempts its mandatory H.264/MP4 fallback.

### Animated image outputs

Select **Animated WebP** or **Animated AVIF** explicitly from `container` to write an animated image rather than a video container. Both choices are deliberately excluded from `codec=Auto` and `container=Auto`; they use their dedicated FFmpeg encoders (`libwebp_anim` and `libaom-av1`) and ignore the `codec` dropdown. Animated image formats cannot mux the connected `AUDIO` value, so the node logs that audio is omitted.

### Bit depth and quality

- `bit_depth=Auto` detects whether the frame values represent 8-bit or 10-bit precision.
- `quality` is passed as CRF/CQ depending on the selected encoder. Lower values preserve more detail and create larger files; the default of 20 is the recommended balance.
- H.264 preview sidecars always use 8-bit for broad browser compatibility.

## Audio

Connect a ComfyUI `AUDIO` value to mux it with the video.

- `audio_codec=Auto` uses Opus for WebM and AAC for MKV/MP4. You can instead select AAC, Opus, or MP3; the node falls back to a container-compatible audio encoder when necessary.
- Select `audio_bitrate` from 64k through 320k (192k by default).
- `crop_to_audio` ends video output at the connected audio duration.
- Preview audio is muted by default, unmutes while the pointer is over the video, and mutes again when the pointer leaves.

## Preview and frame controls

The completed video is shown inside the node with source-native aspect ratio, play/pause, seek, Autoplay, and first/last-frame controls. Preview audio automatically unmutes while the pointer is over the video and mutes again when it leaves; there is no separate sound toggle.

`save_first_frame` and `save_last_frame` write the selected source frame as a native-resolution PNG beside the encoded video. The names retain the complete video basename, including the counter and optional audio marker:

```text
video_00001.mp4
video_00001-first-frame.png
video_00001-last-frame.png

video_00002-audio.webm
video_00002-audio-first-frame.png
video_00002-audio-last-frame.png
```

No browser download or popup is used. The encoded video and each selected PNG are included in the `ui.images` payload, so all generated files appear in ComfyUI Assets.

Many Chromium/Linux installations cannot decode H.265 in a canvas/video element. For a H.265 output, the node creates an H.264 MP4 sidecar used only by the ComfyUI preview; the returned output path remains the requested H.265 file.

`pingpong` appends reverse interior frames, giving a seamless forward/reverse loop. `pass_frames` returns the encoded frame sequence instead of an empty `IMAGE` result.

## Metadata and filenames

- `save_metadata` embeds ComfyUI prompt/workflow metadata where the selected container supports it.
- `filename_prefix` supports ComfyUI date placeholders such as `%date%`, `%date:yyyy-MM-dd%`, and `%date:hhmmss%`.
- `save_output=false` writes under ComfyUI temporary output instead of the normal output directory.

## Console logging

The node always emits concise progress to the ComfyUI CLI with the `[DaSiWa Enhanced Video Combine]` prefix: input summary, Auto codec/container tests, selected video encoder, resolved audio encoder and bitrate, output path, and any audio fallback or selected frame exports. There is no logging-level toggle.

Example:

```text
[DaSiWa Enhanced Video Combine] Encode 48f 1920x1080@24fps 8-bit; codec=Auto, container=Auto, audio=yes.
[DaSiWa Enhanced Video Combine] Auto test: AV1/WebM.
[DaSiWa Enhanced Video Combine] Encoded AV1/WebM via av1_nvenc; audio=libopus/192k -> video_00001-audio.webm.
[DaSiWa Enhanced Video Combine] Output: .../video_00001-audio.webm (AV1, av1_nvenc, 8-bit).
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Auto falls back to another codec | The higher-priority codec or its containers could not encode on this host. The final `No usable encoder` error includes the compact attempts when every fallback fails. |
| No usable encoder was found | Install an FFmpeg build with at least a software encoder such as `libx264`; check the detailed CLI log. |
| H.265 output has no preview | The H.264 preview-sidecar encode failed. The final H.265 output remains valid; check the CLI log for the preview fallback error. |
| WebM mux fails | WebM requires VP8/VP9/AV1 video and compatible audio. Use `container=Auto` or select VP9/AV1. |
