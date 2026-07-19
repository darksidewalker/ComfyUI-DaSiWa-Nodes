# DaSiWa Enhanced Video Combine

DaSiWa Enhanced Video Combine converts a ComfyUI `IMAGE` batch into a video, optionally muxes a connected `AUDIO` value, and provides an in-node playback preview after execution.

Click the small `?` at the right of the node title for the same concise in-app reference.

## Quick start

1. Connect an `IMAGE` batch and set `frame_rate`.
2. Leave `codec` and `container` on `Auto` for host-aware selection.
3. Optionally connect `AUDIO`.
4. Run the workflow. The node returns the generated file path and, when enabled, the input frame batch for downstream nodes.

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

### Bit depth and quality

- `bit_depth=Auto` detects whether the frame values represent 8-bit or 10-bit precision.
- `quality` is passed as CRF/CQ depending on the selected encoder. Lower values preserve more detail and create larger files; the default of 20 is the recommended balance.
- H.264 preview sidecars always use 8-bit for broad browser compatibility.

## Audio

Connect a ComfyUI `AUDIO` value to mux it with the video.

- `audio_codec=Auto` uses Opus for WebM and AAC for MKV/MP4.
- `audio_bitrate` sets the target audio bitrate.
- `crop_to_audio` ends video output at the connected audio duration.

## Preview and frame controls

The completed video is shown inside the node with source-native aspect ratio, play/pause, seek, mute, and optional first/last-frame PNG export.

Many Chromium/Linux installations cannot decode H.265 in a canvas/video element. For a H.265 output, the node creates an H.264 MP4 sidecar used only by the ComfyUI preview; the returned output path remains the requested H.265 file.

`pingpong` appends reverse interior frames, giving a seamless forward/reverse loop. `pass_frames` returns the encoded frame sequence instead of an empty `IMAGE` result.

## Metadata and filenames

- `save_metadata` embeds ComfyUI prompt/workflow metadata where the selected container supports it.
- `filename_prefix` supports ComfyUI date placeholders such as `%date%`, `%date:yyyy-MM-dd%`, and `%date:hhmmss%`.
- `save_output=false` writes under ComfyUI temporary output instead of the normal output directory.

## Console logging

The node logs every encode to the ComfyUI CLI with the `[DaSiWa Enhanced Video Combine]` prefix.

- `Standard` logs the input summary, Auto codec/container tests, selected encoder, and final output path.
- `Verbose` additionally lists FFmpeg encoders absent from the host and the compact failure reason for each runtime encoder attempt.

Example:

```text
[DaSiWa Enhanced Video Combine] Encode 48f 1920x1080@24fps 8-bit; codec=Auto, container=Auto, audio=yes.
[DaSiWa Enhanced Video Combine] Auto test: AV1/WebM.
[DaSiWa Enhanced Video Combine] Encoded AV1/WebM via av1_nvenc -> video_00001-audio.webm.
[DaSiWa Enhanced Video Combine] Output: .../video_00001-audio.webm (AV1, av1_nvenc, 8-bit).
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Auto falls back to another codec | The higher-priority codec or its containers could not encode on this host. Select `Verbose` to see the attempted and missing encoders. |
| No usable encoder was found | Install an FFmpeg build with at least a software encoder such as `libx264`; check the detailed CLI log. |
| H.265 output has no preview | The H.264 preview-sidecar encode failed. The final H.265 output remains valid; check the CLI log for the preview fallback error. |
| WebM mux fails | WebM requires VP8/VP9/AV1 video and compatible audio. Use `container=Auto` or select VP9/AV1. |
