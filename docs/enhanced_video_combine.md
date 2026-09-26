# DaSiWa Enhanced Video Combine

DaSiWa Enhanced Video Combine converts a ComfyUI `IMAGE` batch into a video, optionally muxes a connected `AUDIO` value, and provides an in-node playback preview after execution. Its enhanced automations select a usable codec, container, encoder, precision, browser-preview path, and generated-asset payload at runtime.

Click the small `?` at the right of the node title for the same concise in-app reference.

## Quick start

1. Connect an `IMAGE` batch and set `frame_rate`.
2. Leave `codec` and `container` on `Auto` for host-aware selection.
3. Optionally connect `AUDIO`.
4. Optionally enable **Save first frame** and/or **Save last frame**.
5. Run the workflow. The node returns the generated file path and, when enabled, the input frame batch for downstream nodes.

```mermaid
flowchart LR
    VD[VAE Decode<br>or sampler] -->|IMAGE batch| EVC[Enhanced Video Combine<br>codec: Auto, container: Auto]
    AL[Audio Loader<br>/ VAE Decode] -->|AUDIO| EVC
    EVC --> out((filename + path))
```

## Enhanced automations

The node is deliberately designed to avoid fixed, host-specific video settings:

- **Audio output uniqueness (fix):** Audio sidecar files were named `video_00001-audio.mp4` (dash), making the counter unparseable by `folder_paths.get_save_image_path`, which expected an underscore after the prefix. Every audio save therefore overwrote the previous one. The separator is now an underscore (`video_00001_audio.mp4`, `video_00002_audio.mp4`, …) so the counter increments correctly.
- **Codec and encoder selection:** `Auto` tests browser-compatible 8-bit AV1/WebM first, then VP9, then H.264. H.265/HEVC is excluded from Auto and must be chosen explicitly. For each candidate the node prefers NVIDIA NVENC, then Intel QSV, AMD AMF, VAAPI, and finally a software encoder. An advertised FFmpeg encoder is not trusted until an actual encode attempt succeeds.
- **PyAV-native execution:** The node uses PyAV 18 and the FFmpeg libraries bundled in its wheel for video encoding, audio muxing, metadata, animated outputs, and preview transcoding. It never launches an `ffmpeg` executable. Hardware candidates are runtime-tested in the existing NVENC → QSV → AMF → VAAPI order; an unavailable device or failed session continues to the next candidate and ultimately to the codec's software encoder.
- **Container safety:** Auto tries compatible container sequences for each codec and prevents an incompatible preference from being the only path. If all selected candidates fail, it attempts the mandatory H.264/MP4 fallback.
- **Source precision:** with an explicit codec, `bit_depth=Auto` detects 8-bit versus 10-bit image-batch quantization. Browser-compatible codec `Auto` always emits 8-bit 4:2:0.
- **Browser preview and seeking:** Every video encoder receives a one-second GOP and an explicit keyframe whenever another elapsed second begins. MP4 outputs use fast-start metadata. For AV1 and H.265 compatibility, the node creates a content-addressed H.264/AAC preview in ComfyUI's temporary directory and serves it with `FileResponse`, including HTTP byte-range support. The requested output file is never changed; cached previews are invalidated automatically when its size or modification time changes.
- **Output freshness:** The output node deliberately executes again on every Queue Prompt, including when upstream IMAGE inputs were cached. This ensures a new video, preview, and any selected frame exports are produced for every queue.
- **Asset publication:** The final video and every selected first/last-frame PNG are published to ComfyUI Assets together.

## Video encoding

### Codec

`Auto` uses browser-compatible 8-bit output and tests codecs in this order, using the first combination FFmpeg can actually encode:

1. AV1
2. VP9
3. H.264 / AVC

H.265 / HEVC is excluded from Auto and must be chosen explicitly.

Within a codec, the node tries hardware encoders first (NVIDIA NVENC, Intel QSV, AMD AMF, VAAPI) followed by the supported software encoder. This is a runtime test, not just an encoder-list check, so an advertised but unusable host encoder falls back safely.

Selecting a specific codec keeps that codec preference, but still uses its hardware-to-software encoder fallback.

### Container

With `container=Auto`:

- AV1 and VP9: WebM, then MKV, then MP4.
- H.264 and H.265: MP4, then MKV.

If every requested combination fails, the node attempts its mandatory H.264/MP4 fallback.

### Animated image outputs

Select **Animated WebP** or **Animated AVIF** explicitly from `container` to write an animated image rather than a video container. Both choices are deliberately excluded from `codec=Auto` and `container=Auto` and ignore the `codec` dropdown. Animated AVIF runtime-tests AV1 encoders in GPU-first order (`av1_nvenc`, Intel QSV, AMD AMF, VAAPI), then falls back to `libsvtav1` or `libaom-av1`. Animated WebP requires the CPU `libwebp_anim` encoder; FFmpeg has no NVENC WebP encoder. Animated image formats cannot mux the connected `AUDIO` value, so the node logs that audio is omitted.

### Bit depth and quality

- `bit_depth=Auto` detects whether the frame values represent 8-bit or 10-bit precision.
- `quality` is passed as CRF/CQ depending on the selected encoder. Lower values preserve more detail and create larger files; the default of 20 is the recommended balance.
- Streamed H.264 previews use 8-bit 4:2:0 for broad browser compatibility.

## Audio

Connect a ComfyUI `AUDIO` value to mux it with the video.

- `audio_codec=Auto` uses Opus for WebM and AAC for MKV/MP4. You can instead select AAC, Opus, or MP3; the node falls back to a container-compatible audio encoder when necessary.
- Select `audio_bitrate` from 64k through 320k (192k by default).
- `crop_to_audio` ends video output at the connected audio duration.
- Preview audio is muted by default, unmutes while the pointer is over the video, and mutes again when the pointer leaves. Check **Mute** to keep the preview permanently silent; the setting is saved with the node and the native player controls cannot un-mute it while it is on.

## Preview and frame controls

The completed video is shown inside a compact framed preview with source-native aspect ratio. Its native canvas player controls appear only while the pointer is over the video, and hide again when it leaves; there is no duplicate play button or timeline. Preview audio follows the same hover behavior. The framed row below the video holds Autoplay, Mute, and first/last-frame export controls. Autoplay and Mute are stored in the node properties, so both are saved with the workflow and restored on load.

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

Many Chromium/Linux installations cannot decode H.265 in a canvas/video element (and can similarly misreport AV1 support while rendering 10-bit hardware streams as black). For H.265 and AV1 outputs, the node streams an FFmpeg-transcoded fragmented H.264 MP4 response; the requested output file is not changed and no preview file is created.

`pingpong` appends reverse interior frames, giving a seamless forward/reverse loop. `pass_frames` returns the encoded frame sequence instead of an empty `IMAGE` result.

## Metadata and filenames

- `save_metadata` embeds ComfyUI prompt/workflow metadata where the selected container supports it.
- `filename_prefix` supports `%seed%` when the optional `seed` input is connected, plus ComfyUI date placeholders such as `%date%`, `%date:yyyy-MM-dd%`, and `%date:hhmmss%`. For example, `video/%date:yyyy-MM-dd%/shot_%seed%` writes the generation seed into the output name. Without a connected seed, `%seed%` is left unchanged for workflow compatibility.
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
| AV1/H.265 output has no preview | The node uses an FFmpeg-streamed H.264 preview for these codecs. If FFmpeg is unavailable or its H.264 stream encode failed, the preview cannot play. The final AV1/H.265 output remains valid. |
| WebM mux fails | WebM requires VP8/VP9/AV1 video and compatible audio. Use `container=Auto` or select VP9/AV1. |
