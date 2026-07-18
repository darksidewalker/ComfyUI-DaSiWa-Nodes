import datetime
import json
import os
import re
import shutil
import subprocess
import tempfile

import torch

import folder_paths


_CODEC_OPTIONS = ["AV1", "VP9", "H.265 (HEVC)", "H.264"]
_CONTAINER_OPTIONS = ["Auto", "WebM", "MKV", "MP4"]
_ENCODER_NAMES = {
    "H.264": ("h264_nvenc", "h264_qsv", "h264_amf", "h264_vaapi", "libx264"),
    "H.265 (HEVC)": ("hevc_nvenc", "hevc_qsv", "hevc_amf", "hevc_vaapi", "libx265"),
    "AV1": ("av1_nvenc", "av1_qsv", "av1_amf", "av1_vaapi", "libsvtav1", "libaom-av1"),
    "VP9": ("vp9_qsv", "vp9_vaapi", "libvpx-vp9"),
}
_CONTAINER_EXTENSIONS = {"WebM": ".webm", "MKV": ".mkv", "MP4": ".mp4"}
_AUDIO_CODEC_OPTIONS = ["Auto", "AAC", "Opus", "MP3"]
_AUDIO_ENCODERS = {"AAC": "aac", "Opus": "libopus", "MP3": "libmp3lame"}
_AUDIO_BITRATE_OPTIONS = ["64k", "96k", "128k", "160k", "192k", "256k", "320k"]


def find_ffmpeg():
    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        return path_ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def detect_bit_depth(images):
    if not torch.is_floating_point(images):
        return 10 if images.element_size() >= 2 else 8

    values = images.detach().to(device="cpu", dtype=torch.float32).flatten()
    if values.numel() == 0:
        return 8
    if values.numel() > 250_000:
        values = values[:: max(1, values.numel() // 250_000)]

    values = values.clamp(0, 1)
    error_8 = torch.mean(torch.abs(values * 255 - torch.round(values * 255)))
    error_10 = torch.mean(torch.abs(values * 1023 - torch.round(values * 1023)))
    return 10 if error_10 < error_8 * 0.8 else 8


def _container_candidates(codec, container):
    if container != "Auto":
        return (container,)
    if codec in {"AV1", "VP9"}:
        return ("WebM", "MKV", "MP4")
    return ("MP4", "MKV")


def _available_encoders(ffmpeg):
    result = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        return set()
    return {line.split()[1] for line in result.stdout.splitlines() if line.startswith(" V")}


def _encoder_arguments(codec, encoder, bit_depth, cq, crf):
    pixel_format = "yuv420p10le" if bit_depth == 10 else "yuv420p"
    if encoder.endswith("_nvenc"):
        return ["-c:v", encoder, "-preset", "p5", "-cq", str(cq), "-pix_fmt", pixel_format]
    if encoder.endswith("_qsv"):
        return ["-c:v", encoder, "-global_quality", str(cq), "-pix_fmt", pixel_format]
    if encoder.endswith("_amf"):
        return ["-c:v", encoder, "-quality", "quality", "-qp_i", str(cq), "-qp_p", str(cq), "-pix_fmt", pixel_format]
    if encoder.endswith("_vaapi"):
        return ["-c:v", encoder, "-qp", str(cq), "-pix_fmt", pixel_format]
    if codec in {"H.264", "H.265 (HEVC)"}:
        return ["-c:v", encoder, "-crf", str(crf), "-preset", "medium", "-pix_fmt", pixel_format]
    if encoder == "libsvtav1":
        return ["-c:v", encoder, "-crf", str(crf), "-preset", "6", "-pix_fmt", pixel_format]
    if codec == "AV1":
        return ["-c:v", encoder, "-crf", str(crf), "-b:v", "0", "-cpu-used", "6", "-pix_fmt", pixel_format]
    return ["-c:v", encoder, "-crf", str(crf), "-b:v", "0", "-deadline", "good", "-pix_fmt", pixel_format]


def _frame_bytes(images, bit_depth):
    frames = images[..., :3].detach().to(device="cpu", dtype=torch.float32).clamp_(0, 1)
    if bit_depth == 10:
        return torch.round(frames * 1023).to(torch.int32).mul_(64).to(torch.uint16).numpy().tobytes()
    return torch.round(frames * 255).to(torch.uint8).numpy().tobytes()


def _pingpong_frames(images, pingpong):
    if not pingpong or len(images) < 3:
        return images
    return torch.cat((images, images[1:-1].flip(0)), dim=0)


def _format_filename_prefix(filename_prefix):
    """Expand ComfyUI-style date placeholders before requesting an output path."""
    now = datetime.datetime.now()

    def replace_date(match):
        fmt = match.group(1) or "yyyyMMdd_HHmmss"
        fmt = fmt.replace("yyyy", "%Y").replace("yy", "%y")
        fmt = fmt.replace("MM", "%m").replace("dd", "%d").replace("DD", "%d")
        fmt = fmt.replace("HH", "%H").replace("hh", "%H")
        fmt = fmt.replace("mm", "%M").replace("ss", "%S")
        return now.strftime(fmt)

    return re.sub(r"%date(?::([^%]+))?%", replace_date, filename_prefix)


def _output_filename(filename, counter, extension, has_audio):
    audio_suffix = "-audio" if has_audio else ""
    return f"{filename}_{counter:05}{audio_suffix}.{extension.lstrip('.')}"


def _metadata_file(prompt, extra_pnginfo):
    if prompt is None and not extra_pnginfo:
        return None

    metadata = {}
    if prompt is not None:
        metadata["prompt"] = prompt
    if extra_pnginfo:
        metadata.update(extra_pnginfo)

    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".ffmeta", delete=False, encoding="utf-8")
    try:
        handle.write(";FFMETADATA1\n")
        for key, value in metadata.items():
            escaped = json.dumps(value, separators=(",", ":"))
            escaped = escaped.replace("\\", "\\\\").replace(";", "\\;")
            escaped = escaped.replace("#", "\\#").replace("=", "\\=").replace("\n", "\\\n")
            handle.write(f"{key}={escaped}\n")
    finally:
        handle.close()
    return handle.name


def _audio_file(audio):
    if audio is None:
        return None, None
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("audio must be a ComfyUI AUDIO value containing waveform and sample_rate.")

    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor):
        raise ValueError("audio waveform must be a torch.Tensor.")
    waveform = waveform.detach().to(device="cpu", dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 3:
        raise ValueError("audio waveform must have shape [batch, channels, samples].")

    sample_rate = int(audio["sample_rate"])
    if sample_rate <= 0 or waveform.shape[-1] == 0:
        raise ValueError("audio must have a positive sample rate and at least one sample.")
    channels = waveform.shape[1]
    interleaved = waveform.transpose(1, 2).reshape(-1, channels).clamp_(-1, 1).numpy()
    handle = tempfile.NamedTemporaryFile(suffix=".f32le", delete=False)
    try:
        handle.write(interleaved.tobytes())
    finally:
        handle.close()
    return (handle.name, sample_rate, channels), len(interleaved) / sample_rate


def _audio_encoder(audio_codec, container):
    if audio_codec == "Auto":
        return "libopus" if container == "WebM" else "aac"
    return _AUDIO_ENCODERS[audio_codec]


def _audio_encoder_candidates(audio_codec, container):
    requested = _audio_encoder(audio_codec, container)
    fallback = {
        "WebM": ("libopus",),
        "MKV": ("aac", "libopus", "libmp3lame", "pcm_s16le"),
        "MP4": ("aac", "libmp3lame"),
    }[container]
    return tuple(dict.fromkeys((requested, *fallback)))


def _encode_with_available_encoder(
    ffmpeg, codec, bit_depth, width, height, frame_rate, payload, output_path,
    container, cq, crf, metadata_path, audio_path=None, audio_duration=None, crop_to_audio=False,
    audio_codec="Auto", audio_bitrate="192k",
):
    available = _available_encoders(ffmpeg)
    attempts = []
    for encoder in _ENCODER_NAMES[codec]:
        if encoder not in available:
            continue
        audio_encoders = _audio_encoder_candidates(audio_codec, container) if audio_path else (None,)
        for selected_audio_encoder in audio_encoders:
            command = [ffmpeg, "-y", "-v", "error"]
            if metadata_path:
                command.extend(["-f", "ffmetadata", "-i", metadata_path])
            command.extend([
                "-f", "rawvideo", "-pix_fmt", "rgb48le" if bit_depth == 10 else "rgb24",
                "-s", f"{width}x{height}", "-framerate", str(frame_rate), "-i", "-",
            ])
            video_input_index = 1 if metadata_path else 0
            if audio_path:
                command.extend(["-f", "f32le", "-ar", str(audio_path[1]), "-ac", str(audio_path[2]), "-i", audio_path[0]])
            if metadata_path:
                command.extend(["-map", f"{video_input_index}:v:0", "-map_metadata", "0"])
            elif audio_path:
                command.extend(["-map", f"{video_input_index}:v:0"])
            if audio_path:
                command.extend(["-map", f"{video_input_index + 1}:a:0", "-c:a", selected_audio_encoder, "-b:a", audio_bitrate])
            command.extend(_encoder_arguments(codec, encoder, bit_depth, cq, crf))
            if crop_to_audio and audio_duration is not None:
                command.extend(["-t", f"{audio_duration:.9f}"])
            if container == "MP4":
                command.extend(["-movflags", "+use_metadata_tags"])
            result = subprocess.run(command + [output_path], input=payload, capture_output=True, timeout=3600)
            if result.returncode == 0:
                if selected_audio_encoder and selected_audio_encoder != _audio_encoder(audio_codec, container):
                    print(f"[DaSiWa Enhanced Video Combine] Audio fallback: {selected_audio_encoder}.")
                return encoder
            attempts.append(f"{encoder}/{selected_audio_encoder or 'no-audio'}: {result.stderr.decode(errors='replace')[:300]}")
    raise RuntimeError("No usable encoder was found. " + " | ".join(attempts))


class DaSiWa_EnhancedVideoCombine:
    DESCRIPTION = (
        "Combines an IMAGE batch into a high-quality video with automatic NVENC/GPU "
        "selection, optional ping-pong playback, ComfyUI workflow metadata, and MP4 fallback."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"description": "Frames to encode as a video."}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 240.0, "step": 0.01}),
                "codec": (_CODEC_OPTIONS, {"default": "AV1"}),
                "container": (_CONTAINER_OPTIONS, {"default": "Auto", "description": "Auto tries the best container first: WebM, then MKV, then MP4 for AV1/VP9; MP4 then MKV for H.264/H.265."}),
                "bit_depth": (["Auto", "8-bit", "10-bit"], {"default": "Auto"}),
                "quality": ("INT", {"default": 20, "min": 0, "max": 51, "description": "Encoding quality for every encoder. 0 is no compression (largest files); higher values increase compression and reduce quality. 20 is the recommended default."}),
                "pingpong": ("BOOLEAN", {"default": False, "description": "Append the interior frames in reverse order for seamless forward/reverse playback."}),
                "save_metadata": ("BOOLEAN", {"default": True, "description": "Embed ComfyUI prompt and workflow metadata for workflow-aware video loaders."}),
                "filename_prefix": ("STRING", {"default": "video_%date:hhmmss%", "description": "Output path/name prefix. Supports %date% and formatted dates such as video/%date:yyyy-MM-dd%/%date:hhmmss%."}),
                "save_output": ("BOOLEAN", {"default": True}),
                "pass_frames": ("BOOLEAN", {"default": False, "description": "Return the encoded frame sequence for downstream processing."}),
                "crop_to_audio": ("BOOLEAN", {"default": False, "description": "When audio is connected, end the output video at the audio duration."}),
                "audio_codec": (_AUDIO_CODEC_OPTIONS, {"default": "Auto", "description": "Audio codec. Auto uses Opus for WebM and AAC for MKV/MP4."}),
                "audio_bitrate": (_AUDIO_BITRATE_OPTIONS, {"default": "192k", "description": "Target bitrate for the connected audio stream."}),
            },
            "optional": {
                "audio": ("AUDIO", {"description": "Optional ComfyUI audio to mux into the encoded video."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "filename")
    FUNCTION = "combine"
    OUTPUT_NODE = True
    CATEGORY = "DaSiWa/Video"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def validate_inputs(self, *args, **kwargs):
        return True

    def combine(
        self, images, frame_rate, codec, container, bit_depth, quality, pingpong,
        save_metadata, filename_prefix, save_output, pass_frames, crop_to_audio=False, audio_codec="Auto",
        audio_bitrate="192k", audio=None, prompt=None,
        extra_pnginfo=None,
    ):
        if images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("images must be an IMAGE batch shaped [frames, height, width, channels] with RGB channels.")

        images = _pingpong_frames(images, pingpong)
        selected_bit_depth = {"8-bit": 8, "10-bit": 10}.get(bit_depth, detect_bit_depth(images))
        output_dir = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
        output_type = "output" if save_output else "temp"
        height, width = images.shape[1:3]
        filename_prefix = _format_filename_prefix(filename_prefix)
        output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, output_dir, width, height)
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("No FFmpeg executable was found. Install FFmpeg or imageio-ffmpeg for the mandatory H.264/MP4 fallback.")

        metadata_path = _metadata_file(prompt, extra_pnginfo) if save_metadata else None
        audio_path, audio_duration = _audio_file(audio)
        payload = _frame_bytes(images, selected_bit_depth)
        attempts = []
        try:
            for selected_container in _container_candidates(codec, container):
                output_path = os.path.join(
                    output_folder,
                    _output_filename(filename, counter, _CONTAINER_EXTENSIONS[selected_container], audio_path is not None),
                )
                try:
                    encoder = _encode_with_available_encoder(
                        ffmpeg, codec, selected_bit_depth, width, height, frame_rate, payload,
                        output_path, selected_container, quality, quality, metadata_path, audio_path, audio_duration, crop_to_audio,
                        audio_codec, audio_bitrate,
                    )
                    break
                except RuntimeError as error:
                    attempts.append(f"{selected_container}: {error}")
            else:
                fallback_path = os.path.join(output_folder, _output_filename(filename, counter, ".mp4", audio_path is not None))
                encoder = _encode_with_available_encoder(
                    ffmpeg, "H.264", selected_bit_depth, width, height, frame_rate, payload,
                    fallback_path, "MP4", quality, quality, metadata_path, audio_path, audio_duration, crop_to_audio,
                    audio_codec, audio_bitrate,
                )
                output_path = fallback_path
                selected_container = "MP4"
        finally:
            if metadata_path:
                os.unlink(metadata_path)
            if audio_path:
                os.unlink(audio_path[0])

        output_frames = images if pass_frames else images[:0]
        mime_type = {"WebM": "video/webm", "MKV": "video/x-matroska", "MP4": "video/mp4"}[selected_container]
        ui = {"gifs": [{"filename": os.path.basename(output_path), "subfolder": subfolder, "type": output_type, "format": mime_type, "width": width, "height": height, "fps": frame_rate}]}
        print(f"[DaSiWa Enhanced Video Combine] Saved {output_path} with {encoder}, {selected_bit_depth}-bit.")
        return {"ui": ui, "result": (output_frames, output_path)}
