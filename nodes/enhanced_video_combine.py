import datetime
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time

import torch
from PIL import Image

import folder_paths


_CODEC_OPTIONS = ["Auto", "AV1", "VP9", "H.265 (HEVC)", "H.264"]
_AUTO_CODEC_CANDIDATES = ("AV1", "H.265 (HEVC)", "VP9", "H.264")
_CONTAINER_OPTIONS = ["Auto", "WebM", "MKV", "MP4", "Animated WebP", "Animated AVIF"]
_ENCODER_NAMES = {
    "H.264": ("h264_nvenc", "h264_qsv", "h264_amf", "h264_vaapi", "libx264"),
    "H.265 (HEVC)": ("hevc_nvenc", "hevc_qsv", "hevc_amf", "hevc_vaapi", "libx265"),
    "AV1": ("av1_nvenc", "av1_qsv", "av1_amf", "av1_vaapi", "libsvtav1", "libaom-av1"),
    "VP9": ("vp9_qsv", "vp9_vaapi", "libvpx-vp9"),
}
_CONTAINER_EXTENSIONS = {"WebM": ".webm", "MKV": ".mkv", "MP4": ".mp4"}
_ANIMATED_IMAGE_SETTINGS = {
    "Animated WebP": (".webp", "libwebp_anim", "image/webp"),
    "Animated AVIF": (".avif", "libaom-av1", "image/avif"),
}
_ANIMATED_AVIF_ENCODERS = ("av1_nvenc", "av1_qsv", "av1_amf", "av1_vaapi", "libsvtav1", "libaom-av1")
_AUDIO_CODEC_OPTIONS = ["Auto", "AAC", "Opus", "MP3"]
_AUDIO_ENCODERS = {"AAC": "aac", "Opus": "libopus", "MP3": "libmp3lame"}
_AUDIO_BITRATE_OPTIONS = ["64k", "96k", "128k", "160k", "192k", "256k", "320k"]
_MAX_RAW_FRAME_CHUNK_BYTES = 64 * 1024 * 1024
_MAX_FFMPEG_STDERR_BYTES = 4 * 1024 * 1024


def _log(message):
    print(f"[DaSiWa Enhanced Video Combine] {message}")


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


def _codec_candidates(codec):
    return _AUTO_CODEC_CANDIDATES if codec == "Auto" else (codec,)


def _animated_image_settings(container):
    settings = _ANIMATED_IMAGE_SETTINGS.get(container)
    return settings[:2] if settings else None


def _animated_image_encoder_candidates(container):
    if container == "Animated AVIF":
        return _ANIMATED_AVIF_ENCODERS
    settings = _animated_image_settings(container)
    return (settings[1],) if settings else ()


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


def _frames_per_chunk(images, bit_depth, max_chunk_bytes):
    bytes_per_frame = images.shape[1] * images.shape[2] * 3 * (2 if bit_depth == 10 else 1)
    return max(1, min(32, max_chunk_bytes // bytes_per_frame))


def _iter_frame_byte_chunks(images, bit_depth, pingpong, max_chunk_bytes=_MAX_RAW_FRAME_CHUNK_BYTES):
    frames_per_chunk = _frames_per_chunk(images, bit_depth, max_chunk_bytes)
    for start in range(0, len(images), frames_per_chunk):
        yield _frame_bytes(images[start:start + frames_per_chunk], bit_depth)
    if pingpong:
        for stop in range(len(images) - 1, 1, -frames_per_chunk):
            start = max(1, stop - frames_per_chunk)
            yield _frame_bytes(images[start:stop].flip(0), bit_depth)


def _encoded_frame_count(images, pingpong):
    return len(images) + (len(images) - 2 if pingpong and len(images) >= 3 else 0)


def _save_frame_exports(images, output_path, save_first_frame, save_last_frame, pingpong=False):
    """Write selected source frames beside the encoded video without browser downloads."""
    if not (save_first_frame or save_last_frame):
        return []

    frame_stem = os.path.splitext(output_path)[0]
    exports = []
    last_frame = images[1] if pingpong and len(images) >= 3 else images[-1]
    for suffix, frame, enabled in (
        ("first", images[0], save_first_frame),
        ("last", last_frame, save_last_frame),
    ):
        if not enabled:
            continue
        path = f"{frame_stem}-{suffix}-frame.png"
        pixels = torch.round(frame[..., :3].detach().to(device="cpu", dtype=torch.float32).clamp(0, 1) * 255).to(torch.uint8).numpy()
        Image.fromarray(pixels, mode="RGB").save(path, "PNG")
        exports.append(path)
    return exports


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


def _run_ffmpeg(command, frame_chunks, progress_callback=None):
    """Run FFmpeg and translate its throttled progress stream into frame progress."""
    if isinstance(frame_chunks, bytes):
        if progress_callback is None:
            return subprocess.run(command, input=frame_chunks, capture_output=True, timeout=3600)
        frame_chunks = lambda: (frame_chunks,)
    if progress_callback is None:
        progress_callback = lambda _encoded_seconds: None

    process = subprocess.Popen(
        [*command[:-1], "-progress", "pipe:2", "-nostats", command[-1]],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    stderr_tail = bytearray()
    last_report = 0.0

    def read_stderr():
        nonlocal last_report
        for raw_line in iter(process.stderr.readline, b""):
            stderr_tail.extend(raw_line)
            if len(stderr_tail) > _MAX_FFMPEG_STDERR_BYTES:
                del stderr_tail[:-_MAX_FFMPEG_STDERR_BYTES]
            try:
                key, value = raw_line.decode(errors="replace").strip().split("=", 1)
                if key == "out_time_us":
                    now = time.monotonic()
                    if now - last_report >= 0.5:
                        progress_callback(int(value) / 1_000_000)
                        last_report = now
            except ValueError:
                pass

    stderr_thread = threading.Thread(target=read_stderr, name="dasiwa-ffmpeg-progress", daemon=True)
    stderr_thread.start()
    stdin = process.stdin
    if stdin is None:
        process.kill()
        process.wait()
        stderr_thread.join()
        raise RuntimeError("Could not open FFmpeg stdin.")
    try:
        for chunk in frame_chunks():
            stdin.write(chunk)
        stdin.close()
        returncode = process.wait(timeout=3600)
    except BrokenPipeError:
        try:
            stdin.close()
        except BrokenPipeError:
            pass
        returncode = process.wait(timeout=3600)
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        stderr_thread.join()
    return subprocess.CompletedProcess(command, returncode, stderr=bytes(stderr_tail))


def _encode_with_available_encoder(
    ffmpeg, codec, bit_depth, width, height, frame_rate, frame_chunks, output_path,
    container, cq, crf, metadata_path, audio_path=None, audio_duration=None, crop_to_audio=False,
    audio_codec="Auto", audio_bitrate="192k", progress_callback=None,
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
            result = _run_ffmpeg(command + [output_path], frame_chunks, progress_callback)
            if result.returncode == 0:
                if selected_audio_encoder and selected_audio_encoder != _audio_encoder(audio_codec, container):
                    _log(f"Audio fallback: {selected_audio_encoder}.")
                audio_details = f"; audio={selected_audio_encoder}/{audio_bitrate}" if selected_audio_encoder else ""
                _log(f"Encoded {codec}/{container} via {encoder}{audio_details} -> {os.path.basename(output_path)}.")
                return encoder
            error_lines = result.stderr.decode(errors="replace").splitlines()
            error = error_lines[0][:180] if error_lines else "unknown FFmpeg error"
            attempts.append(f"{encoder}/{selected_audio_encoder or 'no-audio'}: {error}")
    raise RuntimeError("No usable encoder was found. " + " | ".join(attempts))


def _encode_animated_image(ffmpeg, container, bit_depth, width, height, frame_rate, frame_chunks, output_path, quality, progress_callback=None):
    available = _available_encoders(ffmpeg)
    attempts = []
    for encoder in _animated_image_encoder_candidates(container):
        if encoder not in available:
            continue
        command = [
            ffmpeg, "-y", "-v", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb48le" if bit_depth == 10 else "rgb24",
            "-s", f"{width}x{height}", "-framerate", str(frame_rate), "-i", "-",
        ]
        if container == "Animated WebP":
            command.extend(["-c:v", encoder, "-loop", "0", "-q:v", str(quality)])
        else:
            command.extend(_encoder_arguments("AV1", encoder, bit_depth, quality, quality))
            command.extend(["-still-picture", "0", "-f", "avif"])
        result = _run_ffmpeg(command + [output_path], frame_chunks, progress_callback)
        if result.returncode == 0:
            _log(f"Encoded {container} via {encoder} -> {os.path.basename(output_path)}.")
            return encoder
        error_lines = result.stderr.decode(errors="replace").splitlines()
        error = error_lines[0][:180] if error_lines else "unknown FFmpeg error"
        attempts.append(f"{encoder}: {error}")
    if not attempts:
        raise RuntimeError(f"FFmpeg does not provide a usable encoder for {container}.")
    raise RuntimeError(f"{container} encode failed. " + " | ".join(attempts))


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
                "codec": (_CODEC_OPTIONS, {"default": "Auto", "description": "Auto tests AV1, H.265, VP9, then H.264 encoders and uses the first working codec."}),
                "container": (_CONTAINER_OPTIONS, {"default": "Auto", "description": "Auto tries video containers only. Select Animated WebP or Animated AVIF manually; these image animations ignore codec selection and cannot include audio."}),
                "bit_depth": (["Auto", "8-bit", "10-bit"], {"default": "Auto"}),
                "quality": ("INT", {"default": 20, "min": 0, "max": 51, "description": "Encoding quality for every encoder. 0 is no compression (largest files); higher values increase compression and reduce quality. 20 is the recommended default."}),
                "log_level": (["Standard", "Verbose"], {"default": "Standard", "description": "Legacy workflow compatibility; logging is always concise."}),
                "pingpong": ("BOOLEAN", {"default": False, "description": "Append the interior frames in reverse order for seamless forward/reverse playback."}),
                "save_metadata": ("BOOLEAN", {"default": True, "description": "Embed ComfyUI prompt and workflow metadata for workflow-aware video loaders."}),
                "filename_prefix": ("STRING", {"default": "video_%date:hhmmss%", "description": "Output path/name prefix. Supports %date% and formatted dates such as video/%date:yyyy-MM-dd%/%date:hhmmss%."}),
                "save_output": ("BOOLEAN", {"default": True}),
                "pass_frames": ("BOOLEAN", {"default": False, "description": "Return the encoded frame sequence for downstream processing."}),
                "crop_to_audio": ("BOOLEAN", {"default": False, "description": "When audio is connected, end the output video at the audio duration."}),
                "audio_codec": (_AUDIO_CODEC_OPTIONS, {"default": "Auto", "description": "Audio codec. Auto uses Opus for WebM and AAC for MKV/MP4."}),
                "audio_bitrate": (_AUDIO_BITRATE_OPTIONS, {"default": "192k", "description": "Target bitrate for the connected audio stream."}),
                "save_first_frame": ("BOOLEAN", {"default": False, "description": "Write the first frame as a PNG beside the encoded video."}),
                "save_last_frame": ("BOOLEAN", {"default": False, "description": "Write the last frame as a PNG beside the encoded video."}),
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

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always encode again when the output node is queued with unchanged frames."""
        return float("nan")

    def validate_inputs(self, *args, **kwargs):
        return True

    def combine(
        self, images, frame_rate, codec, container, bit_depth, quality, pingpong,
        save_metadata, filename_prefix, save_output, pass_frames, crop_to_audio=False, audio_codec="Auto",
        audio_bitrate="192k", log_level="Standard", save_first_frame=False, save_last_frame=False, audio=None, prompt=None,
        extra_pnginfo=None,
    ):
        if images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("images must be an IMAGE batch shaped [frames, height, width, channels] with RGB channels.")

        try:
            import comfy.utils

            progress_bar = comfy.utils.ProgressBar(_encoded_frame_count(images, pingpong))
        except ImportError:
            progress_bar = None

        def report_encode_progress(encoded_seconds):
            if progress_bar is not None:
                progress_bar.update_absolute(min(_encoded_frame_count(images, pingpong), max(0, int(encoded_seconds * frame_rate))))

        if progress_bar is not None:
            progress_bar.update_absolute(0)
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
        frame_chunks = lambda: _iter_frame_byte_chunks(images, selected_bit_depth, pingpong)
        attempts = []
        _log(
            f"Encode {_encoded_frame_count(images, pingpong)}f {width}x{height}@{frame_rate:g}fps {selected_bit_depth}-bit; "
            f"codec={codec}, container={container}, audio={'yes' if audio_path else 'no'}."
        )
        try:
            animated_settings = _animated_image_settings(container)
            if animated_settings:
                if audio_path:
                    _log(f"{container} does not support audio; connected audio is omitted.")
                output_path = os.path.join(output_folder, _output_filename(filename, counter, animated_settings[0], False))
                encoder = _encode_animated_image(
                    ffmpeg, container, selected_bit_depth, width, height, frame_rate, frame_chunks, output_path, quality,
                    report_encode_progress,
                )
                selected_container = container
                selected_codec = container
            else:
                for selected_codec in _codec_candidates(codec):
                    for selected_container in _container_candidates(selected_codec, container):
                        if codec == "Auto":
                            _log(f"Auto test: {selected_codec}/{selected_container}.")
                        output_path = os.path.join(
                            output_folder,
                            _output_filename(filename, counter, _CONTAINER_EXTENSIONS[selected_container], audio_path is not None),
                        )
                        try:
                            encoder = _encode_with_available_encoder(
                                ffmpeg, selected_codec, selected_bit_depth, width, height, frame_rate, frame_chunks,
                                output_path, selected_container, quality, quality, metadata_path, audio_path, audio_duration, crop_to_audio,
                                audio_codec, audio_bitrate, report_encode_progress,
                            )
                            break
                        except RuntimeError as error:
                            attempts.append(f"{selected_codec}/{selected_container}: {error}")
                            if codec == "Auto":
                                _log(f"Auto miss: {selected_codec}/{selected_container}; trying next.")
                    else:
                        continue
                    break
                else:
                    fallback_path = os.path.join(output_folder, _output_filename(filename, counter, ".mp4", audio_path is not None))
                    encoder = _encode_with_available_encoder(
                        ffmpeg, "H.264", selected_bit_depth, width, height, frame_rate, frame_chunks,
                        fallback_path, "MP4", quality, quality, metadata_path, audio_path, audio_duration, crop_to_audio,
                        audio_codec, audio_bitrate, report_encode_progress,
                    )
                    output_path = fallback_path
                    selected_container = "MP4"
                    selected_codec = "H.264"
        finally:
            if metadata_path:
                os.unlink(metadata_path)
            if audio_path:
                os.unlink(audio_path[0])

        output_frames = _pingpong_frames(images, pingpong) if pass_frames else images[:0]
        if progress_bar is not None:
            progress_bar.update_absolute(_encoded_frame_count(images, pingpong))
        frame_exports = _save_frame_exports(images, output_path, save_first_frame, save_last_frame, pingpong)
        animated_settings = _animated_image_settings(selected_container)
        preview_path = output_path
        preview_codec = selected_codec
        if selected_codec == "H.265 (HEVC)":
            preview_path = f"{os.path.splitext(output_path)[0]}-preview.mp4"
            try:
                _encode_with_available_encoder(
                    ffmpeg, "H.264", 8, width, height, frame_rate,
                    lambda: _iter_frame_byte_chunks(images, 8, pingpong), preview_path,
                    "MP4", quality, quality, None,
                )
                preview_codec = "H.264"
            except RuntimeError as error:
                if os.path.exists(preview_path):
                    os.unlink(preview_path)
                preview_path = output_path
                _log(f"Browser preview fallback failed: {error}")
        preview_container = "MP4" if preview_codec == "H.264" and preview_path != output_path else selected_container
        mime_types = {"WebM": "video/webm", "MKV": "video/x-matroska", "MP4": "video/mp4", **{name: settings[2] for name, settings in _ANIMATED_IMAGE_SETTINGS.items()}}
        preview_mime_type = mime_types[preview_container]
        output_mime_type = mime_types[selected_container]
        assets = [{"filename": os.path.basename(output_path), "subfolder": subfolder, "type": output_type, "format": output_mime_type, "width": width, "height": height}]
        assets.extend(
            {"filename": os.path.basename(path), "subfolder": subfolder, "type": output_type, "format": "image/png", "width": width, "height": height}
            for path in frame_exports
        )
        ui = {"images": assets}
        if not animated_settings:
            ui["gifs"] = [{"filename": os.path.basename(preview_path), "subfolder": subfolder, "type": output_type, "format": preview_mime_type, "width": width, "height": height, "fps": frame_rate}]
        _log(f"Output: {output_path} ({selected_codec}, {encoder}, {selected_bit_depth}-bit).")
        for path in frame_exports:
            _log(f"Frame export: {path}.")
        return {"ui": ui, "result": (output_frames, output_path)}
