import asyncio
import datetime
import hashlib
import os
import re

import torch
from PIL import Image

import folder_paths
try:
    from aiohttp import web
    from server import PromptServer
except ImportError:
    web = None
    PromptServer = None
try:
    from .helper_logging import log_dasiwa
    from . import helper_pyav_video
except ImportError:
    from helper_logging import log_dasiwa
    import helper_pyav_video


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
    log_dasiwa("Enhanced Video Combine", message)


def _preview_source_path(filename, subfolder, output_type):
    """Resolve an output asset without allowing traversal or alternate roots."""
    if output_type != "output" or not filename or filename != os.path.basename(filename) or ".." in subfolder:
        return None
    output_dir = folder_paths.get_directory_by_type("output")
    if not output_dir:
        return None
    try:
        root = os.path.realpath(output_dir)
        candidate = os.path.realpath(os.path.join(root, subfolder, filename))
        if os.path.commonpath((candidate, root)) != root or not os.path.isfile(candidate):
            return None
    except (OSError, ValueError):
        return None
    return candidate


def _preview_cache_path(source):
    stat = os.stat(source)
    identity = f"{os.path.realpath(source)}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    cache_dir = os.path.join(folder_paths.get_temp_directory(), "dasiwa-video-previews")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, hashlib.sha256(identity).hexdigest() + ".mp4")


if PromptServer is not None:
    @PromptServer.instance.routes.get("/dasiwa/enhanced-video-preview")
    async def enhanced_video_preview(request):
        """Transcode unsupported outputs to fragmented H.264 MP4 without a sidecar file."""
        source = _preview_source_path(
            request.rel_url.query.get("filename", ""),
            request.rel_url.query.get("subfolder", ""),
            request.rel_url.query.get("type", "output"),
        )
        if source is None:
            return web.Response(status=404)
        try:
            preview_path = _preview_cache_path(source)
            if not os.path.isfile(preview_path):
                await asyncio.to_thread(helper_pyav_video.transcode_preview, source, preview_path)
        except Exception as error:
            return web.Response(status=503, text=f"Preview transcode failed: {error}")
        return web.FileResponse(
            preview_path,
            headers={"Content-Type": "video/mp4", "Cache-Control": "private, max-age=3600"},
        )


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


def _auto_container_candidates(codec, container):
    if container != "Auto":
        return _container_candidates(codec, container)
    return {"AV1": ("WebM",), "VP9": ("WebM",), "H.264": ("MP4",)}[codec]


def _codec_candidates(codec):
    # H.265 is intentionally explicit-only: unlike AV1/VP9/H.264 it is not a
    # broadly browser-compatible WebM/MP4 fallback.
    return ("AV1", "VP9", "H.264") if codec == "Auto" else (codec,)


def _selected_bit_depth(codec, bit_depth, images):
    requested_bit_depth = {"8-bit": 8, "10-bit": 10}.get(bit_depth)
    if requested_bit_depth is not None:
        return requested_bit_depth
    # Auto uses AV1 first and must therefore avoid 10-bit browser decoder gaps.
    return 8 if codec == "Auto" else detect_bit_depth(images)


def _animated_image_settings(container):
    settings = _ANIMATED_IMAGE_SETTINGS.get(container)
    return settings[:2] if settings else None


def _animated_image_encoder_candidates(container):
    if container == "Animated AVIF":
        return _ANIMATED_AVIF_ENCODERS
    settings = _animated_image_settings(container)
    return (settings[1],) if settings else ()


def _available_encoders(_backend=None):
    return helper_pyav_video.available_encoders()


def _frame_arrays(images, bit_depth, pingpong):
    frames = _pingpong_frames(images, pingpong)
    for image in frames:
        pixels = image[..., :3].detach().to(device="cpu", dtype=torch.float32).clamp(0, 1)
        if bit_depth == 10:
            yield torch.round(pixels * 1023).to(torch.int32).mul_(64).to(torch.uint16).numpy()
        else:
            yield torch.round(pixels * 255).to(torch.uint8).numpy()


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


def _format_filename_prefix(filename_prefix, seed=None):
    """Expand supported output-name placeholders without changing unknown tokens."""
    now = datetime.datetime.now()

    def replace_date(match):
        fmt = match.group(1) or "yyyyMMdd_HHmmss"
        fmt = fmt.replace("yyyy", "%Y").replace("yy", "%y")
        fmt = fmt.replace("MM", "%m").replace("dd", "%d").replace("DD", "%d")
        fmt = fmt.replace("HH", "%H").replace("hh", "%H")
        fmt = fmt.replace("mm", "%M").replace("ss", "%S")
        return now.strftime(fmt)

    formatted = re.sub(r"%date(?::([^%]+))?%", replace_date, filename_prefix)
    return formatted.replace("%seed%", str(int(seed))) if seed is not None else formatted


def _output_filename(filename, counter, extension, has_audio):
    # Use an underscore (not a dash) for the audio marker: ComfyUI's
    # get_save_image_path counter-scanner only parses digits when the char
    # right after the filename prefix is "_". A dash makes the digit block
    # unparseable, so the counter resets to 1 and audio outputs overwrite.
    audio_suffix = "_audio" if has_audio else ""
    return f"{filename}_{counter:05}{audio_suffix}.{extension.lstrip('.')}"


def _metadata_file(prompt, extra_pnginfo):
    metadata = {}
    if prompt is not None:
        metadata["prompt"] = prompt
    if extra_pnginfo:
        metadata.update(extra_pnginfo)
    return metadata or None


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
    planar = waveform.permute(1, 0, 2).reshape(channels, -1).contiguous().clamp_(-1, 1).numpy()
    return (planar, sample_rate), planar.shape[1] / sample_rate


def _audio_encoder(audio_codec, container):
    if isinstance(audio_codec, bool):
        audio_codec = "Auto"
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
    _backend, codec, bit_depth, width, height, frame_rate, frames, output_path,
    container, cq, crf, metadata, audio=None, audio_duration=None, crop_to_audio=False,
    audio_codec="Auto", audio_bitrate="192k", progress_callback=None,
):
    available = _available_encoders()
    attempts = []
    for encoder in _ENCODER_NAMES[codec]:
        if encoder not in available:
            continue
        audio_encoders = _audio_encoder_candidates(audio_codec, container) if audio else (None,)
        for selected_audio_encoder in audio_encoders:
            if selected_audio_encoder and selected_audio_encoder not in available:
                continue
            try:
                helper_pyav_video.encode_attempt(
                    output_path, container, encoder, width, height, frame_rate, bit_depth, cq,
                    frames, metadata, audio, selected_audio_encoder, audio_bitrate,
                    crop_to_audio, progress_callback,
                )
                if selected_audio_encoder and selected_audio_encoder != _audio_encoder(audio_codec, container):
                    _log(f"Audio fallback: {selected_audio_encoder}.")
                audio_details = f"; audio={selected_audio_encoder}/{audio_bitrate}" if selected_audio_encoder else ""
                _log(f"Encoded {codec}/{container} via {encoder}{audio_details} -> {os.path.basename(output_path)}.")
                return encoder
            except Exception as error:
                attempts.append(f"{encoder}/{selected_audio_encoder or 'no-audio'}: {str(error)[:180]}")
    raise RuntimeError("No usable encoder was found. " + " | ".join(attempts))


def _encode_animated_image(_backend, container, bit_depth, width, height, frame_rate, frames, output_path, quality, progress_callback=None):
    available = _available_encoders()
    attempts = []
    for encoder in _animated_image_encoder_candidates(container):
        if encoder not in available:
            continue
        try:
            helper_pyav_video.encode_attempt(
                output_path, container, encoder, width, height, frame_rate,
                8 if container == "Animated WebP" else bit_depth, quality,
                frames, progress_callback=progress_callback,
            )
            _log(f"Encoded {container} via {encoder} -> {os.path.basename(output_path)}.")
            return encoder
        except Exception as error:
            attempts.append(f"{encoder}: {str(error)[:180]}")
    if not attempts:
        raise RuntimeError(f"PyAV does not provide a usable encoder for {container}.")
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
                "codec": (_CODEC_OPTIONS, {"default": "Auto", "description": "Auto prefers browser-compatible 8-bit AV1/WebM, then VP9 and H.264 fallbacks. Choose H.265 explicitly when required."}),
                "container": (_CONTAINER_OPTIONS, {"default": "Auto", "description": "Auto tries video containers only. Select Animated WebP or Animated AVIF manually; these image animations ignore codec selection and cannot include audio."}),
                "bit_depth": (["Auto", "8-bit", "10-bit"], {"default": "Auto"}),
                "quality": ("INT", {"default": 20, "min": 0, "max": 51, "description": "Encoding quality for every encoder. 0 is no compression (largest files); higher values increase compression and reduce quality. 20 is the recommended default."}),
                "log_level": (["Standard", "Verbose"], {"default": "Standard", "description": "Legacy workflow compatibility; logging is always concise."}),
                "pingpong": ("BOOLEAN", {"default": False, "description": "Append the interior frames in reverse order for seamless forward/reverse playback."}),
                "save_metadata": ("BOOLEAN", {"default": True, "description": "Embed ComfyUI prompt and workflow metadata for workflow-aware video loaders."}),
                "filename_prefix": ("STRING", {"default": "video_%date:hhmmss%", "description": "Output path/name prefix. Supports %seed%, %date%, and formatted dates such as video/%date:yyyy-MM-dd%/%date:hhmmss%. Connect the optional seed input to expand %seed%."}),
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
                "seed": ("INT", {"forceInput": True, "description": "Optional generation seed used to expand %seed% in filename_prefix."}),
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
        audio_bitrate="192k", log_level="Standard", save_first_frame=False, save_last_frame=False, audio=None, seed=None,
        prompt=None, extra_pnginfo=None,
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
        selected_bit_depth = _selected_bit_depth(codec, bit_depth, images)
        output_dir = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
        output_type = "output" if save_output else "temp"
        height, width = images.shape[1:3]
        filename_prefix = _format_filename_prefix(filename_prefix, seed=seed)
        output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, output_dir, width, height)
        metadata = _metadata_file(prompt, extra_pnginfo) if save_metadata else None
        audio_data, audio_duration = _audio_file(audio)
        frames = lambda: _frame_arrays(images, selected_bit_depth, pingpong)
        attempts = []
        _log(
            f"Encode {_encoded_frame_count(images, pingpong)}f {width}x{height}@{frame_rate:g}fps {selected_bit_depth}-bit; "
            f"codec={codec}, container={container}, audio={'yes' if audio_data else 'no'}."
        )
        try:
            animated_settings = _animated_image_settings(container)
            if animated_settings:
                if audio_data:
                    _log(f"{container} does not support audio; connected audio is omitted.")
                output_path = os.path.join(output_folder, _output_filename(filename, counter, animated_settings[0], False))
                encoder = _encode_animated_image(
                    None, container, selected_bit_depth, width, height, frame_rate,
                    lambda: _frame_arrays(images, 8 if container == "Animated WebP" else selected_bit_depth, pingpong),
                    output_path, quality,
                    report_encode_progress,
                )
                selected_container = container
                selected_codec = container
                if container == "Animated WebP":
                    selected_bit_depth = 8
            else:
                for selected_codec in _codec_candidates(codec):
                    container_candidates = _auto_container_candidates(selected_codec, container) if codec == "Auto" else _container_candidates(selected_codec, container)
                    for selected_container in container_candidates:
                        if codec == "Auto":
                            _log(f"Auto test: {selected_codec}/{selected_container}.")
                        output_path = os.path.join(
                            output_folder,
                            _output_filename(filename, counter, _CONTAINER_EXTENSIONS[selected_container], audio_data is not None),
                        )
                        try:
                            encoder = _encode_with_available_encoder(
                                None, selected_codec, selected_bit_depth, width, height, frame_rate, frames,
                                output_path, selected_container, quality, quality, metadata, audio_data, audio_duration, crop_to_audio,
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
                    fallback_path = os.path.join(output_folder, _output_filename(filename, counter, ".mp4", audio_data is not None))
                    encoder = _encode_with_available_encoder(
                        None, "H.264", selected_bit_depth, width, height, frame_rate, frames,
                        fallback_path, "MP4", quality, quality, metadata, audio_data, audio_duration, crop_to_audio,
                        audio_codec, audio_bitrate, report_encode_progress,
                    )
                    output_path = fallback_path
                    selected_container = "MP4"
                    selected_codec = "H.264"
        finally:
            pass

        output_frames = _pingpong_frames(images, pingpong) if pass_frames else images[:0]
        if progress_bar is not None:
            progress_bar.update_absolute(_encoded_frame_count(images, pingpong))
        frame_exports = _save_frame_exports(images, output_path, save_first_frame, save_last_frame, pingpong)
        animated_settings = _animated_image_settings(selected_container)
        mime_types = {"WebM": "video/webm", "MKV": "video/x-matroska", "MP4": "video/mp4", **{name: settings[2] for name, settings in _ANIMATED_IMAGE_SETTINGS.items()}}
        output_mime_type = mime_types[selected_container]
        main_asset = {"filename": os.path.basename(output_path), "subfolder": subfolder, "type": output_type, "format": output_mime_type, "width": width, "height": height, "codec": selected_codec, "bit_depth": selected_bit_depth, "container": selected_container}
        frame_assets = [
            {"filename": os.path.basename(path), "subfolder": subfolder, "type": output_type, "format": "image/png", "width": width, "height": height}
            for path in frame_exports
        ]
        # Real video containers (WebM/MKV/MP4) aren't PIL-openable images: only animated
        # image formats (GIF/WebP/AVIF) belong in "images", or /view 500s trying to decode them.
        ui = {"images": ([main_asset] if animated_settings else []) + frame_assets}
        if not animated_settings:
            ui["gifs"] = [{"filename": os.path.basename(output_path), "subfolder": subfolder, "type": output_type, "format": output_mime_type, "codec": selected_codec, "bit_depth": selected_bit_depth, "container": selected_container, "width": width, "height": height, "fps": frame_rate}]
        _log(f"Output: {output_path} ({selected_codec}, {encoder}, {selected_bit_depth}-bit).")
        for path in frame_exports:
            _log(f"Frame export: {path}.")
        return {"ui": ui, "result": (output_frames, output_path)}
