import contextlib
import json
import math
import os
import tempfile
import uuid
from fractions import Fraction

import av
import numpy as np
from av.video.frame import PictureType


FORMAT_NAMES = {"WebM": "webm", "MKV": "matroska", "MP4": "mp4", "Animated WebP": "webp", "Animated AVIF": "avif"}
DEVICE_TYPES = {"_nvenc": "cuda", "_qsv": "qsv", "_amf": "amf", "_vaapi": "vaapi", "_videotoolbox": "videotoolbox"}


def available_encoders():
    encoders = set()
    for name in av.codecs_available:
        try:
            av.Codec(name, "w")
            encoders.add(name)
        except (av.FFmpegError, ValueError):
            pass
    return encoders


def device_type_for_encoder(name):
    return next((device for suffix, device in DEVICE_TYPES.items() if name.endswith(suffix)), None)


def video_options(codec, encoder, quality):
    q = str(quality)
    if encoder.endswith("_nvenc"):
        return {"preset": "p5", "cq": q}
    if encoder.endswith("_qsv"):
        return {"global_quality": q}
    if encoder.endswith("_amf"):
        return {"quality": "quality", "qp_i": q, "qp_p": q}
    if encoder.endswith("_vaapi"):
        return {"qp": q}
    if encoder in {"libx264", "libx265"}:
        return {"crf": q, "preset": "medium"}
    if encoder == "libsvtav1":
        return {"crf": q, "preset": "6"}
    if encoder == "libaom-av1":
        return {"crf": q, "b": "0", "cpu-used": "6"}
    if encoder == "libvpx-vp9":
        return {"crf": q, "b": "0", "deadline": "good"}
    if encoder in {"libwebp_anim", "libwebp"}:
        # libwebp quality runs 100 (best) -> 0; map the shared 0 (best) -> 51 CRF-style slider onto it.
        return {"quality": str(round(100 - min(int(quality), 51) * 100 / 51))}
    return {}


def _rate(value):
    return Fraction(float(value)).limit_denominator(1_000_000)


def _is_second_boundary(index, rate):
    if index == 0:
        return True
    return (Fraction(index, 1) / rate).numerator // (Fraction(index, 1) / rate).denominator > (Fraction(index - 1, 1) / rate).numerator // (Fraction(index - 1, 1) / rate).denominator


def _video_frame(array, index, rate, bit_depth, force_keyframe=True):
    frame = av.VideoFrame.from_ndarray(array, format="rgb48le" if bit_depth == 10 else "rgba" if array.shape[-1] == 4 else "rgb24")
    frame.pts = index
    frame.time_base = Fraction(rate.denominator, rate.numerator)
    if force_keyframe and _is_second_boundary(index, rate):
        frame.pict_type = PictureType.I
    return frame


def _hardware_accel(encoder):
    device_type = device_type_for_encoder(encoder)
    if not device_type:
        return None
    kwargs = {"device_type": device_type, "allow_software_fallback": False}
    if device_type == "vaapi" and os.path.exists("/dev/dri/renderD128"):
        kwargs["device"] = "/dev/dri/renderD128"
    return av.codec.hwaccel.HWAccel(**kwargs)


def _configure_video_stream(container, encoder, width, height, frame_rate, bit_depth, quality, alpha=False):
    kwargs = {"rate": _rate(frame_rate), "options": video_options("", encoder, quality)}
    hwaccel = _hardware_accel(encoder)
    if hwaccel is not None:
        kwargs["hwaccel"] = hwaccel
    stream = container.add_stream(encoder, **kwargs)
    stream.width = width
    stream.height = height
    stream.codec_context.gop_size = max(1, math.ceil(float(_rate(frame_rate))))
    if hwaccel is None:
        stream.pix_fmt = "yuva420p" if alpha else "yuv420p10le" if bit_depth == 10 else "yuv420p"
    else:
        device_type = device_type_for_encoder(encoder)
        codec = av.Codec(encoder, "w")
        hardware_format = next(
            (
                config.format.name for config in codec.hardware_configs
                if config.format is not None and config.device_type.name == device_type
                and config.format.name in {"cuda", "qsv", "amf", "vaapi", "videotoolbox"}
            ),
            None,
        )
        stream.pix_fmt = hardware_format or ("p010le" if bit_depth == 10 else "nv12")
        stream.codec_context.sw_format = "p010le" if bit_depth == 10 else "nv12"
    return stream


def _audio_layout(channels):
    layouts = {1: "mono", 2: "stereo", 3: "2.1", 4: "quad", 5: "5.0", 6: "5.1", 7: "6.1", 8: "7.1"}
    if channels not in layouts:
        raise ValueError(f"PyAV audio encoding supports 1 through 8 channels, not {channels}.")
    return layouts[channels]


def _audio_rate(encoder, source_rate):
    codec = av.Codec(encoder, "w")
    rates = sorted(codec.audio_rates or [])
    if not rates:
        return source_rate
    if encoder == "libopus" and source_rate == 32000 and 48000 in rates:
        return 48000  # Opus decodes at 48 kHz; avoid a lossy 32 -> 24 -> 48 kHz path.
    return min(rates, key=lambda candidate: abs(candidate - source_rate))


def _configure_audio_stream(container, encoder, source_rate, channels, bitrate):
    target_rate = _audio_rate(encoder, source_rate)
    stream = container.add_stream(encoder, rate=target_rate)
    stream.layout = _audio_layout(channels)
    stream.bit_rate = int(str(bitrate).rstrip("k")) * 1000 if str(bitrate).endswith("k") else int(bitrate)
    return stream, target_rate


def segment_sample_interval(interval, sample_rate):
    if not isinstance(interval, dict) or set(interval) != {"start", "end"}:
        raise ValueError("Invalid H3 segment audio interval.")
    bounds = []
    for key in ("start", "end"):
        pair = interval[key]
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(type(value) is not int for value in pair) or pair[0] < 0 or pair[1] <= 0):
            raise ValueError("H3 segment audio boundaries must be nonnegative rational times.")
        bounds.append(Fraction(*pair))
    start, end = bounds
    if end <= start:
        raise ValueError("H3 segment audio interval must have positive duration.")
    return round(end * sample_rate) - round(start * sample_rate)


def _encode_audio(container, stream, audio, target_rate):
    waveform, source_rate = audio[:2]
    expected = None
    if len(audio) == 3:
        interval = audio[2]
        if waveform.shape[1] != segment_sample_interval(interval, source_rate):
            raise ValueError("H3 segment audio length differs from its source sample interval.")
        expected = segment_sample_interval(interval, target_rate)
    channels = waveform.shape[0]
    layout = _audio_layout(channels)
    codec_formats = [fmt.name for fmt in (av.Codec(stream.codec_context.name, "w").audio_formats or [])]
    target_format = codec_formats[0] if codec_formats else "fltp"
    resampler = av.AudioResampler(format=target_format, layout=layout, rate=target_rate, frame_size=stream.codec_context.frame_size or 1024)
    def converted_frames():
        for start in range(0, waveform.shape[1], 32768):
            chunk = np.ascontiguousarray(waveform[:, start:start + 32768], dtype=np.float32)
            frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout=layout)
            frame.sample_rate = source_rate
            frame.pts = start
            frame.time_base = Fraction(1, source_rate)
            yield from resampler.resample(frame)
        yield from resampler.resample(None)

    total = written = 0
    converted = None
    for converted in converted_frames():
        total += converted.samples
        if expected is not None:
            keep = min(converted.samples, max(0, expected - written))
            if not keep:
                continue
            if keep != converted.samples:
                array = converted.to_ndarray()
                stride = 1 if converted.format.is_planar else channels
                converted = av.AudioFrame.from_ndarray(np.ascontiguousarray(array[:, :keep * stride]), format=target_format, layout=layout)
            converted.sample_rate = target_rate
            converted.pts = written
            converted.time_base = Fraction(1, target_rate)
        written += converted.samples
        for packet in stream.encode(converted):
            container.mux(packet)
    if expected is not None:
        # Only global-grid rounding plus at most one resampler sample may differ.
        resampled = Fraction(waveform.shape[1] * target_rate, source_rate)
        if converted is None or abs(total - round(resampled)) > 1 or abs(total - expected) > math.ceil(Fraction(target_rate, source_rate)):
            raise ValueError("H3 segment resampled audio differs meaningfully from its exact interval.")
        if written < expected:
            planar = av.AudioFormat(target_format).is_planar
            dtype = converted.to_ndarray().dtype
            shape = (channels, expected - written) if planar else (1, (expected - written) * channels)
            padding = av.AudioFrame.from_ndarray(np.zeros(shape, dtype=dtype), format=target_format, layout=layout)
            padding.sample_rate = target_rate
            padding.pts = written
            padding.time_base = Fraction(1, target_rate)
            for packet in stream.encode(padding):
                container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)


def _metadata(container, metadata):
    for key, value in (metadata or {}).items():
        container.metadata[str(key)] = json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def encode_attempt(output_path, container_name, encoder, width, height, frame_rate, bit_depth, quality, frames, metadata=None, audio=None, audio_encoder=None, audio_bitrate="192k", crop_to_audio=False, progress_callback=None, alpha=False):
    attempt = f"{output_path}.attempt-{uuid.uuid4().hex}"
    container = None
    try:
        # FFmpeg 7.1 (PyAV 18) writes MP4 edit-list durations in the movie timescale, so the
        # default 1000 ticks/s rounds a track length to milliseconds for readers and for the
        # H3 continuity readback. A 48 kHz movie timescale keeps the sample grid exact.
        options = {"MP4": {"movflags": "+faststart+use_metadata_tags", "movie_timescale": "48000"},
                   "Animated WebP": {"loop": "0"}}.get(container_name, {})
        container = av.open(attempt, "w", format=FORMAT_NAMES[container_name], options=options)
        _metadata(container, metadata)
        video = _configure_video_stream(container, encoder, width, height, frame_rate, bit_depth, quality, alpha)
        audio_stream = None
        target_audio_rate = None
        if audio is not None and audio_encoder:
            audio_stream, target_audio_rate = _configure_audio_stream(container, audio_encoder, audio[1], audio[0].shape[0], audio_bitrate)
        audio_duration = Fraction(audio[0].shape[1], audio[1]) if audio is not None else None
        rate = _rate(frame_rate)
        encoded = 0
        for index, array in enumerate(frames()):
            if crop_to_audio and audio_duration is not None and Fraction(index * rate.denominator, rate.numerator) >= audio_duration:
                break
            for packet in video.encode(_video_frame(array, index, rate, bit_depth)):
                container.mux(packet)
            encoded += 1
            if progress_callback:
                progress_callback(encoded / float(rate))
        for packet in video.encode(None):
            container.mux(packet)
        if audio_stream is not None:
            _encode_audio(container, audio_stream, audio, target_audio_rate)
        container.close()
        container = None
        with av.open(attempt, "r") as check:
            if not check.streams.video:
                raise RuntimeError("Encoded output has no video stream.")
        os.replace(attempt, output_path)
    except Exception:
        if container is not None:
            with contextlib.suppress(Exception):
                container.close()
        with contextlib.suppress(OSError):
            os.unlink(attempt)
        raise


def _transcode_audio(input_container, output, input_stream, output_stream):
    layout = input_stream.codec_context.layout.name
    resampler = av.AudioResampler(
        format="fltp", layout=layout, rate=output_stream.codec_context.sample_rate,
        frame_size=output_stream.codec_context.frame_size or 1024,
    )
    for frame in input_container.decode(audio=0):
        for converted in resampler.resample(frame):
            for packet in output_stream.encode(converted):
                output.mux(packet)
    for converted in resampler.resample(None):
        for packet in output_stream.encode(converted):
            output.mux(packet)
    for packet in output_stream.encode(None):
        output.mux(packet)


def transcode_preview(source, destination=None):
    if destination is None:
        handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        final_path = handle.name
        handle.close()
        path = final_path
    else:
        final_path = destination
        path = f"{destination}.attempt-{uuid.uuid4().hex}"
    try:
        with av.open(source, "r") as input_container:
            input_video = input_container.streams.video[0]
            input_audio = input_container.streams.audio[0] if input_container.streams.audio else None
            output = av.open(path, "w", format="mp4", options={"movflags": "+faststart", "movie_timescale": "48000"})
            rate = input_video.average_rate or Fraction(24, 1)
            video = output.add_stream("libx264", rate=rate, options={"preset": "ultrafast", "crf": "24"})
            video.width = input_video.width
            video.height = input_video.height
            video.pix_fmt = "yuv420p"
            video.codec_context.gop_size = max(1, math.ceil(float(rate)))
            audio = None
            if input_audio is not None:
                audio_rate = input_audio.codec_context.sample_rate or 48000
                audio = output.add_stream("aac", rate=audio_rate)
                audio.layout = input_audio.codec_context.layout.name
            for index, frame in enumerate(input_container.decode(video=0)):
                if _is_second_boundary(index, rate):
                    frame.pict_type = PictureType.I
                for packet in video.encode(frame):
                    output.mux(packet)
            for packet in video.encode(None):
                output.mux(packet)
            if audio is not None:
                with av.open(source, "r") as audio_input:
                    _transcode_audio(audio_input, output, audio_input.streams.audio[0], audio)
            output.close()
        if destination is not None:
            os.replace(path, final_path)
        return final_path
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(path)
        raise
