"""Pure contracts shared by the MiniMax H3 Director nodes.

This module deliberately has no ComfyUI imports so its compatibility rules can be
regression-tested from a source checkout.
"""
from dataclasses import dataclass
from typing import Any
import os

import numpy as np
import torch
from PIL import Image

from .nodes_scaling import DaSiWa_TorchResize

FPS = 24
AUDIO_LATENT_FPS = 40
CANVAS_MULTIPLE = 16
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
MIN_REF_SECONDS = 2.0
MAX_REF_SECONDS = 15.0
MAX_REF_TOTAL_SECONDS = 15.0


@dataclass(frozen=True)
class NormalizedGuide:
    mode: str
    prompt: str
    resolved_prompt: str
    width: int
    height: int
    length: int
    ref_image_size: str = "match"
    first_frame: Any = None
    last_frame: Any = None
    ref_images: dict = None
    ref_videos: dict = None
    ref_video_audios: dict = None
    ref_audios: dict = None
    minimax_ref_items: Any = None
    selection_stamp: float = 0.0

    def __post_init__(self):
        for name in ("ref_images", "ref_videos", "ref_video_audios", "ref_audios"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, {})


def align_frame_count(n: int) -> int:
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length: int) -> tuple[int, int, int]:
    frame_count = align_frame_count(length)
    return frame_count, video_latent_t(frame_count), round(frame_count / FPS * AUDIO_LATENT_FPS)


def adapt_canvas(width: int, height: int) -> tuple[int, int]:
    return scale_canvas_to_short_edge(width, height)


def scale_canvas_to_short_edge(width: int, height: int, short_edge: int = BASE_SHORT_EDGE) -> tuple[int, int]:
    """Preserve a source aspect while setting its shorter output side for H3."""
    if width <= 0 or height <= 0:
        raise ValueError("reference dimensions must be positive")
    scale = float(short_edge) / min(width, height)
    scaled_width, scaled_height = width * scale, height * scale
    return (
        max(CANVAS_MULTIPLE, round(scaled_width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(scaled_height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


INPUT_SCALING_MODES = ("Off", "Auto", "Target", "Fit", "Fill and crop", "Fit and pad", "Long side with divisible crop")


def scale_input_media(image, behavior: str, target_width: int, target_height: int):
    """Apply Director input preprocessing through the existing DaSiWa Torch Resize node."""
    behavior = str(behavior or "Off")
    if behavior not in INPUT_SCALING_MODES:
        raise ValueError(f"unsupported MiniMax Director input scaling behavior: {behavior}")
    if behavior == "Off":
        return image
    if not hasattr(image, "shape") or len(image.shape) != 4:
        return image
    resize = DaSiWa_TorchResize()
    if behavior == "Auto":
        source_short_edge = min(int(image.shape[1]), int(image.shape[2]))
        if source_short_edge <= REF_IMAGE_SHORT_EDGE:
            return image
        return resize.resize(image, "Multiplier", "Fit", target_width, target_height,
                             REF_IMAGE_SHORT_EDGE / source_short_edge, "Lanczos", True,
                             CANVAS_MULTIPLE, "0, 0, 0", "center", 0, 16.0, 64)[0]
    aspect_mode = "Stretch" if behavior == "Target" else behavior
    return resize.resize(image, "Target resolution", aspect_mode, target_width, target_height,
                         1.0, "Lanczos", True, CANVAS_MULTIPLE, "0, 0, 0", "center",
                         0, 16.0, 64)[0]


def assemble_prompt(prompt: str = "", prompt_blocks=None) -> str:
    parts = []
    if prompt and prompt.strip():
        parts.append(prompt)
    blocks = prompt_blocks or []
    ordered = sorted(enumerate(blocks), key=lambda pair: (float(pair[1].get("start", 0)), int(pair[1].get("order", pair[0])), pair[0]))
    for _, block in ordered:
        if block.get("enabled", True) and str(block.get("text", "")).strip():
            parts.append(str(block["text"]))
    return "\n".join(parts)


def _duration(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} duration is invalid") from exc
    if not MIN_REF_SECONDS <= result <= MAX_REF_SECONDS:
        raise ValueError(f"{label} must be between 2 and 15 seconds (got {result:g})")
    return result


def validate_reference_limits(images=(), videos=(), audios=(), *, audio_has_visual=True):
    images = list(images or [])
    videos = list(videos or [])
    audios = list(audios or [])
    if len(images) > 9:
        raise ValueError("REF2VA supports at most 9 images")
    if len(videos) > 3:
        raise ValueError("REF2VA supports at most 3 video clips")
    if len(audios) > 3:
        raise ValueError("REF2VA supports at most 3 audio clips")
    if len(images) + len(videos) + len(audios) > 12:
        raise ValueError("REF2VA supports at most 12 reference files")
    if audios and not audio_has_visual:
        raise ValueError("REF2VA audio must be accompanied by an image or video")

    video_total = sum(_duration(item.get("duration"), f"video {i + 1}") for i, item in enumerate(videos))
    audio_total = sum(_duration(item.get("duration"), f"audio {i + 1}") for i, item in enumerate(audios))
    if video_total > MAX_REF_TOTAL_SECONDS:
        raise ValueError("REF2VA video duration total must not exceed 15 seconds")
    if audio_total > MAX_REF_TOTAL_SECONDS:
        raise ValueError("REF2VA audio duration total must not exceed 15 seconds")
    return video_total, audio_total


def normalize_guide(data: dict) -> NormalizedGuide:
    if not isinstance(data, dict):
        raise ValueError("MiniMax Director guide must be a dictionary")
    mode = data.get("mode", "FL2VA")
    if mode not in {"T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA", "Image Inpaint"}:
        raise ValueError(f"unsupported MiniMax Director mode: {mode}")
    prompt = str(data.get("prompt", ""))
    resolved = str(data.get("resolved_prompt") or assemble_prompt(prompt, data.get("prompt_blocks")))
    common = dict(
        mode=mode, prompt=prompt, resolved_prompt=resolved,
        width=int(data.get("width", 1344)), height=int(data.get("height", 768)),
        length=int(data.get("length", 124)), ref_image_size=data.get("ref_image_size", "match"),
        minimax_ref_items=data.get("minimax_ref_items"), selection_stamp=float(data.get("selection_stamp", 0) or 0),
    )
    if mode in {"T2VA", "I2VA", "FL2VA", "L2VA"}:
        if data.get("ref_images") or data.get("ref_videos") or data.get("ref_audios") or data.get("ref_video_audios"):
            raise ValueError(f"{mode} accepts only its optional image keyframes")
        return NormalizedGuide(**common, first_frame=data.get("first_frame"), last_frame=data.get("last_frame"))
    if mode == "Image Inpaint":
        if data.get("ref_images") or data.get("ref_videos") or data.get("ref_audios") or data.get("ref_video_audios"):
            raise ValueError("Image Inpaint accepts one image keyframe and no video/audio references")
        if data.get("first_frame") is None:
            raise ValueError("Image Inpaint requires one image keyframe")
        return NormalizedGuide(**common, first_frame=data.get("first_frame"), last_frame=None)
    return NormalizedGuide(
        **common,
        ref_images=data.get("ref_images") or {}, ref_videos=data.get("ref_videos") or {},
        ref_video_audios=data.get("ref_video_audios") or {}, ref_audios=data.get("ref_audios") or {},
    )


def build_reference_plan(ref_images: dict, ref_videos: dict, ref_video_audios: dict, ref_audios: dict):
    items, blocks = [], []
    for _, image in (ref_images or {}).items():
        if image is not None:
            items.append({"type": "image", "data": image})
    for name, video in (ref_videos or {}).items():
        if video is None:
            continue
        suffix = name.rsplit("_", 1)[-1]
        audio = (ref_video_audios or {}).get(f"ref_video_audio_{suffix}")
        if audio is not None:
            items.append({"type": "audio", "data": audio, "paired_video": name})
        items.append({"type": "video", "data": video, "paired_audio": audio is not None})
    for _, audio in (ref_audios or {}).items():
        if audio is not None:
            items.append({"type": "audio", "data": audio})
    return items, blocks


def resolve_input_path(relative_path: str, input_directory: str) -> str:
    """Resolve a workflow media path without allowing input-directory escape."""
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("media path must be a non-empty string")
    root = os.path.realpath(input_directory)
    candidate = os.path.realpath(os.path.join(root, relative_path))
    if os.path.commonpath((root, candidate)) != root:
        raise ValueError("media path escapes the ComfyUI input directory")
    if not os.path.isfile(candidate):
        raise ValueError(f"media file does not exist: {relative_path}")
    return candidate


def load_image(path: str, input_directory: str):
    with Image.open(resolve_input_path(path, input_directory)) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def decode_audio_frame(frame) -> np.ndarray:
    """Return one decoded audio frame as float samples shaped (channels, samples).

    PyAV yields (channels, samples) for planar formats (fltp — MP3, AAC) but a
    single interleaved row for packed ones (s16 — PCM WAV). Left interleaved, a
    stereo clip measures twice its real length and every crop offset lands on
    the wrong sample.

    Integer formats are scaled by their maximum magnitude, not their maximum
    value, so a full-scale -32768 lands on exactly -1.0 instead of overshooting
    the unit range.
    """
    samples = frame.to_ndarray()
    if samples.ndim == 1:
        samples = samples[None, :]
    channels = samples.shape[-1] // frame.samples if frame.samples else 1
    if channels > 1 and samples.shape[0] == 1:
        samples = np.ascontiguousarray(samples.reshape(-1, channels).T)
    if samples.dtype.kind == "u":
        midpoint = 2.0 ** (8 * samples.dtype.itemsize - 1)
        samples = (samples.astype(np.float32) - midpoint) / midpoint
    elif samples.dtype.kind != "f":
        samples = samples.astype(np.float32) / -float(np.iinfo(samples.dtype).min)
    return samples


def load_audio(path: str, input_directory: str, *, trim_start: float = 0.0,
               trim_end: float | None = None):
    full_path = resolve_input_path(path, input_directory)
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("MiniMax H3 audio references require the PyAV package") from exc
    if trim_start < 0 or (trim_end is not None and trim_end <= trim_start):
        raise ValueError("audio trim range is invalid")
    container = av.open(full_path)
    try:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("audio file contains no audio stream")
        sample_rate = int(stream.rate or 48_000)
        chunks = []
        for frame in container.decode(stream):
            timestamp = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0
            frame_end = timestamp + float(frame.samples) / sample_rate
            if frame_end <= trim_start or (trim_end is not None and timestamp >= trim_end):
                continue
            samples = decode_audio_frame(frame)
            start = max(0, int(round((trim_start - timestamp) * sample_rate)))
            end = samples.shape[-1] if trim_end is None else min(samples.shape[-1], int(round((trim_end - timestamp) * sample_rate)))
            if end > start:
                chunks.append(torch.from_numpy(samples[:, start:end]).float())
        if not chunks:
            raise ValueError("audio trim range produced no samples")
        return {"waveform": torch.cat(chunks, dim=-1).unsqueeze(0), "sample_rate": sample_rate}
    finally:
        container.close()


def audio_duration(audio: dict) -> float:
    waveform = audio["waveform"]
    return float(waveform.shape[-1]) / float(audio["sample_rate"])


def load_embedded_video_audio(path: str, input_directory: str, *, trim_start: float = 0.0,
                              trim_end: float | None = None):
    """Decode the embedded audio stream of a video into a cropped ComfyUI audio dict."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("MiniMax H3 embedded video audio requires the PyAV package") from exc
    if trim_start < 0 or (trim_end is not None and trim_end <= trim_start):
        raise ValueError("embedded video audio trim range is invalid")
    container = av.open(resolve_input_path(path, input_directory))
    try:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("video file contains no audio stream")
        sample_rate = int(stream.rate or 48_000)
        chunks = []
        for frame in container.decode(stream):
            timestamp = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0
            frame_end = timestamp + float(frame.samples) / sample_rate
            if frame_end <= trim_start or (trim_end is not None and timestamp >= trim_end):
                continue
            array = decode_audio_frame(frame)
            start = max(0, int(round((trim_start - timestamp) * sample_rate)))
            end = array.shape[-1] if trim_end is None else min(array.shape[-1], int(round((trim_end - timestamp) * sample_rate)))
            if end > start:
                chunks.append(torch.from_numpy(array[:, start:end]).float())
        if not chunks:
            raise ValueError("embedded video audio trim range produced no samples")
        # decode_audio_frame() already returns unit-range floats, so the former
        # int16 rescale here would only ever fire on correctly scaled audio.
        return {"waveform": torch.cat(chunks, dim=-1).unsqueeze(0), "sample_rate": sample_rate}
    finally:
        container.close()


def load_video(path: str, input_directory: str, *, trim_start: float = 0.0,
               trim_end: float | None = None, target_fps: float = FPS):
    """Decode a trimmed video to ComfyUI IMAGE frames at a fixed FPS.

    Output frames are chosen by presentation time, not by source index: for
    every 1/target_fps-second tick inside [trim_start, trim_end) the source
    frame with the nearest PTS is emitted (VHS-style force_rate). A 60 fps
    clip therefore fills a 24 fps batch across the whole trim window instead
    of taking the first few seconds of source frames and stretching them.
    Sources slower than the target rate repeat frames (nearest neighbor).

    PyAV is intentionally imported lazily because image-only workflows must not
    require a video dependency at node import time.
    """
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("MiniMax H3 video references require the PyAV package") from exc
    if trim_start < 0 or (trim_end is not None and trim_end <= trim_start):
        raise ValueError("video trim range is invalid")
    container = av.open(resolve_input_path(path, input_directory))
    try:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise ValueError("video file contains no video stream")
        duration = (float(stream.duration * stream.time_base)
                    if stream.duration is not None and stream.time_base is not None else None)
        if duration is None and container.duration is not None:
            duration = float(container.duration) / float(av.time_base)
        if duration is None:
            raise ValueError("video duration could not be determined")
        end = duration if trim_end is None else min(float(trim_end), duration)
        if end <= trim_start:
            raise ValueError("video trim range is empty")
        timestamps = np.arange(float(trim_start), end, 1.0 / target_fps)
        timestamps = timestamps[timestamps < end]
        if timestamps.size == 0:
            raise ValueError("video trim range produced no frames")
        frames = []
        source_times = []
        for frame in container.decode(stream):
            timestamp = float(frame.pts * frame.time_base) if frame.pts is not None else None
            if timestamp is None or timestamp < trim_start or timestamp >= end:
                continue
            frames.append(torch.from_numpy(frame.to_rgb().to_ndarray()).float() / 255.0)
            source_times.append(timestamp)
        if not frames:
            raise ValueError("video trim range produced no frames")
        source = torch.stack(frames)
        # Sample by presentation time, not by source index: for each target
        # tick pick the source frame with the nearest PTS (VHS-style
        # force_rate). The former (timestamps * target_fps) index collapsed
        # to 0, 1, 2, ..., so a 60 fps source played ~2.5x slow and its tail
        # was never used. Nearest-match also repeats frames when the source
        # runs slower than the target rate.
        ticks = torch.as_tensor(timestamps, dtype=torch.float64)
        times = torch.as_tensor(source_times, dtype=torch.float64)
        pos = torch.searchsorted(times, ticks)
        right = pos.clamp(max=len(times) - 1)
        left = (pos - 1).clamp(min=0)
        nearest = torch.where((ticks - times[left]).abs() <= (times[right] - ticks).abs(), left, right)
        return source[nearest]
    finally:
        container.close()
