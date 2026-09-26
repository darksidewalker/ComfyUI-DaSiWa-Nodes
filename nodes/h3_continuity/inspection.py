"""Read-only source preparation: metadata/header checks, timing and scratch budget."""
import math
import shutil
import tempfile

from .core import continuation_timing, mode_family, safe_id

VIDEO_MODES = {"T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"}


def normalized_frames(seconds):
    frames = max(5, math.floor(float(seconds) * 24 + 0.5))
    return frames + (5 - frames) % 17


def import_scratch_bytes(seconds, width, height):
    """Conservative raw RGB + stereo float PCM + 10%/64 MiB working reserve.

    It is a disk estimate, not a RAM/VRAM promise. VFR/media duration errors
    still require ffmpeg's ordinary error handling.
    """
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Invalid source duration.")
    width, height = int(width), int(height)
    if width < 32 or height < 32 or width % 32 or height % 32:
        raise ValueError("H3 source canvas must have both edges divisible by 32.")
    rgb = math.ceil(seconds * 24 + 1) * width * height * 3
    pcm = round(normalized_frames(seconds) / 24 * 40) * 800 * 2 * 4
    return math.ceil((rgb + pcm) * 1.1) + 64 * 1024**2


def check_import_space(seconds, width, height, directory=None):
    required = import_scratch_bytes(seconds, width, height)
    available = shutil.disk_usage(directory or tempfile.gettempdir()).free
    if required > available:
        raise ValueError(f"Insufficient temporary disk space for source import: approximately {required / 1024**3:.2f} GiB required, {available / 1024**3:.2f} GiB free. Shorten the source, reduce the canvas, or change the temporary directory.")
    return {"estimated_scratch_bytes": required, "scratch_free_bytes": available}


def inspect_source(settings, mode, width, height, store=None):
    from .video_source import read_manifest, input_video, import_cache_id
    from .core import ClipStore
    store = store or ClipStore()
    session = safe_id(settings["session"])
    if session == "_imports":
        raise ValueError("_imports is reserved for uploaded videos.")
    if settings.get("source_kind", "checkpoint") == "checkpoint":
        metadata = store.inspect(session, safe_id(settings["source_id"]))
        return metadata, True, False
    if settings.get("source_kind") != "video":
        raise ValueError("Select a checkpoint or video source.")
    metadata = read_manifest(store, safe_id(settings["source_video_id"]))
    input_video(metadata["filename"])  # Missing upload is detected before model work.
    clip_id = import_cache_id(metadata, {"mode": mode, "width": width, "height": height})
    if store.member(session, clip_id, "clip.json").is_file():
        try:
            return store.inspect(session, clip_id), True, False
        except (OSError, ValueError) as exc:
            raise ValueError("The cached import is unavailable. Start a new session to re-encode this video. " + str(exc)) from exc
    seconds = float(metadata["seconds"])
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Invalid uploaded video duration.")
    return {**metadata, "frames": normalized_frames(seconds), "seconds": normalized_frames(seconds) / 24,
            "original_seconds": seconds}, False, True


def checkpoint_issues(metadata, mode, width, height, external_canvas=False):
    issues = []
    if mode_family(mode) != metadata["mode_family"]:
        issues.append("Source uses " + metadata["mode_family"].upper() + "; match its model family before queuing.")
    if not external_canvas and (int(width), int(height)) != (metadata["width"], metadata["height"]):
        issues.append(f"Source canvas is {metadata['width']} × {metadata['height']}; match the canvas before queuing.")
    return issues


def preflight(body, store=None):
    if not isinstance(body, dict):
        raise ValueError("Source check must be a JSON object.")
    mode, width, height = body.get("mode"), int(body.get("width", 0)), int(body.get("height", 0))
    issues, notes = [], []
    if mode not in VIDEO_MODES:
        issues.append("Choose an H3 video model mode.")
    if float(body.get("frame_rate", 24)) != 24:
        issues.append("Continuity requires 24 fps.")
    external = bool(body.get("external_canvas"))
    if external:
        notes.append("Canvas comes from connected inputs; its size is checked when queued.")
    elif width < 32 or height < 32 or width % 32 or height % 32:
        issues.append("Set a canvas divisible by 32 on both axes.")
    settings = {"session": safe_id(body.get("session", "")), "source_kind": body.get("source_kind", "checkpoint"),
                "source_id": body.get("source_id", ""), "source_video_id": body.get("source_id", "")}
    metadata, exact, needs_import = inspect_source(settings, mode, width, height, store)
    preferred = int(body.get("overlap_frames", 22))
    if preferred not in (5, 22, 39, 56, 73):
        raise ValueError("Invalid preferred context.")
    timing = continuation_timing(body.get("duration"), preferred, metadata["frames"])
    if exact:
        issues.extend(checkpoint_issues(metadata, mode, width, height, external))
    if timing["overlap_frames"] < preferred:
        notes.append(f"Context reduced to {timing['overlap_frames']} frames to fit this source/window.")
    storage = {}
    if needs_import and not external and not issues:
        try:
            storage = check_import_space(metadata["original_seconds"], width, height)
        except ValueError as exc:
            issues.append(str(exc))
    public = {k: v for k, v in metadata.items() if k not in {"output_path", "prompt", "idea"}}
    return {"ok": not issues, "issues": issues, "notes": notes, "source": public,
            "source_frames_exact": exact, "needs_import": needs_import, "timing": timing,
            "total_seconds": (metadata["frames"] + timing["extension_frames"]) / 24, **storage}


def resolve_runtime_source(settings, mode, width, height, frame_rate=24, store=None):
    """Director-side checks; Guide repeats source loading and actual import alignment."""
    body = {"session": settings["session"], "source_kind": settings["source_kind"],
            "source_id": settings["source_video_id"] if settings["source_kind"] == "video" else settings["source_id"],
            "mode": mode, "width": width, "height": height, "frame_rate": frame_rate,
            "duration": settings["duration_seconds"], "overlap_frames": settings["overlap_frames"]}
    result = preflight(body, store)
    if result["issues"]:
        raise ValueError(" ".join(result["issues"]))
    # Do not cap the preference against a probe estimate; Guide knows decoded length.
    if result["source_frames_exact"]:
        return {**settings, **result["timing"]}
    return settings
