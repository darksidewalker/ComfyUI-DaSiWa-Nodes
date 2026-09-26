"""Pure policy, native continuation adapter, and immutable checkpoint storage."""
from __future__ import annotations

import copy
import json
import math
import os
import re
import threading
import time
import uuid
from pathlib import Path

import torch
from comfy.nested_tensor import NestedTensor
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .vendor.continuation_nodes import (
    MiniMaxH3GuidedContinuationWindow,
    MiniMaxH3LatentTailGuide,
    MiniMaxH3AppendContinuation,
    validate_h3_av_latent,
    _require_native_arbitrary_guides,
)

VERSION = "1.2.5"
DEFAULT_PROMPT = (
    "Continue the same uninterrupted shot naturally. Preserve the subjects' identity, "
    "clothing, positions, lighting and environment. Maintain the established motion "
    "direction, camera trajectory and ambient sound. Do not restart the action, "
    "repeat completed dialogue, introduce a cut, fade, title, freeze or loop."
)
_ID = re.compile(r"^[a-zA-Z0-9_-]{1,80}$")
_LOCK = threading.RLock()


def safe_id(value):
    value = str(value)
    if not _ID.fullmatch(value):
        raise ValueError("Invalid continuity session/clip ID (letters, numbers, _ and - only).")
    return value


def continuation_timing(duration_seconds, overlap_frames=22, source_frames=None):
    """Duration means newly visible time. Snap to the nearest 17-frame step.

    Python and the frontend use floor(x + .5), including exact half steps.
    Context can shrink to fit a 15-second request or a short source, but new
    visible frames are never silently truncated to make room for context.
    """
    seconds = float(duration_seconds)
    if not math.isfinite(seconds) or not 0 < seconds <= 15:
        raise ValueError("Continuity Duration must be greater than 0 and at most 15 seconds.")
    extension = max(17, math.floor(seconds * 24 / 17 + 0.5) * 17)
    available = min(int(overlap_frames), 362 - extension)
    if source_frames is not None:
        available = min(available, int(source_frames))
    overlap = next((n for n in (73, 56, 39, 22, 5) if n <= available), None)
    if overlap is None:
        raise ValueError("The source must contain at least 5 H3 frames.")
    return {"duration_seconds": seconds, "overlap_frames": overlap,
            "extension_frames": extension, "added_seconds": extension / 24,
            "window_frames": overlap + extension}


def parse_settings(raw, duration_seconds=None):
    value = json.loads(raw) if isinstance(raw, str) else copy.deepcopy(raw)
    if not isinstance(value, dict):
        raise ValueError("Continuity settings must be a JSON object.")
    version = 3 if int(value.get("version", 2)) >= 3 else 2
    kind = value.get("source_kind", "checkpoint")
    selected = value.get("source_video_id") if kind == "video" else value.get("source_id")
    operation = ("continue" if selected else "new") if version == 3 else value.get("operation", "new")
    if operation not in {"new", "continue"}:
        raise ValueError("Continuity operation must be new or continue.")
    # The Director's opt-in state starts with an empty session. Ordinary New
    # takes must keep working before the user ever touches Continuity.
    session = value.get("session") or "mythicalchemy"
    session = safe_id(session)
    if session == "_imports":
        raise ValueError("_imports is reserved for uploaded source videos.")
    overlap = int(value.get("overlap_frames", 22))
    if overlap not in (5, 22, 39, 56, 73):
        raise ValueError("Use 5, 22, 39, 56 or 73 context frames.")
    if version == 3 and operation == "continue":
        requested = duration_seconds if duration_seconds is not None else value.get("duration_seconds", 5)
        value.update(continuation_timing(requested, overlap))
        overlap = value["overlap_frames"]
    extension = 119 if version == 3 and operation == "new" else int(value.get("extension_frames", 119))
    if extension < 17 or extension % 17 or overlap + extension > 362:
        raise ValueError("New frames must be a multiple of 17; context + new frames must be <= 362.")
    source = value.get("source_id", "")
    source_kind = value.get("source_kind", "checkpoint")
    if source_kind not in {"checkpoint", "video"}:
        raise ValueError("Select a checkpoint or uploaded video as the continuity source.")
    video_id = value.get("source_video_id", "")
    if operation == "continue":
        selected = video_id if source_kind == "video" else source
        if not selected:
            raise ValueError("Select a completed checkpoint or choose a start video before continuing.")
        safe_id(selected)
    prompt = str(value.get("continuation_prompt", "" if version == 3 else DEFAULT_PROMPT)).strip()
    idea = str(value.get("idea", "")).strip()
    if len(prompt) > 50000 or len(idea) > 12000:
        raise ValueError("Continuation text is too long.")
    if operation == "continue" and not prompt and version < 3:
        raise ValueError("Enter a continuation prompt or use Prefill.")
    capture = value.get("capture", False)
    if not isinstance(capture, bool):
        raise ValueError("Continuity capture must be true or false.")
    use_references = value.get("use_references", False)
    if not isinstance(use_references, bool):
        raise ValueError("Use Director references must be true or false.")
    return {**value, "version": version, "operation": operation, "capture": capture, "session": session,
            "source_kind": source_kind, "source_video_id": video_id, "use_references": use_references,
            "source_id": source, "overlap_frames": overlap, "extension_frames": extension,
            "continuation_prompt": prompt, "idea": idea}


def compose_prompt(settings):
    """Describe the sampled window, including its invisible leading context."""
    head = settings["overlap_frames"] / 24
    visible = settings["extension_frames"] / 24
    text = settings["continuation_prompt"]
    if settings.get("version", 2) >= 3:
        text = DEFAULT_PROMPT + ("\nNext action: " + text if text else "")
    idea = settings.get("idea", "").strip()
    if idea:
        text += "\nNext action: " + idea
    return (
        f"This is a continuation window. Its opening {head:.3f} seconds are hidden "
        "overlapping context from the end of the preceding shot. Match that context's "
        "motion and sound before advancing. "
        f"The following {visible:.3f} seconds are the new visible continuation.\n\n{text}"
    )


def mode_family(mode):
    return "ref2va" if mode == "REF2VA" else "fl2va"


def prepare_continuation(previous, metadata, guide, settings):
    _require_native_arbitrary_guides()
    video, _, _ = validate_h3_av_latent(previous, name="previous clip")
    if metadata.get("fps", 24) != 24:
        raise ValueError("Source clip must use native 24 fps.")
    if guide.get("mode") == "Image Inpaint":
        raise ValueError("Continuity requires a video mode, not Image Inpaint.")
    if mode_family(guide["mode"]) != metadata["mode_family"]:
        raise ValueError("Source and Director use different model families. Restore the source video mode.")
    width, height = video.shape[4] * 16, video.shape[3] * 16
    if (guide["width"], guide["height"]) != (width, height):
        raise ValueError(f"Source canvas is {width} x {height}. Restore that Director canvas before continuing.")
    window = MiniMaxH3GuidedContinuationWindow.execute(
        previous, settings["overlap_frames"], settings["extension_frames"])
    target, length, video_tokens, audio_tokens, _, _ = window.result
    updated = dict(guide)
    resolved = compose_prompt(settings)
    updated.update(length=length, prompt=resolved, resolved_prompt=resolved,
                   prompt_blocks=[], first_frame=None, last_frame=None)
    # The Director Guide normalizer gives resolved_prompt priority. Reference media
    # and RefMods are retained in REF2VA; only competing endpoint guides are removed.
    return updated, target, {"overlap_video_tokens": video_tokens,
                            "overlap_audio_tokens": audio_tokens}


def add_tail(positive, previous, target, layout):
    return MiniMaxH3LatentTailGuide.execute(
        positive, previous, target, layout["overlap_video_tokens"],
        layout["overlap_audio_tokens"])[0]


def append_tail(previous, sampled, layout):
    # ComfyUI samplers may promote the working latent dtype; canonicalize both
    # streams to the immutable source dtype/device before strict native validation.
    pv, pa, _ = validate_h3_av_latent(previous, name="source")
    sv, sa = sampled["samples"].tensors
    sampled = {"samples": NestedTensor((sv.to(pv), sa.to(pa)))}
    return MiniMaxH3AppendContinuation.execute(
        previous, sampled, layout["overlap_video_tokens"],
        layout["overlap_audio_tokens"], 0, 0)[0]


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class ClipStore:
    """IDs never denote 'latest' during execution: every queue pins a parent."""
    def __init__(self, root=None):
        if root is None:
            import folder_paths
            root = Path(folder_paths.get_output_directory()) / "df_h3_continuity"
        self.root = Path(root).resolve()

    def session_dir(self, session):
        path = self.root / safe_id(session)
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Session path escapes the continuity store.")
        return path

    def clip_dir(self, session, clip):
        path = self.session_dir(session) / safe_id(clip)
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Clip path escapes the continuity store.")
        return path

    def member(self, session, clip, name):
        directory = self.clip_dir(session, clip).resolve()
        path = (directory / name).resolve()
        if path.parent != directory:
            raise ValueError("Checkpoint member escapes its directory.")
        return path

    def metadata(self, session, clip, ready=True):
        data = json.loads(self.member(session, clip, "clip.json").read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("session") != session or data.get("clip_id") != clip:
            raise ValueError("Checkpoint identity does not match its directory.")
        if data.get("status") not in {"staged", "ready"}:
            raise ValueError("Invalid checkpoint status.")
        if ready and data["status"] != "ready":
            raise ValueError("The source clip has not finished exporting successfully.")
        frames = data.get("frames")
        if type(frames) is not int or frames < 5 or (frames - 5) % 17:
            raise ValueError("Invalid checkpoint frame count.")
        if any(type(data.get(k)) is not int or data[k] <= 0 for k in ("width", "height", "created_ns")):
            raise ValueError("Invalid checkpoint dimensions or timestamp.")
        if data["status"] == "ready" and (type(data.get("completed_ns")) is not int or data["completed_ns"] <= 0):
            raise ValueError("Invalid checkpoint completion timestamp.")
        if data.get("fps") != 24 or not isinstance(data.get("seconds"), (float, int)) or not math.isfinite(data["seconds"]) or abs(data["seconds"] - frames / 24) > 1e-6:
            raise ValueError("Invalid checkpoint duration or frame rate.")
        if data.get("mode") not in {"T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"} or data.get("mode_family") != mode_family(data["mode"]):
            raise ValueError("Invalid checkpoint model family.")
        if not isinstance(data.get("provenance", {}), dict):
            raise ValueError("Invalid checkpoint provenance.")
        return data

    def inspect(self, session, clip, ready=True):
        """Read metadata and safetensors headers, never materialize AV tensors."""
        data = self.metadata(session, clip, ready)
        path = self.member(session, clip, "latent.safetensors")
        try:
            with safe_open(str(path), framework="pt", device="cpu") as file:
                if set(file.keys()) != {"video", "audio"}:
                    raise ValueError("Expected video and audio streams.")
                v, a = file.get_slice("video"), file.get_slice("audio")
                vs, aus = v.get_shape(), a.get_shape()
                if len(vs) != 5 or vs[:2] != [1, 24] or vs[2] < 2 or (vs[2] - 2) % 5:
                    raise ValueError("Invalid H3 video shape.")
                frames = (vs[2] - 2) // 5 * 17 + 5
                if aus != [1, 32, 2, round(frames / 24 * 40)]:
                    raise ValueError("Invalid H3 audio shape or phase.")
                if (frames, vs[4] * 16, vs[3] * 16) != (data["frames"], data["width"], data["height"]):
                    raise ValueError("Tensor shape differs from checkpoint metadata.")
                if v.get_dtype() not in {"F16", "BF16", "F32", "F64"} or a.get_dtype() not in {"F16", "BF16", "F32", "F64"}:
                    raise ValueError("Checkpoint streams must be floating point.")
        except Exception as exc:
            raise ValueError(f"Checkpoint {clip[:12]} is missing or invalid: {exc}") from exc
        return {**data, "latent_bytes": path.stat().st_size}

    def load(self, session, clip):
        metadata = self.inspect(session, clip)
        tensors = load_file(str(self.member(session, clip, "latent.safetensors")), device="cpu")
        latent = {"samples": NestedTensor((tensors["video"], tensors["audio"]))}
        _, _, frames = validate_h3_av_latent(latent, name="saved clip")
        if frames != metadata["frames"]:
            raise ValueError("Checkpoint frame count differs from its metadata.")
        return latent, metadata

    def stage(self, latent, context):
        video, audio, frames = validate_h3_av_latent(latent, name="completed clip")
        session, clip = context["session"], context["run_id"]
        directory = self.clip_dir(session, clip)
        directory.mkdir(parents=True, exist_ok=False)
        temp = directory / "latent.safetensors.tmp"
        try:
            save_file({"video": video.detach().cpu().contiguous(),
                       "audio": audio.detach().cpu().contiguous()}, str(temp))
            os.replace(temp, directory / "latent.safetensors")
            data = {"schema": 1, "package_version": VERSION, "clip_id": clip,
                    "session": session, "status": "staged", "created_ns": time.time_ns(),
                    "parent_id": context.get("source_id", ""), "frames": frames,
                    "fps": 24, "seconds": frames / 24,
                    "width": video.shape[4] * 16, "height": video.shape[3] * 16,
                    "mode": context["mode"], "mode_family": mode_family(context["mode"]),
                    "prompt": context["resolved_prompt"], "idea": context.get("idea", ""),
                    "overlap_frames": context.get("overlap_frames", 0),
                    "extension_frames": context.get("extension_frames", 0),
                    "provenance": context.get("provenance", {}), "thumbnails": []}
            atomic_json(directory / "clip.json", data)
        finally:
            temp.unlink(missing_ok=True)
        return {"session": session, "clip_id": clip}

    def publish(self, ticket, output_path, thumbnails=(), preview_warning=""):
        session, clip = ticket["session"], ticket["clip_id"]
        with _LOCK:
            data = self.inspect(session, clip, ready=False)
            data.update(status="ready", output_path=str(output_path),
                        completed_ns=time.time_ns(), thumbnails=list(thumbnails),
                        preview_warning=preview_warning)
            atomic_json(self.clip_dir(session, clip) / "clip.json", data)
            atomic_json(self.session_dir(session) / "latest.json", {"clip_id": clip})
        return data

    def list_clips(self, session, selected_id=None):
        directory = self.session_dir(session)
        clips, unavailable, staged = [], 0, 0
        for path in directory.glob("*/clip.json"):
            try:
                data = self.metadata(session, path.parent.name, ready=False)
                if data["status"] == "staged":
                    staged += 1
                    continue
                data = self.inspect(session, path.parent.name)
                data.pop("output_path", None)
                clips.append(data)
            except (ValueError, OSError, KeyError, TypeError):
                unavailable += 1
        clips.sort(key=lambda x: (x["completed_ns"], x["clip_id"]), reverse=True)
        # An imported source is reusable, but is not a generated output.
        outputs = [c for c in clips if c.get("provenance", {}).get("kind") != "imported_video"]
        visible = clips[:200]
        if selected_id and not any(c["clip_id"] == selected_id for c in visible):
            visible += [c for c in clips[200:] if c["clip_id"] == selected_id]
        return {"clips": visible, "latest_id": outputs[0]["clip_id"] if outputs else "",
                "ready_count": len(clips), "output_count": len(outputs), "import_count": len(clips) - len(outputs),
                "latent_bytes": sum(c["latent_bytes"] for c in clips),
                "unavailable_count": unavailable, "staged_count": staged,
                "latest_completed_ns": clips[0]["completed_ns"] if clips else 0}

    def list_sessions(self):
        summaries = []
        if not self.root.exists():
            return {"sessions": [], "session_count": 0}
        for directory in self.root.iterdir():
            if not directory.is_dir() or directory.name == "_imports":
                continue
            try:
                result = self.list_clips(safe_id(directory.name))
            except (ValueError, OSError):
                continue
            if result["ready_count"] or result["staged_count"] or result["unavailable_count"]:
                summaries.append({"session": directory.name, **{k: v for k, v in result.items() if k != "clips"}})
        summaries.sort(key=lambda x: (x["latest_completed_ns"], x["session"]), reverse=True)
        return {"sessions": summaries[:200], "session_count": len(summaries)}
