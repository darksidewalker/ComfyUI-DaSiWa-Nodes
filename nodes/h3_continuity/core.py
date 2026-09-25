"""Pure policy, native continuation adapter, and immutable checkpoint storage."""
from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

import torch
from comfy.nested_tensor import NestedTensor
from safetensors.torch import load_file, save_file

from .vendor.continuation_nodes import (
    MiniMaxH3GuidedContinuationWindow,
    MiniMaxH3LatentTailGuide,
    MiniMaxH3AppendContinuation,
    validate_h3_av_latent,
    _require_native_arbitrary_guides,
)

VERSION = "1.0.0"
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


def parse_settings(raw):
    value = json.loads(raw) if isinstance(raw, str) else copy.deepcopy(raw)
    if not isinstance(value, dict):
        raise ValueError("Continuity settings must be a JSON object.")
    operation = value.get("operation", "new")
    if operation not in {"new", "continue"}:
        raise ValueError("Continuity operation must be new or continue.")
    session = safe_id(value.get("session", "mythicalchemy"))
    overlap = int(value.get("overlap_frames", 22))
    extension = int(value.get("extension_frames", 119))
    if overlap not in (5, 22, 39, 56, 73):
        raise ValueError("Use 5, 22, 39, 56 or 73 context frames.")
    if extension < 17 or extension % 17 or overlap + extension > 362:
        raise ValueError("New frames must be a multiple of 17; context + new frames must be <= 362.")
    source = value.get("source_id", "")
    if operation == "continue":
        if not source:
            raise ValueError("Click Use last completed or select a source clip before continuing.")
        safe_id(source)
    prompt = str(value.get("continuation_prompt", DEFAULT_PROMPT)).strip()
    idea = str(value.get("idea", "")).strip()
    if len(prompt) > 50000 or len(idea) > 12000:
        raise ValueError("Continuation text is too long.")
    if operation == "continue" and not prompt:
        raise ValueError("Enter a continuation prompt or use Prefill.")
    capture = value.get("capture", False)
    if not isinstance(capture, bool):
        raise ValueError("Continuity capture must be true or false.")
    return {**value, "version": 1, "operation": operation, "capture": capture, "session": session,
            "source_id": source, "overlap_frames": overlap, "extension_frames": extension,
            "continuation_prompt": prompt, "idea": idea}


def compose_prompt(settings):
    """Describe the sampled window, including its invisible leading context."""
    head = settings["overlap_frames"] / 24
    visible = settings["extension_frames"] / 24
    text = settings["continuation_prompt"]
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

    def metadata(self, session, clip, ready=True):
        directory = self.clip_dir(session, clip)
        data = json.loads((directory / "clip.json").read_text(encoding="utf-8"))
        if data.get("session") != session or data.get("clip_id") != clip:
            raise ValueError("Checkpoint identity does not match its directory.")
        if ready and data.get("status") != "ready":
            raise ValueError("The source clip has not finished exporting successfully.")
        return data

    def load(self, session, clip):
        metadata = self.metadata(session, clip)
        tensors = load_file(str(self.clip_dir(session, clip) / "latent.safetensors"), device="cpu")
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
            data = self.metadata(session, clip, ready=False)
            data.update(status="ready", output_path=str(output_path),
                        completed_ns=time.time_ns(), thumbnails=list(thumbnails),
                        preview_warning=preview_warning)
            atomic_json(self.clip_dir(session, clip) / "clip.json", data)
            atomic_json(self.session_dir(session) / "latest.json", {"clip_id": clip})
        return data

    def list_clips(self, session):
        directory = self.session_dir(session)
        if not directory.exists():
            return {"clips": [], "latest_id": ""}
        clips = []
        for path in directory.glob("*/clip.json"):
            try:
                data = self.metadata(session, path.parent.name)
                data.pop("output_path", None)
                clips.append(data)
            except (ValueError, OSError):
                continue
        clips.sort(key=lambda x: x["completed_ns"], reverse=True)
        return {"clips": clips[:200], "latest_id": clips[0]["clip_id"] if clips else ""}
