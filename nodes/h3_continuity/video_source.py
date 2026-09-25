"""Import ordinary videos through the connected H3 VAEs during queue execution.

HTTP preparation only probes media and extracts previews. No model is loaded there.
The source is content-addressed; replacing an uploaded file cannot change a queued
take. Native VAE calls preserve ComfyUI's model loading/offloading behaviour.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from comfy.nested_tensor import NestedTensor

from .core import ClipStore, atomic_json, mode_family, safe_id
from .media import make_tail_thumbnails
from .vendor.continuation_nodes import validate_h3_av_latent

IMPORT_VERSION = 1


def executable(name):
    value = shutil.which(name)
    if not value:
        raise ValueError(f"Install {name} on PATH to import an ordinary video.")
    return value


def run_media(args, timeout=1800, interrupt=False):
    # Disk spooling avoids accumulating conversion logs in RAM. Queue cancellation
    # terminates ffmpeg instead of leaving a potentially long conversion running.
    check = (lambda: None)
    if interrupt:
        from comfy.model_management import throw_exception_if_processing_interrupted
        check = throw_exception_if_processing_interrupted
    check()
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=output, stderr=errors)
        try:
            started = time.monotonic()
            while process.poll() is None:
                check()
                if time.monotonic() - started > timeout:
                    raise subprocess.TimeoutExpired(args, timeout)
                try:
                    process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    pass
            check()
        except BaseException:
            process.kill()
            process.wait()
            raise
        if process.returncode:
            errors.seek(max(0, errors.tell() - 2000))
            raise ValueError("Video conversion failed: " + errors.read().decode(errors="replace")[-1200:])
        output.seek(0)
        return output.read()


def input_video(filename, input_root=None):
    if input_root is None:
        import folder_paths
        input_root = folder_paths.get_input_directory()
    root = Path(input_root).resolve()
    if not isinstance(filename, str) or not filename or "\x00" in filename:
        raise ValueError("Select a video uploaded to ComfyUI input.")
    relative = Path(filename.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Video must be inside ComfyUI input.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Source video is missing. Select it again in the Director.")
    return path


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_video(path):
    data = json.loads(run_media([
        executable("ffprobe"), "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(path)], timeout=45))
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    if not video:
        raise ValueError("The selected file has no decodable video stream.")
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Could not determine the source video's duration.")
    numerator, _, denominator = str(video.get("avg_frame_rate", "0/1")).partition("/")
    fps = float(numerator) / (float(denominator or 1) or 1)
    return {"width": int(video["width"]), "height": int(video["height"]),
            "fps": fps, "seconds": duration, "video_stream": int(video["index"]),
            "has_audio": any(s.get("codec_type") == "audio" for s in streams)}


def manifest_dir(store, source_id):
    return store.clip_dir("_imports", safe_id(source_id))


def read_manifest(store, source_id):
    data = json.loads((manifest_dir(store, source_id) / "source.json").read_text(encoding="utf-8"))
    if data.get("clip_id") != source_id:
        raise ValueError("Imported video identity does not match its manifest.")
    return data


def prepare_video(filename, store=None, input_root=None):
    store = store or ClipStore()
    path = input_video(filename, input_root)
    digest = file_digest(path)
    # Include the upload name so two identical uploads never share a mutable path.
    source_id = hashlib.sha256((filename + "\0" + digest).encode()).hexdigest()
    directory = manifest_dir(store, source_id)
    if (directory / "source.json").exists():
        return read_manifest(store, source_id)
    info = probe_video(path)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        thumbnails = make_tail_thumbnails(path, directory)
        warning = ""
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        thumbnails, warning = [], str(exc)
    if file_digest(path) != digest:
        raise ValueError("The source changed during preparation. Select it again.")
    data = {**info, "clip_id": source_id, "filename": filename, "sha256": digest,
            "kind": "video", "prompt": "", "thumbnails": thumbnails, "preview_warning": warning}
    atomic_json(directory / "source.json", data)
    return data


def encode_video_chunks(frames, leading, total, vae):
    """124-frame public VAE calls, advancing by 119 frames (7 native chunks).

    Native H3 encodes independent 17-frame chunks, then drops three tokens only
    at the global end. Five look-ahead frames make each call valid (17k+5);
    remove their two tokens from every nonfinal call. This avoids loading the
    entire decoded movie as a floating point tensor.
    """
    pieces = []
    for start in range(0, total, 119):
        from comfy.model_management import throw_exception_if_processing_interrupted
        throw_exception_if_processing_interrupted()
        stop = min(start + 124, total)
        indices = np.clip(np.arange(start, stop) - leading, 0, len(frames) - 1)
        pixels = torch.from_numpy(np.array(frames[indices], copy=True)).float().div_(255)
        latent = vae.encode(pixels)
        final = stop == total
        expected = ((stop - start - 5) // 17) * 5 + 2
        if latent.ndim != 5 or latent.shape[:2] != (1, 24) or latent.shape[2] != expected:
            raise ValueError("Connect the native MiniMax H3 video VAE: unexpected encoded shape.")
        pieces.append((latent if final else latent[:, :, :-2]).detach().cpu().clone())
        if final:
            break
    return torch.cat(pieces, dim=2)


def encode_source(path, info, width, height, vae, audio_vae):
    if width < 32 or height < 32 or width % 32 or height % 32:
        raise ValueError("H3 source canvas must have both edges divisible by 32.")
    if audio_vae is None:
        raise ValueError("Connect the MiniMax H3 audio VAE to import video, including silent video.")
    if int(getattr(audio_vae, "audio_sample_rate", 32000)) != 32000:
        raise ValueError("Source import requires the MiniMax H3 audio VAE (32000 Hz).")
    ffmpeg = executable("ffmpeg")
    with tempfile.TemporaryDirectory(prefix="dasiwa-h3-import-") as work:
        work = Path(work)
        raw_video = work / "video.rgb"
        # ffmpeg applies rotation, timebase conversion and aspect-preserving fit.
        vf = (f"fps=24,scale={width}:{height}:force_original_aspect_ratio=decrease,"
              f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1")
        run_media([ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(path),
                   "-map", f"0:{info['video_stream']}", "-an", "-vf", vf,
                   "-pix_fmt", "rgb24", "-f", "rawvideo", str(raw_video)], interrupt=True)
        size = raw_video.stat().st_size
        frame_bytes = width * height * 3
        if not size or size % frame_bytes:
            raise ValueError("The source video did not decode to complete RGB frames.")
        actual = size // frame_bytes
        total = max(5, actual)
        total += (5 - total) % 17
        leading = total - actual
        # Preserve the real ending. Grid padding is at the beginning, never the tail.
        frames = np.memmap(raw_video, dtype=np.uint8, mode="r", shape=(actual, height, width, 3))
        try:
            video = encode_video_chunks(frames, leading, total, vae)
        finally:
            frames._mmap.close()
        audio_tokens = round(total / 24 * 40)
        audio_samples = audio_tokens * 800
        audio_path = work / "audio.f32"
        if info["has_audio"]:
            leading_samples = round(leading / 24 * 32000)
            af = (f"aresample=32000:async=1:first_pts=0,adelay={leading_samples}S:all=1,"
                  f"apad,atrim=end_sample={audio_samples}")
            run_media([ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(path),
                       "-map", "0:a:0", "-vn", "-af", af, "-ac", "2", "-ar", "32000",
                       "-f", "f32le", str(audio_path)], interrupt=True)
            waveform = np.fromfile(audio_path, dtype="<f4")
            if waveform.size != audio_samples * 2:
                raise ValueError("Could not align source audio with its video.")
            waveform = torch.from_numpy(waveform.reshape(1, audio_samples, 2))
        else:
            # A zero latent is not encoded silence. Use the actual audio encoder.
            waveform = torch.zeros((1, audio_samples, 2), dtype=torch.float32)
        from comfy.model_management import throw_exception_if_processing_interrupted
        throw_exception_if_processing_interrupted()
        audio = audio_vae.encode(waveform).detach().cpu()
        latent = {"samples": NestedTensor((video, audio))}
        _, _, verified_frames = validate_h3_av_latent(latent, name="imported video")
        if verified_frames != total:
            raise ValueError("The connected H3 VAEs returned an incompatible AV duration.")
        return latent, {"decoded_frames": actual, "leading_pad_frames": leading,
                        "fit": "contain", "audio": "encoded" if info["has_audio"] else "encoded silence"}


def import_checkpoint(settings, guide, vae, audio_vae, store=None, input_root=None):
    store = store or ClipStore()
    manifest = read_manifest(store, settings["source_video_id"])
    path = input_video(manifest["filename"], input_root)
    if file_digest(path) != manifest["sha256"]:
        raise ValueError("The selected source video changed. Select it again before continuing.")
    key = [IMPORT_VERSION, manifest["clip_id"], guide["width"], guide["height"], mode_family(guide["mode"])]
    clip_id = "import_" + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:48]
    directory = store.clip_dir(settings["session"], clip_id)
    if (directory / "clip.json").exists():
        # An immutable source checkpoint is reused on rerolls of the same upload/canvas.
        return (*store.load(settings["session"], clip_id), clip_id)
    latent, normalization = encode_source(path, manifest, guide["width"], guide["height"], vae, audio_vae)
    if file_digest(path) != manifest["sha256"]:
        raise ValueError("Source video changed during encoding. Select it again.")
    context = {"session": settings["session"], "run_id": clip_id, "mode": guide["mode"],
               "resolved_prompt": "", "provenance": {"kind": "imported_video",
               "source_video_id": manifest["clip_id"], "filename": manifest["filename"],
               "sha256": manifest["sha256"], "normalization": normalization}}
    # Stage under an execution-owned temporary ID, publish, then atomically expose
    # the deterministic directory. A failed encode never poisons the cache key.
    import uuid
    staging_id = "importing_" + uuid.uuid4().hex
    context["run_id"] = staging_id
    ticket = store.stage(latent, context)
    staged = store.clip_dir(settings["session"], staging_id)
    try:
        for name in manifest["thumbnails"]:
            shutil.copyfile(manifest_dir(store, manifest["clip_id"]) / name, staged / name)
        data = store.metadata(settings["session"], staging_id, ready=False)
        data.update(clip_id=clip_id, status="ready", output_path=str(path),
                    completed_ns=data["created_ns"], thumbnails=manifest["thumbnails"])
        atomic_json(staged / "clip.json", data)
        staged.rename(directory)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return latent, data, clip_id
