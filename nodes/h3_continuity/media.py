"""Small, chronological tail thumbnails for visual prompt assistance."""
from pathlib import Path
import shutil
import subprocess


def make_tail_thumbnails(video, directory):
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("ffmpeg is not on PATH; video continuity still works, but visual analysis is unavailable.")
    directory = Path(directory)
    pattern = directory / "tail-%02d.jpg"
    process = subprocess.run([
        executable, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-sseof", "-2", "-i", str(video), "-an", "-vf",
        "fps=2,scale=640:640:force_original_aspect_ratio=decrease", "-frames:v", "4",
        "-q:v", "3", str(pattern)], capture_output=True, timeout=45, check=False)
    if process.returncode:
        raise RuntimeError("Could not extract tail previews: " + process.stderr.decode(errors="replace")[:400])
    names = [p.name for p in sorted(directory.glob("tail-*.jpg"))]
    if not names:
        raise RuntimeError("The exported file provided no decodable tail frames.")
    return names


def ensure_tail_thumbnails(metadata, directory):
    """Lazy visual evidence; no extraction during upload, capture or text-only Forge.

    Keep this cache separate from the immutable checkpoint/manifest metadata.
    Source identity is rechecked even when JPEGs already exist.
    """
    from .core import atomic_json
    from .video_source import input_video, file_digest
    import folder_paths
    import json
    directory = Path(directory).resolve()
    if metadata.get("kind") == "video":
        path = input_video(metadata["filename"])
        if file_digest(path) != metadata["sha256"]:
            raise ValueError("Source video changed; select it again before analysis.")
    else:
        path = Path(metadata.get("output_path") or "").resolve()
        roots = [Path(folder_paths.get_output_directory()).resolve(), Path(folder_paths.get_temp_directory()).resolve(), Path(folder_paths.get_input_directory()).resolve()]
        if not any(path.is_relative_to(root) for root in roots) or not path.is_file():
            raise ValueError("Source video is unavailable for visual analysis.")
        provenance = metadata.get("provenance") or {}
        if provenance.get("kind") == "imported_video" and file_digest(path) != provenance.get("sha256"):
            raise ValueError("Source video changed; select it again before analysis.")
    signature = [str(path), path.stat().st_size, path.stat().st_mtime_ns]
    cache = directory / "tail-cache.json"
    if cache.is_file():
        try:
            data = json.loads(cache.read_text())
            names = data.get("thumbnails", []) if isinstance(data, dict) else []
            if isinstance(data, dict) and data.get("signature") == signature and isinstance(names, list) and names and all(isinstance(n, str) and (directory / n).resolve().parent == directory and n.endswith(".jpg") and (directory / n).is_file() for n in names):
                return names
        except (OSError, ValueError, TypeError):
            pass  # Optional preview cache: regenerate after a partial or invalid write.
    directory.mkdir(parents=True, exist_ok=True)
    for image in directory.glob("tail-*.jpg"):
        image.unlink()
    names = make_tail_thumbnails(path, directory)
    atomic_json(cache, {"signature": signature, "thumbnails": names})
    return names
