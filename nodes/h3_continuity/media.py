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
