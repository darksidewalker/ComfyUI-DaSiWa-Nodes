"""Package metadata for the ComfyUI About badge.

Reads the version straight from pyproject.toml with a regex so the pack
loads on Python 3.10 (tomllib is 3.11+; Stability Matrix ships 3.10.11).
"""
import re
from pathlib import Path

from aiohttp import web
from server import PromptServer

_VERSION_RE = re.compile(r'^version\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)


def _read_version() -> str:
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    match = _VERSION_RE.search(text)
    return match.group(1) if match else "unknown"


@PromptServer.instance.routes.get("/dasiwa/version")
async def dasiwa_version(request):
    return web.json_response({"version": _read_version()}, headers={"Cache-Control": "no-store"})
