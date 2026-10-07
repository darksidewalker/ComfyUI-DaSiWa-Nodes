"""Package metadata for the ComfyUI About badge."""
from pathlib import Path
import tomllib

from aiohttp import web
from server import PromptServer


@PromptServer.instance.routes.get("/dasiwa/version")
async def dasiwa_version(request):
    with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as metadata:
        version = tomllib.load(metadata)["project"]["version"]
    return web.json_response({"version": version}, headers={"Cache-Control": "no-store"})
