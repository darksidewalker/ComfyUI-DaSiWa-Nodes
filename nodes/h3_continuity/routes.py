"""Read-only H3 continuity listings, bounded tail JPEGs and Forge drafts."""
import asyncio
from aiohttp import web
from .core import ClipStore, safe_id


def register_routes(server=None):
    if server is None:
        from server import PromptServer
        server = PromptServer.instance
    routes = server.routes

    @routes.get("/df_h3_continuity/session/{session}")
    async def session(request):
        try:
            result = await asyncio.to_thread(ClipStore().list_clips, safe_id(request.match_info["session"]))
            return web.json_response(result)
        except (ValueError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.get("/df_h3_continuity/tail/{session}/{clip}/{index}")
    async def tail(request):
        try:
            store = ClipStore()
            session_id = safe_id(request.match_info["session"])
            clip_id = safe_id(request.match_info["clip"])
            meta = await asyncio.to_thread(store.metadata, session_id, clip_id)
            index = int(request.match_info["index"])
            thumbnails = meta.get("thumbnails", [])
            if index < 0 or index >= len(thumbnails):
                raise ValueError("Preview index out of range.")
            directory = store.clip_dir(session_id, clip_id).resolve()
            path = (directory / thumbnails[index]).resolve()
            if path.parent != directory or path.suffix != ".jpg" or not path.is_file():
                raise ValueError("Invalid preview file.")
            return web.FileResponse(path)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=404)

    @routes.post("/df_h3_continuity/analyze")
    async def analyze(request):
        from .. import h3_forge
        def release_memory():
            from ..nodes_llm import _release_all_model_memory
            _release_all_model_memory()
        if request.content_length is not None and request.content_length > 65536:
            return web.json_response({"error": "Analyze request exceeds 64 KiB."}, status=413)
        if server.prompt_queue.get_tasks_remaining() > 0:
            return web.json_response({"error": "A workflow is running. Draft before queuing."}, status=409)
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("Analyze request must be an object.")
            session_id, clip_id = safe_id(body["session"]), safe_id(body["clip_id"])
            idea = body.get("idea", "")
            model = body.get("model", "")
            settings = body.get("settings", {})
            if not isinstance(idea, str) or len(idea) > 12000 or not isinstance(model, str) or not isinstance(settings, dict):
                raise ValueError("Invalid idea, model or Forge settings.")
            store = ClipStore()
            metadata = await asyncio.to_thread(store.metadata, session_id, clip_id)
            result = await asyncio.to_thread(
                h3_forge.generate_continuity_draft, metadata, idea,
                store.clip_dir(session_id, clip_id), model, settings, release_memory)
            return web.json_response(result)
        except (ValueError, KeyError, OSError, TypeError, h3_forge.ForgeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
