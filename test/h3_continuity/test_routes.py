import asyncio
from types import SimpleNamespace
from aiohttp import web
from dasiwa_continuity_test import core


def handlers(monkeypatch, tmp_path):
    from dasiwa_nodes_test.h3_continuity import routes
    store = core.ClipStore(tmp_path)
    monkeypatch.setattr(routes, "ClipStore", lambda: store)
    class Registry:
        def __init__(self): self.handlers = {}
        def get(self, path):
            def decorator(fn): self.handlers[path] = fn; return fn
            return decorator
        post = get
    registry = Registry()
    routes.register_routes(SimpleNamespace(routes=registry, prompt_queue=SimpleNamespace(get_tasks_remaining=lambda: 0)))
    return registry.handlers, store


def test_empty_session_list_and_invalid_id(monkeypatch, tmp_path):
    routes, _ = handlers(monkeypatch, tmp_path)
    session = routes["/df_h3_continuity/session/{session}"]
    good = asyncio.run(session(SimpleNamespace(match_info={"session": "valid"})))
    assert good.status == 200 and good.text == '{"clips": [], "latest_id": ""}'
    bad = asyncio.run(session(SimpleNamespace(match_info={"session": "../escape"})))
    assert bad.status == 400


def test_tail_rejects_traversal_and_missing_previews(monkeypatch, tmp_path):
    routes, store = handlers(monkeypatch, tmp_path)
    tail = routes["/df_h3_continuity/tail/{session}/{clip}/{index}"]
    from test_core import ready
    ticket = ready(store)
    request = lambda index: SimpleNamespace(match_info={"session": "test", "clip": ticket["clip_id"], "index": index})
    assert asyncio.run(tail(request("../0"))).status == 404
    assert asyncio.run(tail(request("0"))).status == 404
    store.publish(ticket, "movie.mp4", ["../../secret.jpg"])
    assert asyncio.run(tail(request("0"))).status == 404


def test_analyze_uses_ready_source_and_forge_without_advancing(monkeypatch, tmp_path):
    from test_core import ready
    handlers_map, store = handlers(monkeypatch, tmp_path)
    ticket = ready(store)
    store.publish(ticket, "movie.mp4")
    calls = []
    from dasiwa_nodes_test import h3_forge
    def draft(meta, idea, directory, model, settings, release_memory):
        calls.append((meta["clip_id"], idea, model, str(directory)))
        return {"prompt": "Continue along the path.", "vision": False, "source_id": meta["clip_id"]}
    monkeypatch.setattr(h3_forge, "generate_continuity_draft", draft)
    async def body():
        return {"session": "test", "clip_id": ticket["clip_id"], "idea": "Walk on", "model": "ollama:test", "settings": {}}
    response = asyncio.run(handlers_map["/df_h3_continuity/analyze"](SimpleNamespace(json=body, content_length=256)))
    assert response.status == 200
    assert response.text.find("Continue along the path.") != -1
    assert calls[0][:3] == (ticket["clip_id"], "Walk on", "ollama:test")
    assert store.list_clips("test")["latest_id"] == ticket["clip_id"]


def test_analyze_rejects_missing_source_and_oversized_body(monkeypatch, tmp_path):
    handlers_map, _store = handlers(monkeypatch, tmp_path)
    analyze = handlers_map["/df_h3_continuity/analyze"]
    async def body():
        return {"session": "test", "clip_id": "missing", "idea": "Next", "model": "ollama:test", "settings": {}}
    assert asyncio.run(analyze(SimpleNamespace(json=body, content_length=256))).status == 400
    assert asyncio.run(analyze(SimpleNamespace(json=body, content_length=65537))).status == 413
