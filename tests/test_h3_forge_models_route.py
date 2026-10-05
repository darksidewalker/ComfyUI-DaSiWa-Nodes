"""Exercise Forge discovery over local HTTP; all model transports are mocked."""
import asyncio
import json
import sys
import threading
import types

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from nodes import h3_forge as forge


def test_models_route_refreshes_operator_cache_off_thread(monkeypatch):
    routes = web.RouteTableDef()
    server = types.SimpleNamespace(routes=routes)
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(
        PromptServer=types.SimpleNamespace(instance=server)))
    calls = []
    caller = threading.get_ident()
    dialog = {"openai_url": "http://dialog-fixture.invalid", "openai_api_key": "fixture-only"}

    def list_models(settings):
        calls.append(("dialog", threading.get_ident(), settings))
        return [{"id": "openai:fixture", "label": "Fixture"}], {}

    def refresh():
        # Deliberately accepts no Settings argument: operator config owns this.
        calls.append(("workflow", threading.get_ident(), None))
        return ["Server: operator-fixture"]

    monkeypatch.setattr(forge, "list_all", list_models)
    monkeypatch.setattr(forge, "refresh_server_model_choices", refresh)
    forge.register_routes()

    async def exercise():
        app = web.Application()
        app.add_routes(routes)
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/dasiwa/h3/forge/models", json={"settings": dialog})
            assert response.status == 200
            payload = json.loads(await response.text())
            assert payload["models"][0]["id"] == "openai:fixture"
            assert payload["errors"] == {}

    asyncio.run(exercise())
    assert [call[0] for call in calls] == ["dialog", "workflow"]
    assert all(call[1] != caller for call in calls)
    assert calls[0][2] == dialog
    assert calls[1][2] is None
