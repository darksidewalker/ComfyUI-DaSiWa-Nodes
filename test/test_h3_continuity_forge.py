"""Continuity drafting reuses Forge's selected model and unload policy."""
from pathlib import Path
import importlib
import sys

import pytest

if "dasiwa_nodes_test" in sys.modules:
    h3_forge = importlib.import_module("dasiwa_nodes_test.h3_forge")
else:
    from nodes import h3_forge


class StubBackend:
    kind = "ollama"
    base = "http://127.0.0.1:11434"

    def __init__(self, *, vision: bool | None = True, response="A rider keeps moving through the forest."):
        self.vision = vision
        self.response = response
        self.calls = []
        self.unloads = []

    def models(self):
        return [{"id": "ollama:test-model", "label": "test-model"}]

    def can_see(self, name):
        return self.vision

    def chat(self, name, system, user, images, sampling, num_ctx, timeout, cancel=None):
        self.calls.append((name, system, user, images))
        return self.response, {}

    def unload(self, name):
        self.unloads.append(name)
        return True


def test_continuity_draft_uses_selected_forge_model_and_unloads(monkeypatch, tmp_path):
    backend = StubBackend(vision=False)
    monkeypatch.setattr(h3_forge, "backends", lambda settings: {"ollama": backend})
    result = h3_forge.generate_continuity_draft(
        {"prompt": "The rider enters the forest.", "thumbnails": []},
        "She slows near a tower.", tmp_path, "ollama:test-model", {},
    )
    assert result["prompt"] == "A rider keeps moving through the forest."
    assert result["vision"] is False
    assert backend.unloads == ["test-model"]
    assert "She slows near a tower." in backend.calls[0][2]
    assert backend.calls[0][3] == []


def test_continuity_draft_rejects_unlisted_model(monkeypatch, tmp_path):
    backend = StubBackend()
    monkeypatch.setattr(h3_forge, "backends", lambda settings: {"ollama": backend})
    with pytest.raises(h3_forge.ForgeError, match="Pick an available"):
        h3_forge.generate_continuity_draft({"prompt": "Prior", "thumbnails": []}, "Next", tmp_path, "ollama:unlisted", {})
    assert backend.calls == []


def test_continuity_draft_uses_only_four_safe_ordered_frames(monkeypatch, tmp_path):
    backend = StubBackend()
    monkeypatch.setattr(h3_forge, "backends", lambda settings: {"ollama": backend})
    monkeypatch.setattr(h3_forge, "_image_b64", lambda path: Path(path).name)
    names = [f"frame-{i}.jpg" for i in range(6)]
    for name in names:
        (tmp_path / name).write_bytes(b"jpeg")
    result = h3_forge.generate_continuity_draft(
        {"prompt": "Earlier scene", "thumbnails": names}, "Move on", tmp_path, "ollama:test-model", {},
    )
    assert result["vision"] is True
    assert backend.calls[0][3] == names[-4:]
    assert backend.unloads == ["test-model"]


def test_continuity_draft_rejects_escaping_preview(monkeypatch, tmp_path):
    backend = StubBackend()
    monkeypatch.setattr(h3_forge, "backends", lambda settings: {"ollama": backend})
    with pytest.raises(h3_forge.ForgeError, match="preview"):
        h3_forge.generate_continuity_draft(
            {"prompt": "Earlier", "thumbnails": ["../secret.jpg"]}, "Next", tmp_path,
            "ollama:test-model", {},
        )
    assert backend.calls == []


def test_continuity_draft_unloads_after_backend_failure(monkeypatch, tmp_path):
    backend = StubBackend(vision=False)
    def fail_chat(*args, **kwargs):
        raise RuntimeError("server disconnected")
    backend.chat = fail_chat
    monkeypatch.setattr(h3_forge, "backends", lambda settings: {"ollama": backend})
    with pytest.raises(RuntimeError, match="server disconnected"):
        h3_forge.generate_continuity_draft(
            {"prompt": "Earlier", "thumbnails": []}, "Next", tmp_path,
            "ollama:test-model", {},
        )
    assert backend.unloads == ["test-model"]
    assert not any(name == "test-model" for _backend, name in h3_forge._FORGE_LOADED)


def test_continuity_openai_retries_text_only_when_vision_is_unsupported(monkeypatch, tmp_path):
    from urllib.error import HTTPError
    from email.message import Message

    backend = StubBackend(vision=None)
    backend.kind = "openai"
    backend.base = "http://127.0.0.1:8080"
    backend.models = lambda: [{"id": "openai:test-model", "label": "test-model"}]
    def chat(name, system, user, images, sampling, num_ctx, timeout, cancel=None):
        backend.calls.append((name, system, user, images))
        if images:
            raise HTTPError("http://127.0.0.1:8080/v1/chat/completions", 400, "no vision", Message(), None)
        return "The rider continues forward.", {}
    backend.chat = chat
    monkeypatch.setattr(h3_forge, "backends", lambda settings: {"openai": backend})
    monkeypatch.setattr(h3_forge, "_image_b64", lambda path: "jpeg-data")
    (tmp_path / "last.jpg").write_bytes(b"jpeg")
    result = h3_forge.generate_continuity_draft(
        {"prompt": "Earlier", "thumbnails": ["last.jpg"]}, "Next", tmp_path,
        "openai:test-model", {},
    )
    assert result["vision"] is False
    assert [bool(call[3]) for call in backend.calls] == [True, False]
    assert backend.unloads == ["test-model"]
