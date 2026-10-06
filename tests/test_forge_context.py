"""Pictures get a larger context where we set it, and a draft that cannot fit says what to do."""
import base64
import io
import sys
from pathlib import Path
from urllib import error as urlerror

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nodes import h3_forge as forge
from nodes import llm_backends as backends


def jpeg_b64(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (90, 120, 160)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


SQUARE = jpeg_b64(1024, 1024)


def test_picture_cost_follows_its_size():
    assert backends.image_tokens(SQUARE) == 32 * 32 + 4
    assert backends.image_tokens(jpeg_b64(1024, 576)) == 32 * 18 + 4
    assert backends.estimate_tokens("a" * 380, "", [SQUARE]) == 100 + 1028


def test_each_picture_adds_its_own_room():
    assert backends.context_for(16384, []) == 16384
    assert backends.context_for(16384, [SQUARE]) == 18432
    assert backends.context_for(16384, [SQUARE] * 3) == 22016
    assert backends.context_for(16384, [SQUARE] * 9) == 32768
    # Each picture's room is more than a square picture costs.
    assert backends.CONTEXT_PER_PICTURE > backends.image_tokens(SQUARE)


def test_only_contexts_we_set_are_checked(monkeypatch):
    huge = "x" * 400000
    for kind, name in (("openai", "m"), ("local", "Qwen3-VL-8B")):
        backends.check_fits(kind, name, huge, "", [], 16384)  # server's own limit / transformers: no estimate
    warnings = []
    monkeypatch.setattr(backends, "log_dasiwa", lambda tag, message: warnings.append(message))
    for kind, name in (("local", "q.gguf"), ("ollama", "qwen3.5:9b")):
        backends.check_fits(kind, name, huge, "", [SQUARE], 32768)
    assert len(warnings) == 2
    assert all("estimate" in warning.lower() for warning in warnings)


def pic(n, path):
    return {"kind": "image", "role": "subject", "easy_role": f"character-{n}", "path": path}


def _labelled(monkeypatch, tmp_path, count, kind="local", name="qwen.gguf"):
    refs = []
    for n in range(1, count + 1):
        (tmp_path / f"p{n}.png").write_bytes(b"x")
        refs.append(pic(n, f"p{n}.png"))
    monkeypatch.setattr(forge, "_image_b64", lambda path: SQUARE)
    sent = []

    class Backend:
        base = "http://127.0.0.1:11434"
        def can_see(self, model):
            return True
        def chat(self, model, system, user, images, sampling, num_ctx, *rest):
            sent.append({"images": len(images), "num_ctx": num_ctx})
            return "===SEGMENT: Subject definitions===\n<Subject 1>: waves\n===SEGMENT: Summary===\nS.\n" \
                   "===SEGMENT: Detailed description===\n[Shot 1] <Subject 1> waves.\n" \
                   "===SEGMENT: Soundscape===\nRoom tone.\n===SEGMENT: Music===\nN/A", {}
        def unload(self, model):
            return True

    monkeypatch.setattr(forge, "backends", lambda settings: {kind: Backend()})
    body = {"mode": "REF2VA", "model": f"{kind}:{name}", "brief": "Character 1 waves.", "duration": 5,
            "easy": True, "see_pictures": True, "references": refs}
    return body, sent


def test_nine_pictures_load_with_the_vision_context(monkeypatch, tmp_path):
    body, sent = _labelled(monkeypatch, tmp_path, 9)
    forge._generate(body, str(tmp_path), None, None)
    assert sent == [{"images": 9, "num_ctx": 32768}]


def test_blind_drafts_keep_the_default_context(monkeypatch, tmp_path):
    body, sent = _labelled(monkeypatch, tmp_path, 9)
    body["see_pictures"] = False
    forge._generate(body, str(tmp_path), None, None)
    assert sent == [{"images": 0, "num_ctx": 16384}]


def test_one_picture_costs_only_its_own_room(monkeypatch, tmp_path):
    body, sent = _labelled(monkeypatch, tmp_path, 1, kind="ollama", name="qwen3-vl:8b")
    forge._generate(body, str(tmp_path), None, None)
    assert sent == [{"images": 1, "num_ctx": 18432}]


def test_a_server_keeps_its_own_context(monkeypatch, tmp_path):
    for kind, name in (("openai", "m"), ("local", "Qwen3-VL-8B")):
        body, sent = _labelled(monkeypatch, tmp_path, 9, kind=kind, name=name)
        forge._generate(body, str(tmp_path), None, None)
        assert sent == [{"images": 9, "num_ctx": 16384}]


def test_approximate_overflow_does_not_block_backend(monkeypatch, tmp_path):
    body, sent = _labelled(monkeypatch, tmp_path, 9, kind="ollama", name="qwen3-vl:8b")
    monkeypatch.setattr(forge, "context_for", lambda num_ctx, images: num_ctx)
    forge._generate(body, str(tmp_path), None, None)
    assert sent == [{"images": 9, "num_ctx": 16384}]


def test_server_context_refusal_is_not_retried_blind(monkeypatch, tmp_path):
    body, sent = _labelled(monkeypatch, tmp_path, 2, kind="openai", name="m")
    calls = []

    class Refusing:
        base = "http://127.0.0.1:8099"
        def can_see(self, model):
            return None
        def chat(self, model, system, user, images, *rest):
            calls.append(len(images))
            raise urlerror.HTTPError("http://x", 400, "Bad Request", {},
                                     io.BytesIO(b'{"error":"the request exceeds the available context size, try increasing it"}'))
        def unload(self, model):
            return True

    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": Refusing()})
    with pytest.raises(backends.ForgeError) as err:
        forge._generate(body, str(tmp_path), None, None)
    assert err.value.code == "too_long" and "--ctx-size" in err.value.message
    assert calls == [2]


def test_llama_cpp_failures_are_said_plainly():
    err = backends.local_context_error(RuntimeError("llama_decode returned 1"), 16384, 3)
    assert err.code == "too_long" and "16,384" in err.message and "fewer pictures" in err.message
    err = backends.local_context_error(RuntimeError("Failed to evaluate chunk: error code 1"), 16384, 9)
    assert err.code == "too_long" and "fewer pictures" in err.message
    err = backends.local_context_error(ValueError("Failed to create llama_context"), 32768, 3)
    assert err.code == "memory" and "untick" in err.message
    assert backends.local_context_error(ValueError("something else"), 16384, 0) is None
    assert backends.server_context_error("model not found") is None


@pytest.mark.parametrize("message", [
    "llama_decode returned -1", "llama_decode returned 2",
    "llama_decode returned 10", "Failed to evaluate chunk: error code -3",
])
def test_unknown_decode_failures_are_preserved(message):
    assert backends.local_context_error(RuntimeError(message), 32768, 9) is None


def test_decode_oom_takes_priority():
    error = backends.local_context_error(
        RuntimeError("llama_decode returned -2: out of memory"), 32768, 9)
    assert error is not None and error.code == "memory"


@pytest.mark.parametrize("message", [
    "invalid context size: must be positive",
    "Failed to allocate KV cache for context size 32768: out of memory",
    "context length configuration is unsupported",
])
def test_server_non_overflow_context_errors_are_preserved(message):
    assert backends.server_context_error(message) is None


def test_continuity_estimate_is_advisory(monkeypatch, tmp_path):
    calls = []

    class Backend:
        base = "http://127.0.0.1:11434"
        def models(self):
            return [{"id": "ollama:test"}]
        def can_see(self, name):
            return False
        def chat(self, name, system, user, images, sampling, context, *rest):
            calls.append(context)
            return "Continue the action.", {}
        def unload(self, name):
            return True

    monkeypatch.setattr(forge, "backends", lambda settings: {"ollama": Backend()})
    result = forge.generate_continuity_draft(
        {"prompt": " scene" * 4000}, " scene" * 2000, str(tmp_path),
        "ollama:test", {}, current_prompt=" scene" * 2000,
        existing_definitions=" scene" * 2000)
    assert calls == [16384]
    assert result["prompt"] == "Continue the action."


@pytest.mark.parametrize("message, expected", [
    ("llama_decode returned 1", "too_long"),
    ("llama_decode returned -2: out of memory", "memory"),
    ("llama_decode returned -1", None),
])
def test_local_chat_maps_errors_and_releases_memory(monkeypatch, message, expected):
    from types import SimpleNamespace
    released = []

    def fail(*args, **kwargs):
        raise RuntimeError(message)

    runtime = SimpleNamespace(
        _resolve_model_path=lambda *args, **kwargs: "model.gguf",
        _messages_for_llama_cpp=lambda *args, **kwargs: [],
        _load_llama_cpp_model=fail,
        _release_all_model_memory=lambda: released.append(True))
    backend = backends.Local()
    monkeypatch.setattr(backend, "_llm", lambda: runtime)
    monkeypatch.setattr(backends, "_half_dtype", lambda: "float16")
    with pytest.raises(backends.ForgeError if expected else RuntimeError) as error:
        backend.chat("model.gguf", "system", "user", [], {}, 16384, 600)
    if expected:
        assert error.value.code == expected
    else:
        assert str(error.value) == message
    assert released == [True]


def test_transformers_memory_names_no_context():
    oom = RuntimeError("CUDA out of memory. Tried to allocate 10.25 GiB.")
    err = backends.local_context_error(oom, 16384, 9, gguf=False)
    assert err.code == "memory" and "9 pictures" in err.message and "untick" in err.message
    assert "context" not in err.message and "16,384" not in err.message
    # llama.cpp wording means nothing to a transformers model.
    assert backends.local_context_error(RuntimeError("llama_decode returned 1"), 16384, 3, gguf=False) is None
