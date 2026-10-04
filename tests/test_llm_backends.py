"""Shared transport policy and workflow security contracts."""
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from PIL import Image
from nodes import llm_backends as backend


def test_workflow_addresses_are_operator_config(monkeypatch):
    monkeypatch.setenv("DASIWA_LLM_OPENAI_URL", "http://trusted:8041/v1/")
    monkeypatch.setenv("DASIWA_LLM_OPENAI_API_KEY", " secret ")
    monkeypatch.delenv("DASIWA_LLM_OLLAMA_URL", raising=False)
    assert backend.workflow_server_settings() == {
        "openai_url":"http://trusted:8041/v1", "openai_api_key":"secret",
        "ollama_url":backend.DEFAULT_OLLAMA}


@pytest.mark.parametrize("value", ["http://user:secret@trusted/v1", "https://trusted/v1?key=secret", "ftp://secret", "http://trusted/#secret", "http://trusted:invalid", "http://trusted/ secret"])
def test_workflow_bad_urls_rejected_without_secrets(monkeypatch, value):
    monkeypatch.setenv("DASIWA_LLM_OPENAI_URL", value)
    with pytest.raises(backend.ForgeError) as error:
        backend.workflow_server_settings()
    assert error.value.code == "bad_url"
    assert "secret" not in str(error.value)
    assert error.value.raw is None


def test_pil_encoder_preserves_sampled_dimensions_and_order():
    images = [Image.new("RGBA", (1200, 3), "red"), Image.new("L", (7, 9), 50)]
    encoded = backend.pil_images_b64(images)
    decoded = [Image.open(io.BytesIO(base64.b64decode(b))) for b in encoded]
    assert [im.size for im in decoded] == [(1200, 3), (7, 9)]
    assert all(im.mode == "RGB" and im.format == "JPEG" for im in decoded)
    assert images[0].mode == "RGBA"


def test_workflow_interrupt_propagates_comfy_interrupt(monkeypatch):
    import comfy.model_management as management
    calls = []
    monkeypatch.setattr(management, "throw_exception_if_processing_interrupted", lambda: calls.append(True))
    assert backend.WorkflowInterrupt().is_set() is False
    assert calls == [True]
    def interrupt():
        raise management.InterruptProcessingException()
    monkeypatch.setattr(management, "throw_exception_if_processing_interrupted", interrupt)
    with pytest.raises(management.InterruptProcessingException):
        backend.WorkflowInterrupt().is_set()


@pytest.mark.parametrize("kind", ["openai", "ollama_server"])
@pytest.mark.parametrize("unload", [False, True])
def test_workflow_remote_payload_uses_only_operator_settings(monkeypatch, kind, unload):
    monkeypatch.setenv("DASIWA_LLM_OPENAI_URL", "http://trusted/v1")
    monkeypatch.setenv("DASIWA_LLM_OPENAI_API_KEY", "secret")
    monkeypatch.setenv("DASIWA_LLM_OLLAMA_URL", "http://trusted")
    response = ['data: {"choices":[{"delta":{"content":"done"}}]}', 'data: [DONE]'] if kind == "openai" else ['{"message":{"content":"done"},"done":true}']
    sent = capture_lines(monkeypatch, response)
    unloaded = []
    cls = backend.OpenAICompatible if kind == "openai" else backend.Ollama
    monkeypatch.setattr(cls, "unload", lambda self, model: unloaded.append(model) or True)
    config = {"backend":kind, "model_path":"served", "cache_mode":"unload_after_run" if unload else "cached",
              "openai_url":"http://evil", "ollama_url":"http://evil", "api_key":"evil",
              "llama_n_ctx":4096, "ollama_timeout":17}
    text, count, status = backend.run_workflow_server(config, "sys", "idea",
        [Image.new("RGB", (5, 7))], 123, 0.2, 0.9, 1.1, 42)
    assert (text, count) == ("done", 1)
    assert status == {"unload_requested":unload, "unloaded":unload}
    assert unloaded == (["served"] if unload else [])
    url, payload, timeout, cancel, headers = sent[0]
    assert url.startswith("http://trusted/")
    assert timeout == 17 and isinstance(cancel, backend.WorkflowInterrupt)
    assert payload["messages"][0]["content"] == "sys"
    if kind == "openai":
        assert payload["messages"][1]["content"][0] == {"type":"text", "text":"idea"}
        assert len(payload["messages"][1]["content"]) == 2
        assert payload["max_tokens"] == 123 and payload["seed"] == 42
        assert payload["temperature"] == 0.2 and payload["top_p"] == 0.9
        assert headers == {"Authorization":"Bearer secret"}
    else:
        assert payload["messages"][1]["content"] == "idea"
        assert len(payload["messages"][1]["images"]) == 1
        assert payload["options"] == {"num_ctx":4096, "num_predict":123, "temperature":0.2,
            "top_p":0.9, "presence_penalty":0, "seed":42, "repeat_penalty":1.1}
        assert payload["keep_alive"] == (0 if unload else "5m")
    assert "secret" not in repr(status)


@pytest.mark.parametrize("model", ["http://evil/key", "https://user:secret@evil", "//evil/model", "", "a\r\nb"])
def test_workflow_model_cannot_be_endpoint(monkeypatch, model):
    monkeypatch.setenv("DASIWA_LLM_OPENAI_URL", "http://trusted")
    sent = capture_lines(monkeypatch, [])
    with pytest.raises(ValueError):
        backend.run_workflow_server({"backend":"openai", "model_path":model, "cache_mode":"cached"},
            "s", "u", [], 12, 0.2, 0.9, 1, -1)
    assert not sent


def test_workflow_openai_requires_operator_url(monkeypatch):
    monkeypatch.delenv("DASIWA_LLM_OPENAI_URL", raising=False)
    sent = capture_lines(monkeypatch, [])
    with pytest.raises(ValueError, match="DASIWA_LLM_OPENAI_URL"):
        backend.run_workflow_server({"backend":"openai", "model_path":"m", "openai_url":"http://evil"},
            "s", "u", [], 12, 0.2, 0.9, 1, -1)
    assert not sent


def test_workflow_finally_unload_never_masks_generation_error(monkeypatch):
    monkeypatch.setenv("DASIWA_LLM_OPENAI_URL", "http://trusted")
    original = TimeoutError("generation timed out")
    calls = []
    def fail(*args, **kwargs):
        raise original
    def unload(self, model):
        calls.append(model)
        raise RuntimeError("unload failed secret")
    monkeypatch.setattr(backend.OpenAICompatible, "chat", fail)
    monkeypatch.setattr(backend.OpenAICompatible, "unload", unload)
    with pytest.raises(TimeoutError) as error:
        backend.run_workflow_server({"backend":"openai", "model_path":"m", "cache_mode":"cached"},
            "s", "u", [], 12, 0.2, 0.9, 1, -1, cleanup_after=True)
    assert error.value is original
    assert calls == ["m"]


@pytest.mark.parametrize("failure", [
    backend.urlerror.HTTPError("http://trusted", 401, "secret", {}, None),
    backend.urlerror.URLError("secret"),
    backend.ForgeError("backend", "server echoed secret", raw="secret"),
    ValueError("malformed secret JSON"),
])
def test_workflow_transport_errors_do_not_expose_secrets(monkeypatch, failure):
    monkeypatch.setenv("DASIWA_LLM_OPENAI_URL", "http://trusted")
    monkeypatch.setenv("DASIWA_LLM_OPENAI_API_KEY", "secret")
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(backend.OpenAICompatible, "chat", fail)
    with pytest.raises(backend.ForgeError) as error:
        backend.run_workflow_server({"backend":"openai", "model_path":"m", "cache_mode":"cached"},
            "s", "u", [], 12, 0.2, 0.9, 1, -1)
    assert "secret" not in str(error.value)
    assert error.value.raw is None
    assert error.value.__suppress_context__


@pytest.mark.parametrize("images", [None, [], ["jpeg1", "jpeg2"]])
def test_gguf_generation_supports_images_without_changing_text_calls(images):
    from nodes import llm_runtime as runtime
    sent = []
    class Model:
        def create_chat_completion(self, **kwargs):
            sent.append(kwargs)
            return {"choices":[{"message":{"content":" done "}}]}
    loaded = SimpleNamespace(model=Model())
    args = (loaded, "sys", "idea", 123, 0.2, 0.9, 1.1, 42)
    text, count = runtime._run_llama_cpp_generation(*args, images_b64=images)
    assert (text, count) == ("done", len(images or []))
    content = "idea" if not images else [{"type":"text", "text":"idea"}] + [
        {"type":"image_url", "image_url":{"url":f"data:image/jpeg;base64,{b}"}} for b in images]
    assert sent == [{"messages":[{"role":"system", "content":"sys"}, {"role":"user", "content":content}],
        "max_tokens":123, "temperature":0.2, "top_p":0.9, "repeat_penalty":1.1, "seed":42}]


@pytest.mark.parametrize("system", ["sys", ""])
@pytest.mark.parametrize("images", [[], ["jpeg"]])
def test_director_local_gguf_uses_shared_messages(monkeypatch, system, images):
    from nodes import llm_runtime as runtime
    expected = [{"role":"system", "content":system}, {"role":"user", "content":
        "idea" if not images else [{"type":"text", "text":"idea"},
            {"type":"image_url", "image_url":{"url":"data:image/jpeg;base64,jpeg"}}]}]
    built = []
    builder = runtime._messages_for_llama_cpp
    def messages(*args, **kwargs):
        result = builder(*args, **kwargs)
        built.append(result)
        return result
    monkeypatch.setattr(runtime, "_messages_for_llama_cpp", messages)
    sent, released = [], []
    class Model:
        def create_chat_completion(self, **kwargs):
            sent.append(kwargs)
            yield {"choices":[{"delta":{"content":"done"}}]}
        def close(self):
            released.append("closed")
    monkeypatch.setattr(runtime, "_resolve_model_path", lambda *a, **k: "m.gguf")
    monkeypatch.setattr(runtime, "_load_llama_cpp_model", lambda *a, **k: SimpleNamespace(model=Model()))
    monkeypatch.setattr(runtime, "_release_all_model_memory", lambda: released.append("released"))
    monkeypatch.setattr(backend.Local, "_mmproj", lambda *a: "mmproj.gguf")
    text, _ = backend.Local().chat("m.gguf", system, "idea", images, {}, 8192, 60)
    assert text == "done" and released == ["closed", "released"]
    assert built == [expected] and sent[0]["messages"] == expected
    assert sent[0]["stream"] is True and sent[0]["max_tokens"] == 3500


def test_shared_gguf_messages_preserve_empty_system_workflow_abi():
    from nodes import llm_runtime as runtime
    assert runtime._messages_for_llama_cpp("", "idea") == [{"role":"user", "content":"idea"}]


@pytest.mark.parametrize("response", [
    'data: {"error":{"message":"secret"}}', 'data: []',
    'data: {"choices":[{"delta":{"content":7}}]}'])
def test_openai_rejects_error_or_malformed_chunks(monkeypatch, response):
    capture_lines(monkeypatch, [response])
    with pytest.raises(backend.ForgeError) as error:
        backend.OpenAICompatible("http://trusted").chat("m", "s", "u", [], {}, 8192, 60)
    assert error.value.code == "backend"


def test_ollama_unload_does_not_log_server_secrets(monkeypatch):
    logs = []
    def fail(*args, **kwargs):
        raise backend.urlerror.URLError("secret")
    monkeypatch.setattr(backend, "_http", fail)
    monkeypatch.setattr(backend, "log_dasiwa", lambda *args: logs.append(args))
    assert backend.Ollama("http://trusted").unload("m") is False
    assert "secret" not in repr(logs)


@pytest.fixture
def fake_http():
    """Isolated deterministic HTTP fixture; never an existing model service."""
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, payload, self.headers.get("Authorization")))
            if payload.get("model") == "auth":
                self.send_response(401)
                self.end_headers()
                return
            if payload.get("model") == "slow":
                time.sleep(0.1)
            self.send_response(200)
            self.end_headers()
            data = (b'data: {"choices":[{"delta":{"content":"done"}}]}\n\n'
                    b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\n'
                    b'data: [DONE]\n\n')
            if self.path == "/api/chat":
                data = b'{"message":{"content":"done"},"done":true}\n'
                if payload.get("model") == "error":
                    data = b'{"error":"fixture refused"}\n'
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("kind", ["openai", "ollama_server"])
def test_workflow_real_http_stream(fake_http, monkeypatch, kind):
    url, calls = fake_http
    monkeypatch.setenv("DASIWA_LLM_OPENAI_URL", url)
    monkeypatch.setenv("DASIWA_LLM_OLLAMA_URL", url)
    monkeypatch.setenv("DASIWA_LLM_OPENAI_API_KEY", "fixture-key")
    text, count, status = backend.run_workflow_server(
        {"backend":kind, "model_path":"served", "cache_mode":"cached"},
        "sys", "idea", [], 27, 0.3, 0.8, 1, -1)
    assert (text, count, status) == ("done", 0, {"unload_requested":False, "unloaded":False})
    assert calls[0][0] == ("/v1/chat/completions" if kind == "openai" else "/api/chat")
    assert calls[0][2] == ("Bearer fixture-key" if kind == "openai" else None)


@pytest.mark.parametrize("model,code", [("auth", "auth"), ("error", "backend")])
def test_workflow_real_http_errors(fake_http, monkeypatch, model, code):
    url, _ = fake_http
    monkeypatch.setenv("DASIWA_LLM_OLLAMA_URL", url)
    with pytest.raises(backend.ForgeError) as error:
        backend.run_workflow_server({"backend":"ollama_server", "model_path":model, "cache_mode":"cached"},
            "sys", "idea", [], 27, 0.3, 0.8, 1, -1)
    assert error.value.code == code


def test_workflow_real_http_timeout(fake_http, monkeypatch):
    url, _ = fake_http
    monkeypatch.setenv("DASIWA_LLM_OPENAI_URL", url)
    with pytest.raises(TimeoutError):
        backend.run_workflow_server({"backend":"openai", "model_path":"slow", "cache_mode":"cached", "ollama_timeout":0.01},
            "sys", "idea", [], 27, 0.3, 0.8, 1, -1)


def test_workflow_interrupt_during_real_stream_unloads(fake_http, monkeypatch):
    import comfy.model_management as management
    url, _ = fake_http
    monkeypatch.setenv("DASIWA_LLM_OPENAI_URL", url)
    unloaded = []
    monkeypatch.setattr(backend.OpenAICompatible, "unload", lambda self, model: unloaded.append(model) or True)
    def interrupt():
        raise management.InterruptProcessingException()
    monkeypatch.setattr(management, "throw_exception_if_processing_interrupted", interrupt)
    with pytest.raises(management.InterruptProcessingException):
        backend.run_workflow_server({"backend":"openai", "model_path":"served", "cache_mode":"unload_after_run"},
            "sys", "idea", [], 27, 0.3, 0.8, 1, -1)
    assert unloaded == ["served"]


def test_director_cancellation_during_real_stream(fake_http):
    url, _ = fake_http
    stop = threading.Event()
    stop.set()
    with pytest.raises(backend.ForgeError) as error:
        backend.OpenAICompatible(url).chat("served", "sys", "idea", [], {}, 8192, 3, stop)
    assert error.value.code == "cancelled"


@pytest.mark.parametrize("response", ['[]', '{"message":"invalid"}', '{"message":{"content":7}}'])
def test_ollama_rejects_malformed_chunks(monkeypatch, response):
    capture_lines(monkeypatch, [response])
    with pytest.raises(backend.ForgeError) as error:
        backend.Ollama("http://trusted").chat("m", "s", "u", [], {}, 8192, 60)
    assert error.value.code == "backend"


def test_ollama_unload_cannot_confirm_release_after_status_failure(monkeypatch):
    def http(url, *args, **kwargs):
        if url.endswith("/api/ps"):
            raise backend.urlerror.URLError("unreachable")
        return {}
    monkeypatch.setattr(backend, "_http", http)
    assert backend.Ollama("http://trusted").unload("m") is False


def capture_lines(monkeypatch, response):
    sent = []
    def lines(url, payload, timeout, cancel, headers=None):
        sent.append((url, payload, timeout, cancel, headers))
        yield from response
    monkeypatch.setattr(backend, "_stream_lines", lines)
    return sent


def test_ollama_workflow_limits_and_lifetime(monkeypatch):
    sent = capture_lines(monkeypatch, ['{"message":{"content":"done"},"done":true}'])
    text, _ = backend.Ollama("http://trusted").chat(
        "m", "s", "u", [], {}, 8192, 60, max_tokens=123,
        keep_alive="5m", seed=42, repetition_penalty=1.1)
    assert text == "done"
    payload = sent[0][1]
    assert payload["options"] == {"num_ctx":8192, "num_predict":123,
        "temperature":0.7, "top_p":0.8, "presence_penalty":0,
        "seed":42, "repeat_penalty":1.1}
    assert payload["keep_alive"] == "5m"


def test_openai_workflow_limit_key_and_usage(monkeypatch):
    sent = capture_lines(monkeypatch, [
        'data: {"choices":[{"delta":{"content":"done"}}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2}}',
        'data: [DONE]', 'data: not-json'])
    text, stats = backend.OpenAICompatible("http://trusted/v1", "secret").chat(
        "m", "s", "u", [], {}, 8192, 60, max_tokens=123, seed=42,
        repetition_penalty=1.2, keep_alive="5m")
    assert text == "done"
    assert stats == {"prompt_tokens":7, "output_tokens":2}
    assert sent[0][0] == "http://trusted/v1/chat/completions"
    assert sent[0][1]["max_tokens"] == 123
    assert sent[0][1]["seed"] == 42
    assert "repetition_penalty" not in sent[0][1]
    assert "keep_alive" not in sent[0][1]
    assert sent[0][4] == {"Authorization":"Bearer secret"}


def test_director_ollama_defaults_are_unchanged(monkeypatch):
    sent = capture_lines(monkeypatch, ['{"done":true}'])
    backend.Ollama("http://trusted").chat("m", "s", "u", [], {}, 8192, 60)
    assert sent[0][1]["options"] == {"num_ctx":8192, "num_predict":3500,
        "temperature":0.7, "top_p":0.8, "presence_penalty":0}
    assert sent[0][1]["keep_alive"] == 0
