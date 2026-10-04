"""The simple Prompt Writer drives the advanced Analyze path with fixed defaults."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nodes import nodes_llm_simple as simple
from nodes import llm_prompt_presets as prompts


def _fake_server(monkeypatch, raw, sent):
    from nodes import llm_backends
    monkeypatch.setenv("DASIWA_LLM_OPENAI_URL", "http://trusted")
    def stream(url, payload, timeout, cancel, headers=None):
        sent.append(payload)
        yield "data: " + json.dumps({"choices": [{"delta": {"content": raw}}]})
        yield "data: [DONE]"
    monkeypatch.setattr(llm_backends, "_stream_lines", stream)
    monkeypatch.setattr(llm_backends.OpenAICompatible, "unload", lambda self, name: False)


def test_schema_is_the_short_list():
    schema = simple.DaSiWa_LLMPromptWriter.INPUT_TYPES()
    assert list(schema["required"]) == ["model", "write_for", "idea", "seed", "keep_loaded"]
    assert list(schema["optional"]) == ["picture", "server_model"]
    assert schema["required"]["write_for"][0] == list(simple.WRITE_FOR)
    assert simple.DaSiWa_LLMPromptWriter.RETURN_NAMES == ("prompt",)


def test_every_target_is_a_real_preset():
    for preset in simple.WRITE_FOR.values():
        assert preset in prompts._SYSTEM_PROMPT_PRESET_LABELS
        assert prompts.preset_spec(preset)["system"]


def test_writes_through_the_server_with_the_guide(monkeypatch):
    sent = []
    _fake_server(monkeypatch, "<think>x</think>\n===SEGMENT: Positive prompt===\nbest_quality, cat_ears", sent)
    (prompt,) = simple.DaSiWa_LLMPromptWriter().write(
        model="None", write_for="Illustrious", idea=" a catgirl ", seed=5, keep_loaded=False,
        server_model="qwen3.5:9b")
    assert prompt == "best quality, cat ears"
    body = sent[0]
    assert body["messages"][0]["content"] == prompts.exported_system("promptforge_illustrious")
    assert body["messages"][1]["content"] == "a catgirl"
    assert body["max_tokens"] == simple.MAX_NEW_TOKENS
    assert body["model"] == "qwen3.5:9b"


def test_server_without_openai_address_uses_ollama(monkeypatch):
    monkeypatch.delenv("DASIWA_LLM_OPENAI_URL", raising=False)
    assert simple.simple_config("None", "qwen3.5:9b", True)["backend"] == "ollama_server"
    assert simple.simple_config("None", "qwen3.5:9b", True)["cache_mode"] == "cached"


def test_local_backend_follows_the_file(monkeypatch):
    monkeypatch.setattr(simple, "_resolve_model_path", lambda name, custom, allow_gguf: f"/models/{name}")
    gguf = simple.simple_config("qwen/model-Q8_0.gguf", "", False)
    assert gguf["backend"] == "llama_cpp" and gguf["cache_mode"] == "unload_after_run"
    assert simple.simple_config("Qwen3-VL-4B", "", False)["backend"] == "transformers"


def test_nothing_to_write_from_says_what_to_do():
    with pytest.raises(ValueError, match="Type an idea"):
        simple.DaSiWa_LLMPromptWriter().write(model="x", write_for="Anima", idea="  ", seed=0, keep_loaded=False)


def test_no_model_says_where_to_put_one(monkeypatch):
    monkeypatch.delenv("DASIWA_LLM_OPENAI_URL", raising=False)
    with pytest.raises(ValueError, match="models/llm"):
        simple.simple_config("None", "", False)
