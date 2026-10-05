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
        sent.append((url, payload))
        yield "data: " + json.dumps({"choices": [{"delta": {"content": raw}}]})
        yield "data: [DONE]"
    monkeypatch.setattr(llm_backends, "_stream_lines", stream)
    monkeypatch.setattr(llm_backends.OpenAICompatible, "unload", lambda self, name: False)


def test_schema_is_the_short_list(monkeypatch):
    monkeypatch.setattr(simple, "model_choices", lambda: ["None"])
    schema = simple.DaSiWa_LLMPromptWriter.INPUT_TYPES()
    assert list(schema["required"]) == ["model", "write_for", "start_with", "idea", "style", "detail", "creativity",
                                        "max_tokens", "seed", "keep_loaded"]
    assert schema["required"]["style"][0][0] == "None" and "Watercolor" in schema["required"]["style"][0]
    for name in ("detail", "creativity"):
        assert (schema["required"][name][1]["min"], schema["required"][name][1]["max"], schema["required"][name][1]["default"]) == (1, 10, 5)
    assert list(schema["optional"]) == ["images"]
    assert schema["required"]["write_for"][0] == list(simple.WRITE_FOR)
    assert simple.DaSiWa_LLMPromptWriter.RETURN_NAMES == ("prompt",)
    assert simple.DaSiWa_LLMPromptWriter.VALIDATE_INPUTS("Ollama: gone:latest") is True


def test_every_target_is_a_real_preset():
    for preset in simple.WRITE_FOR.values():
        assert preset in prompts._SYSTEM_PROMPT_PRESET_LABELS
        assert prompts.preset_spec(preset)["system"]


def test_one_list_holds_local_ollama_and_server_models(monkeypatch):
    from nodes import llm_backends
    monkeypatch.setenv("DASIWA_LLM_OPENAI_URL", "http://trusted:8099")
    monkeypatch.delenv("DASIWA_LLM_OLLAMA_URL", raising=False)
    monkeypatch.setattr(llm_backends, "comfy_settings", lambda: {})
    monkeypatch.setattr(simple, "_list_llm_models", lambda: ["qwen/model-Q8_0.gguf"])
    asked = []
    monkeypatch.setattr(llm_backends.Ollama, "models", lambda self, timeout=10: asked.append(("ollama", self.base, timeout)) or [{"id": "ollama:qwen3.5:9b", "label": "qwen3.5:9b (9B)"}])
    monkeypatch.setattr(llm_backends.OpenAICompatible, "models", lambda self, timeout=10: asked.append(("openai", self.base, timeout)) or [{"id": "openai:Qwen3-VL-8B", "label": "Qwen3-VL-8B"}])
    llm_backends.refresh_server_model_choices()
    assert simple.model_choices() == ["qwen/model-Q8_0.gguf", "Ollama: qwen3.5:9b", "Server: Qwen3-VL-8B"]
    assert asked == [("ollama", "http://127.0.0.1:11434", 2), ("openai", "http://trusted:8099", 2)]


def test_unreachable_servers_leave_the_local_list(monkeypatch):
    from nodes import llm_backends
    monkeypatch.delenv("DASIWA_LLM_OPENAI_URL", raising=False)
    monkeypatch.setattr(llm_backends, "comfy_settings", lambda: {})
    monkeypatch.setattr(simple, "_list_llm_models", lambda: ["None"])
    def down(self, timeout=10):
        raise OSError("connection refused")
    monkeypatch.setattr(llm_backends.Ollama, "models", down)
    with llm_backends._model_choices_condition:
        llm_backends._model_choices_cache.clear()
    llm_backends.refresh_server_model_choices()
    assert simple.model_choices() == ["None"]


def test_writes_through_the_server_with_the_guide(monkeypatch):
    sent = []
    _fake_server(monkeypatch, "<think>x</think>\n===SEGMENT: Positive prompt===\nbest_quality, cat_ears", sent)
    (prompt,) = simple.DaSiWa_LLMPromptWriter().write(
        model="Server: qwen3.5:9b", write_for="Illustrious", idea=" a catgirl ", seed=5, keep_loaded=False)
    assert prompt == "best quality, cat ears"
    body = sent[0][1]
    assert body["messages"][0]["content"] == prompts.exported_system("promptforge_illustrious")
    assert body["messages"][1]["content"] == "a catgirl"
    assert body["max_tokens"] == simple.DEFAULT_MAX_TOKENS
    assert body["temperature"] == 0.70
    assert body["model"] == "qwen3.5:9b"


def test_sliders_and_the_front_of_the_prompt(monkeypatch):
    sent = []
    _fake_server(monkeypatch, "===SEGMENT: Positive prompt===\nmasterpiece, 1girl, my_lora, cat_ears", sent)
    (prompt,) = simple.DaSiWa_LLMPromptWriter().write(
        model="Server: m", write_for="Illustrious", idea="a catgirl", seed=5, keep_loaded=False,
        start_with="masterpiece, my_lora, (detailed:1.2)", detail=9, creativity=1, max_tokens=900)
    # The person's text untouched, then the model's tags minus repeats.
    assert prompt == "masterpiece, my_lora, (detailed:1.2), 1girl, cat ears"
    body = sent[0][1]
    assert body["max_tokens"] == 900 and body["temperature"] == 0.30
    user = body["messages"][1]["content"]
    assert user.startswith("a catgirl\n\nCreativity 1 of 10: Add nothing.")
    assert "Detail 9 of 10: richly detailed" in user


def test_style_carries_tags_or_words_by_dialect():
    anima, krea = prompts.preset_spec("promptforge_anima"), prompts.preset_spec("promptforge_krea2")
    assert simple.style_line("Watercolor", anima) == r"Style: Watercolor - include these tags: watercolor \(medium\), traditional media, soft edges"
    assert simple.style_line("Watercolor", krea).startswith("Style: Watercolor - carry this look through the description: watercolor, paper texture")
    # No tag list of its own: a tag model gets the words.
    assert simple.style_line("Digital art", anima).startswith("Style: Digital art - include these tags: digital painting")
    assert simple.style_line("None", anima) == "" and simple.style_line("Missing", krea) == ""
    assert simple.request_text("idea", 5, 5, "Style: X") == "idea\n\nStyle: X"


def test_standard_sliders_add_no_lines():
    assert simple.request_text("idea", 5, 5) == "idea"
    assert simple.request_text("", 5, 5) == simple.PICTURE_ONLY
    assert [simple.CREATIVITY[n][0] for n in range(1, 11)] == [0.30, 0.45, 0.55, 0.62, 0.70, 0.78, 0.85, 0.92, 1.00, 1.10]


def test_front_on_prose_and_anima():
    krea = prompts.preset_spec("promptforge_krea2")
    assert simple.assemble("A woman at a stall.", krea, "ohwx woman") == "ohwx woman, A woman at a stall."
    assert simple.assemble("A woman at a stall.", krea, "") == "A woman at a stall."
    anima = prompts.preset_spec("promptforge_anima")
    assert "quality" not in anima
    out = simple.assemble("safe, 1girl, best quality, She walks, slowly, past the stall.", anima, "masterpiece, best_quality, score_7")
    assert out == "masterpiece, best_quality, score_7, safe, 1girl, She walks, slowly, past the stall."
    assert simple.assemble("safe, 1girl", anima, "") == "safe, 1girl"


def test_a_batch_is_sampled_up_to_eight_from_the_first(monkeypatch):
    from nodes import nodes_llm
    asked = []
    def prepare(images, max_frames, frame_stride, frame_strategy, resize_max_px, resize_algorithm):
        asked.append((images, max_frames, frame_strategy))
        return []
    monkeypatch.setattr(nodes_llm, "_prepare_images", prepare)
    _fake_server(monkeypatch, "===SEGMENT: Positive prompt===\nA dancer.", [])
    simple.DaSiWa_LLMPromptWriter().write(model="Server: m", write_for="Wan 2.2", idea="", seed=0, keep_loaded=False,
                                          images="batch")
    assert asked == [("batch", 8, "evenly_spaced")]
    assert nodes_llm._select_frame_indices(20, 8, 1, "evenly_spaced")[0] == 0


def test_prefixes_pick_the_backend(monkeypatch):
    monkeypatch.setattr(simple, "_resolve_model_path", lambda name, custom, allow_gguf: f"/models/{name}")
    ollama = simple.simple_config("Ollama: qwen3.5:9b", True)
    assert (ollama["backend"], ollama["model_path"], ollama["cache_mode"]) == ("ollama_server", "qwen3.5:9b", "cached")
    assert simple.simple_config("Server: a/b:Q8_0", False)["backend"] == "openai"
    gguf = simple.simple_config("qwen/model-Q8_0.gguf", False)
    assert gguf["backend"] == "llama_cpp" and gguf["cache_mode"] == "unload_after_run"
    assert simple.simple_config("Qwen3-VL-4B", False)["backend"] == "transformers"


def test_stopped_ollama_says_what_to_do(monkeypatch):
    from nodes import llm_backends
    def refused(*args, **kwargs):
        raise llm_backends.ForgeError("connection", "Could not reach the configured workflow server.")
    monkeypatch.setattr(simple.DaSiWa_LLMPromptWriter, "_analyze", staticmethod(refused))
    with pytest.raises(ValueError, match="Could not reach Ollama. Start it, check qwen3.5:9b"):
        simple.DaSiWa_LLMPromptWriter().write(model="Ollama: qwen3.5:9b", write_for="Anima", idea="x", seed=0, keep_loaded=False)


def _settings_file(monkeypatch, tmp_path, values):
    import types
    (tmp_path / "default").mkdir()
    (tmp_path / "default" / "comfy.settings.json").write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(get_user_directory=lambda: str(tmp_path)))
    monkeypatch.setenv("DASIWA_LLM_ALLOW_SETTINGS", "1")
    for env in ("DASIWA_LLM_OLLAMA_URL", "DASIWA_LLM_OPENAI_URL", "DASIWA_LLM_OPENAI_API_KEY"):
        monkeypatch.delenv(env, raising=False)


def test_servers_come_from_comfyui_settings_when_operator_opts_in(monkeypatch, tmp_path):
    from nodes import llm_backends
    _settings_file(monkeypatch, tmp_path, {
        "DaSiWa.H3Forge.OllamaURL": "http://192.168.0.50:11434/",
        "DaSiWa.H3Forge.OpenAIURL": "http://127.0.0.1:8099",
        "DaSiWa.H3Forge.OpenAIKey": "k",
        "Comfy.Other": "ignored",
    })
    assert llm_backends.workflow_server_settings() == {
        "ollama_url": "http://192.168.0.50:11434", "openai_url": "http://127.0.0.1:8099", "openai_api_key": "k"}


def test_environment_wins_over_settings(monkeypatch, tmp_path):
    from nodes import llm_backends
    _settings_file(monkeypatch, tmp_path, {"DaSiWa.H3Forge.OpenAIURL": "http://127.0.0.1:8099"})
    monkeypatch.setenv("DASIWA_LLM_OPENAI_URL", "http://10.0.0.2:1234/v1")
    settings = llm_backends.workflow_server_settings()
    assert settings["openai_url"] == "http://10.0.0.2:1234/v1"
    assert settings["ollama_url"] == "http://127.0.0.1:11434"


def test_no_settings_file_means_local_ollama_only(monkeypatch, tmp_path):
    import types
    from nodes import llm_backends
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(get_user_directory=lambda: str(tmp_path / "missing")))
    for env in ("DASIWA_LLM_OLLAMA_URL", "DASIWA_LLM_OPENAI_URL", "DASIWA_LLM_OPENAI_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    assert llm_backends.workflow_server_settings() == {
        "ollama_url": "http://127.0.0.1:11434", "openai_url": "", "openai_api_key": ""}


def _selector_defaults():
    from nodes import nodes_llm
    kwargs = {}
    for name, spec in nodes_llm.DaSiWa_LLMModelSelector.INPUT_TYPES()["required"].items():
        kwargs[name] = spec[1]["default"] if len(spec) > 1 and "default" in spec[1] else spec[0][0]
    return nodes_llm, kwargs


def test_advanced_selector_lists_server_models_after_local(monkeypatch):
    from nodes import llm_backends, nodes_llm
    monkeypatch.setattr(nodes_llm, "server_model_choices", lambda: ["Ollama: qwen3.5:9b", "Server: m"])
    choices = nodes_llm.DaSiWa_LLMModelSelector.INPUT_TYPES()["required"]["model"][0]
    assert choices[0] == "None" and choices[-2:] == ["Ollama: qwen3.5:9b", "Server: m"]
    assert nodes_llm.DaSiWa_LLMModelSelector.VALIDATE_INPUTS("Ollama: gone") is True


@pytest.mark.parametrize("choice,backend,name", [
    ("Ollama: qwen3.5:9b", "ollama_server", "qwen3.5:9b"),
    ("Server: hf.co/a/b:Q8_0", "openai", "hf.co/a/b:Q8_0"),
])
def test_advanced_selector_server_choice_sets_its_backend(monkeypatch, choice, backend, name):
    nodes_llm, kwargs = _selector_defaults()
    kwargs.update(model=choice, backend="transformers")
    (config,) = nodes_llm.DaSiWa_LLMModelSelector().select(**kwargs)
    assert (config["backend"], config["model_path"]) == (backend, name)


def test_advanced_selector_old_server_fields_still_work(monkeypatch):
    nodes_llm, kwargs = _selector_defaults()
    kwargs.update(model="None", backend="ollama_server")
    (config,) = nodes_llm.DaSiWa_LLMModelSelector().select(**kwargs, server_model="qwen3.5:9b")
    assert (config["backend"], config["model_path"]) == ("ollama_server", "qwen3.5:9b")


def test_nothing_to_write_from_says_what_to_do():
    with pytest.raises(ValueError, match="Type an idea"):
        simple.DaSiWa_LLMPromptWriter().write(model="x", write_for="Anima", idea="  ", seed=0, keep_loaded=False)


def test_no_model_says_where_to_put_one():
    with pytest.raises(ValueError, match="models/llm"):
        simple.simple_config("None", False)
