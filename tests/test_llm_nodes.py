"""Characterize the saved-workflow ABI before shared-owner extraction."""
import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nodes import nodes_llm as llm


def test_custom_system_and_both_text_sources():
    assert llm._compose_user_text("custom", " sys ", " idea ", " linked ") == (
        "sys", "idea\n\nlinked")


def test_preset_does_not_append_custom_system():
    system, user = llm._compose_user_text("enhance_video_wan22", "DO NOT APPEND", "idea", "")
    assert system == llm._SYSTEM_PROMPT_PRESETS["enhance_video_wan22"]
    assert "DO NOT APPEND" not in system
    assert user == "idea"


def test_native_prompt_and_legacy_text_socket():
    schema = llm.DaSiWa_LLMAnalyze.INPUT_TYPES()
    assert schema["required"]["prompt"][0] == "STRING"
    assert schema["required"]["prompt"][1]["multiline"] is True
    assert not schema["required"]["prompt"][1].get("forceInput", False)
    assert schema["optional"]["text_input"][0] == "STRING"
    assert schema["optional"]["text_input"][1]["forceInput"] is True
    assert llm.DaSiWa_LLMAnalyze.RETURN_TYPES == ("STRING", "STRING")
    assert llm.DaSiWa_LLMAnalyze.RETURN_NAMES == ("response", "info")


def test_legacy_contract_fixture_preserves_prefixes_and_presets():
    baseline = json.loads((Path(__file__).parent / "fixtures" / "llm_legacy_contract.json").read_text())
    for name, expected in baseline["nodes"].items():
        cls = getattr(llm, name)
        schema = cls.INPUT_TYPES()
        for group in ("required", "optional"):
            old = expected[group]
            assert list(schema.get(group, {}))[:len(old)] == old
        for field in ("RETURN_TYPES", "RETURN_NAMES", "FUNCTION", "CATEGORY"):
            value = getattr(cls, field)
            assert (list(value) if isinstance(value, tuple) else value) == expected[field]
    for key, value in baseline["presets"].items():
        assert llm._SYSTEM_PROMPT_PRESETS[key] == value
    assert list(llm._SYSTEM_PROMPT_PRESETS)[:len(baseline["presets"])] == list(baseline["presets"])


def test_unknown_legacy_preset_keeps_empty_system_fallback():
    assert llm._compose_user_text("missing", "ignore", "idea", "linked") == ("", "idea\n\nlinked")


def node_defaults(cls):
    result = {}
    for name, spec in cls.INPUT_TYPES()["required"].items():
        if len(spec) > 1 and "default" in spec[1]:
            result[name] = spec[1]["default"]
        elif isinstance(spec[0], list):
            result[name] = spec[0][0]
    return result


def test_remote_selector_does_not_resolve_local_model(monkeypatch):
    def fail_path(*args, **kwargs):
        raise AssertionError("Remote models must not resolve local files")
    monkeypatch.setattr(llm, "_resolve_model_path", fail_path)
    for backend in ("openai", "ollama_server"):
        kwargs = node_defaults(llm.DaSiWa_LLMModelSelector)
        kwargs.update(backend=backend, server_model="served-model")
        config, = llm.DaSiWa_LLMModelSelector().select(**kwargs)
        assert config["backend"] == backend
        assert config["model_path"] == "served-model"
        assert not {"openai_url", "ollama_url", "api_key", "openai_api_key"} & config.keys()


def test_h3_widgets_append_after_legacy_optional_inputs():
    schema = llm.DaSiWa_LLMAnalyze.INPUT_TYPES()
    assert list(schema["optional"]) == ["images", "text_input", "h3_mode", "h3_duration"]
    assert schema["optional"]["h3_mode"][0] == ["T2VA", "I2VA", "FL2VA", "L2VA"]
    assert schema["optional"]["h3_mode"][1]["default"] == "T2VA"
    assert schema["optional"]["h3_duration"][1]["default"] == 10.0


def test_remote_analyze_uses_shared_transport_without_local_unload(monkeypatch):
    from nodes import llm_backends
    sent = []
    monkeypatch.setenv("DASIWA_LLM_OPENAI_URL", "http://trusted/v1")
    monkeypatch.setenv("DASIWA_LLM_OPENAI_API_KEY", "private-key")

    def stream(url, payload, timeout, cancel, headers=None):
        sent.append((url, payload, headers))
        yield 'data: {"choices":[{"delta":{"content":"raw custom response"}}]}'
        yield 'data: [DONE]'
    monkeypatch.setattr(llm_backends, "_stream_lines", stream)
    monkeypatch.setattr(llm_backends.OpenAICompatible, "unload", lambda self, name: False)
    monkeypatch.setattr(llm, "_release_all_model_memory", lambda: (_ for _ in ()).throw(AssertionError("Remote unload cleared local cache")))
    config = {"backend": "openai", "model_path": "served-model", "cache_mode": "unload_after_run"}
    kwargs = node_defaults(llm.DaSiWa_LLMAnalyze)
    kwargs.update(llm_config=config, prompt="idea", system_prompt="custom sys", text_input="linked", max_new_tokens=123, repetition_penalty=1.1)
    response, info = llm.DaSiWa_LLMAnalyze().analyze(**kwargs)
    assert response == "raw custom response"
    assert sent[0][1]["messages"] == [{"role": "system", "content": "custom sys"}, {"role": "user", "content": "idea\n\nlinked"}]
    assert sent[0][1]["max_tokens"] == 123
    assert "remote_unloaded=False" in info
    assert "repetition_penalty_supported=False" in info
    assert "private-key" not in info


@pytest.mark.parametrize("preset,label", [
    ("promptforge_krea2", "Enhanced prompt"),
    ("promptforge_wan22", "Positive prompt"),
    ("promptforge_ltx", "Enhanced paragraph"),
    ("promptforge_anima", "Positive prompt"),
    ("promptforge_illustrious", "Positive prompt"),
])
def test_analyze_projects_only_model_prompt(monkeypatch, preset, label):
    from nodes import llm_backends
    monkeypatch.setenv("DASIWA_LLM_OPENAI_URL", "http://trusted")
    raw = f"<think>private reasoning</think>\n===SEGMENT: {label}===\nA brass workshop."
    def stream(*args, **kwargs):
        yield "data: " + json.dumps({"choices": [{"delta": {"content": raw}}]})
        yield "data: [DONE]"
    monkeypatch.setattr(llm_backends, "_stream_lines", stream)
    kwargs = node_defaults(llm.DaSiWa_LLMAnalyze)
    kwargs.update(llm_config={"backend": "openai", "model_path": "m", "cache_mode": "cached"}, system_prompt_preset=preset, prompt="idea")
    response, _ = llm.DaSiWa_LLMAnalyze().analyze(**kwargs)
    assert response == "A brass workshop."


def test_analyze_h3_uses_prepared_images_and_shared_projection(monkeypatch):
    from nodes import llm_backends, h3_prompting
    monkeypatch.setenv("DASIWA_LLM_OPENAI_URL", "http://trusted")
    sent = []
    raw = "===SEGMENT: Detailed description===\nA dancer turns.\n===SEGMENT: Soundscape===\nFootsteps.\n===SEGMENT: Music===\nN/A"
    def stream(url, payload, *args):
        sent.append(payload)
        yield "data: " + json.dumps({"choices": [{"delta": {"content": raw}}]})
        yield "data: [DONE]"
    monkeypatch.setattr(llm_backends, "_stream_lines", stream)
    kwargs = node_defaults(llm.DaSiWa_LLMAnalyze)
    kwargs.update(llm_config={"backend": "openai", "model_path": "m", "cache_mode": "cached"}, system_prompt_preset="promptforge_h3", prompt="A dancer turns", text_input="in a workshop", h3_duration=5.0)
    response, _ = llm.DaSiWa_LLMAnalyze().analyze(**kwargs)
    assert response.startswith("integrated_multimodal_description:")
    assert "overall_soundscape: Footsteps." in response
    assert "===SEGMENT:" not in response
    assert sent[0]["messages"][0]["content"] == h3_prompting.load_bundle()["modes"]["T2VA"]["system"]
    assert "Duration: 5.0 sec" in sent[0]["messages"][1]["content"]
    assert "in a workshop" in sent[0]["messages"][1]["content"]


def test_gguf_analyze_forwards_sampled_images(monkeypatch):
    from PIL import Image
    sampled = [Image.new("RGB", (12, 8))]
    monkeypatch.setattr(llm, "_prepare_images", lambda *args: sampled)
    monkeypatch.setattr(llm, "_find_mmproj", lambda path: "matching-mmproj.gguf")
    monkeypatch.setattr(llm, "_load_llama_cpp_model", lambda config, need_vision: "loaded-vision" if need_vision else "wrong")
    captured = []
    def generate(loaded, system, user, tokens, temperature, top_p, penalty, seed, images_b64=None):
        captured.append((loaded, images_b64))
        return "a brass workshop", len(images_b64)
    monkeypatch.setattr(llm, "_run_llama_cpp_generation", generate)
    kwargs = node_defaults(llm.DaSiWa_LLMAnalyze)
    kwargs.update(llm_config={"backend": "llama_cpp", "model_path": "local.gguf", "cache_mode": "cached"})
    response, info = llm.DaSiWa_LLMAnalyze().analyze(**kwargs)
    assert captured[0][0] == "loaded-vision"
    assert len(captured[0][1]) == 1
    assert response == "a brass workshop"
    assert "images_sent=1" in info


@pytest.mark.parametrize("mode,count", [("T2VA", 1), ("I2VA", 0), ("L2VA", 2), ("FL2VA", 1)])
def test_h3_invalid_image_count_fails_before_backend_load(monkeypatch, mode, count):
    from PIL import Image
    monkeypatch.setattr(llm, "_prepare_images", lambda *args: [Image.new("RGB", (8, 8))] * count)
    def never(*args, **kwargs):
        raise AssertionError("Invalid H3 inputs must not load a model")
    monkeypatch.setattr(llm, "_load_transformers_model", never)
    kwargs = node_defaults(llm.DaSiWa_LLMAnalyze)
    kwargs.update(llm_config={"backend": "transformers"}, system_prompt_preset="promptforge_h3", h3_mode=mode)
    with pytest.raises(ValueError, match="image"):
        llm.DaSiWa_LLMAnalyze().analyze(**kwargs)


@pytest.mark.parametrize("has_projector", [True, False])
def test_gguf_vision_analyze_real_loader_boundary(monkeypatch, tmp_path, has_projector):
    import types
    from PIL import Image
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model fixture; never loaded by llama.cpp")
    projector = tmp_path / "model-mmproj.gguf"
    if has_projector:
        projector.write_bytes(b"projector fixture")
    selection = node_defaults(llm.DaSiWa_LLMModelSelector)
    selection.update(backend="llama_cpp", custom_path=str(model), cache_mode="cached", llama_n_gpu_layers=0)
    config, = llm.DaSiWa_LLMModelSelector().select(**selection)
    captured = {}
    class FakeLlama:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def create_chat_completion(self, **kwargs):
            captured["generation"] = kwargs
            return {"choices": [{"message": {"content": "fixture caption"}}]}
    package = types.ModuleType("llama_cpp")
    package.Llama = FakeLlama
    formats = types.ModuleType("llama_cpp.llama_chat_format")
    formats.MTMDChatHandler = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "llama_cpp", package)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", formats)
    monkeypatch.setattr(llm, "_prepare_images", lambda *args: [Image.new("RGB", (8, 8))])
    monkeypatch.setattr(llm, "_LLM_CACHE", {})
    from nodes import llm_runtime
    monkeypatch.setattr(llm_runtime, "_LLM_CACHE", {})
    kwargs = node_defaults(llm.DaSiWa_LLMAnalyze)
    kwargs.update(llm_config=config)
    if not has_projector:
        with pytest.raises(ValueError, match="matching mmproj"):
            llm.DaSiWa_LLMAnalyze().analyze(**kwargs)
        assert not captured
        return
    response, info = llm.DaSiWa_LLMAnalyze().analyze(**kwargs)
    assert response == "fixture caption"
    assert "images_sent=1" in info
    assert captured["chat_handler"]["clip_model_path"] == str(projector)
    assert captured["generation"]["messages"][1]["content"][1]["type"] == "image_url"
    assert "llama_mmproj_path" not in config
