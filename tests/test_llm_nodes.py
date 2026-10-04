"""Characterize the saved-workflow ABI before shared-owner extraction."""
import json
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
