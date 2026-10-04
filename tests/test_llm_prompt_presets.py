"""Prompt-owner helpers use real exported systems and the shared H3 bundle."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nodes import h3_prompting
from nodes import llm_prompt_presets as prompts


def test_h3_uses_existing_bundle_exactly():
    system, user = prompts.h3_request("A dancer turns", "T2VA", 10, 0)
    assert system == h3_prompting.load_bundle()["modes"]["T2VA"]["system"]
    assert "A dancer turns" in user
    assert "Duration: 10 sec" in user
    assert "Omit aspect ratio" in user


@pytest.mark.parametrize("mode,count,required", [
    ("T2VA", 1, 0), ("I2VA", 0, 1), ("I2VA", 2, 1),
    ("FL2VA", 1, 2), ("FL2VA", 3, 2), ("L2VA", 0, 1),
])
def test_h3_requires_exact_sampled_image_count(mode, count, required):
    with pytest.raises(ValueError, match=f"exactly {required}"):
        prompts.h3_request("A dancer turns", mode, 10, count)


@pytest.mark.parametrize("mode", ["REF2VA", "EASY", "continuity", "unknown"])
def test_h3_rejects_director_only_modes(mode):
    with pytest.raises(ValueError, match="Director"):
        prompts.h3_request("A dancer turns", mode, 10, 0)


def test_h3_fl2va_attaches_keyframes_in_order():
    system, user = prompts.h3_request("A door opens", "FL2VA", 10, 2)
    assert system == h3_prompting.load_bundle()["modes"]["FL2VA"]["system"]
    assert user.index("- <Picture 1>") < user.index("- <Picture 2>")
    assert "<Picture 1>, <Picture 2> are attached to this message, in that order" in user
    assert "keyframe" in user
    assert "Omit aspect ratio" in user
    assert "not visible" not in user


@pytest.mark.parametrize("mode", ["T2VA", "I2VA", "FL2VA", "L2VA"])
def test_h3_response_projects_shared_description_sound_and_music(mode):
    values = {
        "Detailed description": "integrated_multimodal_description: [Shot 1] A dancer turns.",
        "Soundscape": "overall_soundscape: Shoes tap the floor.",
        "Music": "non_diegetic_music: A gentle piano melody.",
    }
    labels = h3_prompting.load_bundle()["modes"][mode]["segments"]
    raw = "<think>hidden</think>\n" + "\n".join(
        f"===SEGMENT: {label}===\n{values[label]}" for label in labels)
    response = prompts.h3_response(raw, mode, 10)
    assert "integrated_multimodal_description: [Shot 1] A dancer turns." in response
    assert "overall_soundscape: Shoes tap the floor." in response
    assert "non_diegetic_music: A gentle piano melody." in response
    assert "===SEGMENT" not in response
    assert "hidden" not in response
    expected = h3_prompting.simple_prompt(
        h3_prompting.builder_fields(h3_prompting.parse_segments(raw, labels), mode), mode, 10)
    assert response == expected


def test_exported_systems_are_loaded_verbatim_with_no_h3_copy():
    path = Path(__file__).resolve().parents[1] / "data" / "llm_prompt_presets.json"
    bundle = json.loads(path.read_text(encoding="utf-8"))
    specs = prompts.load_exported_presets()
    assert specs == bundle["presets"]
    assert list(specs) == ["promptforge_wan22", "promptforge_ltx", "promptforge_krea2",
                           "promptforge_anima", "promptforge_illustrious"]
    assert "promptforge_h3" not in specs
    for preset, spec in specs.items():
        assert prompts.exported_system(preset) == spec["system"]


def test_exported_bundle_rejects_unknown_version(tmp_path, monkeypatch):
    path = tmp_path / "unsupported.json"
    path.write_text(json.dumps({"schema_version": 2, "presets": {}}), encoding="utf-8")
    monkeypatch.setattr(prompts, "_BUNDLE_PATH", path, raising=False)
    with pytest.raises(ValueError, match="Unsupported"):
        prompts.load_exported_presets()


@pytest.mark.parametrize("preset", ["promptforge_wan22", "promptforge_ltx", "promptforge_krea2",
                                    "promptforge_anima", "promptforge_illustrious"])
def test_prompt_projection_removes_scaffolding(preset):
    label = prompts.load_exported_presets()[preset]["output"]
    raw = f"<think>hidden</think>\n===SEGMENT: {label}===\n A brass workshop. \n===SEGMENT: Extra===\nIgnore this."
    assert prompts.prompt_response(preset, raw) == "A brass workshop."


@pytest.mark.parametrize("raw,code", [
    ("unstructured chat", "no_segments"),
    ("===SEGMENT: Wrong===\nA workshop.", "missing_segments"),
    ("===SEGMENT: Refused===\nNo.", "refused"),
])
def test_prompt_projection_preserves_parser_errors(raw, code):
    with pytest.raises(h3_prompting.ForgeError) as error:
        prompts.prompt_response("promptforge_krea2", raw)
    assert error.value.code == code


def test_prompt_projection_rejects_empty_primary_segment():
    with pytest.raises(ValueError, match="empty prompt segment"):
        prompts.prompt_response("promptforge_krea2", "===SEGMENT: Enhanced prompt===\n \n")


def test_preset_labels_append_new_ids_after_exact_legacy_contract():
    baseline = json.loads((Path(__file__).parent / "fixtures" / "llm_legacy_contract.json").read_text())
    legacy = baseline["presets"]
    assert prompts._SYSTEM_PROMPT_PRESETS == legacy
    assert prompts._SYSTEM_PROMPT_PRESET_LABELS == list(legacy) + [
        "promptforge_wan22", "promptforge_ltx", "promptforge_krea2",
        "promptforge_anima", "promptforge_illustrious", "promptforge_h3",
    ]


@pytest.mark.parametrize("preset", ["promptforge_wan22", "promptforge_ltx", "promptforge_krea2",
                                    "promptforge_anima", "promptforge_illustrious"])
def test_exported_composition_ignores_custom_widget_and_preserves_text_order(preset):
    assert prompts._compose_user_text(preset, "DO NOT APPEND", " idea ", " linked ") == (
        prompts.exported_system(preset), "idea\n\nlinked")


def test_h3_composition_can_defer_resolution_until_images_are_prepared(monkeypatch):
    def premature_resolution(*args):
        pytest.fail("H3 system resolution must wait for prepared reference images")
    monkeypatch.setattr(prompts, "_resolve_system_prompt", premature_resolution)
    assert prompts._compose_user_text("promptforge_h3", "ignore", " idea ", " linked ",
                                      resolve_system=False) == ("", "idea\n\nlinked")


def test_legacy_composition_and_unknown_fallback_do_not_load_exported_bundle(monkeypatch):
    def unnecessary_export_load():
        pytest.fail("Legacy and custom prompts must not depend on the exported artifact")
    monkeypatch.setattr(prompts, "load_exported_presets", unnecessary_export_load)
    assert prompts._compose_user_text("custom", " sys ", " idea ", " linked ") == (
        "sys", "idea\n\nlinked")
    for preset, system in prompts._SYSTEM_PROMPT_PRESETS.items():
        if preset != "custom":
            assert prompts._compose_user_text(preset, "ignore", "idea", "") == (system, "idea")
    assert prompts._compose_user_text("missing", "ignore", "idea", "linked") == ("", "idea\n\nlinked")
