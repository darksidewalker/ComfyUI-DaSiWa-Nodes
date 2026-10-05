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


@pytest.mark.parametrize("preset,name,output,tag_style", [
    ("promptforge_wan22", "wan", "Positive prompt", None),
    ("promptforge_ltx", "ltx", "Enhanced paragraph", None),
    ("promptforge_krea2", "krea2", "Enhanced prompt", None),
    ("promptforge_anima", "anima", "Positive prompt", "space"),
    ("promptforge_illustrious", "illustrious", "Positive prompt", "space"),
])
def test_model_presets_are_skill_guides(preset, name, output, tag_style):
    spec = prompts.preset_spec(preset)
    assert spec["model"] == name
    assert spec["output"] == output and spec["segments"] == [output]
    assert spec.get("tag_style") == tag_style
    assert not spec["system"].startswith("---")
    assert f"===SEGMENT: {output}===" in spec["system"]
    assert prompts.exported_system(preset) == spec["system"]


def test_h3_has_no_guide_copy():
    assert "promptforge_h3" not in prompts._SKILL_PRESETS
    assert not (prompts._SKILLS_DIR / "h3.md").exists()


def test_skill_frontmatter_survives_crlf(tmp_path, monkeypatch):
    (tmp_path / "x.md").write_bytes(b"---\r\nname: x\r\noutput: Positive prompt\r\n---\r\n\r\n# Guide\r\n")
    monkeypatch.setattr(prompts, "_SKILLS_DIR", tmp_path)
    spec = prompts.load_skill("x")
    assert spec["output"] == "Positive prompt" and spec["system"] == "# Guide"


def test_skill_without_output_is_rejected(tmp_path, monkeypatch):
    (tmp_path / "x.md").write_text("---\nname: x\n---\n# Guide\n", encoding="utf-8")
    monkeypatch.setattr(prompts, "_SKILLS_DIR", tmp_path)
    with pytest.raises(ValueError, match="output segment"):
        prompts.load_skill("x")


@pytest.mark.parametrize("preset", ["promptforge_wan22", "promptforge_ltx", "promptforge_krea2",
                                    "promptforge_anima", "promptforge_illustrious"])
def test_prompt_projection_removes_scaffolding(preset):
    label = prompts.preset_spec(preset)["output"]
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


def test_legacy_composition_and_unknown_fallback_do_not_load_guides(monkeypatch):
    def unnecessary_guide_load(name):
        pytest.fail("Legacy and custom prompts must not depend on the guide files")
    monkeypatch.setattr(prompts, "load_skill", unnecessary_guide_load)
    assert prompts._compose_user_text("custom", " sys ", " idea ", " linked ") == (
        "sys", "idea\n\nlinked")
    for preset, system in prompts._SYSTEM_PROMPT_PRESETS.items():
        if preset != "custom":
            assert prompts._compose_user_text(preset, "ignore", "idea", "") == (system, "idea")
    assert prompts._compose_user_text("missing", "ignore", "idea", "linked") == ("", "idea\n\nlinked")


@pytest.mark.parametrize("raw,expected", [
    ("masterpiece, best_quality, cat_ears, tank_top", "masterpiece, best quality, cat ears, tank top"),
    ("sousou no_frieren, 1girl", "sousou no frieren, 1girl"),
    (r"crane_\(machine\)", r"crane \(machine\)"),
    ("(cat_ears:1.2)", "(cat ears:1.2)"),
    ("o_o, x_x, ^_^, >_<", "o_o, x_x, ^_^, >_<"),
    ("score_9, score_8_up, 1girl", "score_9, score_8_up, 1girl"),
    ("A woman stands, smiling.", "A woman stands, smiling."),
])
def test_space_underscores_matches_promptforge(raw, expected):
    assert prompts.space_underscores(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("masterpiece, 1girl, solo, makima (chainsaw man), chainsaw man, rain",
     r"masterpiece, 1girl, solo, makima \(chainsaw man\), chainsaw man, rain"),
    ("makima_(chainsaw_man)", r"makima_\(chainsaw_man\)"),
    ("(chibi:2), 1girl", "(chibi:2), 1girl"),
    ("(glitch:1.5), 1girl", "(glitch:1.5), 1girl"),
    ("( chibi : 2 ), 1girl", "( chibi : 2 ), 1girl"),
    ("(a:2), (b:3)", "(a:2), (b:3)"),
    ("2b (nier:automata), nier:automata", r"2b \(nier:automata\), nier:automata"),
    ("(re:zero)", r"\(re:zero\)"),
    (r"makima \(chainsaw man\)", r"makima \(chainsaw man\)"),
    ("masterpiece, best quality, 1girl, solo", "masterpiece, best quality, 1girl, solo"),
    ("", ""),
])
def test_escape_literal_parens_matches_promptforge(raw, expected):
    assert prompts.escape_literal_parens(raw) == expected
    assert prompts.escape_literal_parens(expected) == expected


def test_tag_presets_escape_a_bare_series():
    raw = "===SEGMENT: Positive prompt===\nwatercolor_(medium), (rain:1.2)"
    assert prompts.prompt_response("promptforge_illustrious", raw) == r"watercolor \(medium\), (rain:1.2)"
    assert prompts.prompt_response("promptforge_krea2", raw.replace("Positive prompt", "Enhanced prompt")) == "watercolor_(medium), (rain:1.2)"


def test_space_style_presets_respell_and_prose_presets_do_not():
    raw = "===SEGMENT: {}===\nbest_quality, plate_armor, score_7"
    for preset in ("promptforge_illustrious", "promptforge_anima"):
        out = prompts.prompt_response(preset, raw.format(prompts.preset_spec(preset)["output"]))
        assert out == "best quality, plate armor, score_7"
    out = prompts.prompt_response("promptforge_krea2", raw.format("Enhanced prompt"))
    assert out == "best_quality, plate_armor, score_7"


def test_prompt_projection_drops_a_copied_code_fence():
    raw = "===SEGMENT: Enhanced prompt===\n```\nA brass workshop.\n```"
    assert prompts.prompt_response("promptforge_krea2", raw) == "A brass workshop."
