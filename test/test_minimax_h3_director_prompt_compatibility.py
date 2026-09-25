"""Regression tests for Director prompt migration and standard prompt output."""
import json

from nodes.helper_minimax_h3_prompt_builder import (
    build_prompt,
    default_builder_state,
    migrate_legacy_prompt,
)
from nodes.nodes_minimax_h3_director import MiniMaxH3Director
from nodes import nodes_minimax_h3_director as director_module


def test_legacy_widget_prompt_is_preserved_losslessly():
    prompt = "Detailed old prompt\n<d>[English] Exact dialogue.</d>"
    builder = default_builder_state("FL2VA")

    assert migrate_legacy_prompt(builder, {}, prompt)
    assert builder["prompt_mode"] == "simple"
    assert build_prompt(builder) == prompt


def test_director_uses_legacy_prompt_when_builder_is_absent():
    prompt = "Former video generation prompt"
    guide = MiniMaxH3Director().build_guide(
        "T2VA", prompt, 1344, 768, 5, "match", "{}", "",
    )[0]

    assert guide["resolved_prompt"] == prompt
    assert guide["builder_state"]["simple_prompt"] == prompt


def test_empty_new_director_prompt_stays_empty():
    guide = MiniMaxH3Director().build_guide(
        "T2VA", "", 1344, 768, 5, "match", "{}", "",
    )[0]
    assert guide["resolved_prompt"] == ""
    assert guide["builder_state"]["simple_prompt"] == ""


def test_old_structured_ref_builder_remains_executable_without_frontend():
    old_builder = _real_ref2va_builder_state()
    guide = MiniMaxH3Director().build_guide(
        "REF2VA", "", 1344, 768, 5, "match", "{}", old_builder,
    )[0]
    assert "red sandstone lion" in guide["resolved_prompt"]
    assert "subject_definitions:" in guide["resolved_prompt"]


def test_new_single_prompt_takes_precedence_over_stale_structured_fields():
    old = json.loads(_real_ref2va_builder_state())
    old.update(prompt_mode="simple", simple_prompt="My new prompt")
    guide = MiniMaxH3Director().build_guide(
        "REF2VA", "My new prompt", 1344, 768, 5, "match", "{}", json.dumps(old),
    )[0]
    assert guide["resolved_prompt"] == "My new prompt"


def test_frontend_single_editor_and_legacy_reference_pack_migration():
    source = open("js/minimax_h3_director.js", encoding="utf-8").read()
    assert "topRow.append(modesSide, spacer, ioSide" in source
    assert "buildSimpleForm(promptPanel);" in source
    assert "Insert Prompt Structure" in source
    assert 'forgeButton.textContent = "Prompt Forge"' in source
    assert "portablePromptText(saved)" in source
    assert 'builderState.simple_prompt = joinText(builderState.simple_prompt, portablePromptText(saved))' in source


def test_frontend_keeps_standard_prompt_widget_serialized():
    source = open("js/minimax_h3_director.js", encoding="utf-8").read()

    assert "promptWidget.value = resolved" in source
    assert "promptWidget.callback?.(resolved)" in source
    assert "state.resolved_prompt = resolved" in source


def _required_order():
    """Return the widget names in the exact order ComfyUI's positional loader
    assigns them, which is the INPUT_TYPES iteration order (required first)."""
    schema = MiniMaxH3Director.INPUT_TYPES()
    names = []
    for section in ("required", "optional"):
        for key in schema.get(section, {}):
            if key in ("fl2va_model", "ref2va_model",
                       "external_width_overwrite", "external_height_overwrite",
                       "external_prompt_overwrite"):
                continue  # sockets, not widgets
            names.append(key)
    return names


def _load_old_widgets_values(values):
    """Positionally map a saved widgets_values array onto the current node's
    widget slots, exactly as the ComfyUI graph loader does."""
    order = _required_order()
    node = MiniMaxH3Director()
    positional = {key: values[i] for i, key in enumerate(order[: len(values)])}
    # Sockets / optional widgets not present in the old save keep their defaults.
    return node, positional


def _real_ref2va_builder_state():
    """A minimal v1 builder_state carrying real REF2VA content (the kind that
    lives in the embedded metadata of the old videos)."""
    return json.dumps({
        "version": 1,
        "mode": "REF2VA",
        "ref": {
            "subject_defs": [{"text": "a red sandstone lion"}],
            "summary_text": "transfer the lion's mane texture onto the character",
            "summary_types": ["reference generation"],
            "retention": [],
            "style_line": "",
            "detail": "",
        },
    })


def test_old_9_value_save_loads_without_crashing_and_preserves_prompt():
    """A save from before frame_rate was a required input has 9 widgets_values
    where position 8 is the removed external_prompt widget (''). After the
    reordering, that stale value lands on frame_rate and must not crash the
    queue; the real builder_state (position 7) must round-trip to a prompt."""
    node, kwargs = _load_old_widgets_values([
        "REF2VA", "", 1344, 768, 5, "match",
        json.dumps({"version": 1, "items": [], "prompt_blocks": []}),
        _real_ref2va_builder_state(),
        "",  # stale 9th value: was external_prompt, now lands on frame_rate
    ])

    guide, *_ , out_frame_rate = node.build_guide(
        **{**kwargs, "frame_rate": kwargs.get("frame_rate", 24.0)}
    )

    # The stale value is coerced to the default instead of raising.
    assert out_frame_rate == 24.0
    # The builder_state is reachable and produced a non-empty prompt.
    assert guide["resolved_prompt"].strip()
    assert "lion" in guide["resolved_prompt"]


def test_mixed_uploaded_media_and_refmod_tags_keep_both_reference_paths(monkeypatch):
    class Latent:
        def __mul__(self, strength):
            return ("scaled", strength)

    monkeypatch.setattr(director_module, "load_refmods", lambda name: [
        (Latent(), {"kind": "image"}), (Latent(), {"kind": "video"}),
        (Latent(), {"kind": "audio"}),
    ])
    monkeypatch.setattr(director_module, "refmod_fingerprint", lambda name: (1, 1))
    media = object()
    state = {"items": [{"type": "image", "value": media, "slot": 0},
                       {"type": "video", "value": media, "slot": 0, "duration": 2},
                       {"type": "audio", "value": media, "slot": 0, "duration": 2}],
             "refmods": [{"slot": 1, "name": "combo", "strength": 1}]}
    monkeypatch.setattr(director_module, "scale_input_media", lambda value, *args: value)
    guide = MiniMaxH3Director().build_guide(
        "REF2VA", "<RefMod 1>", 1344, 768, 5, "match", json.dumps(state, default=lambda _: "media"), "",
    )[0]
    assert list(guide["ref_images"]) == ["ref_image_1"]
    assert list(guide["ref_videos"]) == ["ref_video_1"]
    assert list(guide["ref_audios"]) == ["ref_audio_1"]
    assert [item["kind"] for item in guide["minimax_ref_items"]] == ["image", "video", "audio"]
    assert "<Picture 2> <Video 2> <Audio 2>" in guide["resolved_prompt"]


def test_out_of_range_frame_rate_still_raises():
    node = MiniMaxH3Director()
    try:
        node.build_guide("REF2VA", "", 1344, 768, 5, "match", "{}", "",
                         frame_rate=9999.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for out-of-range frame_rate")

