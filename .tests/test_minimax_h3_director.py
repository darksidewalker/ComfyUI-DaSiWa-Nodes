import json
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from nodes.helper_minimax_h3_director import (
    audio_duration, load_audio, load_embedded_video_audio, load_video,
    normalize_guide, scale_canvas_to_short_edge, scale_input_media,
)
from nodes.helper_minimax_h3_prompt_builder import (
    PROMPT_MODES, build_base_prompt, build_prompt, build_ref_prompt, default_builder_state,
)
from nodes import nodes_minimax_h3_director as director
from nodes import nodes_minimax_h3_director_guide as director_guide


def test_base_prompt_builder_includes_fl2va_alignment_and_schema():
    state = default_builder_state("FL2VA")
    state.update({"duration": 5, "p2_shot": 2, "imd": "[Shot 1] A lantern rises.", "soundscape": "wind", "music": "N/A"})

    prompt = build_prompt(state)

    assert "Picture 1 (from Shot 1) aligns with the 0.00-second mark" in prompt
    assert "Picture 2 (from Shot 2) aligns with the 5.17-second mark" in prompt
    assert "integrated_multimodal_description: [Shot 1] A lantern rises." in prompt
    assert "overall_soundscape: wind" in prompt


def test_minimax_auto_canvas_sets_the_short_edge_to_768_on_a_16_pixel_grid():
    assert scale_canvas_to_short_edge(1920, 1080) == (1360, 768)
    assert scale_canvas_to_short_edge(1080, 1920) == (768, 1360)
    assert scale_canvas_to_short_edge(1024, 1024) == (768, 768)


def test_director_input_scaling_reuses_torch_resize_without_upscaling_auto_inputs():
    torch = __import__("torch")
    image = torch.zeros((1, 3000, 6000, 3))
    small_image = torch.zeros((1, 100, 200, 3))

    auto = scale_input_media(image, "Auto", 1024, 768)
    small_auto = scale_input_media(small_image, "Auto", 1024, 768)
    target = scale_input_media(image, "Target", 1024, 768)
    off = scale_input_media(image, "Off", 1024, 768)

    assert auto.shape == (1, 2048, 4096, 3)
    assert small_auto is small_image
    assert target.shape == (1, 768, 1024, 3)
    assert off is image


def test_ref_prompt_builder_emits_all_required_sections():
    state = default_builder_state("REF2VA")
    state["ref"].update({
        "subject_defs": [{"text": "Picture 1: a red fox"}],
        "summary_text": "Animate the fox.",
        "retention": [{"label": "Picture 1", "context": "fox", "marker": "fully_preserved", "note": "keep fur"}],
        "style_line": "Cinematic.", "detail": "[Shot 1] The fox turns.",
        "soundscape": "forest", "music": "N/A",
    })

    prompt = build_prompt(state)

    for section in ("subject_definitions:", "summary:", "retention_analysis:", "detailed_description:", "overall_soundscape:", "non_diegetic_music:"):
        assert section in prompt
    assert "Picture 1 (fox): fully_preserved - keep fur" in prompt


def test_director_builds_a_ref2va_guide_from_the_default_builder_state():
    """REF2VA defaults carry no top-level imd — prompt_payload must cope."""
    builder = default_builder_state("REF2VA")

    guide = director.MiniMaxH3Director().build_guide(
        "REF2VA", "", 1344, 768, 5, "match", json.dumps({"items": []}),
        json.dumps(builder))[0]

    assert guide["prompt_payload"]["is_ref_mode"] is True
    assert guide["prompt_payload"]["imd"] == ""


def test_director_emits_v2_consolidated_prompt_for_i2va_builder_state():
    state = {"items": [{"type": "image", "value": "opening.png", "slot": 0}]}
    builder = default_builder_state("I2VA")
    builder.update({"imd": "A bright room.", "soundscape": "birds", "music": "N/A"})

    guide, _, resolved, _, _, _, fl2va_requested, inpaint_requested, _ = director.MiniMaxH3Director().build_guide(
        "I2VA", "legacy", 1344, 768, 5, "match", json.dumps(state), json.dumps(builder)
    )

    assert guide["version"] == 2
    assert guide["first_frame"] == "opening.png"
    assert guide["last_frame"] is None
    assert guide["prompt_payload"]["full_prompt"] == resolved
    assert "integrated_multimodal_description: A bright room." in resolved
    assert fl2va_requested and not inpaint_requested


def test_director_blank_external_prompt_falls_back_to_builder():
    """Structured mode: a blank/whitespace external prompt must not bypass the builder."""
    builder = default_builder_state("T2VA")
    builder.update({"imd": "[Shot 1] A calm lake at dawn.", "soundscape": "light wind", "music": "N/A"})

    def resolved_for(external_prompt_overwrite):
        return director.MiniMaxH3Director().build_guide(
            "T2VA", "", 1344, 768, 5, "match", json.dumps({"items": []}),
            json.dumps(builder), None, None, external_prompt_overwrite=external_prompt_overwrite
        )[2]

    # Blank string and whitespace-only must both resolve via the builder (non-empty).
    for blank in ("", "   "):
        fallback = resolved_for(blank)
        assert fallback != blank
        assert "integrated_multimodal_description: [Shot 1] A calm lake at dawn." in fallback

    # A real external prompt still overrides the builder.
    assert resolved_for("MY OWN PROMPT") == "MY OWN PROMPT"


def test_director_external_dimension_overwrites_replace_canvas_before_guide_construction():
    guide, _, resolved, output_width, output_height, *_ = director.MiniMaxH3Director().build_guide(
        "I2VA", "internal prompt", 1344, 768, 5, "match",
        json.dumps({"items": [{"type": "image", "value": "opening.png", "slot": 0}]}),
        external_width_overwrite=1023,
        external_height_overwrite=577,
        external_prompt_overwrite="external prompt",
    )

    assert (output_width, output_height) == (1023, 577)
    assert (guide["width"], guide["height"]) == (1023, 577)
    assert resolved == "external prompt"


def test_director_external_dimension_overwrites_require_a_complete_pair():
    with pytest.raises(ValueError, match="both external width overwrite and external height overwrite"):
        director.MiniMaxH3Director().build_guide(
            "T2VA", "", 1344, 768, 5, "match", json.dumps({"items": []}),
            external_width_overwrite=1024,
        )


def test_prompt_modes_constant_lists_both_styles():
    assert set(PROMPT_MODES) == {"simple", "structured"}


def test_base_mode_defaults_to_structured_when_prompt_mode_absent():
    state = default_builder_state("FL2VA")
    state.update({"imd": "[Shot 1] A lantern rises.", "soundscape": "wind", "music": "N/A"})
    prompt = build_prompt(state)
    # structured = the official sectioned layout (with the FL2VA alignment head).
    assert "How the reference pictures align" in prompt
    assert "integrated_multimodal_description: [Shot 1] A lantern rises." in prompt
    # simple would not carry the alignment head.
    assert prompt == build_base_prompt(state)


def test_base_mode_simple_flattens_fields_without_alignment_head():
    state = default_builder_state("FL2VA")
    state.update({"imd": "[Shot 1] A lantern rises.", "soundscape": "wind", "music": "N/A", "prompt_mode": "simple"})
    prompt = build_prompt(state)
    assert "How the reference pictures align" not in prompt
    assert "integrated_multimodal_description: [Shot 1] A lantern rises." in prompt
    assert "overall_soundscape: wind" in prompt
    assert "non_diegetic_music: N/A" in prompt
    # simple keeps one header per line, no blank-line sectioning.
    assert "\n\n" not in prompt


def test_simple_prompt_mode_uses_the_single_saved_prompt_field():
    state = default_builder_state("REF2VA")
    state.update({"prompt_mode": "simple", "simple_prompt": "A red fox walks through a quiet forest."})
    state["ref"].update({"subject_definitions": "This must not be rendered."})

    assert build_prompt(state) == "A red fox walks through a quiet forest."


def test_prompt_mode_simple_is_lossless_vs_structured_fields():
    """Every non-empty field present in structured must appear in simple."""
    state = default_builder_state("FL2VA")
    state.update({"imd": "[Shot 1] A lantern rises.", "soundscape": "wind and birds", "music": "soft strings"})
    structured = build_prompt(state)
    simple = build_prompt({**state, "prompt_mode": "simple"})
    for field in ("[Shot 1] A lantern rises.", "wind and birds", "soft strings"):
        assert field in structured
        assert field in simple


def test_ref2va_mode_simple_flattens_sectioned_layout():
    state = default_builder_state("REF2VA")
    state["ref"].update({
        "subject_definitions": "<Subject 1> is a red fox.",
        "summary": "[reference generation] Animate the fox.",
        "retention_analysis": "<Subject 1>: fully_preserved - keep fur",
        "detailed_description": "Cinematic. [Shot 1] The fox turns.",
        "soundscape": "forest", "music": "N/A",
    })
    structured = build_prompt(state)
    simple = build_prompt({**state, "prompt_mode": "simple"})
    # structured keeps the sectioned "header:\nvalue" layout.
    assert "subject_definitions:\n<Subject 1> is a red fox." in structured
    # simple flattens to "header: value" on one line each.
    assert "subject_definitions: <Subject 1> is a red fox." in simple
    assert "summary: [reference generation] Animate the fox." in simple
    assert "retention_analysis: <Subject 1>: fully_preserved - keep fur" in simple
    # and simple drops the multi-line sectioning.
    assert "subject_definitions:\n" not in simple


def test_prompt_mode_invalid_or_non_string_falls_back_to_structured():
    state = default_builder_state("T2VA")
    state.update({"imd": "A room.", "soundscape": "quiet", "music": "N/A"})
    assert build_prompt({**state, "prompt_mode": "bogus"}) == build_prompt(state)
    assert build_prompt({**state, "prompt_mode": 5}) == build_prompt(state)
    assert build_prompt({**state, "prompt_mode": None}) == build_prompt(state)
    # case-insensitive + whitespace-tolerant.
    assert build_prompt({**state, "prompt_mode": "  Simple  "}) == build_prompt({**state, "prompt_mode": "simple"})


def test_director_end_to_end_honors_prompt_mode_from_builder_state():
    base = default_builder_state("T2VA")
    base.update({"imd": "[Shot 1] A calm lake at dawn.", "soundscape": "light wind", "music": "N/A"})

    def resolved_with(prompt_mode):
        builder = dict(base)
        if prompt_mode is not None:
            builder["prompt_mode"] = prompt_mode
        return director.MiniMaxH3Director().build_guide(
            "T2VA", "", 1344, 768, 5, "match", json.dumps({"items": []}),
            json.dumps(builder), None, None,
        )[2]

    structured = resolved_with("structured")
    simple = resolved_with("simple")
    # default (absent) == structured, backward compatible.
    assert resolved_with(None) == structured
    # structured keeps the sectioned body, simple does not carry double-newlines.
    assert "integrated_multimodal_description: [Shot 1] A calm lake at dawn." in structured
    assert "\n\n" in structured
    assert "integrated_multimodal_description: [Shot 1] A calm lake at dawn." in simple
    assert "\n\n" not in simple


def test_guider_routes_l2va_to_image_to_video_with_only_last_frame(monkeypatch):
    calls = []

    class NativeImageToVideo:
        @staticmethod
        def execute(*args):
            calls.append(args)
            return ["conditioning"], {"samples": np.zeros((1, 2, 3))}

    monkeypatch.setattr(director_guide, "_native_node", lambda _name: NativeImageToVideo)
    guide = {"version": 2, "mode": "L2VA", "width": 1344, "height": 768, "length": 124, "last_frame": "closing"}

    director_guide.MiniMaxH3DirectorGuide().apply(object(), object(), guide)

    assert calls[0][-2:] == (None, "closing")


def test_guider_routes_ref2va_by_name_against_the_current_native_signature(monkeypatch):
    """REF2VA uses named arguments, preserving Core's prompt-before-VAE order."""
    calls = []

    class NativeReferenceToVideo:
        @staticmethod
        def execute(clip, prompt, width, height, length, ref_image_size="match", vae=None,
                    audio_vae=None, ref_images=None, ref_videos=None,
                    ref_video_audios=None, ref_audios=None):
            calls.append({
                "clip": clip, "prompt": prompt, "width": width, "height": height,
                "length": length, "ref_image_size": ref_image_size, "vae": vae,
                "audio_vae": audio_vae, "ref_images": ref_images, "ref_videos": ref_videos,
                "ref_video_audios": ref_video_audios, "ref_audios": ref_audios,
            })
            return ["conditioning"], {"samples": np.zeros((1, 2, 3))}

    monkeypatch.setattr(director_guide, "_native_node", lambda _name: NativeReferenceToVideo)
    guide = {
        "version": 2, "mode": "REF2VA", "width": 1344, "height": 768, "length": 124,
        "resolved_prompt": "A red fox walks through a quiet forest.",
        "ref_image_size": "match",
        "ref_images": {"ref_image_1": "fox.png"}, "ref_videos": {},
        "ref_video_audios": {}, "ref_audios": {},
    }
    clip, vae, audio_vae = object(), object(), object()

    positive, latent, context = director_guide.MiniMaxH3DirectorGuide().apply(clip, vae, guide, audio_vae)

    assert context == {"disabled": True}
    assert positive == ["conditioning"]
    assert latent["samples"].shape == (1, 2, 3)
    assert calls == [{
        "clip": clip, "prompt": "A red fox walks through a quiet forest.",
        "width": 1344, "height": 768, "length": 124, "ref_image_size": "match",
        "vae": vae, "audio_vae": audio_vae, "ref_images": {"ref_image_1": "fox.png"},
        "ref_videos": {}, "ref_video_audios": {}, "ref_audios": {},
    }]


def test_director_ui_exposes_the_derived_frame_slots():
    source = Path("js/minimax_h3_director.js").read_text()

    assert 'mode() === "I2VA" ? [0]' in source
    assert 'mode() === "FL2VA" ? [0, 1]' in source
    assert 'mode() === "L2VA" ? [0, 1]' in source
    assert 'mode() === "T2VA" ? []' in source


def test_director_preview_overlay_is_clickable_and_closes_with_escape():
    source = Path("js/minimax_h3_director.js").read_text()

    assert "function openPreview(item)" in source
    assert "function closePreview(discard = false)" in source
    assert 'previewCloseFn = event => { if (event.key === "Escape") closePreview(); };' in source
    assert "openPreview(item);" in source


def test_director_clip_clicks_do_not_commit_a_reorder_without_pointer_movement():
    source = Path("js/minimax_h3_director.js").read_text()

    assert "let dragged = false;" in source
    assert "if (!dragged) return;" in source


def test_director_text_fields_preserve_the_comfyui_run_shortcut():
    source = Path("js/minimax_h3_director.js").read_text()

    assert 'if ((event.ctrlKey || event.metaKey) && event.key === "Enter") return;' in source


def test_director_dom_ui_forwards_wheel_events_to_the_comfyui_canvas():
    source = Path("js/minimax_h3_director.js").read_text()

    assert 'timeline.addEventListener("wheel", event =>' in source
    assert "const canvas = app.canvas?.canvas;" in source
    assert 'canvas.dispatchEvent(new WheelEvent("wheel", {' in source


def test_director_paste_replaces_selected_media_without_changing_its_slot():
    source = Path("js/minimax_h3_director.js").read_text()

    assert "const replacementFile = files.find(file =>" in source
    assert "await replaceSelectedFile(replacementFile, selected);" in source
    assert "async function replaceSelectedFile(file, selected)" in source
    assert "laneForItem(selected) !== lane" in source
    assert "s.items[index] = item" in source


def test_director_recognizes_wav_aliases_and_falls_back_to_riff_metadata_for_editing():
    source = Path("js/minimax_h3_director.js").read_text()

    assert '"wav", "wave"' in source
    assert 'const mimeType = String(file.type || "").toLowerCase();' in source
    assert "function wavDurationFromBuffer(buffer)" in source
    assert "async function probeWavDuration(value)" in source
    assert "data.getUint32(0, false) !== 0x52494646" in source
    assert "return duration ?? (type === \"audio\" ? await probeWavDuration(value) : null);" in source


def test_director_preview_crop_range_can_be_dragged_without_resizing():
    source = Path("js/minimax_h3_director.js").read_text()

    assert 'dragging = "range"' in source
    assert "cropDragOffset = pct - sPct" in source
    assert "MARKER_GRAB_RADIUS_PX" in source
    assert "const width = parseFloat(rangeTe.value) - parseFloat(rangeTs.value);" in source
    assert "rangeTe.value = start + width;" in source


def test_director_preview_can_play_only_the_current_crop_range():
    source = Path("js/minimax_h3_director.js").read_text()

    assert 'playCropBtn.textContent = "▶ Play crop"' in source
    assert "media.currentTime = start;" in source
    assert "if (cropPlayback && media.currentTime >= Number(teInput.value))" in source
    assert "media.currentTime = Number(teInput.value);" in source
    assert "media.pause();\n        media.currentTime = start;" not in source


def test_director_uses_the_native_h3_frame_grid_for_guide_and_output_length():
    guide, output_length, *_ = director.MiniMaxH3Director().build_guide(
        "FL2VA", "overall_soundscape: quiet room tone", 1344, 768, 5, "match", "{}"
    )

    assert guide["length"] == 124
    assert output_length == 124


def test_director_exposes_frame_rate_output_and_validates_range():
    node = director.MiniMaxH3Director()

    # Default frame_rate is 24 and is emitted as the final output.
    *_, default_fps = node.build_guide("FL2VA", "", 1344, 768, 5, "match", "{}")
    assert default_fps == 24.0
    assert isinstance(default_fps, float)

    # A selected frame_rate passes straight through to the output.
    *_, chosen_fps = node.build_guide("FL2VA", "", 1344, 768, 5, "match", "{}", frame_rate=60)
    assert chosen_fps == 60.0

    # Fractional rates pass through, while the declared bounds are enforced.
    *_, fractional = node.build_guide("FL2VA", "", 1344, 768, 5, "match", "{}", frame_rate=23.976)
    assert fractional == 23.976
    *_, low = node.build_guide("FL2VA", "", 1344, 768, 5, "match", "{}", frame_rate=0.1)
    assert low == 0.1
    *_, high = node.build_guide("FL2VA", "", 1344, 768, 5, "match", "{}", frame_rate=240)
    assert high == 240.0
    for out_of_range in (0.09, 240.01):
        with pytest.raises(ValueError, match="frame_rate must be between 0.1 and 240"):
            node.build_guide("FL2VA", "", 1344, 768, 5, "match", "{}", frame_rate=out_of_range)


def test_director_input_types_keep_legacy_widget_order_and_put_frame_rate_last():
    required = director.MiniMaxH3Director.INPUT_TYPES()["required"]
    assert required["frame_rate"] == ("FLOAT", {"default": 24.0, "min": 0.1, "max": 240.0, "step": 0.01})
    keys = list(required)
    # The eight widgets that predate frame_rate must keep their original
    # positional order so that older saved widgets_values arrays still map to
    # the right slots; frame_rate is appended last so it never shifts them.
    assert keys[:8] == ["mode", "prompt", "width", "height", "duration",
                       "ref_image_size", "timeline_data", "builder_state"]
    assert keys.index("frame_rate") == len(keys) - 1


def test_director_keeps_only_external_prompt_overwrite_as_its_prompt_input():
    optional = director.MiniMaxH3Director.INPUT_TYPES()["optional"]
    assert "external_prompt_overwrite" in optional
    assert "external_prompt" not in optional


def test_fl2va_slot_two_without_slot_one_is_the_closing_frame():
    closing_frame = "closing.png"
    state = {"items": [{
        "type": "image", "value": closing_frame, "slot": 1, "order": 0,
    }]}

    guide, *_ = director.MiniMaxH3Director().build_guide(
        "FL2VA", "", 1344, 768, 5, "match", json.dumps(state)
    )

    assert guide["first_frame"] is None
    assert guide["last_frame"] == closing_frame


def test_fl2va_slots_map_opening_and_closing_independent_of_item_order():
    opening_frame = "opening.png"
    closing_frame = "closing.png"
    state = {"items": [
        {"type": "image", "value": closing_frame, "slot": 1, "order": 0},
        {"type": "image", "value": opening_frame, "slot": 0, "order": 1},
    ]}

    guide, *_ = director.MiniMaxH3Director().build_guide(
        "FL2VA", "", 1344, 768, 5, "match", json.dumps(state)
    )

    assert guide["first_frame"] == opening_frame
    assert guide["last_frame"] == closing_frame


def test_l2va_prefers_the_new_slot_one_but_preserves_legacy_slot_zero_workflows():
    preferred_closing_frame = "slot-one-closing.png"
    legacy_closing_frame = "slot-zero-closing.png"

    guide, *_ = director.MiniMaxH3Director().build_guide(
        "L2VA", "", 1344, 768, 5, "match", json.dumps({"items": [
            {"type": "image", "value": legacy_closing_frame, "slot": 0},
            {"type": "image", "value": preferred_closing_frame, "slot": 1},
        ]})
    )
    assert guide["last_frame"] == preferred_closing_frame

    legacy_guide, *_ = director.MiniMaxH3Director().build_guide(
        "L2VA", "", 1344, 768, 5, "match", json.dumps({"items": [
            {"type": "image", "value": legacy_closing_frame, "slot": 0},
        ]})
    )
    assert legacy_guide["last_frame"] == legacy_closing_frame


def test_ref2va_image_references_follow_their_displayed_slots_after_reordering():
    state = {"items": [
        {"type": "image", "value": "picture-1.png", "slot": 0, "order": 0},
        {"type": "image", "value": "picture-2.png", "slot": 2, "order": 1},
        {"type": "image", "value": "picture-3.png", "slot": 1, "order": 2},
    ]}

    guide, *_ = director.MiniMaxH3Director().build_guide(
        "REF2VA", "", 1344, 768, 5, "match", json.dumps(state)
    )

    assert guide["ref_images"] == {
        "ref_image_1": "picture-1.png",
        "ref_image_2": "picture-3.png",
        "ref_image_3": "picture-2.png",
    }


def test_director_logs_selected_mode_and_model(capsys):
    model = object()

    director.MiniMaxH3Director().build_guide(
        "FL2VA", "", 1344, 768, 5, "match", "{}", fl2va_model=model
    )

    log = capsys.readouterr().out
    assert "mode=FL2VA" in log
    assert "requested_model=fl2va_model" in log
    assert "passed_model=builtins.object" in log
    assert "frames=124" in log


def test_director_guide_logs_native_upstream_and_passed_outputs(monkeypatch, capsys):
    class NativeImageToVideo:
        @staticmethod
        def execute(*_args):
            return ["conditioning"], {"samples": np.zeros((1, 2, 3))}

    monkeypatch.setattr(director_guide, "_native_node", lambda _name: NativeImageToVideo)
    guide = {"mode": "FL2VA", "width": 1344, "height": 768, "length": 124}

    director_guide.MiniMaxH3DirectorGuide().apply(object(), object(), guide)

    log = capsys.readouterr().out
    assert "upstream=MiniMaxH3ImageToVideo" in log
    assert "passed forward from MiniMaxH3ImageToVideo" in log
    assert "latent={samples:(1, 2, 3)}" in log


def test_load_audio_applies_timeline_crop(tmp_path):
    path = tmp_path / "reference.wav"
    samples = np.zeros(20 * 8_000, dtype=np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8_000)
        output.writeframes(samples.tobytes())

    audio = load_audio(path.name, str(tmp_path), trim_start=2, trim_end=17)

    assert audio_duration(audio) == 15


def _write_packed_stereo_wav(path, seconds, sample_rate=8_000, frames=None):
    """Write a PCM s16 stereo file — a packed (interleaved) PyAV sample format."""
    if frames is None:
        frames = np.zeros((seconds * sample_rate, 2), dtype=np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(frames.tobytes())


def test_load_audio_deinterleaves_packed_stereo(tmp_path):
    path = tmp_path / "stereo.wav"
    _write_packed_stereo_wav(path, seconds=10)

    audio = load_audio(path.name, str(tmp_path))

    assert audio["waveform"].shape[1] == 2
    assert audio_duration(audio) == 10


def test_load_audio_crops_packed_stereo_on_real_seconds(tmp_path):
    path = tmp_path / "stereo.wav"
    _write_packed_stereo_wav(path, seconds=20)

    audio = load_audio(path.name, str(tmp_path), trim_start=2, trim_end=17)

    assert audio["waveform"].shape[1] == 2
    assert audio_duration(audio) == 15


def test_load_embedded_video_audio_deinterleaves_packed_stereo(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "stereo_packed.mkv"
    sample_rate = 8_000
    with av.open(str(path), "w") as output:
        stream = output.add_stream("pcm_s16le", rate=sample_rate)
        stream.layout = "stereo"
        for _ in range(10 * sample_rate // 1024):
            frame = av.AudioFrame.from_ndarray(np.zeros((1, 1024 * 2), dtype=np.int16), format="s16", layout="stereo")
            frame.sample_rate = sample_rate
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)

    audio = load_embedded_video_audio(path.name, str(tmp_path))

    assert audio["waveform"].shape[1] == 2
    assert audio_duration(audio) == pytest.approx(10, abs=0.1)


def test_load_audio_keeps_full_scale_pcm_inside_the_unit_range(tmp_path):
    path = tmp_path / "fullscale.wav"
    sample_rate = 8_000
    frames = np.tile(np.array([[-32768, 32767]], dtype=np.int16), (3 * sample_rate, 1))
    _write_packed_stereo_wav(path, seconds=3, sample_rate=sample_rate, frames=frames)

    audio = load_audio(path.name, str(tmp_path))

    assert float(audio["waveform"].abs().max()) <= 1.0
    assert float(audio["waveform"].min()) == pytest.approx(-1.0, abs=1e-4)


def test_load_audio_centres_unsigned_pcm_on_silence(tmp_path):
    """8-bit PCM is unsigned: its silence sits at 128, not at 0."""
    path = tmp_path / "unsigned.wav"
    sample_rate = 8_000
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(1)
        output.setframerate(sample_rate)
        output.writeframes(np.full(3 * sample_rate, 128, dtype=np.uint8).tobytes())

    waveform = load_audio(path.name, str(tmp_path))["waveform"]

    assert float(waveform.abs().max()) == pytest.approx(0.0, abs=1e-6)


def test_load_embedded_video_audio_keeps_its_amplitude_at_full_scale(tmp_path):
    """A single -32768 sample must not trip the legacy int16 rescale on the rest."""
    av = pytest.importorskip("av")
    path = tmp_path / "fullscale.mkv"
    sample_rate = 8_000
    block = np.full((1, 1024), 16_384, dtype=np.int16)
    block[0, 0] = -32_768
    with av.open(str(path), "w") as output:
        stream = output.add_stream("pcm_s16le", rate=sample_rate)
        stream.layout = "mono"
        for _ in range(3 * sample_rate // 1024):
            frame = av.AudioFrame.from_ndarray(block, format="s16", layout="mono")
            frame.sample_rate = sample_rate
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)

    waveform = load_embedded_video_audio(path.name, str(tmp_path))["waveform"]

    assert float(waveform.abs().max()) <= 1.0
    assert float(waveform.max()) == pytest.approx(0.5, abs=1e-3)


def test_load_embedded_video_audio_applies_the_video_crop(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "reference.m4a"
    sample_rate = 8_000
    with av.open(str(path), "w") as output:
        stream = output.add_stream("aac", rate=sample_rate)
        stream.layout = "mono"
        for _ in range(14 * sample_rate // 1024):
            frame = av.AudioFrame.from_ndarray(np.zeros((1, 1024), dtype=np.int16), format="s16", layout="mono")
            frame.sample_rate = sample_rate
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)

    audio = load_embedded_video_audio(path.name, str(tmp_path), trim_start=2, trim_end=12)

    assert audio["sample_rate"] == sample_rate
    assert audio_duration(audio) == pytest.approx(10, abs=0.1)


def test_load_audio_decodes_m4a_aac_with_pyav(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "reference.m4a"
    sample_rate = 8_000
    with av.open(str(path), "w") as output:
        stream = output.add_stream("aac", rate=sample_rate)
        stream.layout = "mono"
        for _ in range(14 * sample_rate // 1024):
            frame = av.AudioFrame.from_ndarray(np.zeros((1, 1024), dtype=np.int16), format="s16", layout="mono")
            frame.sample_rate = sample_rate
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)

    audio = load_audio(path.name, str(tmp_path), trim_start=2, trim_end=12)

    assert audio["sample_rate"] == sample_rate
    assert audio_duration(audio) == pytest.approx(10, abs=0.1)


def test_attached_video_soundtrack_uses_video_crop(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(
        get_input_directory=lambda: str(tmp_path)))
    monkeypatch.setattr(director, "load_video", lambda *args, **kwargs: np.zeros((240, 1, 1, 3)))
    monkeypatch.setattr(director, "load_audio", lambda *args, **kwargs: calls.append(kwargs) or {"waveform": np.zeros((1, 1)), "sample_rate": 1})
    monkeypatch.setattr(director, "audio_duration", lambda audio: 10)

    state = {"items": [{"type": "video", "value": "reference.mp4", "audio": "reference.wav", "trim_start": 2, "trim_end": 12}]}
    director.MiniMaxH3Director().build_guide("REF2VA", "", 1344, 768, 5, "match", json.dumps(state))

    assert calls == [{"trim_start": 2.0, "trim_end": 12.0}]


def test_embedded_video_media_modes_select_requested_streams(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(
        get_input_directory=lambda: str(tmp_path)))
    monkeypatch.setattr(director, "load_video", lambda *args, **kwargs: calls.append(("video", kwargs)) or np.zeros((240, 1, 1, 3)))
    monkeypatch.setattr(director, "load_embedded_video_audio", lambda *args, **kwargs: calls.append(("audio", kwargs)) or {"waveform": np.zeros((1, 1, 80_000)), "sample_rate": 8_000})
    monkeypatch.setattr(director, "load_image", lambda *args: np.zeros((1, 1, 1, 3)))
    monkeypatch.setattr(director, "audio_duration", lambda audio: 10)

    node = director.MiniMaxH3Director()
    common = {"type": "video", "value": "reference.mp4", "trim_start": 2, "trim_end": 12}

    video_guide = node.build_guide("REF2VA", "", 1344, 768, 5, "match", json.dumps({"items": [{**common, "media_mode": "video"}]}))[0]
    assert list(video_guide["ref_videos"]) == ["ref_video_1"]
    assert video_guide["ref_video_audios"] == {}
    assert calls == [("video", {"trim_start": 2.0, "trim_end": 12.0, "target_fps": 24.0})]

    calls.clear()
    audio_guide = node.build_guide("REF2VA", "", 1344, 768, 5, "match", json.dumps({"items": [
        {"type": "image", "value": "visual.png"}, {**common, "media_mode": "audio"},
    ]}))[0]
    assert audio_guide["ref_videos"] == {}
    assert list(audio_guide["ref_audios"]) == ["ref_audio_1"]
    assert calls == [("audio", {"trim_start": 2.0, "trim_end": 12.0})]

    calls.clear()
    combined_guide = node.build_guide("REF2VA", "", 1344, 768, 5, "match", json.dumps({"items": [{**common, "media_mode": "video_audio"}]}))[0]
    assert list(combined_guide["ref_videos"]) == ["ref_video_1"]
    assert list(combined_guide["ref_video_audios"]) == ["ref_video_audio_1"]
    assert calls == [
        ("video", {"trim_start": 2.0, "trim_end": 12.0, "target_fps": 24.0}),
        ("audio", {"trim_start": 2.0, "trim_end": 12.0}),
    ]


def test_each_video_audio_pair_uses_its_matching_upstream_autogrow_key(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(
        get_input_directory=lambda: str(tmp_path)))
    monkeypatch.setattr(director, "load_video", lambda *args, **kwargs: np.zeros((120, 1, 1, 3)))
    monkeypatch.setattr(director, "load_embedded_video_audio", lambda path, *args, **kwargs: {
        "waveform": np.full((1, 1, 40_000), 1 if path == "one.mp4" else 2), "sample_rate": 8_000,
    })
    monkeypatch.setattr(director, "audio_duration", lambda audio: 5)

    guide = director.MiniMaxH3Director().build_guide("REF2VA", "", 1344, 768, 5, "match", json.dumps({"items": [
        {"type": "video", "value": "one.mp4", "media_mode": "video_audio", "trim_start": 2, "trim_end": 7},
        {"type": "video", "value": "two.mp4", "media_mode": "video_audio", "trim_start": 3, "trim_end": 8},
    ]}))[0]

    assert list(guide["ref_videos"]) == ["ref_video_1", "ref_video_2"]
    assert list(guide["ref_video_audios"]) == ["ref_video_audio_1", "ref_video_audio_2"]
    assert guide["ref_video_audios"]["ref_video_audio_1"]["waveform"][0, 0, 0] == 1
    assert guide["ref_video_audios"]["ref_video_audio_2"]["waveform"][0, 0, 0] == 2


def test_load_video_falls_back_to_container_duration_when_stream_duration_is_missing(monkeypatch, tmp_path):
    class Frame:
        time_base = 1

        def __init__(self, pts):
            self.pts = pts

        def to_rgb(self):
            return self

        def to_ndarray(self):
            return np.zeros((1, 1, 3), dtype=np.uint8)

    class Container:
        duration = 3_000_000
        streams = [types.SimpleNamespace(type="video", duration=None, time_base=1)]

        def decode(self, stream):
            return iter([Frame(0), Frame(1), Frame(2)])

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "av", types.SimpleNamespace(
        time_base=1_000_000,
        open=lambda path: Container(),
    ))
    (tmp_path / "durationless.mp4").touch()

    frames = load_video("durationless.mp4", str(tmp_path), target_fps=1)

    assert frames.shape[0] == 3


def test_v1_normalizer_accepts_image_inpaint_with_single_first_frame():
    state = normalize_guide({"mode": "Image Inpaint", "width": 1024, "height": 1024,
                             "length": 5, "first_frame": "img", "last_frame": None})
    assert state.mode == "Image Inpaint"
    assert state.first_frame == "img"
    assert state.last_frame is None
    assert state.length == 5
    assert state.ref_images == {} and state.ref_videos == {} and state.ref_audios == {}


def test_v1_normalizer_rejects_image_inpaint_with_references():
    with pytest.raises(ValueError, match="no video/audio references"):
        normalize_guide({"mode": "Image Inpaint", "first_frame": "img",
                         "ref_images": {"ref_image_1": "x"}})


def test_v1_normalizer_rejects_image_inpaint_without_first_frame():
    with pytest.raises(ValueError, match="requires one image keyframe"):
        normalize_guide({"mode": "Image Inpaint", "first_frame": None})


def test_director_builds_image_inpaint_guide():
    timeline = json.dumps({"items": [{"id": "a", "type": "image", "slot": 0, "order": 0, "value": "in.png"}],
                           "prompt_blocks": []})
    guide, length, resolved, w, h, model, fl2va_req, inpaint_req, fps = director.MiniMaxH3Director().build_guide(
        "Image Inpaint", "p", 768, 768, 5, "match", timeline, "", fl2va_model="M")
    assert guide["mode"] == "Image Inpaint"
    assert guide["length"] == 5
    assert guide["first_frame"] == "in.png"
    assert guide["last_frame"] is None
    assert length == 5
    assert fl2va_req is True and inpaint_req is True
    assert model == "M"


def test_director_rejects_image_inpaint_without_exactly_one_image():
    timeline = json.dumps({"items": [{"id": "a", "type": "video", "slot": 0, "order": 0, "value": "v.mp4"}],
                           "prompt_blocks": []})
    with pytest.raises(ValueError, match="accepts image references only"):
        director.MiniMaxH3Director().build_guide("Image Inpaint", "p", 768, 768, 5, "match", timeline, "")
    with pytest.raises(ValueError, match="requires exactly one enabled image reference"):
        director.MiniMaxH3Director().build_guide("Image Inpaint", "p", 768, 768, 5, "match",
                                                  json.dumps({"items": [], "prompt_blocks": []}), "")


def test_guider_routes_image_inpaint_to_5frame_image_to_video(monkeypatch):
    calls = []

    class NativeImageToVideo:
        @staticmethod
        def execute(*args):
            calls.append(args)
            return ["conditioning"], {"samples": np.zeros((1, 2, 3))}

    monkeypatch.setattr(director_guide, "_native_node", lambda _name: NativeImageToVideo)
    guide = {"version": 2, "mode": "Image Inpaint", "width": 768, "height": 768,
             "length": 5, "first_frame": "img", "resolved_prompt": "p"}

    positive, latent, context = director_guide.MiniMaxH3DirectorGuide().apply(object(), object(), guide)

    assert context == {"disabled": True}
    assert positive == ["conditioning"]
    assert latent["samples"].shape == (1, 2, 3)
    # native call: (clip, vae, prompt, width, height, 5, first_frame, None)
    assert calls[0][2:] == ("p", 768, 768, 5, "img", None)


def test_director_v1_ui_exposes_image_inpaint_mode():
    source = Path("js/minimax_h3_director.js").read_text()

    assert '"T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA", "Image Inpaint"' in source
    assert 'mode() === "Image Inpaint"' in source


def test_director_exposes_inpaint_requested_instead_of_ref2va_requested():
    node = director.MiniMaxH3Director()

    assert node.RETURN_NAMES == (
        "guide", "duration", "positive_prompt", "width", "height", "model",
        "fl2va_requested", "inpaint_requested", "frame_rate",
    )

    inpaint = node.build_guide("Image Inpaint", "", 1024, 1024, 5, "match",
                               json.dumps({"items": [{"type": "image", "value": "a.png", "slot": 0}]}))
    base = node.build_guide("FL2VA", "", 1024, 1024, 5, "match", json.dumps({"items": []}))
    ref = node.build_guide("REF2VA", "", 1024, 1024, 5, "match", json.dumps({"items": []}))

    assert inpaint[7] is True
    assert base[7] is False
    assert ref[7] is False
