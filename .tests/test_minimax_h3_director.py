import json
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from nodes.helper_minimax_h3_director import audio_duration, load_audio, load_embedded_video_audio, load_video
from nodes.helper_minimax_h3_prompt_builder import build_prompt, default_builder_state
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

    guide, _, resolved, _, _, _, fl2va_requested, ref2va_requested = director.MiniMaxH3Director().build_guide(
        "I2VA", "legacy", 1344, 768, 5, "match", json.dumps(state), json.dumps(builder)
    )

    assert guide["version"] == 2
    assert guide["first_frame"] == "opening.png"
    assert guide["last_frame"] is None
    assert guide["prompt_payload"]["full_prompt"] == resolved
    assert "integrated_multimodal_description: A bright room." in resolved
    assert fl2va_requested and not ref2va_requested


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


def test_director_ui_exposes_the_derived_frame_slots():
    source = Path("js/minimax_h3_director.js").read_text()

    assert 'mode() === "I2VA" ? [0]' in source
    assert 'mode() === "FL2VA" ? [0, 1]' in source
    assert 'mode() === "L2VA" ? [1]' in source
    assert 'mode() === "T2VA" ? []' in source


def test_director_preview_overlay_is_clickable_and_closes_with_escape():
    source = Path("js/minimax_h3_director.js").read_text()

    assert "function openPreview(item)" in source
    assert "function closePreview()" in source
    assert 'previewCloseFn = event => { if (event.key === "Escape") closePreview(); };' in source
    assert "openPreview(item);" in source


def test_director_clip_clicks_do_not_commit_a_reorder_without_pointer_movement():
    source = Path("js/minimax_h3_director.js").read_text()

    assert "let dragged = false;" in source
    assert "if (!dragged) return;" in source


def test_director_uses_the_native_h3_frame_grid_for_guide_and_output_length():
    guide, output_length, *_ = director.MiniMaxH3Director().build_guide(
        "FL2VA", "overall_soundscape: quiet room tone", 1344, 768, 5, "match", "{}"
    )

    assert guide["length"] == 124
    assert output_length == 124


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
    assert calls == [("video", {"trim_start": 2.0, "trim_end": 12.0})]

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
        ("video", {"trim_start": 2.0, "trim_end": 12.0}),
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
