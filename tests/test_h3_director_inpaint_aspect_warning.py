"""Image Inpaint must warn when the keyframe aspect differs from the canvas."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nodes  # Resolve the local package before adding ComfyUI's nodes.py.
sys.path.append(str(ROOT.parent / "ComfyUI"))
from nodes import nodes_minimax_h3_director as director


def _build(monkeypatch, keyframe, width, height, input_scaling):
    monkeypatch.setattr(director, "scale_input_media", lambda image, *args: image)
    monkeypatch.setattr(director, "load_image", lambda path, directory: keyframe)
    logs = []
    monkeypatch.setattr(director, "log_dasiwa", lambda component, message: logs.append(message))
    timeline = json.dumps({"version": 1, "items": [
        {"type": "image", "slot": 0, "order": 0, "value": "unused"}]})
    director.MiniMaxH3Director().build_guide(
        "Image Inpaint", "", width, height, 5, "match", timeline,
        builder_state=json.dumps({"resolution": {"input_scaling": input_scaling}}),
        fl2va_model=object())
    return [m for m in logs if "stretches" in m]


def test_mismatched_auto_keyframe_warns(monkeypatch):
    # 16:9 keyframe into a 1:1 canvas: native H3 plain-stretches it.
    warnings = _build(monkeypatch, torch.zeros(1, 576, 1024, 3), 768, 768, "Auto")
    assert len(warnings) == 1 and "Fill and crop" in warnings[0]


def test_matched_or_pre_cropped_keyframes_stay_silent(monkeypatch):
    assert _build(monkeypatch, torch.zeros(1, 768, 768, 3), 768, 768, "Auto") == []
    # Aspect matches the canvas (16:16 grid rounding included) via a pre-scaled keyframe.
    assert _build(monkeypatch, torch.zeros(1, 432, 768, 3), 768, 432, "Fill and crop") == []
