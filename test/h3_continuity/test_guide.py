import copy
import math
import torch
import pytest
from comfy.nested_tensor import NestedTensor
from dasiwa_nodes_test.nodes_minimax_h3_director_guide import MiniMaxH3DirectorGuide
from dasiwa_nodes_test.h3_continuity import core
from test_core import latent, context, ready


def settings(**kw):
    return core.parse_settings({"session": "test", "capture": True, **kw})


def test_guide_appends_context_without_changing_old_slots(monkeypatch):
    seen = []
    positive, target = object(), object()
    monkeypatch.setattr(MiniMaxH3DirectorGuide, "_apply_native", lambda self, *a: (seen.append(a), (positive, target))[1], raising=False)
    node = MiniMaxH3DirectorGuide()
    result = node.apply("clip", "vae", {"mode": "T2VA"}, "audio")
    assert result == (positive, target, {"disabled": True})
    assert len(seen) == 1
    assert MiniMaxH3DirectorGuide.RETURN_TYPES[:2] == ("CONDITIONING", "LATENT")
    assert MiniMaxH3DirectorGuide.RETURN_TYPES[2] == "DF_H3_CONTINUITY_CONTEXT"


def test_capture_new_creates_unique_run_and_preserves_new_prompt(monkeypatch):
    monkeypatch.setattr(MiniMaxH3DirectorGuide, "_apply_native", lambda self, *a: ("cond", "latent"), raising=False)
    guide = {"mode": "T2VA", "resolved_prompt": "New prompt", "frame_rate": 24, "continuity": settings()}
    a = MiniMaxH3DirectorGuide().apply(None, None, guide)
    b = MiniMaxH3DirectorGuide().apply(None, None, guide)
    assert a[:2] == ("cond", "latent")
    assert a[2]["run_id"] != b[2]["run_id"]
    assert a[2]["resolved_prompt"] == guide["resolved_prompt"] == "New prompt"
    assert a[2]["source_id"] == ""


def test_continue_uses_pinned_parent_and_active_native_conditioning(monkeypatch, tmp_path):
    store = core.ClipStore(tmp_path)
    ticket = ready(store)
    original_store = core.ClipStore
    monkeypatch.setattr(core, "ClipStore", lambda: store)
    import dasiwa_nodes_test.nodes_minimax_h3_director_guide as guide_module
    monkeypatch.setattr(guide_module, "ClipStore", lambda: store, raising=False)
    seen = []
    def native(self, clip, vae, guide, audio_vae=None):
        seen.append(guide)
        return [[torch.zeros(1, 2, 3), {"minimax_refs": [1]}]], object()
    monkeypatch.setattr(MiniMaxH3DirectorGuide, "_apply_native", native, raising=False)
    guide = {"mode": "REF2VA", "width": 48, "height": 32, "frame_rate": 24,
             "resolved_prompt": "New prompt", "continuity": settings(operation="continue", source_id=ticket["clip_id"], continuation_prompt="Continue")}
    old = copy.deepcopy(guide)
    positive, target, ctx = MiniMaxH3DirectorGuide().apply(None, None, guide)
    assert len(seen) == 1 and seen[0]["length"] == 141
    assert "Continue" in seen[0]["resolved_prompt"]
    assert "minimax_keyframes" in positive[0][1]
    assert target["samples"].tensors[0].count_nonzero() == 0
    assert ctx["source_id"] == ticket["clip_id"] and ctx["layout"]["overlap_video_tokens"] == 7
    assert guide == old


def test_invalid_rate_fails_before_native(monkeypatch):
    monkeypatch.setattr(MiniMaxH3DirectorGuide, "_apply_native", lambda *a: pytest.fail("native called"), raising=False)
    with pytest.raises(ValueError, match="24 fps"):
        MiniMaxH3DirectorGuide().apply(None, None, {"mode": "T2VA", "frame_rate": 30, "continuity": settings()})


def test_capture_requeues_but_uncaptured_refmod_fingerprint_is_preserved(monkeypatch):
    assert math.isnan(MiniMaxH3DirectorGuide.IS_CHANGED(None, None, {"mode": "T2VA", "continuity": settings()}))
    import dasiwa_nodes_test.nodes_minimax_h3_director_guide as module
    monkeypatch.setattr(module, "refmod_fingerprint", lambda name: (3, 4))
    assert MiniMaxH3DirectorGuide.IS_CHANGED(None, None, {"minimax_ref_items": [{"name": "ref"}]}) == (("ref", (3, 4)),)
