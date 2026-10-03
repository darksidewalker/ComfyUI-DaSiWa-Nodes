"""Checkpoint identity must not depend on cached Director Guide outputs."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nodes
sys.path.append(str(ROOT.parent / "ComfyUI"))
from comfy.nested_tensor import NestedTensor
from nodes.h3_continuity.core import ClipStore
from nodes.h3_continuity import nodes as continuity_nodes


def latent(value=0, frames=5):
    video_t = (frames - 5) // 17 * 5 + 2
    audio_t = round(frames / 24 * 40)
    return {"samples": NestedTensor((torch.full((1, 24, video_t, 1, 1), float(value)),
                                     torch.full((1, 32, 2, audio_t), float(value))))}


@pytest.mark.parametrize("operation", ["new", "continue"])
@pytest.mark.parametrize("legacy_run_id", [False, True])
def test_rerun_with_cached_context_keeps_distinct_immutable_checkpoints(
        tmp_path, monkeypatch, operation, legacy_run_id):
    store = ClipStore(tmp_path)
    monkeypatch.setattr(continuity_nodes, "ClipStore", lambda: store)
    source = {"session": "test", "run_id": "parent", "mode": "REF2VA",
              "resolved_prompt": "Original shot"}
    parent = store.stage(latent(), source)
    store.publish(parent, tmp_path / "parent.mp4")
    parent_bytes = store.member("test", "parent", "latent.safetensors").read_bytes()
    context = {"session": "test", "operation": operation, "mode": "REF2VA",
               "resolved_prompt": "Continue walking", "source_id": "parent" if operation == "continue" else "",
               "layout": {"overlap_video_tokens": 2, "overlap_audio_tokens": 8}}
    if legacy_run_id:
        context["run_id"] = "cached-guide-id"
    original = dict(context)
    node = continuity_nodes.DaSiWaH3ContinuityAppend()
    frames = 22 if operation == "continue" else 5
    sampled1, sampled2 = latent(1, frames), latent(2, frames)
    combined1, first = node.commit(sampled1, context)
    store.publish(first, tmp_path / "first.mp4")
    first_bytes = store.member("test", first["clip_id"], "latent.safetensors").read_bytes()
    combined2, second = node.commit(sampled2, context)
    store.publish(second, tmp_path / "second.mp4")
    if operation == "new":
        assert combined1 is sampled1 and combined2 is sampled2
    else:
        assert torch.count_nonzero(combined2["samples"].tensors[0][:, :, :2]) == 0
        assert torch.all(combined2["samples"].tensors[0][:, :, 2:] == 2)
    assert first["clip_id"] != second["clip_id"]
    assert context == original
    assert store.member("test", "parent", "latent.safetensors").read_bytes() == parent_bytes
    assert store.member("test", first["clip_id"], "latent.safetensors").read_bytes() == first_bytes
    for ticket in (first, second):
        saved, metadata = store.load(ticket["session"], ticket["clip_id"])
        assert metadata["parent_id"] == context["source_id"]
        assert metadata["status"] == "ready"
    saved, _ = store.load(second["session"], second["clip_id"])
    assert torch.equal(saved["samples"].tensors[0], combined2["samples"].tensors[0])
    assert torch.equal(saved["samples"].tensors[1], combined2["samples"].tensors[1])


def test_disabled_capture_does_not_stage(tmp_path, monkeypatch):
    def unexpected_store():
        raise AssertionError("Disabled capture must not access checkpoint storage")
    monkeypatch.setattr(continuity_nodes, "ClipStore", unexpected_store)
    sampled = latent()
    result, ticket = continuity_nodes.DaSiWaH3ContinuityAppend().commit(sampled, {"disabled": True})
    assert result is sampled and ticket == {"disabled": True}
