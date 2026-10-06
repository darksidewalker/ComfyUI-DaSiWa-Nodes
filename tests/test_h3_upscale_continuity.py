"""CPU continuity upscale contracts with actual staged/published checkpoints."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT))
if "--cpu" not in sys.argv:
    sys.argv.append("--cpu")

# A dedicated package name avoids collision with ComfyUI's own nodes.py.
import importlib
import types
package = types.ModuleType("dasiwa_upscale_test_nodes")
package.__path__ = [str(ROOT / "nodes")]
sys.modules[package.__name__] = package
from comfy.nested_tensor import NestedTensor
helper = importlib.import_module("dasiwa_upscale_test_nodes.h3_upscale_continuity")
ClipStore = helper.ClipStore


def streams(tokens=12, audio_tokens=65, size=1):
    return torch.arange(24 * tokens * size * size, dtype=torch.float32).reshape(1, 24, tokens, size, size), torch.arange(32 * 2 * audio_tokens, dtype=torch.float32).reshape(1, 32, 2, audio_tokens)


@pytest.fixture
def source(tmp_path, monkeypatch):
    store = ClipStore(tmp_path)
    video, audio = streams(7, 37)
    ticket = store.stage({"samples": NestedTensor((video, audio))},
                         dict(session="test", run_id="source", mode="T2VA", resolved_prompt="walk"))
    store.publish(ticket, tmp_path / "source.mp4")
    monkeypatch.setattr(helper, "ClipStore", lambda: store)
    def forbidden(*args, **kwargs):
        raise AssertionError("Source tensor loading or checkpoint writes are forbidden")
    monkeypatch.setattr(store, "load", forbidden)
    monkeypatch.setattr(store, "stage", forbidden)
    monkeypatch.setattr(store, "publish", forbidden)
    context = dict(operation="continue", session="test", source_id="source",
                   overlap_frames=5, extension_frames=17,
                   layout=dict(overlap_video_tokens=2, overlap_audio_tokens=9))
    return store, context


@pytest.mark.parametrize("context", [None, {}, {"disabled": True, "operation": "continue"}, {"operation": "new"}])
def test_inactive_is_ordinary_path(context, monkeypatch):
    monkeypatch.setattr(helper, "ClipStore", lambda: pytest.fail("inactive context accessed storage"))
    assert helper.continuity_plan(None, None, context) is None


def test_native_plan_reads_headers_only_and_preserves_checkpoint(source):
    store, context = source
    before = {p: p.read_bytes() for p in store.root.rglob("*") if p.is_file()}
    video, audio = streams()
    plan = helper.continuity_plan(video, audio, context)
    assert {k: plan[k] for k in ("refine_start_token", "source_tokens", "frame_offset", "audio_start", "source_audio_tokens")} == dict(refine_start_token=5, source_tokens=7, frame_offset=17, audio_start=28, source_audio_tokens=37)
    assert plan["source_audio_tokens"] - plan["audio_start"] == 9
    assert round(5 * 40 / 24) == 8  # Isolated-overlap rounding is not global rounding.
    assert plan["explanation"]
    assert before == {p: p.read_bytes() for p in store.root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("tokens,audio_tokens,size", [(7, 37, 1), (12, 64, 1), (11, 65, 1), (12, 65, 2), (17, 93, 1)])
def test_reject_miswired_cumulative_latent(source, tokens, audio_tokens, size):
    _, context = source
    with pytest.raises(ValueError):
        helper.continuity_plan(*streams(tokens, audio_tokens, size), context)


@pytest.mark.parametrize("field,value", [("overlap_video_tokens", 1), ("overlap_video_tokens", 3), ("overlap_video_tokens", 12), ("overlap_audio_tokens", 8), ("overlap_video_tokens", True)])
def test_reject_invalid_native_overlap(source, field, value):
    _, context = source
    context["layout"][field] = value
    with pytest.raises(ValueError):
        helper.continuity_plan(*streams(), context)


@pytest.mark.parametrize("field,value", [("extension_frames", 0), ("extension_frames", 18), ("extension_frames", "17"), ("overlap_frames", 22)])
def test_reject_inconsistent_context_timing(source, field, value):
    _, context = source
    context[field] = value
    with pytest.raises(ValueError):
        helper.continuity_plan(*streams(), context)


def test_source_inspection_keeps_native_security_and_readiness(source):
    _, context = source
    context["source_id"] = "../source"
    with pytest.raises(ValueError):
        helper.continuity_plan(*streams(), context)


def test_full_source_overlap_starts_at_native_zero_boundary(source):
    _, context = source
    context.update(overlap_frames=22)
    context["layout"].update(overlap_video_tokens=7, overlap_audio_tokens=37)
    plan = helper.continuity_plan(*streams(), context)
    assert plan["refine_start_token"] == plan["frame_offset"] == plan["audio_start"] == 0


@pytest.mark.parametrize("stream,axis", [("video", 0), ("video", 1), ("audio", 1), ("audio", 2)])
def test_native_validator_rejects_malformed_packed_shapes(source, stream, axis):
    _, context = source
    video, audio = streams()
    if stream == "video":
        video = video.repeat_interleave(2, dim=axis)
    else:
        audio = audio.repeat_interleave(2, dim=axis)
    with pytest.raises(ValueError):
        helper.continuity_plan(video, audio, context)


def test_real_store_rejects_unready_source(source):
    import json
    store, context = source
    path = store.member("test", "source", "clip.json")
    metadata = json.loads(path.read_text())
    metadata["status"] = "staged"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="not finished exporting"):
        helper.continuity_plan(*streams(), context)


def test_real_store_rejects_metadata_header_mismatch(source):
    import json
    store, context = source
    path = store.member("test", "source", "clip.json")
    metadata = json.loads(path.read_text())
    metadata["width"] *= 2
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Tensor shape differs"):
        helper.continuity_plan(*streams(), context)


def test_each_entry_preserves_refs_and_replaces_tail_with_learned_upscale(source):
    _, context = source
    video, audio = streams()
    plan = helper.continuity_plan(video, audio, context)
    upscaled, _ = streams(size=2)
    tail = dict(resolved_frame_index=0, latent=torch.full((1, 24, 2, 1, 1), -99.),
                audio_latent=torch.full((1, 32, 2, 9), -99.), label="native")
    extra = dict(resolved_frame_index=21, latent=torch.ones(1, 24, 1, 1, 1), label="last")
    refs, unknown, text = [{"ref": "retain"}], {"opaque": "retain"}, torch.ones(1)
    metadata = dict(minimax_keyframes=[extra, tail], minimax_refs=refs, unknown=unknown)
    second = dict(minimax_keyframes=[dict(resolved_frame_index=3, latent=torch.zeros(1, 24, 1, 1, 1))], second=True)
    incoming = [[text, metadata], (text, second)]
    result = helper.align_continuity_conditioning(incoming, upscaled, audio, plan)
    assert result is not incoming and result[0][0] is text
    assert result[0][1]["minimax_refs"] is refs and result[0][1]["unknown"] is unknown
    first_guides = result[0][1]["minimax_keyframes"]
    assert [g["resolved_frame_index"] for g in first_guides] == [38, 17]
    assert first_guides[0]["latent"] is extra["latent"]
    learned = first_guides[1]
    assert learned["label"] == "native"
    assert torch.equal(learned["latent"], upscaled[:, :, 5:7])
    assert torch.equal(learned["audio_latent"], audio[..., 28:37])
    assert learned["audio_latent"].shape[-1] == 9
    assert isinstance(result[1], tuple)
    assert result[1][1]["second"] is True
    assert [g["resolved_frame_index"] for g in result[1][1]["minimax_keyframes"]] == [20, 17]
    assert len(metadata["minimax_keyframes"]) == 2 and len(second["minimax_keyframes"]) == 1
    assert tail["resolved_frame_index"] == 0 and torch.all(tail["latent"] == -99)
    assert extra["resolved_frame_index"] == 21
    learned["latent"].zero_()
    learned["audio_latent"].zero_()
    assert torch.count_nonzero(upscaled[:, :, 5:7]) > 0
    assert torch.count_nonzero(audio[..., 28:37]) > 0


def test_negative_shifts_without_inserting_or_replacing_tail(source):
    _, context = source
    video, audio = streams()
    plan = helper.continuity_plan(video, audio, context)
    original_tail = torch.full((1, 24, 2, 1, 1), -1.)
    original_audio = torch.full((1, 32, 2, 9), -1.)
    incoming = [[None, {"minimax_keyframes": [dict(resolved_frame_index=0, latent=original_tail, audio_latent=original_audio)]}], [None, {"opaque": 1}]]
    result = helper.align_continuity_conditioning(incoming, video, audio, plan, inject_tail=False)
    guides = result[0][1]["minimax_keyframes"]
    assert len(guides) == 1 and guides[0]["resolved_frame_index"] == 17
    assert guides[0]["latent"] is original_tail and guides[0]["audio_latent"] is original_audio
    assert "minimax_keyframes" not in result[1][1]
    assert incoming[0][1]["minimax_keyframes"][0]["resolved_frame_index"] == 0


def test_missing_positive_tail_is_inserted_and_inactive_alignment_is_identity(source):
    _, context = source
    video, audio = streams()
    plan = helper.continuity_plan(video, audio, context)
    incoming = [[None, {"minimax_refs": []}]]
    assert helper.align_continuity_conditioning(incoming, video, audio, None) is incoming
    result = helper.align_continuity_conditioning(incoming, video, audio, plan)
    assert result[0][1]["minimax_keyframes"][0]["resolved_frame_index"] == 17
    assert "minimax_keyframes" not in incoming[0][1]
