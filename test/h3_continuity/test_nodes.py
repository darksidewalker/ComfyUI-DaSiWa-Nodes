import pytest
import torch
from comfy.nested_tensor import NestedTensor
from dasiwa_continuity_test import core
from test_core import latent, context, ready


def test_only_two_technical_node_types_registered():
    from dasiwa_continuity_test import nodes
    assert set(nodes.NODE_CLASS_MAPPINGS) == {"DaSiWaH3ContinuityAppend", "DaSiWaH3ContinuityPublish"}
    assert nodes.DaSiWaH3ContinuityAppend.RETURN_TYPES == ("LATENT", "DF_H3_CONTINUITY_TICKET")
    assert nodes.DaSiWaH3ContinuityPublish.OUTPUT_NODE


def test_disabled_append_passthrough_and_no_checkpoint(monkeypatch, tmp_path):
    from dasiwa_continuity_test import nodes
    monkeypatch.setattr(nodes, "ClipStore", lambda: core.ClipStore(tmp_path))
    sample = latent()
    combined, ticket = nodes.DaSiWaH3ContinuityAppend().commit(sample, {"disabled": True})
    assert combined is sample and ticket == {"disabled": True}
    assert list(tmp_path.iterdir()) == []


def test_append_branches_from_pinned_parent_and_preserves_prefix(monkeypatch, tmp_path):
    from dasiwa_continuity_test import nodes
    store = core.ClipStore(tmp_path)
    monkeypatch.setattr(nodes, "ClipStore", lambda: store)
    parent = ready(store)
    previous, meta = store.load("test", parent["clip_id"])
    settings = core.parse_settings({"session": "test", "operation": "continue", "source_id": parent["clip_id"]})
    _, target, layout = core.prepare_continuation(previous, meta, {"mode": "REF2VA", "width": 48, "height": 32}, settings)
    newer = ready(store, latent(243))
    combined, ticket = nodes.DaSiWaH3ContinuityAppend().commit(target, context(operation="continue", source_id=parent["clip_id"], layout=layout))
    assert core.validate_h3_av_latent(combined, name="combined")[2] == 243
    assert torch.equal(combined["samples"].tensors[0][:, :, :previous["samples"].tensors[0].shape[2]], previous["samples"].tensors[0])
    assert store.metadata("test", ticket["clip_id"], ready=False)["parent_id"] == parent["clip_id"]
    assert store.list_clips("test")["latest_id"] == newer["clip_id"]


def test_publish_requires_real_nonempty_export(monkeypatch, tmp_path):
    from dasiwa_continuity_test import nodes
    import folder_paths
    store = core.ClipStore(tmp_path / "continuity")
    monkeypatch.setattr(nodes, "ClipStore", lambda: store)
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path / "temp"))
    ticket = store.stage(latent(), context())
    for path in (tmp_path / "missing.mp4", tmp_path / "empty.mp4", tmp_path.parent / "outside.mp4"):
        if path.name == "empty.mp4": path.touch()
        with pytest.raises(ValueError, match="valid output"):
            nodes.DaSiWaH3ContinuityPublish().publish(str(path), ticket)
    assert store.list_clips("test")["clips"] == []
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"export")
    monkeypatch.setattr(nodes, "make_tail_thumbnails", lambda *a: (_ for _ in ()).throw(RuntimeError("ffmpeg unavailable")), raising=False)
    result = nodes.DaSiWaH3ContinuityPublish().publish(str(movie), ticket)
    assert "Saved 124 frames" in result["result"][0]
    assert store.metadata("test", ticket["clip_id"])["preview_warning"]
