"""CPU network, safe checkpoint boundary and scoped native lifecycle tests."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dasiwa_h3_backend_test", ROOT / "nodes/h3_latent_upscale.py")
h3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h3)


@pytest.fixture(scope="module")
def native():
    # Never initialize a GPU: force CPU before importing ComfyUI model management.
    sys.path.insert(0, str(ROOT.parents[1]))
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import folder_paths
    import comfy.model_management as mm
    from safetensors.torch import save_file
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield folder_paths, mm, save_file
    torch.set_num_threads(threads)


@pytest.fixture
def checkpoint(tmp_path, monkeypatch, native):
    folders, mm, save_file = native
    monkeypatch.setitem(folders.folder_names_and_paths, h3._FOLDER, ([str(tmp_path)], set(h3._EXTENSIONS)))
    model = h3._LatentResizer3D(channels=32, in_blocks=2, out_blocks=1,
                               temporal_every=1, temporal_kernel=3, dropout=0)
    save_file(model.state_dict(), str(tmp_path / "h3.safetensors"))
    return "h3.safetensors", model


def test_names_are_clean_native_candidates(monkeypatch):
    folders = SimpleNamespace(folder_names_and_paths={h3._FOLDER: (["models"], {".pth"})},
        get_filename_list=lambda _: ["sub/h3.sft", "H3.SAFETENSORS", "ltx2.safetensors",
                                     "h3.pth", "sub/h3.sft", "random.safetensors"])
    monkeypatch.setitem(sys.modules, "folder_paths", folders)
    assert h3.upscale_model_names() == ["H3.SAFETENSORS", "random.safetensors", "sub/h3.sft"]
    assert ".pth" in folders.folder_names_and_paths[h3._FOLDER][1]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
def test_actual_native_cpu_network_shape_dtype_and_cleanup(checkpoint, native, dtype):
    name, _ = checkpoint
    _, mm, _ = native
    # Keep unrelated registry entries intact. No global load/free-memory permitted.
    unrelated = object()
    mm.current_loaded_models.append(unrelated)
    before = list(mm.current_loaded_models)
    owner = h3.H3LatentUpscaler(name, "cpu")
    try:
        assert owner.model is None
        video = torch.randn(2, 24, 3, 2, 3, dtype=dtype)
        original = video.clone()
        with owner:
            result = owner.upscale(video, 3, 5)
            assert result.shape == (2, 24, 3, 3, 5)
            assert result.dtype == dtype and result.device == video.device
            assert torch.isfinite(result).all()
            assert owner.dtype == torch.float32
            assert owner.temporal_halo == 3
            assert not result.requires_grad
            assert torch.equal(video, original)
            assert owner._loaded in mm.current_loaded_models
        owner.close()
        assert owner.model is None and owner.patcher is None
        assert mm.current_loaded_models == before
        with pytest.raises(RuntimeError, match="closed"):
            owner.upscale(video, 3, 5)
    finally:
        owner.close()
        mm.current_loaded_models.remove(unrelated)


def test_network_matches_normalization_and_scale_embedding(checkpoint):
    name, reference = checkpoint
    video = torch.randn(1, 24, 2, 3, 2).transpose(-1, -2)
    assert not video.is_contiguous()
    mean, std = h3._make_norm_tensors("cpu", torch.float32)
    reference.eval()
    with torch.inference_mode():
        expected = reference((video - mean) / std, scale=(4/2 + 5/3)/2,
                             target_size=(2, 4, 5)) * std + mean
    with h3.H3LatentUpscaler(name, "cpu") as owner:
        result = owner.upscale(video, 4, 5)
    torch.testing.assert_close(result, expected)


def test_noop_does_not_resolve_or_load(monkeypatch):
    monkeypatch.setattr(h3, "_resolve_model_path", lambda _: pytest.fail("unexpected load"))
    video = torch.randn(1, 24, 2, 3, 4)
    with h3.H3LatentUpscaler("missing.pth", "cpu") as owner:
        assert owner.upscale(video, 3, 4) is video
        assert owner.model is None


@pytest.mark.parametrize("shape,targets", [((1, 128, 2, 3, 4), (5, 6)),
    ((1, 24, 3, 4), (5, 6)), ((1, 24, 2, 3, 4), (2, 8)),
    ((1, 24, 2, 3, 4), (4, 0)), ((1, 24, 2, 3, 4), (4.5, 8)),
    ((1, 24, 2, 3, 4), (True, 8))])
def test_invalid_layout_or_targets_fail_before_load(shape, targets):
    with h3.H3LatentUpscaler("unused.sft", "cpu") as owner:
        with pytest.raises(ValueError):
            owner.upscale(torch.zeros(shape), *targets)
        assert owner.model is None


@pytest.mark.parametrize("name", ["evil.pth", "../h3.sft", "/abs/h3.sft", "bad.ckpt"])
def test_unsafe_names_rejected_before_resolver(name, monkeypatch):
    monkeypatch.setattr(h3, "_folders", lambda: pytest.fail("unsafe path reached resolver"))
    with pytest.raises(ValueError):
        h3._resolve_model_path(name)


def test_missing_and_resolved_pickle_path_fail(checkpoint, tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        h3._resolve_model_path("missing.sft")
    bad = tmp_path / "bad.pth"
    bad.write_bytes(b"not a checkpoint")
    monkeypatch.setattr(h3, "_folders", lambda: SimpleNamespace(get_full_path=lambda *_: str(bad)))
    with pytest.raises(ValueError, match="Resolved"):
        h3._resolve_model_path("safe.sft")


def test_symlink_to_pickle_rejected(checkpoint, tmp_path):
    (tmp_path / "bad.pth").write_bytes(b"pickle")
    (tmp_path / "link.sft").symlink_to(tmp_path / "bad.pth")
    with pytest.raises(ValueError, match="Resolved"):
        h3._resolve_model_path("link.sft")


@pytest.mark.parametrize("attn,cadence,kernel", [(False, 0, 5), (False, 3, 7), (True, 1, 3), (True, 2, 5)])
def test_exact_architecture_module_indices(attn, cadence, kernel):
    model = h3._LatentResizer3D(channels=32, in_blocks=4, out_blocks=3,
        temporal_every=cadence, temporal_kernel=kernel, attn=attn, dropout=0)
    sd, config = h3._validated_state_dict(model.state_dict())
    with torch.device("meta"):
        rebuilt = h3._LatentResizer3D(**config)
    assert list(rebuilt.state_dict()) == list(sd)
    assert [tuple(v.shape) for v in rebuilt.state_dict().values()] == [tuple(v.shape) for v in sd.values()]


@pytest.mark.parametrize("failure", ["channels", "missing", "extra", "index", "kernel", "shape"])
def test_strict_checkpoint_validation(failure):
    model = h3._LatentResizer3D(channels=32, in_blocks=1, out_blocks=1, temporal_kernel=3)
    sd = dict(model.state_dict())
    if failure == "channels":
        sd["conv_in.weight"] = torch.empty(32, 128, 3, 3, 3)
    elif failure == "missing":
        del sd["conv_out.bias"]
    elif failure == "extra":
        sd["unrelated"] = torch.empty(1)
    elif failure == "index":
        sd = {k.replace("in_blocks.0.", "in_blocks.8."): v for k, v in sd.items()}
    elif failure == "kernel":
        sd["in_blocks.1.dwconv.weight"] = torch.empty(32, 1, 4, 1, 1)
    else:
        sd["embed.0.weight"] = torch.empty(63, 1)
    with pytest.raises(ValueError):
        h3._validated_state_dict(sd)


def test_safe_loader_and_prefix(checkpoint, native, monkeypatch):
    name, model = checkpoint
    import comfy.utils
    calls = []
    def load(path, **kwargs):
        calls.append((path, kwargs))
        return {"upscaler." + k: v for k, v in model.state_dict().items()}
    monkeypatch.setattr(comfy.utils, "load_torch_file", load)
    with h3.H3LatentUpscaler(name, "cpu") as owner:
        owner.upscale(torch.zeros(1, 24, 2, 2, 2), 3, 3)
        owner.upscale(torch.zeros(1, 24, 2, 2, 2), 3, 3)
    assert len(calls) == 1
    assert calls[0][1] == {"safe_load": True, "device": torch.device("cpu")}


@pytest.mark.parametrize("kernels", [[], [3, 7]])
def test_temporal_halo_from_exact_checkpoint_modules(checkpoint, native, tmp_path, kernels):
    _, mm, save_file = native
    layouts = {"in_blocks": [("residual", None)], "out_blocks": [("residual", None)]}
    layouts["in_blocks"] += [("temporal", kernel) for kernel in kernels]
    model = h3._LatentResizer3D(channels=32, block_layouts=layouts)
    save_file(model.state_dict(), str(tmp_path / "halo.sft"))
    with h3.H3LatentUpscaler("halo.sft", "cpu") as owner:
        assert owner.temporal_halo == sum((kernel - 1) // 2 for kernel in kernels)
        assert owner.upscale(torch.zeros(1, 24, 2, 2, 2), 3, 3).shape == (1, 24, 2, 3, 3)


def test_scoped_reload_after_native_offload(checkpoint, native, monkeypatch):
    name, _ = checkpoint
    _, mm, _ = native
    monkeypatch.setattr(mm, "load_models_gpu", lambda *_args, **_kwargs: pytest.fail("global load"))
    monkeypatch.setattr(mm, "free_memory", lambda *_args, **_kwargs: pytest.fail("global eviction"))
    video = torch.randn(1, 24, 2, 2, 2)
    with h3.H3LatentUpscaler(name, "cpu") as owner:
        first = owner.upscale(video, 3, 3)
        entry = owner._loaded
        entry.model_unload()
        mm.current_loaded_models.remove(entry)
        second = owner.upscale(video, 3, 3)
        assert any(item is entry for item in mm.current_loaded_models)
        torch.testing.assert_close(first, second)
    assert not any(item is entry for item in mm.current_loaded_models)


def test_failed_native_load_releases_owner(checkpoint, native, monkeypatch):
    name, _ = checkpoint
    _, mm, _ = native
    before = list(mm.current_loaded_models)
    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic allocation failure")
    monkeypatch.setattr(mm.LoadedModel, "model_load", fail)
    owner = h3.H3LatentUpscaler(name, "cpu")
    with pytest.raises(RuntimeError, match="allocation failure"):
        owner.upscale(torch.zeros(1, 24, 2, 2, 2), 3, 3)
    assert owner.model is None and owner.patcher is None
    assert mm.current_loaded_models == before


@pytest.mark.parametrize("precision,dtype", [("bf16",torch.bfloat16),("fp16",torch.float16),("fp32",torch.float32)])
def test_explicit_precision_runs_real_cpu_network(checkpoint,precision,dtype):
    name,_=checkpoint
    with h3.H3LatentUpscaler(name,"cpu",precision=precision) as owner:
        result=owner.upscale(torch.randn(1,24,2,2,2),3,3)
        assert owner.dtype==dtype
        assert result.dtype==torch.float32
        assert torch.isfinite(result).all()


def test_auto_honors_comfy_force_fp16():
    mm=SimpleNamespace(args=SimpleNamespace(force_fp16=True),
        should_use_bf16=lambda **_: True,should_use_fp16=lambda **_: True)
    assert h3._compute_dtype(torch.device("cuda"),mm)==torch.float16


def test_unknown_precision_fails_before_loading():
    with pytest.raises(ValueError,match="precision"):
        h3.H3LatentUpscaler("unused.sft","cpu",precision="invalid")


def test_dtype_selection_uses_backend_policy():
    mm = SimpleNamespace(should_use_bf16=lambda **_: True, should_use_fp16=lambda **_: True)
    assert h3._compute_dtype(torch.device("cpu"), mm) == torch.float32
    assert h3._compute_dtype(torch.device("cuda"), mm) == torch.bfloat16
    mm.should_use_bf16 = lambda **_: False
    assert h3._compute_dtype(torch.device("mps"), mm) == torch.float16
    mm.should_use_fp16 = lambda **_: False
    assert h3._compute_dtype(torch.device("xpu"), mm) == torch.float32
