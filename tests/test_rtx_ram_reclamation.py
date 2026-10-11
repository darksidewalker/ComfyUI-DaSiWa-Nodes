"""Host-pressure regression tests; no CUDA or large output allocation."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ComfyUI"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nodes"))
import nodes_rtx_upscaler_refiner as rtx
import nodes_rtx_frame_generation as fg


@pytest.mark.parametrize("allocator", [rtx._allocate_output_tensor, fg._dasiwa_fg_allocate_output])
def test_ram_pressure_reclaims_then_rechecks(monkeypatch, allocator):
    available = [512 * 1024 ** 2]
    requests = []
    def release(target, free_active=False):
        requests.append((target, free_active))
        available[0] = 3 * 1024 ** 3
        return 0  # Capacity, not a callback's claimed byte count, decides success.
    monkeypatch.setattr(rtx, "available_ram_bytes", lambda: available[0])
    monkeypatch.setattr(rtx, "_memory_management", SimpleNamespace(extra_ram_release=release), raising=False)
    monkeypatch.setattr(rtx, "_model_management", None)
    out, path = allocator((1, 8, 8, 3), torch.float32, torch.device("cpu"))
    assert out.shape == (1, 8, 8, 3) and path is None
    assert requests == [(1024 ** 3 + 2 * 768 + 768, False)]


@pytest.mark.parametrize("enough_ram,enabled", [(True, True), (False, False)])
def test_no_host_eviction_without_pressure_or_permission(monkeypatch, enough_ram, enabled):
    monkeypatch.setattr(rtx, "available_ram_bytes", lambda: (3 * 1024 ** 3 if enough_ram else 0))
    def forbidden(*a, **kw):
        pytest.fail("unexpected host reclamation")
    monkeypatch.setattr(rtx, "_memory_management", SimpleNamespace(extra_ram_release=forbidden))
    monkeypatch.setattr(rtx, "_model_management", SimpleNamespace(free_pins=forbidden))
    if enough_ram:
        rtx._allocate_output_tensor((1, 8, 8, 3), torch.float32, torch.device("cpu"), auto_unload_models=enabled)
    else:
        with pytest.raises(RuntimeError, match="enable use_mmap"):
            rtx._allocate_output_tensor((1, 8, 8, 3), torch.float32, torch.device("cpu"), auto_unload_models=enabled)


def test_pins_then_active_cache_fallback_without_global_unload(monkeypatch):
    calls = []
    available = [0]
    def release(target, free_active=False):
        calls.append(("cache", free_active))
        if free_active:
            available[0] = 3 * 1024 ** 3
    def pins(size, loaded=False):
        calls.append(("pins", {"loaded": loaded}))
        assert size > 0
    def forbidden(*a, **kw):
        pytest.fail("must not reset executor or globally offload GPU weights into RAM")
    monkeypatch.setattr(rtx, "available_ram_bytes", lambda: available[0])
    monkeypatch.setattr(rtx, "_memory_management", SimpleNamespace(extra_ram_release=release), raising=False)
    monkeypatch.setattr(rtx, "_model_management", SimpleNamespace(free_pins=pins, unload_all_models=forbidden, free_memory=forbidden))
    out, _ = rtx._allocate_output_tensor((1, 8, 8, 3), torch.float32, torch.device("cpu"))
    assert out.device.type == "cpu"
    assert calls == [("cache", False), ("pins", {"loaded": True}), ("cache", True)]


@pytest.mark.parametrize("allow_mmap", [False, True])
def test_unreclaimable_input_still_obeys_mmap_permission(monkeypatch, allow_mmap):
    requests = []
    monkeypatch.setattr(rtx, "available_ram_bytes", lambda: 0)
    monkeypatch.setattr(rtx, "_memory_management", SimpleNamespace(extra_ram_release=lambda *a, **kw: requests.append(kw)))
    monkeypatch.setattr(rtx, "_model_management", None)
    marker = object()
    monkeypatch.setattr(rtx, "allocate_cpu_output", lambda *a, **kw: (marker, "mapped") if kw["force_mmap"] else pytest.fail("implicit mmap"))
    if allow_mmap:
        assert rtx._allocate_output_tensor((1, 8, 8, 3), torch.float32, torch.device("cpu"), allow_mmap=True) == (marker, "mapped")
    else:
        with pytest.raises(RuntimeError, match="enable use_mmap"):
            rtx._allocate_output_tensor((1, 8, 8, 3), torch.float32, torch.device("cpu"))
    assert requests == [{"free_active": False}, {"free_active": True}]


def test_older_comfy_without_host_hooks_retains_clear_failure(monkeypatch):
    monkeypatch.setattr(rtx, "available_ram_bytes", lambda: 0)
    monkeypatch.setattr(rtx, "_memory_management", None)
    monkeypatch.setattr(rtx, "_model_management", SimpleNamespace())
    with pytest.raises(RuntimeError, match="enable use_mmap"):
        rtx._allocate_output_tensor((1, 8, 8, 3), torch.float32, torch.device("cpu"))


@pytest.mark.parametrize("active", [False, True])
def test_native_comfy_cache_eviction_releases_storage(monkeypatch, active):
    import weakref
    from comfy import memory_management
    from comfy_execution import caching
    cache = caching.RAMPressureCache(caching.CacheKeySetID)
    tensor = torch.ones(8, 8, 3)
    reference = weakref.ref(tensor)
    cache.cache["old"] = SimpleNamespace(outputs=[[tensor]])
    cache.used_generation["old"] = cache.generation if active else cache.generation - 1
    cache.timestamps["old"] = 0
    del tensor
    availability = lambda: 0 if cache.cache else 3 * 1024 ** 3
    monkeypatch.setattr(caching, "virtual_memory_available", availability)
    monkeypatch.setattr(rtx, "available_ram_bytes", availability)
    monkeypatch.setattr(memory_management, "extra_ram_release_callback", cache.ram_release)
    monkeypatch.setattr(rtx, "_memory_management", memory_management)
    monkeypatch.setattr(rtx, "_model_management", None)
    out, path = rtx._allocate_output_tensor((1, 8, 8, 3), torch.float32, torch.device("cpu"))
    assert out.shape == (1, 8, 8, 3) and path is None
    assert not cache.cache and reference() is None
    assert cache.active_evictions is active


def test_reported_large_output_reclaim_target_includes_growing_reserve(monkeypatch):
    shape = (220, 2000, 2000, 3)  # About 9.83 GiB; no large tensor is allocated.
    available = [int(6.37 * 1024 ** 3)]
    targets = []
    def release(target, free_active=False):
        targets.append(target)
        available[0] = target
    monkeypatch.setattr(rtx, "available_ram_bytes", lambda: available[0])
    monkeypatch.setattr(rtx, "_memory_management", SimpleNamespace(extra_ram_release=release))
    monkeypatch.setattr(rtx, "_model_management", None)
    marker = object()
    original_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda s, **kw: original_empty(s, **kw) if s == () else marker)
    assert rtx._allocate_output_tensor(shape, torch.float32, torch.device("cpu")) == (marker, None)
    assert len(targets) == 1
    assert rtx.tensor_nbytes(shape, torch.float32) <= rtx._dasiwa_rtx_ram_budget(shape, torch.float32)[2]


def test_pin_api_without_loaded_parameter(monkeypatch):
    available = [0]
    requests = []
    def pins(size):
        requests.append(size)
        available[0] = 3 * 1024 ** 3
    monkeypatch.setattr(rtx, "available_ram_bytes", lambda: available[0])
    monkeypatch.setattr(rtx, "_memory_management", None)
    monkeypatch.setattr(rtx, "_model_management", SimpleNamespace(free_pins=pins))
    out, _ = rtx._allocate_output_tensor((1, 8, 8, 3), torch.float32, torch.device("cpu"))
    assert out.device.type == "cpu" and len(requests) == 1
