"""Pressure, permission and native cache eviction without large allocations."""
import weakref
from types import SimpleNamespace

import pytest
import torch

from test_enhanced_video_memory import video


def hooks(monkeypatch, memory, models):
    import comfy
    monkeypatch.setattr(comfy, "memory_management", memory, raising=False)
    monkeypatch.setattr(comfy, "model_management", models, raising=False)


def test_no_eviction_with_enough_ram(video, monkeypatch):
    monkeypatch.setattr(video, "available_ram_bytes", lambda: 200)
    def forbidden(*args, **kwargs):
        pytest.fail("unnecessary release")
    hooks(monkeypatch, SimpleNamespace(extra_ram_release=forbidden), SimpleNamespace(free_pins=forbidden))
    assert video._reclaim_encode_ram(100) == 200


@pytest.mark.parametrize("stop_stage", ["inactive", "pins", "active", None])
def test_reclaim_stages_recheck_real_capacity(video, monkeypatch, stop_stage):
    available = [0]
    calls = []
    def release(target, free_active=False):
        stage = "active" if free_active else "inactive"
        calls.append(stage)
        assert target == 100
        if stage == stop_stage:
            available[0] = 150
        return 100000  # A reported count must not replace actual availability.
    def pins(size, loaded=False):
        calls.append("pins")
        assert size == 100 and loaded
        if stop_stage == "pins":
            available[0] = 150
    def forbidden(*a, **kw):
        pytest.fail("global model unload or GPU offload")
    hooks(monkeypatch, SimpleNamespace(extra_ram_release=release),
          SimpleNamespace(free_pins=pins, unload_all_models=forbidden, free_memory=forbidden))
    monkeypatch.setattr(video, "available_ram_bytes", lambda: available[0])
    assert video._reclaim_encode_ram(100) == (150 if stop_stage else 0)
    expected = ["inactive", "pins", "active"]
    assert calls == (expected[:expected.index(stop_stage) + 1] if stop_stage else expected)


def test_older_hooks_without_loaded_parameter(video, monkeypatch):
    available = [0]
    def pins(size):
        assert size == 100
        available[0] = 150
    hooks(monkeypatch, SimpleNamespace(), SimpleNamespace(free_pins=pins))
    monkeypatch.setattr(video, "available_ram_bytes", lambda: available[0])
    assert video._reclaim_encode_ram(100) == 150


def test_missing_hooks_warn_but_do_not_abort(video, monkeypatch):
    hooks(monkeypatch, SimpleNamespace(), SimpleNamespace())
    monkeypatch.setattr(video, "available_ram_bytes", lambda: 0)
    assert video._reclaim_encode_ram(100) == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_permission_and_throttled_progress(video, monkeypatch, enabled):
    clock = [0]
    calls = []
    monkeypatch.setattr(video.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(video, "_reclaim_encode_ram", lambda target: calls.append(clock[0]))
    def encode(*args, **kwargs):
        progress = args[-1]
        for t in [1, 4, 5, 6, 9, 10]:
            clock[0] = t
            progress(0)
        return "test"
    monkeypatch.setattr(video, "_encode_with_available_encoder", encode)
    video.DaSiWa_EnhancedVideoCombine().combine(
        torch.zeros(3, 16, 16, 3), 24, "H.264", "MP4", "8-bit", 20,
        False, False, "test", True, False, free_memory_if_needed=enabled,
    )
    assert calls == ([0, 5, 10] if enabled else [])
    option = video.DaSiWa_EnhancedVideoCombine.INPUT_TYPES()["optional"]["free_memory_if_needed"]
    assert option[1]["default"] is False and option[1]["tooltip"]


@pytest.mark.parametrize("active", [False, True])
def test_native_cache_releases_storage_but_preserves_live_input(video, monkeypatch, active):
    from comfy import memory_management
    from comfy_execution import caching
    cache = caching.RAMPressureCache(caching.CacheKeySetID)
    tensor = torch.ones(8, 8, 3)
    ref = weakref.ref(tensor)
    cache.cache["old"] = SimpleNamespace(outputs=[[tensor]])
    cache.used_generation["old"] = cache.generation if active else cache.generation - 1
    cache.timestamps["old"] = 0
    live_input = torch.ones(3, 8, 8, 3)
    cache.cache["input"] = SimpleNamespace(outputs=[[live_input]])
    cache.used_generation["input"] = cache.generation
    cache.timestamps["input"] = 1
    del tensor
    availability = lambda: 0 if "old" in cache.cache else 200
    monkeypatch.setattr(caching, "virtual_memory_available", availability)
    monkeypatch.setattr(video, "available_ram_bytes", availability)
    monkeypatch.setattr(memory_management, "extra_ram_release_callback", cache.ram_release)
    hooks(monkeypatch, memory_management, SimpleNamespace())
    assert video._reclaim_encode_ram(100) == 200
    assert ref() is None
    assert torch.equal(live_input, torch.ones_like(live_input))
    assert cache.active_evictions is active


def test_target_charges_workspace_not_full_input(video):
    images = torch.zeros(5, 16, 16, 3)
    base = video._encode_ram_target(images, None, True, False)
    assert base == 1024 ** 3 + 16 * 16 * 16 * 3 * 4
    assert video._encode_ram_target(images, None, True, True) == base + 2 * images.numel() * 4


class LazyAudioMap:  # VideoHelperSuite's audio: keys()/__getitem__, no dict base
    def __init__(self, data):
        self.data = data
    def keys(self):
        return self.data.keys()
    def __getitem__(self, key):
        return self.data[key]


def test_audio_file_accepts_mapping_like_audio(video):
    audio = LazyAudioMap({"waveform": torch.ones(1, 2, 800), "sample_rate": 16000})
    (planar, rate), duration = video._audio_file(audio)
    assert planar.shape == (2, 800) and rate == 16000 and duration == 0.05
