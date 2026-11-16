"""Pressure reclamation uses only the current executor's inactive RAM cache."""
import pytest
from test_h3_transition import transition, clips, spec
import comfy.memory_management as memory


def test_inactive_cache_reclaimed_before_transition_decode(tmp_path, monkeypatch):
    clips(tmp_path)
    reserve = 8 * 1024**3
    available = [reserve]
    calls = []
    monkeypatch.setattr(transition.helper_batch_output, 'available_ram_bytes', lambda: available[0])
    monkeypatch.setattr(transition.helper_batch_output, 'total_ram_bytes', lambda: 64 * 1024**3)
    def release(target, free_active=False):
        calls.append((target, free_active))
        available[0] = target
        return target - reserve
    monkeypatch.setattr(memory, 'extra_ram_release', release)
    monkeypatch.setattr(transition.helper_batch_output, 'unload_all_comfy_models',
                        lambda: pytest.fail('No model unloading'))
    data = transition.transition_spec(spec())
    edges = transition.clip_edges(data, input_root=tmp_path, output_root=tmp_path)
    required = 22 * 64 * 64 * 3 * 10 + 22 * 2 * 2000 * 4 * 3 + 64 * 64 * 12 + 64 * 64 * 16
    assert calls == [(required + reserve, False)]
    assert edges['tail'].shape == edges['head'].shape == (22, 64, 64, 3)
