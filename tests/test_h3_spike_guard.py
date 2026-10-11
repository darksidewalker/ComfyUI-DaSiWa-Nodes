"""CPU-only tests; load the helper without importing the node pack."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'ComfyUI'))
from comfy.cli_args import args
args.cpu = True  # Before model_management import: never initialize CUDA.
from comfy import model_management as mm


@pytest.fixture
def guard(monkeypatch):
    path = Path(__file__).parents[1] / 'nodes' / 'h3_spike_guard.py'
    assert path.exists(), 'spike guard helper is not implemented'
    spec = importlib.util.spec_from_file_location('dasiwa_spike_guard_test_only', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(mm, 'get_free_memory', lambda device: 10**12)
    def forbidden(*args, **kwargs):
        raise AssertionError('helper must not free/offload global models')
    monkeypatch.setattr(mm, 'free_memory', forbidden)
    monkeypatch.setattr(mm, 'unload_all_models', forbidden)
    return module


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_clean_is_exact_same_tensor(guard, dtype):
    source = torch.randn(1, 24, 3, 12, 12, generator=torch.Generator().manual_seed(4)).to(dtype)
    output = source.repeat_interleave(3, -2).repeat_interleave(3, -1)
    before = output.clone()
    result, report = guard.spike_guard(source, output)
    assert result is output
    assert torch.equal(result, before)
    assert 'clean' in report


def burned(guard, sign=1, dtype=torch.float32, batch=1):
    source = torch.randn(batch, 24, 5, 12, 12, generator=torch.Generator().manual_seed(41))
    output = source.repeat_interleave(3, -2).repeat_interleave(3, -1)
    baseline = output.clone()
    direction = torch.tensor(guard.BURN_DIRECTION).view(1, 24, 1, 1)
    output[:, :, 2, 15:18, 15:18] += sign * 30 * direction
    return source.to(dtype), output.to(dtype), baseline.to(dtype)


def test_positive_artifact_is_reduced(guard):
    source, output, baseline = burned(guard)
    before = (output - baseline).square().sum()
    result, report = guard.spike_guard(source, output)
    assert result is output
    assert (result - baseline).square().sum() < before * .3
    assert 'repaired' in report


def test_zero_strength_is_honestly_off_without_memory_read(guard, monkeypatch):
    source, output, _ = burned(guard)
    before = output.clone()
    def forbidden(*args):
        raise AssertionError('off must not query capacity or allocate')
    monkeypatch.setattr(mm, 'get_free_memory', forbidden)
    result, report = guard.spike_guard(source, output, strength=0)
    assert result is output
    assert torch.equal(output, before)
    assert 'off' in report and 'repaired' not in report
    assert 'output footprint 0 pixels' in report


def test_both_mode_reduces_negative_signature_default_does_not(guard):
    source, output, baseline = burned(guard, sign=-1)
    before = output.clone()
    guard.spike_guard(source, output)
    assert torch.equal(output, before)
    result, report = guard.spike_guard(source, output, mode='both')
    assert (result - baseline).square().sum() < (before - baseline).square().sum() * .3
    assert 'mode=both' in report


def test_memory_refusal_happens_before_float32_conversion(guard, monkeypatch):
    source, output, _ = burned(guard, dtype=torch.float16)
    before = output.clone()
    queries = []
    def capacity(device):
        queries.append(device)
        return 1
    monkeypatch.setattr(mm, 'get_free_memory', capacity)
    def forbidden_conversion(*args, **kwargs):
        raise AssertionError('capacity must be checked before float32 conversion')
    monkeypatch.setattr(torch.Tensor, 'to', forbidden_conversion)
    with pytest.raises(MemoryError, match=r'spike guard.*CPU.*required.*available'):
        guard.spike_guard(source, output)
    assert queries == [torch.device('cpu')]
    assert torch.equal(output, before)


def test_memory_budget_includes_workspaces_but_streams_output(guard):
    source = (2, 24, 5, 12, 12)
    output = (2, 24, 5, 36, 36)
    cost = guard.spike_guard_memory_bytes(source, output)
    source_bytes = torch.empty(source, device='meta').numel() * 4
    one_frame_bytes = torch.empty(output[:2] + output[3:], device='meta').numel() * 4
    assert cost >= source_bytes * 2 + one_frame_bytes * 2
    # Temporal growth cannot charge for a full float32 output copy.
    long_source = (2, 24, 50, 12, 12)
    long_output = (2, 24, 50, 36, 36)
    spatial_large = (2, 24, 50, 72, 72)
    delta_long = guard.spike_guard_memory_bytes(long_source, spatial_large) - guard.spike_guard_memory_bytes(long_source, long_output)
    delta_short = guard.spike_guard_memory_bytes(source, (2, 24, 5, 72, 72)) - cost
    assert delta_long == delta_short


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_dtype_and_untouched_pixels_are_exact(guard, dtype):
    source, output, baseline = burned(guard, dtype=dtype)
    if dtype == torch.float64:
        # Values deliberately not representable as float32: unaffected pixels
        # must not be rounded by the correction's one-frame work buffer.
        output.add_(1e-10)
    before = output.clone()
    result, _ = guard.spike_guard(source, output)
    assert result.dtype == dtype and result is output
    assert torch.equal(result[:, :, :, :3, :], before[:, :, :, :3, :])
    assert (result.float() - baseline.float()).square().sum() < (before.float() - baseline.float()).square().sum()


def test_diagnostics_measure_actual_non2x_halo_and_frames(guard):
    import ast
    import re
    source, output, _ = burned(guard, batch=2)
    before = output.clone()
    result, report = guard.spike_guard(source, output)
    changed = (result != before).any(dim=1)
    frames = [i for i in range(output.shape[2]) if bool(changed[:, i].any())]
    reported = ast.literal_eval(re.search(r'changed latent frames (\[[^\]]*\])', report).group(1))
    assert reported == frames
    spatial = changed.any(dim=(0, 1))
    ys = spatial.any(dim=1).nonzero().flatten()
    xs = spatial.any(dim=0).nonzero().flatten()
    box = tuple(int(x) for x in re.search(r'output area x(\d+)-(\d+) y(\d+)-(\d+)', report).groups())
    assert box == (int(xs.min()) * 16, (int(xs.max()) + 1) * 16, int(ys.min()) * 16, (int(ys.max()) + 1) * 16)
    # Original burn is output latent [15:18,15:18]; correction's true halo
    # extends beyond that. No hard-coded source-grid x32 bounding box.
    assert box[0] < 15 * 16 and box[1] > 18 * 16
    pixels = int(re.search(r'output footprint (\d+) pixels', report).group(1))
    assert pixels == int(changed.sum()) * 256


@pytest.mark.parametrize('cancel_at', [1, 3, 8, 13, 16, 20])
def test_cancellation_propagates_in_processing_loops(guard, monkeypatch, cancel_at):
    source, output, _ = burned(guard)
    calls = 0
    def interrupt():
        nonlocal calls
        calls += 1
        if calls == cancel_at:
            raise mm.InterruptProcessingException()
    monkeypatch.setattr(mm, 'throw_exception_if_processing_interrupted', interrupt)
    with pytest.raises(mm.InterruptProcessingException):
        guard.spike_guard(source, output)


def test_helper_import_does_not_initialize_model_management():
    import subprocess
    path = Path(__file__).parents[1] / 'nodes' / 'h3_spike_guard.py'
    script = ("import importlib.util, sys, torch; "
              f"sys.path.insert(0, {str(ROOT.parent / 'ComfyUI')!r}); "
              "from comfy.cli_args import args; args.cpu=True; "
              f"s=importlib.util.spec_from_file_location('guard_import_probe', {str(path)!r}); "
              "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
              "assert 'comfy.model_management' not in sys.modules; "
              "assert not torch.cuda.is_initialized()")
    subprocess.run([sys.executable, '-c', script], check=True, capture_output=True, text=True)


@pytest.mark.parametrize('mode', ['absolute', '', None])
def test_invalid_modes_are_rejected(guard, mode):
    source, output, _ = burned(guard)
    with pytest.raises(ValueError, match='mode'):
        guard.spike_guard(source, output, mode=mode)


def submitted_reference(source, output, direction, sensitivity=4.3, strength=1.0):
    """Talon v4's full-video float32 algorithm, independently retained for parity."""
    import torch.nn.functional as F
    src = source.float().cpu()
    b, c, t, sh, sw = src.shape
    oh, ow = output.shape[-2:]
    sd = src.std(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1e-6)
    flat = output.float().cpu().permute(0, 2, 1, 3, 4).reshape(-1, c, oh, ow)
    small = F.interpolate(flat, size=(sh, sw), mode='area')
    err = (small.reshape(b, t, c, sh, sw).permute(0, 2, 1, 3, 4) - src) / sd
    shift = err.mean(dim=(0, 2, 3, 4), keepdim=True)
    score = ((err - shift) * torch.tensor(direction).view(1, c, 1, 1, 1)).sum(dim=1, keepdim=True)
    centre = score.median()
    spread = (score - centre).abs().median() * 1.4826
    score = (score - centre) / spread.clamp_min(1e-6)
    region = (score > sensitivity).float()
    allowed = (score > sensitivity * .58).float()
    for _ in range(96):
        grown = F.max_pool3d(region, 3, 1, 1) * allowed
        if torch.equal(grown, region):
            break
        region = grown
    low = sensitivity * .58
    soft = ((score - low) / max(1e-6, sensitivity - low)).clamp(0, 1) * region
    weight = F.max_pool3d(soft, 3, 1, 1)
    weight = F.avg_pool3d(F.pad(weight, (1, 1, 1, 1, 0, 0), mode='replicate'), (1, 3, 3), 1)
    for i in range(t):
        if float(weight[:, :, i].max()) <= 0:
            continue
        cur = output[:, :, i].float().cpu()
        target = src[:, :, i] + shift[:, :, 0] * sd[:, :, 0]
        for _ in range(2):
            fix = (target - F.interpolate(cur, size=(sh, sw), mode='area')) * weight[:, :, i] * strength
            cur = cur + F.interpolate(fix, size=(oh, ow), mode='bilinear', align_corners=False)
        output[:, :, i] = cur.to(output.dtype)
    return output


@pytest.mark.parametrize('batch', [1, 2])
def test_float32_frame_streaming_matches_submitted_algorithm(guard, batch):
    source, output, _ = burned(guard, batch=batch)
    # Nonzero ordinary residual produces a genuine robust spread, not just the
    # MAD clamp used by the exact-repeat fixture.
    noise = torch.randn(output.shape, generator=torch.Generator().manual_seed(7)) * .01
    output.add_(noise)
    expected = submitted_reference(source, output.clone(), guard.BURN_DIRECTION, strength=.7)
    result, _ = guard.spike_guard(source, output, strength=.7)
    torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)


def test_cpu_float32_statistics_ignore_ambient_tensor_defaults(guard):
    source, output, _ = burned(guard)
    expected, _ = guard.spike_guard(source, output.clone())
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device('meta'):
            result, _ = guard.spike_guard(source, output)
    finally:
        torch.set_default_dtype(previous)
    assert torch.equal(result, expected)


def test_only_source_and_one_output_frame_are_converted(guard, monkeypatch):
    source, output, _ = burned(guard, dtype=torch.float16)
    source_before = source.clone()
    original_to = torch.Tensor.to
    converted_shapes = []
    def tracked_to(tensor, *args, **kwargs):
        converted_shapes.append(tuple(tensor.shape))
        assert tuple(tensor.shape) != tuple(output.shape), 'full output conversion is forbidden'
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', tracked_to)
    guard.spike_guard(source, output)
    assert tuple(source.shape) in converted_shapes
    assert tuple(output.shape[:2] + output.shape[3:]) in converted_shapes
    assert torch.equal(source, source_before)
