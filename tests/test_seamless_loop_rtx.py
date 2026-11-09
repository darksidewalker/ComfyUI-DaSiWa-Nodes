"""RTX loop contracts; real SDK coverage is separate from stub tests."""
import contextlib
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'nodes'))


def load_loop():
    spec = importlib.util.spec_from_file_location('nodes_seamless_loop', ROOT / 'nodes/nodes_seamless_loop.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rtx_backend_uses_overlap_without_native_checkpoint(monkeypatch):
    loop = load_loop()
    images = torch.linspace(0, 1, 24)[:, None, None, None].expand(24, 8, 8, 3).clone()
    lifecycle = []
    calls = []

    @contextlib.contextmanager
    def rtx(shape, mode, device_id):
        lifecycle.append((shape, mode, device_id))
        def interpolate(a, b, t):
            calls.append(t)
            return torch.lerp(a, b, t)
        try:
            yield interpolate
        finally:
            lifecycle.append('closed')

    monkeypatch.setattr(loop, '_load_rtx_interpolator', rtx, raising=False)
    monkeypatch.setattr(loop, '_load_interpolator', lambda *args: pytest.fail('RTX must not load a checkpoint'))
    result, _ = loop.DaSiWa_SeamlessLoop().execute(images, exact_endpoint=True,
        interpolation_backend='RTX Frame Generation', rtx_mode='MEDIUM', rtx_device_id=1)
    start, end, k, _ = loop._analyze(loop._preview(images))
    middle = end - start - 2*k
    assert torch.equal(result[:middle], images[start+k:end-k])
    assert torch.equal(result[0], result[-1])
    assert result.shape == (end-start-k+1, 8, 8, 3)
    assert calls and all(0 < t < 1 for t in calls)
    assert lifecycle == [(images.shape, 'MEDIUM', 1), 'closed']


def test_rtx_schema_preserves_legacy_slots_without_models(monkeypatch):
    import folder_paths
    monkeypatch.setattr(folder_paths, 'get_filename_list', lambda category: [])
    loop = load_loop()
    optional = loop.DaSiWa_SeamlessLoop.INPUT_TYPES()['optional']
    assert list(optional) == ['model_name', 'exact_endpoint', 'interpolation_backend', 'rtx_mode', 'rtx_device_id',
                              'audio', 'frame_rate', 'audio_loop']
    assert optional['model_name'][0], 'RTX must be queueable with no native checkpoints'
    assert optional['interpolation_backend'][1]['default'] == 'RIFE / FILM'
    assert all(options.get('tooltip') for _, options in optional.values())
    assert loop.DaSiWa_SeamlessLoop.IS_CHANGED(None, 'missing.safetensors',
        interpolation_backend='RTX Frame Generation') == 'RTX Frame Generation'


@pytest.mark.parametrize('fail', [False, True])
def test_rtx_adapter_owns_borrowed_output_and_closes(monkeypatch, fail):
    import nodes_rtx_frame_generation as fg
    from types import SimpleNamespace
    from enum import IntEnum
    loop = load_loop()
    calls = []
    class Mode(IntEnum):
        HIGH = 2
    class Effect:
        def __init__(self, mode, automatic_shot_change_detection_enabled, device, image_encoding):
            assert (mode, automatic_shot_change_detection_enabled, device, image_encoding) == (Mode.HIGH, False, 0, 0)
            self.buffer = torch.empty(3, 8, 8)
        def load(self):
            assert (self.input_width, self.input_height) == (8, 8)
        def run_at_timestep(self, previous, current, timestep, *, shot_change, non_blocking, stream_ptr):
            assert (shot_change, non_blocking, stream_ptr) == (False, False, 0)
            calls.append(timestep)
            if fail:
                raise RuntimeError('SDK failure')
            self.buffer.copy_(torch.lerp(previous, current, timestep))
            return SimpleNamespace(image=self.buffer)
        def close(self):
            calls.append('closed')
            self.buffer.zero_()
    monkeypatch.setattr(fg, '_dasiwa_fg_sdk', lambda: (Effect, Mode, SimpleNamespace(RGB8=0), False))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'device', lambda *args: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: None)
    monkeypatch.setattr(fg, '_dasiwa_fg_stream', lambda *args: contextlib.nullcontext())
    prepare = fg._dasiwa_fg_prepare
    monkeypatch.setattr(fg, '_dasiwa_fg_prepare', lambda image, device, packed: prepare(image, 'cpu', packed))
    if fail:
        with pytest.raises(RuntimeError, match='SDK failure'):
            with loop._load_rtx_interpolator((12, 8, 8, 3), 'HIGH', 0) as interpolate:
                interpolate(torch.zeros(8, 8, 3), torch.ones(8, 8, 3), .25)
    else:
        with loop._load_rtx_interpolator((12, 8, 8, 3), 'HIGH', 0) as interpolate:
            result = interpolate(torch.zeros(8, 8, 3), torch.ones(8, 8, 3), .25)
        assert torch.equal(result, torch.full_like(result, .25))
        assert calls[:-1] == [.25, .75]
    assert calls[-1] == 'closed'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Real NVIDIA SDK requires CUDA')
@pytest.mark.parametrize('input_device', ['cpu', 'cuda'])
@pytest.mark.parametrize('short', [False, True])
@pytest.mark.parametrize('custom_stream', [False, True])
def test_real_rtx_loop(input_device, short, custom_stream):
    loop = load_loop()
    h, w = 128, 192
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    frames = []
    for i in range(2 if short else 16):
        frame = torch.zeros(h, w, 3)
        frame[..., 0] = (((x - (50+2*i)).square() + (y-64).square()) < 400).float() * .8
        frame[..., 1] = (x % 17).float() / 50
        frame[..., 2] = (y % 19).float() / 50
        frames.append(frame)
    images = torch.stack(frames).to(input_device)
    original = images.clone()
    stream = torch.cuda.Stream() if custom_stream else torch.cuda.default_stream()
    with torch.cuda.stream(stream):
        result, _ = loop.DaSiWa_SeamlessLoop().execute(images, exact_endpoint=True,
            interpolation_backend='RTX Frame Generation')
    torch.cuda.synchronize()
    assert torch.isfinite(result).all()
    assert 0 <= result.min() <= result.max() <= 1
    assert result.shape[1:] == images.shape[1:]
    assert result.dtype == images.dtype
    assert torch.equal(result[0], result[-1])
    assert torch.equal(images, original)
    if short:
        assert result.shape[0] == 6
        assert torch.equal(result[:2].to(input_device), images)
        assert not torch.equal(result[2], result[3])
    else:
        start, end, k, _ = loop._analyze(loop._preview(images))
        middle = end-start-2*k
        assert result.shape[0] == end-start-k+1
        assert torch.equal(result[:middle].to(input_device), images[start+k:end-k])
        assert not torch.equal(result[middle+1], result[middle])

