import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'nodes'))


def load_loop():
    spec = importlib.util.spec_from_file_location('nodes_seamless_loop', ROOT / 'nodes/nodes_seamless_loop.py')
    assert spec and spec.loader and Path(spec.origin).exists(), 'Seamless Loop node is missing'
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_batch_stays_pixel_exact_without_loading_model():
    loop = load_loop()
    images = torch.full((12, 13, 17, 3), 0.1234567)
    result, _ = loop.DaSiWa_SeamlessLoop().execute(images, 'missing.safetensors')
    assert torch.equal(result, images)
    assert result.data_ptr() != images.data_ptr()


def test_moving_batch_closes_exactly_and_preserves_original_middle(monkeypatch):
    loop = load_loop()
    images = torch.stack([torch.full((13, 17, 3), i / 39) for i in range(40)])
    calls = []
    def interpolate(a, b, t):
        calls.append(t)
        return torch.lerp(a, b, t)
    monkeypatch.setattr(loop, '_load_interpolator', lambda name, shape: interpolate, raising=False)
    result, _ = loop.DaSiWa_SeamlessLoop().execute(images, 'rife.safetensors', True)
    assert torch.equal(result[0], result[-1])
    assert result.shape[0] >= 20
    assert result.shape[1:] == images.shape[1:]
    assert result.dtype == images.dtype
    assert calls and all(0 < t < 1 for t in calls)
    assert any(torch.equal(result[0], frame) for frame in images)
    assert torch.isfinite(result).all()
    assert torch.equal(images[0], torch.zeros_like(images[0]))


def test_schema_native_combo_and_registration(monkeypatch):
    loop = load_loop()
    import folder_paths
    monkeypatch.setattr(folder_paths, 'get_filename_list', lambda category: ['rife.safetensors', 'legacy.pth'])
    schema = loop.DaSiWa_SeamlessLoop.INPUT_TYPES()
    assert set(schema['required']) == {'images'}
    assert schema['optional']['model_name'][0] == ['rife.safetensors']
    assert not schema['optional']['model_name'][1].get('forceInput', False)
    assert not schema['optional']['model_name'][1].get('socketless', False)
    assert 'interp_model' not in schema['optional']
    frontend = (ROOT / 'js/seamless_loop.js').read_text()
    assert '["model_name", "exact_endpoint"]' not in frontend
    source = (ROOT / '__init__.py').read_text()
    assert '"DaSiWa_SeamlessLoop": DaSiWa_SeamlessLoop' in source


@pytest.mark.parametrize('exact', [True, False])
@pytest.mark.parametrize('cyclic', [True, False])
def test_actual_native_rife_inference_odd_size(exact, cyclic):
    import os
    import folder_paths
    checkpoint = Path(os.environ.get('DASIWA_LOOP_TEST_MODEL', '/data/GitHub/AI-Ressources/models/frame_interpolation/rife_v4.26.safetensors'))
    if not checkpoint.is_file():
        pytest.skip('Set DASIWA_LOOP_TEST_MODEL to an installed native RIFE safetensors checkpoint')
    folder_paths.add_model_folder_path('frame_interpolation', str(checkpoint.parent))
    loop = load_loop()
    # A textured moving object, not a fabricated model output.
    y, x = torch.meshgrid(torch.arange(67), torch.arange(73), indexing='ij')
    images = []
    for i in range(24):
        frame = torch.zeros(67, 73, 3)
        center = 25 + (8 * math.sin(i / 24 * 2 * math.pi) if cyclic else .7*i)
        mask = (x - center).square() + (y-33).square() < 100
        frame[..., 0] = mask.float() * .8
        frame[..., 1] = ((x % 9) / 40).float()
        frame[..., 2] = ((y % 11) / 40).float()
        images.append(frame)
    images = torch.stack(images)
    result, _ = loop.DaSiWa_SeamlessLoop().execute(images, model_name=checkpoint.name, exact_endpoint=exact)
    assert torch.isfinite(result).all()
    assert 0 <= result.min() <= result.max() <= 1
    assert result.shape[1:] == images.shape[1:]
    assert result.dtype == images.dtype
    assert torch.equal(result[0], result[-1]) == exact
    start, end, k, _ = loop._analyze(loop._preview(images))
    middle = end - start - 2*k
    assert torch.equal(result[-1 if not exact else -2], images[start+k-1])
    assert torch.equal(result[0], images[start+k])
    if middle:
        assert torch.equal(result[middle-1], images[end-k-1])
        assert torch.equal(result[middle], images[end-k])
    # Verify the circular playback step has exactly the source's adjacent-frame motion.
    last = result[-1 if not exact else -2]
    assert torch.equal(result[0] - last, images[start+k] - images[start+k-1])
    if not cyclic:
        raw_wrap = (images[0] - images[-1]).square().mean()
        closed_wrap = (result[0] - last).square().mean()
        assert closed_wrap < raw_wrap
        transition = result[middle:middle+k]
        transition_steps = (transition[1:] - transition[:-1]).square().mean((1, 2, 3))
        assert transition_steps.max() < raw_wrap, 'Interpolated transition must reduce the original hard seam on this fixture'


@pytest.mark.parametrize('n', [1, 2, 3, 5, 6, 7])
def test_short_batches_are_valid_and_do_not_mutate_input(monkeypatch, n):
    loop = load_loop()
    images = torch.stack([torch.full((1, 1, 3), i / max(n, 1)) for i in range(n)])
    original = images.clone()
    monkeypatch.setattr(loop, '_load_interpolator', lambda *args: lambda a, b, t: torch.lerp(a, b, t))
    result, _ = loop.DaSiWa_SeamlessLoop().execute(images, 'rife.safetensors', True)
    assert torch.equal(result[0], result[-1])
    assert torch.equal(images, original)
    assert torch.isfinite(result).all()


def test_invalid_pixels_are_rejected_before_loading_model():
    loop = load_loop()
    images = torch.zeros(4, 8, 8, 3)
    images[1, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='NaN'):
        loop.DaSiWa_SeamlessLoop().execute(images, 'missing.safetensors')


def test_already_closed_batch_is_not_reinterpolated():
    loop = load_loop()
    images = torch.cat([torch.rand(5, 8, 8, 3), torch.zeros(1, 8, 8, 3)])
    images[-1] = images[0]
    result, _ = loop.DaSiWa_SeamlessLoop().execute(images, 'missing.safetensors', False)
    assert torch.equal(result, images[:-1])
    result, _ = loop.DaSiWa_SeamlessLoop().execute(images, 'missing.safetensors', True)
    assert torch.equal(result, images)


def test_interpolator_keeps_patcher_alive_and_handles_film_alignment(monkeypatch):
    import weakref
    import folder_paths
    import comfy.model_management as mm
    from comfy_extras.nodes_frame_interpolation import FrameInterpolationModelLoader
    loop = load_loop()
    refs = []
    class FilmLikeModel:
        def memory_used_forward(self, shape, dtype):
            return 1
        def __call__(self, a, b, timestep):
            assert a.shape[-2:] == (128, 128)
            return torch.lerp(a, b, timestep)
    class Patcher:
        load_device = torch.device('cpu')
        model = FilmLikeModel()
        def model_dtype(self):
            return torch.float32
    def load(name):
        patcher = Patcher()
        refs.append(weakref.ref(patcher))
        return (patcher,)
    monkeypatch.setattr(folder_paths, 'get_full_path_or_raise', lambda *args: 'fake.safetensors')
    monkeypatch.setattr(FrameInterpolationModelLoader, 'execute', load)
    monkeypatch.setattr(mm, 'load_models_gpu', lambda *args, **kwargs: None)
    interpolate = loop._load_interpolator('film.safetensors', (8, 67, 73, 3))
    assert refs[0]() is not None, 'Native patcher must survive through inference'
    result = interpolate(torch.zeros(67, 73, 3), torch.ones(67, 73, 3), .5)
    assert torch.equal(result, torch.full_like(result, .5))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU-resident input requires CUDA')
def test_gpu_input_obeys_native_cpu_intermediate_policy(monkeypatch):
    import comfy.model_management as mm
    loop = load_loop()
    images = torch.linspace(0, 1, 24, device='cuda')[:, None, None, None].expand(24, 8, 8, 3).clone()
    monkeypatch.setattr(mm, 'intermediate_device', lambda: torch.device('cpu'))
    monkeypatch.setattr(loop, '_load_interpolator', lambda *args: lambda a, b, t: torch.lerp(a, b, t))
    result, _ = loop.DaSiWa_SeamlessLoop().execute(images, 'rife.safetensors', True)
    assert result.device.type == 'cpu'
    assert result.dtype == images.dtype
    assert torch.equal(result[0], result[-1])


def test_model_discovery_and_loading_use_native_interpolation_paths(monkeypatch, tmp_path):
    import os
    import folder_paths
    checkpoint = Path(os.environ.get('DASIWA_LOOP_TEST_MODEL', '/data/GitHub/AI-Ressources/models/frame_interpolation/rife_v4.26.safetensors'))
    if not checkpoint.is_file():
        pytest.skip('Set DASIWA_LOOP_TEST_MODEL to a native RIFE checkpoint')
    interpolation = tmp_path / 'frame_interpolation'
    upscale = tmp_path / 'external_upscale_models'
    interpolation.mkdir()
    upscale.mkdir()
    (interpolation / 'rife.safetensors').symlink_to(checkpoint)
    monkeypatch.setattr(folder_paths, 'folder_names_and_paths', {
        'frame_interpolation': ([str(interpolation)], {'.safetensors'}),
        'upscale_models': ([str(upscale)], {'.safetensors'}),
    })
    monkeypatch.setattr(folder_paths, 'filename_list_cache', {})
    loop = load_loop()
    schema = loop.DaSiWa_SeamlessLoop.INPUT_TYPES()
    assert 'rife.safetensors' in schema['optional']['model_name'][0]
    assert loop.DaSiWa_SeamlessLoop.IS_CHANGED(None, 'rife.safetensors') == (checkpoint.stat().st_mtime_ns, checkpoint.stat().st_size)
    with torch.inference_mode():
        interpolate = loop._load_interpolator('rife.safetensors', (12, 67, 73, 3))
        result = interpolate(torch.zeros(67, 73, 3), torch.ones(67, 73, 3), .5)
    assert result.shape == (67, 73, 3)
    assert torch.isfinite(result).all()


def test_exact_endpoint_defaults_off(monkeypatch):
    loop = load_loop()
    assert loop.DaSiWa_SeamlessLoop.INPUT_TYPES()['optional']['exact_endpoint'][1]['default'] is False
    images = torch.linspace(0, 1, 24)[:, None, None, None].expand(24, 8, 8, 3).clone()
    monkeypatch.setattr(loop, '_load_interpolator', lambda *args: lambda a, b, t: torch.lerp(a, b, t))
    default, _ = loop.DaSiWa_SeamlessLoop().execute(images, 'rife.safetensors')
    explicit, _ = loop.DaSiWa_SeamlessLoop().execute(images, 'rife.safetensors', False)
    exact, _ = loop.DaSiWa_SeamlessLoop().execute(images, 'rife.safetensors', True)
    assert torch.equal(default, explicit)
    assert torch.equal(exact[:-1], default)
    assert torch.equal(exact[-1], default[0])
    assert not torch.equal(default[0], default[-1])


def test_visible_name_is_seamless_loop():
    assert '\"DaSiWa_SeamlessLoop\": \"Seamless Loop\"' in (ROOT / '__init__.py').read_text()

