"""CPU contract tests for DaSiWa H3 Hi-Res Regenerate; fakes test wiring, not visual quality."""
import importlib.util
import inspect
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'ComfyUI'))
import comfy.cli_args
comfy.cli_args.args.cpu = True
import comfy.lora
import comfy.model_base
import comfy.model_management as mm
import folder_paths
from comfy.nested_tensor import NestedTensor

PACKAGE = '_dasiwa_h3_regenerate'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / 'nodes')]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(PACKAGE + '.node', ROOT / 'nodes/nodes_minimax_h3_regenerate.py')
r = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = r
spec.loader.exec_module(r)
u = sys.modules[PACKAGE + '.nodes_minimax_h3_tiled_upscale']
helpers = __import__(PACKAGE + '.h3_tiled_sampling', fromlist=['*'])
upscale = __import__(PACKAGE + '.h3_latent_upscale', fromlist=['*'])

LAYER = 'diffusion_model.blocks.0.attn.out_proj'
NATIVE_LORA = {f'{LAYER}.lora_A.weight': torch.ones(2, 4), f'{LAYER}.lora_B.weight': torch.ones(4, 2)}


class TestH3Base:
    pass


class FakePatcher:
    def __init__(self):
        self.model = TestH3Base()
        self.model_options = {'transformer_options': {}}
        self.patches = []
        self.clones = []

    def model_size(self): return 0
    def loaded_size(self): return 0
    def current_loaded_device(self): return torch.device('cpu')
    def set_model_unet_function_wrapper(self, wrapper): self.model_options['model_function_wrapper'] = wrapper

    def clone(self):
        cloned = FakePatcher()
        cloned.model = self.model
        cloned.patches = list(self.patches)
        self.clones.append(cloned)
        return cloned

    def add_patches(self, patches, strength):
        self.patches.append((dict(patches), strength))
        return list(patches)


@pytest.mark.parametrize('size,target,expected', [
    ((1344, 768), '2K', [(2560, 1440)]),          # the reference draft
    ((1344, 768), '1080P', [(1920, 1088)]),
    ((1344, 768), '4K', [(2560, 1440), (3840, 2176)]),
    ((1024, 576), '2K', [(2560, 1440)]),
    ((768, 1344), '2K', [(1440, 2560)]),          # portrait keeps its orientation
    ((768, 768), '2K', [(1440, 1440)]),           # non-16:9 keeps its aspect, no stretch
    ((2560, 1440), '4K', [(3840, 2176)]),         # no same-size intermediate pass
])
def test_stages_follow_reference_grid(size, target, expected):
    assert r.regenerate_stages(*size, target) == expected


def test_sigmas_are_the_published_three_step_schedule():
    assert r.SIGMAS == (0.909375, 6 / 7, 2 / 3, 0.0)


def test_diffusers_keys_are_detected():
    assert r.is_diffusers_lora({'transformer.transformer_blocks.0.attn.to_q.lora_A.weight': 0})
    assert not r.is_diffusers_lora(NATIVE_LORA)


@pytest.fixture
def node(monkeypatch):
    monkeypatch.setattr(mm, 'get_torch_device', lambda: torch.device('cpu'))
    monkeypatch.setattr(mm, 'get_free_memory', lambda *a, **k: 64 * 1024**3)
    monkeypatch.setattr(mm, 'unload_model_and_clones', lambda *a, **k: None)
    monkeypatch.setattr(comfy.model_base, 'MiniMaxH3', TestH3Base)
    monkeypatch.setattr(comfy.lora, 'model_lora_keys_unet', lambda model, key_map: {LAYER: LAYER + '.weight'})
    monkeypatch.setattr(helpers, 'plan_tiles', lambda *a, **k: dict(tile_width=4096, tile_height=4096, overlap=0,
                        chunk_tokens=10, temporal_overlap_tokens=5, explanation='TEST plan'))
    instance = r.DaSiWaH3HiResRegenerate()
    instance.lora = dict(NATIVE_LORA)
    monkeypatch.setattr(instance, 'load_lora', lambda name: instance.lora)
    return instance


def latent(t=22, h=48, w=84):
    return {'samples': NestedTensor((torch.randn(1, 24, t, h, w), torch.randn(1, 32, 2, 120)))}


def test_original_target_refines_at_input_size_without_upscale(node, monkeypatch):
    events, calls = [], []
    monkeypatch.setattr(r, 'make_telemetry', lambda: events.append)
    monkeypatch.setattr(upscale, 'H3LatentUpscaler', lambda *a, **k: pytest.fail('loaded unused upscaler'))
    monkeypatch.setattr(u, 'resize_video', lambda *a: pytest.fail('resized Original target'))
    def sample(working, positive, negative, cfg, samples, mask, noise, sampler, sigmas, seed):
        calls.append((working, cfg, sigmas))
        return samples
    monkeypatch.setattr(u, 'sample_chunk', sample)
    source, model = latent(2, 6, 10), FakePatcher()
    out, report = node.regenerate(model, [], source, 'regen.safetensors',
                                  target='Original', upscale_model='missing-unused.safetensors')
    assert out['samples'].tensors[0].shape == source['samples'].tensors[0].shape
    assert out['samples'].tensors[1] is source['samples'].tensors[1]
    assert model.patches == [] and calls
    assert all(cfg == 1.0 and torch.equal(sigmas, torch.tensor(r.SIGMAS)) for _,cfg,sigmas in calls)
    assert [e['phase'] for e in events] == ['lora','lora','refine','refine']
    assert all(e['width'] == 160 and e['height'] == 96 for e in events if e['phase'] == 'refine')
    assert r.regenerate_size(160,96,'Original') == (160,96)
    assert r.regenerate_stages(160,96,'Original') == [(160,96)]
    assert '160x96 → 160x96' in report


def test_learned_upscaler_still_runs_at_same_size(node, monkeypatch):
    stages, name = [], 'up.safetensors'
    monkeypatch.setattr(upscale, 'upscale_model_names', lambda: [name])
    class Backend:
        temporal_halo, sharpness_tiers, dtype = 0, False, torch.float32
        def __init__(self, *a, **k): pass
        def upscale(self, video, h, w):
            stages.append((h, w))
            return video
        def close(self): pass
    monkeypatch.setattr(upscale, 'H3LatentUpscaler', Backend)
    monkeypatch.setattr(u, 'sample_chunk', lambda working, positive, negative, cfg, samples, *a: samples)
    u.DaSiWaH3TiledUpscale().upscale(FakePatcher(), [], latent(2, 6, 10), upscale_model=name, steps=3, denoise=1.0,
                                     seed=0, cfg=1.0, sampler_name='euler', sigmas=torch.tensor(r.SIGMAS),
                                     sizes=[(160, 96)])
    assert stages == [(6, 10)]


def test_pinned_sigmas_euler_cfg1_and_lora_only_on_clone(node, monkeypatch):
    calls, stages, events = [], [], []
    monkeypatch.setattr(r, 'make_telemetry', lambda: events.append, raising=False)
    name = 'minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors'
    monkeypatch.setattr(upscale, 'upscale_model_names', lambda: [name])

    class Backend:
        temporal_halo, sharpness_tiers, dtype = 1, False, torch.float32
        def __init__(self, *a, **k): pass
        def upscale(self, video, h, w):
            stages.append((h, w))
            return u.resize_video(video, h, w)
        def close(self): pass
    monkeypatch.setattr(upscale, 'H3LatentUpscaler', Backend)

    def sample(working, positive, negative, cfg, samples, mask, noise, sampler, sigmas, seed):
        calls.append((working, negative, cfg, sampler, sigmas))
        return samples
    monkeypatch.setattr(u, 'sample_chunk', sample)
    import comfy.samplers
    euler = comfy.samplers.sampler_object('euler')
    monkeypatch.setattr(comfy.samplers, 'sampler_object', lambda n: euler if n == 'euler' else pytest.fail(n))
    model = FakePatcher()
    source = latent()
    out, plan = node.regenerate(model, [[torch.zeros(1, 3, 4), {}]], source, 'regen.safetensors',
                                strength=0.8, upscale_model=name, target='4K')
    assert model.patches == []                       # caller's generation chain stays unpatched
    working = calls[0][0]
    assert working.patches[0][1] == 0.8 and set(working.patches[0][0]) == {LAYER + '.weight'}
    assert all(torch.equal(c[4], torch.tensor(r.SIGMAS)) for c in calls)
    assert all(c[1] is None and c[2] == 1.0 and c[3] is euler for c in calls)
    assert stages[:2] == [(90, 160), (136, 240)]     # 4K via 2K inside each temporal window
    assert [(e['phase'], e['outcome']) for e in events[:2]] == [('lora', 'started'), ('lora', 'completed')]
    up = [e for e in events if e['phase'] == 'latent-upscale']
    assert [(e['pass'], e['width'], e['height'], e['outcome']) for e in up[:4]] == [
        (1, 2560, 1440, 'started'), (1, 2560, 1440, 'completed'),
        (2, 3840, 2176, 'started'), (2, 3840, 2176, 'completed')]
    refine = [e for e in events if e['phase'] == 'refine']
    assert len(refine) == len(calls) * 2
    assert all(e['steps'] == 3 and e['cfg'] == 1.0 and e['chunk'] <= e['chunks'] for e in refine)
    assert all(e['elapsed_ms'] >= 0 for e in events)
    assert all(e['spatial_tiling'] and not e['spatial_split'] for e in refine)
    assert all('regen.safetensors' not in str(e) for e in events)
    assert out['samples'].tensors[0].shape == (1, 24, 22, 136, 240)
    assert out['samples'].tensors[1] is source['samples'].tensors[1]
    assert '1344x768 → 2560x1440 → 3840x2176' in plan and 'sigmas=[0.909375, 0.857143, 0.666667, 0]' in plan
    assert 'lora=regen.safetensors (1 layers, strength 0.8)' in plan


@pytest.mark.parametrize('lora,match', [
    ({'transformer.transformer_blocks.0.attn.to_q.lora_A.weight': torch.ones(2, 4)}, 'Diffusers keys'),
    ({'unet.foo.lora_A.weight': torch.ones(2, 4), 'unet.foo.lora_B.weight': torch.ones(4, 2)}, 'matches no layer'),
])
def test_unusable_loras_fail_before_sampling(node, monkeypatch, lora, match):
    node.lora = lora
    monkeypatch.setattr(u, 'sample_chunk', lambda *a: pytest.fail('sampled'))
    with pytest.raises(ValueError, match=match):
        node.regenerate(FakePatcher(), [], latent(), 'x.safetensors')


def test_non_native_model_masks_and_controlnet_are_rejected(node, monkeypatch):
    monkeypatch.setattr(u, 'sample_chunk', lambda *a: pytest.fail('sampled'))
    other = FakePatcher()
    other.model = object()
    with pytest.raises(ValueError, match='native MiniMax H3'):
        node.regenerate(other, [], latent(), 'x')
    masked = latent(2, 4, 4)
    masked['noise_mask'] = torch.ones(1)
    with pytest.raises(ValueError, match='Masked/inpaint'):
        node.regenerate(FakePatcher(), [], masked, 'x', upscale_model='interpolation')
    with pytest.raises(ValueError, match='ControlNet'):
        node.regenerate(FakePatcher(), [[torch.zeros(1), {'control': object()}]], latent(2, 4, 4), 'x',
                        upscale_model='interpolation')


@pytest.mark.parametrize('phase_name', ['lora', 'latent-upscale', 'refine'])
@pytest.mark.parametrize('cancelled', [False, True])
def test_phase_failure_is_reported_once_and_original_error_is_preserved(node, monkeypatch, phase_name, cancelled):
    events = []
    monkeypatch.setattr(r, 'make_telemetry', lambda: events.append)
    monkeypatch.setattr(r, 'regenerate_stages', lambda *a: [(128, 64)])
    error = mm.InterruptProcessingException() if cancelled else RuntimeError('private model path and prompt')

    def fail(*args, **kwargs):
        raise error

    if phase_name == 'lora':
        monkeypatch.setattr(node, 'load_lora', fail)
    elif phase_name == 'latent-upscale':
        monkeypatch.setattr(u, 'resize_video', fail)
    else:
        monkeypatch.setattr(u, 'sample_chunk', fail)
    with pytest.raises(type(error)) as caught:
        node.regenerate(FakePatcher(), [], latent(2, 4, 4), 'private.safetensors')
    assert caught.value is error
    affected = [e for e in events if e['phase'] == phase_name]
    assert [e['outcome'] for e in affected] == ['started', 'cancelled' if cancelled else 'failed']
    assert all('private' not in str(e) for e in events)
    if phase_name != 'refine':
        assert not any(e['phase'] == 'refine' for e in events)


def test_continuity_prefix_is_not_reported_as_refined(node, monkeypatch):
    events, sampled = [], []
    monkeypatch.setattr(r, 'make_telemetry', lambda: events.append)
    monkeypatch.setattr(r, 'regenerate_stages', lambda *a: [(128, 64)])
    continuity_helpers = __import__(PACKAGE + '.h3_upscale_continuity', fromlist=['*'])
    monkeypatch.setattr(continuity_helpers, 'continuity_plan', lambda *a: {
        'refine_start_token': 10, 'explanation': 'test continuity'})
    monkeypatch.setattr(continuity_helpers, 'align_continuity_conditioning', lambda positive, *a, **kw: positive)
    monkeypatch.setattr(u, 'continuity_refine_mask', lambda part, *a, **kw: torch.ones_like(part))
    def sample(model, pos, neg, cfg, samples, *args):
        sampled.append(samples)
        return samples
    monkeypatch.setattr(u, 'sample_chunk', sample)
    source = latent(22, 4, 4)
    out, _ = node.regenerate(FakePatcher(), [], source, 'private.safetensors', continuity_context={})
    starts = [e for e in events if e['phase'] == 'refine' and e['outcome'] == 'started']
    assert starts and all(e['chunk'] > 1 for e in starts)
    assert len(starts) == len(sampled)
    assert torch.equal(out['samples'].tensors[0][:, :, :10],
                       u.resize_video(source['samples'].tensors[0], 4, 8)[:, :, :10])


def test_telemetry_failure_cannot_change_processing(node, monkeypatch):
    def broken(data):
        raise RuntimeError('diagnostic transport failed')
    monkeypatch.setattr(r, 'make_telemetry', lambda: broken)
    monkeypatch.setattr(r, 'regenerate_stages', lambda *a: [(128, 64)])
    monkeypatch.setattr(u, 'sample_chunk', lambda model, pos, neg, cfg, samples, *args: samples)
    source = latent(2, 4, 4)
    out, _ = node.regenerate(FakePatcher(), [], source, 'private.safetensors')
    assert out['samples'].tensors[0].shape == (1, 24, 2, 4, 8)
    assert out['samples'].tensors[1] is source['samples'].tensors[1]


def test_phase_transport_is_directed_scoped_private_and_best_effort(monkeypatch, caplog):
    import uuid
    from comfy_execution.utils import CurrentNodeContext
    import server
    telemetry = __import__(PACKAGE + '.h3_regenerate_telemetry', fromlist=['*'])
    sent = []
    instance = types.SimpleNamespace(client_id='owned-client', send_sync=lambda *args: sent.append(args))
    monkeypatch.setattr(server.PromptServer, 'instance', instance, raising=False)
    prompt = str(uuid.uuid4())
    caplog.set_level('INFO')
    with CurrentNodeContext(prompt_id=prompt, node_id='U'):
        emit = telemetry.make_telemetry()
        with telemetry.phase(emit, 'refine', steps=3, cfg=1.0):
            pass
    assert len(sent) == 2
    assert all(event == 'dasiwa_h3_regenerate_stage' and sid == 'owned-client' for event, data, sid in sent)
    assert all(data['prompt_id'] == prompt and data['node'] == 'U' for event, data, sid in sent)
    assert [data['outcome'] for event, data, sid in sent] == ['started', 'completed']
    assert 'phase=refine' in caplog.text and prompt in caplog.text

    # No active executor context means local logs only, not a broadcast.
    with telemetry.phase(telemetry.make_telemetry(), 'lora'):
        pass
    assert len(sent) == 2
    instance.client_id = None
    with CurrentNodeContext(prompt_id=prompt, node_id='U'):
        with telemetry.phase(telemetry.make_telemetry(), 'lora'):
            pass
    assert len(sent) == 2

    def broken(*args):
        raise RuntimeError('private credentials and model path')
    instance.client_id = 'owned-client'
    instance.send_sync = broken
    with CurrentNodeContext(prompt_id=prompt, node_id='U'):
        with telemetry.phase(telemetry.make_telemetry(), 'lora'):
            pass
    assert caplog.text.count('telemetry transport unavailable') == 1
    assert 'private credentials' not in caplog.text


def test_schema_and_defaults(monkeypatch):
    monkeypatch.setattr(folder_paths, 'get_filename_list', lambda kind: ['regen.safetensors'])
    monkeypatch.setattr(upscale, 'upscale_model_names',
                        lambda: ['x_2d_conv_v1.safetensors', 'H3/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors'])
    schema = r.DaSiWaH3HiResRegenerate.INPUT_TYPES()
    assert list(schema['required']) == ['model', 'conditioning', 'latent', 'lora_name', 'strength',
                                        'upscale_model', 'target', 'seed']
    assert list(schema['optional']) == ['director_guide', 'vae', 'start_image', 'noise',
                                        'continuity_context', 'spatial_tiling']
    assert schema['required']['upscale_model'][1]['default'].endswith('3d_conv_v1_bf16.safetensors')
    assert schema['required']['target'] == (['1080P', '2K', '4K', 'Original'], schema['required']['target'][1])
    assert schema['required']['target'][1]['default'] == '2K'
    params = inspect.signature(r.DaSiWaH3HiResRegenerate.regenerate).parameters
    assert params['target'].default == '2K' and params['strength'].default == 1.0
