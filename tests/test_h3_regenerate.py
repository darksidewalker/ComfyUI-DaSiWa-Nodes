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
    def add_wrapper_with_key(self, kind, key, fn):
        self.__dict__.setdefault('wrappers',{}).setdefault(kind,{}).setdefault(key,[]).append(fn)
    def remove_wrappers_with_key(self, kind, key):
        self.__dict__.setdefault('wrappers',{}).get(kind,{}).pop(key,None)

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


@pytest.mark.parametrize('controls,expected', [
    ({}, r.SIGMAS),
    ({'denoise': 0.5, 'upscale_sharpness': 6}, r.SIGMAS[-3:]),
    ({'denoise': 0.05, 'upscale_sharpness': 6}, r.SIGMAS[-1:]),
    ({'denoise': 0.0, 'upscale_sharpness': 6}, r.SIGMAS[-1:]),
])
def test_original_target_refines_at_input_size_without_upscale(node, monkeypatch, controls, expected):
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
                                  target='Original', upscale_model='missing-unused.safetensors', **controls)
    assert torch.equal(out['samples'].tensors[0], source['samples'].tensors[0])
    assert out['samples'].tensors[1] is source['samples'].tensors[1]
    assert model.patches == [] and bool(calls) == (len(expected) > 1)
    assert all(cfg == 1.0 and torch.equal(sigmas, torch.tensor(expected)) for _,cfg,sigmas in calls)
    assert [e['phase'] for e in events] == ['lora','lora'] + (['refine','refine'] if calls else [])
    assert all(e['width'] == 160 and e['height'] == 96 for e in events if e['phase'] == 'refine')
    assert r.regenerate_size(160,96,'Original') == (160,96)
    assert r.regenerate_stages(160,96,'Original') == [(160,96)]
    assert '160x96 → 160x96' in report and 'sharpness=n/a' in report


@pytest.mark.parametrize('controls,expected', [
    ({}, r.SIGMAS),
    ({'denoise': 0.0}, r.SIGMAS[-1:]),
    ({'denoise': 0.05}, r.SIGMAS[-1:]),
    ({'denoise': 0.2}, r.SIGMAS[-2:]),
    ({'denoise': 0.5}, r.SIGMAS[-3:]),
    ({'denoise': 1.0}, r.SIGMAS),
])
def test_refine_strength_uses_native_schedule_tail_and_preserves_audio(node, monkeypatch, controls, expected):
    from comfy_extras.nodes_custom_sampler import BasicScheduler, SplitSigmasDenoise
    split, sampled, endpoints = [], [], []
    native_split = SplitSigmasDenoise.execute

    def split_sigmas(sigmas, denoise):
        assert sigmas.tolist() == list(r.SIGMAS)
        split.append(denoise)
        return native_split(sigmas, denoise)

    monkeypatch.setattr(SplitSigmasDenoise, 'execute', staticmethod(split_sigmas))
    monkeypatch.setattr(BasicScheduler, 'execute', lambda *a, **k: pytest.fail('replaced pinned schedule'))
    monkeypatch.setattr(r, 'regenerate_stages', lambda *a: [(128, 64)])

    def encode(conditioning, *args, **kwargs):
        endpoints.append(conditioning)
        return conditioning, 'test endpoints'

    monkeypatch.setattr(u, 'encode_endpoints', encode)

    def sample(working, positive, negative, cfg, samples, mask, noise, sampler, sigmas, seed):
        sampled.append(sigmas)
        assert cfg == 1.0 and negative is None and seed == 37
        assert working.patches[0][1] == 0.0  # LoRA strength zero must not disable diffusion.
        assert torch.count_nonzero(mask.tensors[1]) == 0
        return NestedTensor((samples.tensors[0] + 1, samples.tensors[1] + 999))

    monkeypatch.setattr(u, 'sample_chunk', sample)
    source, model = latent(2, 4, 4), FakePatcher()
    source['batch_index'] = [3]
    out, report = node.regenerate(model, [], source, 'regen.safetensors', seed=37, strength=0.0, **controls)
    retained = len(expected) - 1
    assert len(sampled) == len(endpoints) == bool(retained)
    assert all(torch.equal(sigmas, torch.tensor(expected)) for sigmas in sampled)
    assert split == ([controls['denoise']] if controls.get('denoise', 1.0) < 1 else [])
    assert model.patches == [] and model.clones[0].patches[0][1] == 0.0
    assert torch.equal(out['samples'].tensors[0],
                       u.resize_video(source['samples'].tensors[0], 4, 8) + bool(retained))
    assert out['samples'].tensors[1] is source['samples'].tensors[1]
    assert out['batch_index'] is source['batch_index']
    assert f'refine_steps={retained}' in report
    if not retained:
        assert 'diffusion skipped (denoise=0)' in report


@pytest.mark.parametrize('controls,tier', [({}, 1)] + [({'upscale_sharpness': tier}, tier) for tier in range(1, 7)])
def test_sharpness_reaches_learned_upscaler_without_diffusion(node, monkeypatch, controls, tier):
    loaded, resized, closed = [], [], []
    name = 'minimax_h3_latent_upscaler_2d_conv_v1.safetensors'
    monkeypatch.setattr(upscale, 'upscale_model_names', lambda: [name])
    monkeypatch.setattr(r, 'regenerate_stages', lambda *a: [(128, 64)])
    monkeypatch.setattr(u, 'sample_chunk', lambda *a: pytest.fail('sampled at denoise=0'))
    monkeypatch.setattr(u, 'encode_endpoints', lambda *a, **k: pytest.fail('encoded endpoints at denoise=0'))

    class Backend:
        temporal_halo, sharpness_tiers, dtype = 0, True, torch.float32
        def __init__(self, selected, device, precision, sharpness):
            loaded.append((selected, device, precision, sharpness))
        def upscale(self, video, h, w):
            resized.append((h, w))
            return u.resize_video(video, h, w)
        def close(self): closed.append(True)

    monkeypatch.setattr(upscale, 'H3LatentUpscaler', Backend)
    source = latent(2, 4, 4)
    out, report = node.regenerate(FakePatcher(), [], source, 'regen.safetensors',
                                  upscale_model=name, denoise=0.0, **controls)
    assert loaded == [(name, torch.device('cpu'), 'auto', tier)]
    assert resized == [(4, 8)] and closed == [True]
    assert torch.equal(out['samples'].tensors[0], u.resize_video(source['samples'].tensors[0], 4, 8))
    assert out['samples'].tensors[1] is source['samples'].tensors[1]
    assert f'sharpness=t{tier}' in report and 'diffusion skipped (denoise=0)' in report


@pytest.mark.parametrize('requested,expected', [(0.0, r.SIGMAS[-1:]), (0.05, r.SIGMAS[-1:]),
                                                 (0.5, r.SIGMAS[-3:]), (1.0, r.SIGMAS)])
def test_wrapper_preserves_positional_inputs_and_passes_controls_to_core(node, monkeypatch, requested, expected):
    model, positive, source = FakePatcher(), [[torch.zeros(1), {'reference': object()}]], latent(2, 4, 4)
    guide, video_vae, first, external_noise, context = {}, object(), torch.zeros(1), object(), {}
    emit = lambda payload: None
    monkeypatch.setattr(r, 'make_telemetry', lambda: emit)
    called = []

    def capture(self, patched, conditioning, latent, *, upscale_model, steps, denoise, seed,
                director_guide, vae, start_image, cfg, sampler_name, noise, continuity_context,
                spatial_tiling, upscale_sharpness, temporal_join, tile_references, tile_memory_estimate,
                weight_residency, max_resident_gb, sigmas, sizes, telemetry):
        called.append(True)
        assert patched is model.clones[0] and patched.patches[0][1] == 0.65 and model.patches == []
        assert conditioning is positive and latent is source
        assert upscale_model == 'up.safetensors' and upscale_sharpness == 6
        assert (temporal_join, tile_references, tile_memory_estimate) == ('crossfade', 'full', True)
        assert (weight_residency, max_resident_gb) == ('auto', 0.0)
        assert steps == max(1, len(expected) - 1) and denoise == (requested if len(expected) > 1 else 0.0)
        assert seed == 43 and cfg == 1.0 and sampler_name == 'euler' and not spatial_tiling
        assert director_guide is guide and vae is video_vae and start_image is first
        assert noise is external_noise and continuity_context is context and telemetry is emit
        assert sigmas == expected and sizes == r.regenerate_stages(64, 64, '4K')
        if requested == 1.0:
            assert sigmas is r.SIGMAS
        return latent, 'core report'

    monkeypatch.setattr(r.DaSiWaH3TiledUpscale, 'upscale', capture)
    out, report = node.regenerate(model, positive, source, 'regen.safetensors', 0.65, 'up.safetensors',
                                  '4K', 43, guide, video_vae, first, external_noise, context, False, requested, 6)
    assert called == [True] and out is source and report.endswith('core report')


@pytest.mark.parametrize('field,value', [
    ('denoise', value) for value in (-0.01, 1.01, float('nan'), float('inf'), -float('inf'),
                                    True, False, '0.5', None, [], {}, 0.5j)
] + [
    ('upscale_sharpness', value) for value in (0, 7, 1.0, 1.5, True, False, '2', None,
                                              float('nan'), float('inf'), [], {})
])
@pytest.mark.parametrize('target', ['2K', 'Original'])
def test_invalid_controls_fail_before_loading_lora(node, monkeypatch, field, value, target):
    monkeypatch.setattr(node, 'load_lora', lambda *a: pytest.fail('loaded refiner before validating controls'))
    with pytest.raises(ValueError, match=field):
        node.regenerate(FakePatcher(), [], latent(2, 4, 4), 'regen.safetensors',
                        target=target, **{field: value})


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
                                        'continuity_context', 'spatial_tiling', 'denoise', 'upscale_sharpness',
                                        'temporal_join', 'tile_references', 'tile_memory_estimate',
                                        'weight_residency', 'max_resident_gb']
    assert schema['optional']['denoise'][:1] == ('FLOAT',)
    assert schema['optional']['denoise'][1]['default'] == 1.0
    assert (schema['optional']['denoise'][1]['min'], schema['optional']['denoise'][1]['max']) == (0.0, 1.0)
    assert schema['optional']['upscale_sharpness'][:1] == ('INT',)
    assert schema['optional']['upscale_sharpness'][1]['default'] == 1
    assert (schema['optional']['upscale_sharpness'][1]['min'], schema['optional']['upscale_sharpness'][1]['max']) == (1, 6)
    for ignored in ('3D', 'interpolation', 'Original'):
        assert ignored in schema['optional']['upscale_sharpness'][1]['tooltip']
    assert schema['required']['upscale_model'][1]['default'].endswith('3d_conv_v1_bf16.safetensors')
    assert schema['required']['target'] == (['1080P', '2K', '4K', 'Original'], schema['required']['target'][1])
    assert schema['required']['target'][1]['default'] == '2K'
    params = inspect.signature(r.DaSiWaH3HiResRegenerate.regenerate).parameters
    assert params['target'].default == '2K' and params['strength'].default == 1.0
    assert list(params) == ['self', 'model', 'conditioning', 'latent', 'lora_name', 'strength', 'upscale_model',
                            'target', 'seed', 'director_guide', 'vae', 'start_image', 'noise',
                            'continuity_context', 'spatial_tiling', 'denoise', 'upscale_sharpness',
                                        'temporal_join', 'tile_references', 'tile_memory_estimate',
                                        'weight_residency', 'max_resident_gb']
    assert params['denoise'].default == 1.0 and params['upscale_sharpness'].default == 1
