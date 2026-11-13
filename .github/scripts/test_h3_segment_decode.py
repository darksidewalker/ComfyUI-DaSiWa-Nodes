"""G0: actual Core temporal stitching and full BigVGAN topology, CPU only."""
from test_h3_segment_backend import core, torch, importlib
from types import SimpleNamespace, ModuleType
import ast
from pathlib import Path
import sys
import comfy
import pytest

# Load the installed Core class without trusting sys.modules['nodes'].
_core_tree = ast.parse((Path(comfy.cli_args.__file__).resolve().parent.parent / 'nodes.py').read_text(encoding='utf-8'))
_core_class = next(n for n in _core_tree.body if isinstance(n, ast.ClassDef) and n.name == 'VAEDecode')
_core_namespace = {}
exec(compile(ast.Module(body=[_core_class], type_ignores=[]), '<Core VAEDecode>', 'exec'), _core_namespace)
CoreVAEDecode = _core_namespace['VAEDecode']
policy = importlib.import_module('h3_segment_test.nodes.h3_continuity.segment_policy')

def video_decoder():
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
    model = MiniMaxH3VideoVAE(encoder=False,decoder=False)
    model.vae_ratio = 1
    model.decoder = SimpleNamespace(out_channels=3)
    # Deliberately nonlocal within each native chunk: detects missing blend context.
    model._adaptive_decode = lambda z: (z[:,:3].repeat_interleave(4,2) + z.mean(2,keepdim=True)[:,:3]) / 10
    model._finalize_pixels = lambda x: x
    return model

@pytest.mark.parametrize('source', [5,22,39,56,73,124,141,158,175,243,362,515])
@pytest.mark.parametrize('added',[17,119,357])
def test_render_window_matches_global_decode_crop(source,added):
    p = policy.render_layout(source,added)
    torch.manual_seed(0)
    z = torch.rand(1,24,(source+added-5)//17*5+2,1,1)
    vae = video_decoder()
    full = vae.decode_temporal(z)
    local = vae.decode_temporal(z[:,:,p['video_start']:])
    assert torch.equal(full[:,:,source:],local[:,:,p['crop_start_frame']:])
    assert p['frame_offset'] % 51 == 0
    assert p['crop_start_frame'] <= 152

@pytest.mark.parametrize('source,added',[(124,119),(243,119),(260,17),(277,357)])
def test_segment_decode_crops_owned_frame_and_sample_outputs(source,added):
    mod = importlib.import_module('h3_segment_test.nodes.h3_continuity.segment_nodes')
    from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE, BigVGAN
    audio_model = MiniMaxH3AudioVAE.__new__(MiniMaxH3AudioVAE)
    torch.nn.Module.__init__(audio_model)
    audio_model.samples_per_latent = 800
    audio_model.decoder = BigVGAN(num_mels=2,upsample_initial_channel=128)
    p = policy.render_layout(source,added)
    v = torch.zeros(1,24,(p['render_frames']-5)//17*5+2,1,1)
    a = torch.zeros(1,32,2,round(p['render_frames']*40/24))
    from comfy.nested_tensor import NestedTensor
    images = torch.arange(p['render_frames']).float()[:,None,None,None].expand(-1,1,1,3).clone()
    waveform = torch.arange(a.shape[-1]*800).float()[None,:,None].expand(1,-1,2).clone()
    vae = SimpleNamespace(first_stage_model=video_decoder(),decode=lambda z:images)
    avae = SimpleNamespace(first_stage_model=audio_model,audio_sample_rate_output=32000,decode=lambda z:waveform)
    context = dict(render_kind='segment',operation='continue',source_frames=source,extension_frames=added,render_plan=p)
    out,audio = mod.DaSiWaH3ContinuitySegmentDecode().decode({'samples':NestedTensor((v,a))},vae,avae,context)
    assert out.shape[0] == added and out[0,0,0,0] == source-p['frame_offset']
    assert audio['waveform'].shape == (1,2,p['crop_samples']) and audio['sample_rate'] == 32000
    assert audio['waveform'][0,0,0] == p['crop_start_sample']
    assert audio['h3_segment_interval'] == {'start':(source,24),'end':(source+added,24)}
    assert out.untyped_storage().nbytes() == out.numel()*out.element_size()
    assert audio['waveform'].untyped_storage().nbytes() == audio['waveform'].numel()*audio['waveform'].element_size()

def test_segment_decode_rejects_unvalidated_audio_topology():
    mod = importlib.import_module('h3_segment_test.nodes.h3_continuity.segment_nodes')
    from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE,BigVGAN
    am = MiniMaxH3AudioVAE.__new__(MiniMaxH3AudioVAE)
    torch.nn.Module.__init__(am)
    am.samples_per_latent = 800
    am.decoder = BigVGAN(num_mels=2,upsample_initial_channel=128)
    vae = SimpleNamespace(first_stage_model=video_decoder())
    avae = SimpleNamespace(first_stage_model=am,audio_sample_rate=32000)
    mod.validate_segment_vaes(vae,avae)
    am.decoder.resblocks[0].convs1[0] = torch.nn.Conv1d(64,64,101,padding=50)
    with pytest.raises(ValueError,match='topology'): mod.validate_segment_vaes(vae,avae)

@pytest.mark.parametrize('frames',[22,39])
def test_disabled_capture_decodes_first_take_without_staging(monkeypatch,frames):
    from comfy_extras.nodes_audio import vae_decode_audio
    from test_h3_segment_backend import latent
    mod = importlib.import_module('h3_segment_test.nodes.h3_continuity.segment_nodes')
    monkeypatch.setattr(mod,'ClipStore',lambda: pytest.fail('Disabled capture must not stage'))
    monkeypatch.setattr(mod,'validate_segment_vaes',lambda *a: pytest.fail('Disabled capture must use Core decode'))
    take = latent(frames)
    ctx = {'disabled':True}  # Guide's uncaptured context has no output_mode.
    render,context,ticket = mod.DaSiWaH3ContinuitySegmentPrepare().prepare(take,ctx)
    vae = SimpleNamespace(decode=lambda z:torch.rand(1,frames,2,2,3,generator=torch.Generator().manual_seed(1)))
    avae = SimpleNamespace(decode=lambda z:torch.rand(1,z.shape[-1]*800,2,generator=torch.Generator().manual_seed(2))*4,audio_sample_rate=32000)
    images,audio = mod.DaSiWaH3ContinuitySegmentDecode().decode(render,vae,avae,context)
    assert render is take and context is ctx and ticket == {'disabled':True}
    assert torch.equal(images,CoreVAEDecode().decode(vae,take)[0])
    core_audio = vae_decode_audio(avae,take)
    assert audio['waveform'].shape == (1,2,take['samples'].tensors[1].shape[-1]*800)
    assert torch.equal(audio['waveform'],core_audio['waveform'])
    assert audio['sample_rate'] == core_audio['sample_rate'] == 32000

def test_audio_window_matches_global_decode_crop():
    from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE, BigVGAN
    model = MiniMaxH3AudioVAE.__new__(MiniMaxH3AudioVAE)
    torch.nn.Module.__init__(model)
    # Same seven upsampling stages, kernels, dilations and alias-free activations.
    model.dec_in_proj = torch.nn.Conv1d(32,2,1)
    model.decoder = BigVGAN(num_mels=2,upsample_initial_channel=128)
    model.register_buffer('latents_mean',torch.zeros(32))
    model.register_buffer('latents_std',torch.ones(32))
    torch.manual_seed(2)
    with torch.no_grad():
        for name,param in model.named_parameters():
            param.fill_(0.) if 'alpha' in name or 'beta' in name else param.uniform_(-.02,.02)
        for source in [243,260,277]:
            p = policy.render_layout(source,17)
            z = torch.randn(1,32,2,round((source+17)*40/24))
            full = model.decode(z)
            local = model.decode(z[...,p['audio_start']:])
            start = round(source*32000/24)
            assert torch.allclose(full[...,start:],local[...,start-p['audio_start']*800:],atol=1e-7,rtol=1e-5)

def test_full_mode_prepare_is_append_and_stage(tmp_path,monkeypatch):
    from test_h3_segment_backend import root, context, latent
    mod = importlib.import_module('h3_segment_test.nodes.h3_continuity.segment_nodes')
    legacy = importlib.import_module('h3_segment_test.nodes.h3_continuity.nodes')
    store = core.ClipStore(tmp_path)
    root(store)
    monkeypatch.setattr(mod,'ClipStore',lambda:store)
    monkeypatch.setattr(legacy,'ClipStore',lambda:store)
    ctx = {**context(),'output_mode':'full','operation':'continue','overlap_frames':22,'extension_frames':119}
    _,sampled,layout = core.prepare_continuation(latent(124),store.inspect('s','root'),dict(mode='REF2VA',width=64,height=64),core.parse_settings({**ctx,'version':2,'continuation_prompt':'new'}))
    ctx['layout'] = layout
    for tensor in sampled['samples'].tensors: tensor.fill_(5.)
    expected,_ = legacy.DaSiWaH3ContinuityAppend().commit(sampled,ctx)
    render,out_ctx,ticket = mod.DaSiWaH3ContinuitySegmentPrepare().prepare(sampled,ctx)
    assert out_ctx is ctx and 'render_kind' not in out_ctx
    for a,b in zip(render['samples'].tensors,expected['samples'].tensors): assert torch.equal(a,b)
    assert store.inspect('s',ticket['clip_id'],ready=False).get('schema') != 2

@pytest.mark.parametrize('shape', [(39,8,8,3), (1,39,8,8,3), (2,39,8,8,3)])
def test_full_mode_decode_matches_core_decoders(monkeypatch,shape):
    monkeypatch.setitem(sys.modules,'nodes',ModuleType('nodes'))
    from test_h3_segment_backend import latent
    from comfy_extras.nodes_audio import vae_decode_audio
    mod = importlib.import_module('h3_segment_test.nodes.h3_continuity.segment_nodes')
    monkeypatch.setattr(mod,'validate_segment_vaes',lambda *a: pytest.fail('Full mode must not require segment VAE checks'))
    take = latent(39)
    torch.manual_seed(0)
    def decode_video(z):
        assert z is take['samples'].tensors[0]
        return torch.rand(*shape,generator=torch.Generator().manual_seed(1))
    vae = SimpleNamespace(decode=decode_video)
    avae = SimpleNamespace(decode=lambda z: torch.rand(1,z.shape[-1]*800,2,generator=torch.Generator().manual_seed(2))*4,audio_sample_rate=32000)
    images,audio = mod.DaSiWaH3ContinuitySegmentDecode().decode(take,vae,avae,{'output_mode':'full','operation':'new'})
    assert torch.equal(images,CoreVAEDecode().decode(vae,take)[0])
    core_audio = vae_decode_audio(avae,take)
    assert torch.equal(audio['waveform'],core_audio['waveform']) and audio['sample_rate'] == core_audio['sample_rate']

@pytest.mark.parametrize('output_mode',['full','segment'])
def test_segment_decode_names_swapped_vae_slots(output_mode):
    from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
    mod = importlib.import_module('h3_segment_test.nodes.h3_continuity.segment_nodes')
    video = SimpleNamespace(first_stage_model=video_decoder())
    audio = SimpleNamespace(first_stage_model=MiniMaxH3AudioVAE.__new__(MiniMaxH3AudioVAE))
    ctx = {'output_mode':output_mode,'render_kind':'segment'} if output_mode == 'segment' else {'output_mode':'full'}
    with pytest.raises(ValueError,match="video VAE is connected to 'audio_vae'"):
        mod.DaSiWaH3ContinuitySegmentDecode().decode(None,video,video,ctx)
    with pytest.raises(ValueError,match="audio VAE is connected to 'vae'"):
        mod.DaSiWaH3ContinuitySegmentDecode().decode(None,audio,audio,ctx)
