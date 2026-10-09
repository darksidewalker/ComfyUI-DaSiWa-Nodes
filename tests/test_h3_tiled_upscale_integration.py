"""Execute the complete node orchestration with explicit sampling test doubles."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent / 'ComfyUI'))
import comfy.cli_args
comfy.cli_args.args.cpu=True
import comfy.model_base
import comfy.model_management as mm
from comfy.nested_tensor import NestedTensor

PACKAGE='_dasiwa_h3_integration'
package=types.ModuleType(PACKAGE)
package.__path__=[str(ROOT/'nodes')]
sys.modules[PACKAGE]=package
spec=importlib.util.spec_from_file_location(PACKAGE+'.node',ROOT/'nodes/nodes_minimax_h3_tiled_upscale.py')
u=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=u
spec.loader.exec_module(u)
helpers=__import__(PACKAGE+'.h3_tiled_sampling',fromlist=['*'])
upscale=__import__(PACKAGE+'.h3_latent_upscale',fromlist=['*'])


class TestH3Base:
    pass


class FakePatcher:
    def __init__(self):
        self.model=TestH3Base()
        self.model_options={'transformer_options':{'sentinel':'preserve'}}
        self.clones=[]
    def model_size(self):
        return 0
    def loaded_size(self):
        return 0
    def current_loaded_device(self):
        return torch.device('cpu')
    def clone(self):
        cloned=FakePatcher()
        self.clones.append(cloned)
        return cloned
    def set_model_unet_function_wrapper(self,wrapper):
        self.model_options['model_function_wrapper']=wrapper


@pytest.fixture
def setup_node(monkeypatch):
    monkeypatch.setattr(mm,'get_torch_device',lambda:torch.device('cpu'))
    monkeypatch.setattr(comfy.model_base,'MiniMaxH3',TestH3Base)
    monkeypatch.setattr(upscale,'upscale_model_names',lambda:[])
    monkeypatch.setattr(helpers,'plan_tiles',lambda *a,**k:dict(tile_width=64,tile_height=64,overlap=32,
                         chunk_tokens=10,temporal_overlap_tokens=5,explanation='TEST forced multiwindow plan'))
    from comfy_extras.nodes_custom_sampler import BasicScheduler
    model=FakePatcher()
    model.schedule_calls=[]
    def schedule_test_double(actual_model,scheduler,steps,denoise):
        assert actual_model is model
        model.schedule_calls.append((scheduler,steps,denoise))
        return (torch.tensor([0.2,0.0]),)
    monkeypatch.setattr(BasicScheduler,'execute',schedule_test_double)
    return model


@pytest.mark.parametrize('spatial,temporal',[(True,True),(False,True),(True,False),(False,False)])
@pytest.mark.parametrize('denoise',[0,0.2])
def test_switches_control_learned_upscale_and_diffusion(setup_node,monkeypatch,spatial,temporal,denoise):
    model=setup_node
    name='test-upscaler.safetensors'
    monkeypatch.setattr(upscale,'upscale_model_names',lambda:[name])
    def plan(shape,*args,**kw):
        assert kw['spatial_tiling'] is spatial
        assert kw['temporal_chunking'] is temporal
        return dict(tile_width=64,tile_height=64,overlap=32 if spatial else 0,
                    chunk_tokens=10 if temporal else shape[2],
                    temporal_overlap_tokens=5 if temporal else 0,explanation='TEST switch plan')
    monkeypatch.setattr(helpers,'plan_tiles',plan)
    lengths,closed,sampled=[],[],[]
    class Backend:
        temporal_halo=1
        dtype=torch.float32
        def __init__(self,*args,**kwargs): pass
        def upscale(self,video,h,w):
            lengths.append(video.shape[2])
            return u.resize_video(video,h,w)
        def close(self): closed.append(True)
    monkeypatch.setattr(upscale,'H3LatentUpscaler',Backend)
    monkeypatch.setattr(mm,'unload_model_and_clones',lambda *a,**k:None)
    def sample(working,positive,negative,cfg,samples,*args):
        assert ('model_function_wrapper' in working.model_options) is spatial
        sampled.append(samples.tensors[0].shape[2])
        return samples
    monkeypatch.setattr(u,'sample_chunk',sample)
    video=torch.randn(1,24,22,2,2)
    audio=torch.randn(1,32,2,120)
    output,report=u.DaSiWaH3TiledUpscale().upscale(model,[],
        {'samples':NestedTensor((video,audio))},upscale_model=name,denoise=denoise,
        sampler=object(),spatial_tiling=spatial,temporal_chunking=temporal)
    assert closed==[True]
    assert len(lengths)>1 if temporal else lengths==[22]
    assert len(sampled)==(len(lengths) if denoise else 0)
    assert output['samples'].tensors[1] is audio
    assert output['samples'].tensors[0].shape==(1,24,22,4,4)
    assert f"spatial_tiling={'on' if spatial else 'off'}" in report
    assert f"temporal_chunking={'on' if temporal else 'off'}" in report
    schema=u.DaSiWaH3TiledUpscale.INPUT_TYPES()['optional']
    assert list(schema)[-2:]==['spatial_tiling','temporal_chunking']
    assert schema['spatial_tiling'][1]['default'] is True
    assert schema['temporal_chunking'][1]['default'] is True


def test_full_node_multichunk_path_preserves_audio_and_upstream_model(setup_node,monkeypatch):
    model=setup_node
    audio=torch.randn(1,32,2,120)
    video=torch.randn(1,24,22,2,2)
    original=video.clone()
    calls=[]
    def sample_test_double(model,positive,negative,cfg,samples,mask,noise_tensor,sampler,sigmas,seed):
        v,a=samples.tensors
        assert cfg==1.0
        torch.testing.assert_close(sigmas,torch.tensor([0.2,0.0]))
        assert torch.count_nonzero(mask.tensors[1])==0
        assert torch.all(mask.tensors[0]==1)
        assert noise_tensor.tensors[0].shape==v.shape
        assert 'model_function_wrapper' in model.model_options
        assert model.model_options['model_function_wrapper'].coordinates == 'local'
        if calls:
            kf=positive[0][1]['minimax_keyframes'][0]
            assert kf['resolved_frame_index']==0
            assert kf['latent'].shape==(1,24,1,4,4)
        calls.append(v.shape[2])
        return NestedTensor((v+0.125,a+999))
    monkeypatch.setattr(u,'sample_chunk',sample_test_double)
    node=u.DaSiWaH3TiledUpscale()
    result,report=node.upscale(model,[[torch.zeros(1,3,4),{}]],
            {'samples':NestedTensor((video,audio)),'tag':'retain'},upscale_model='interpolation',
            sampler=object())
    assert result['samples'].tensors[0].shape==(1,24,22,4,4)
    assert result['samples'].tensors[1] is audio
    assert torch.equal(video,original)
    assert result['tag']=='retain'
    assert calls==[10,10,10,7]
    assert model.schedule_calls==[('simple',1,0.2)]
    assert 'model_function_wrapper' not in model.model_options
    assert all('model_function_wrapper' not in c.model_options for c in model.clones)
    assert model.model_options['transformer_options']=={'sentinel':'preserve'}
    assert 'temporal=4 chunks' in report


def test_zero_denoise_executes_upscale_without_sampler(setup_node,monkeypatch):
    def forbidden(*a,**k):
        raise AssertionError('denoise zero must not sample')
    monkeypatch.setattr(u,'sample_chunk',forbidden)
    video=torch.arange(24*2*2*2).reshape(1,24,2,2,2).float()
    audio=torch.randn(1,32,2,8)
    output,report=u.DaSiWaH3TiledUpscale().upscale(setup_node,[[torch.zeros(1,3,4),{}]],
                   {'samples':NestedTensor((video,audio))},denoise=0,upscale_model='interpolation')
    assert output['samples'].tensors[0].shape==(1,24,2,4,4)
    assert output['samples'].tensors[1] is audio
    assert 'diffusion skipped' in report


def test_cancellation_cleans_cloned_wrapper(setup_node,monkeypatch):
    def interrupted(*a,**k):
        raise RuntimeError('TEST cancellation in sampling')
    monkeypatch.setattr(u,'sample_chunk',interrupted)
    with pytest.raises(RuntimeError,match='TEST cancellation'):
        u.DaSiWaH3TiledUpscale().upscale(setup_node,[[torch.zeros(1,3,4),{}]],
                 {'samples':NestedTensor((torch.zeros(1,24,2,2,2),torch.zeros(1,32,2,8)))},
                 upscale_model='interpolation',sampler=object())
    assert all('model_function_wrapper' not in c.model_options for c in setup_node.clones)


def test_existing_bbaudio_wrapper_is_rejected_not_overwritten(setup_node):
    wrapper=object()
    setup_node.model_options['model_function_wrapper']=wrapper
    with pytest.raises(ValueError,match='unwrapped H3 model'):
        u.DaSiWaH3TiledUpscale().upscale(setup_node,[],{})
    assert setup_node.model_options['model_function_wrapper'] is wrapper


def test_refine_schema_and_callable_defaults_without_external_sigmas():
    import inspect
    schema=u.DaSiWaH3TiledUpscale.INPUT_TYPES()
    assert list(schema['required']) == ['model','conditioning','latent','scale','upscale_model','steps','denoise','seed']
    assert list(schema['optional']) == ['director_guide','vae','start_image','negative','cfg','sampler_name',
        'scheduler','memory_budget_mb','sampler','noise','continuity_context','upscale_precision',
        'continuity_soft_refine','continuity_mask_strength','spatial_tiling','temporal_chunking']
    assert u.DaSiWaH3TiledUpscale.RETURN_TYPES == ('LATENT','STRING')
    assert 'sigmas' not in schema['required']
    assert 'sigmas' not in schema['optional']
    params=inspect.signature(u.DaSiWaH3TiledUpscale.upscale).parameters
    assert 'sigmas' not in params
    for name in ['target_width','target_height']:
        assert name not in schema['required'] and name not in schema['optional']
        assert name not in params
    assert schema['optional']['upscale_precision'][0]==['auto','bf16','fp16','fp32']
    assert params['upscale_precision'].default=='auto'
    for name,expected in [('steps',1),('denoise',0.2),('cfg',1.0)]:
        config=schema['required'].get(name,schema['optional'].get(name))
        assert config[1]['default']==expected
        assert params[name].default==expected


def test_model_selector_has_only_explicit_choices(setup_node,monkeypatch):
    monkeypatch.setattr(upscale,'upscale_model_names',lambda:['h3.safetensors'])
    choices=u.DaSiWaH3TiledUpscale.INPUT_TYPES()['required']['upscale_model'][0]
    assert choices==['interpolation','h3.safetensors']


@pytest.mark.parametrize('choice',['auto','missing.safetensors'])
def test_unknown_checkpoint_never_falls_back_to_interpolation(setup_node,choice):
    latent={'samples':NestedTensor((torch.zeros(1,24,2,2,2),torch.zeros(1,32,2,8)))}
    with pytest.raises(ValueError,match='checkpoint is not available'):
        u.DaSiWaH3TiledUpscale().upscale(setup_node,[],latent,
                                      upscale_model=choice,denoise=0)


@pytest.mark.parametrize('denoise',[0,0.2])
def test_plan_is_printed_with_dasiwa_cli_prefix(setup_node,capsys,denoise,monkeypatch):
    monkeypatch.setattr(u,'sample_chunk',lambda model,positive,negative,cfg,samples,*args: samples)
    video=torch.randn(1,24,2,2,2)
    audio=torch.randn(1,32,2,8)
    _,plan=u.DaSiWaH3TiledUpscale().upscale(setup_node,[],
        {'samples':NestedTensor((video,audio))},upscale_model='interpolation',
        denoise=denoise,sampler=object())
    output=capsys.readouterr().out
    assert '[DaSiWa MiniMaxH3 Enhanced Upscale]' in output
    assert plan in output


@pytest.mark.parametrize('matching_tail',[False,True])
def test_pending_continuity_reserve_bounds_effective_guides(matching_tail):
    from _dasiwa_h3_integration.h3_upscale_continuity import align_continuity_conditioning
    plan=dict(refine_start_token=5,source_tokens=7,frame_offset=17,audio_start=28,source_audio_tokens=37)
    video=torch.randn(1,24,12,8,8)
    audio=torch.randn(1,32,2,65)
    kfs=[dict(resolved_frame_index=0,latent=torch.randn(1,24,2 if matching_tail else 1,2,2))]
    if matching_tail:
        kfs[0]['audio_latent']=torch.randn(1,32,2,9)
    source=[[torch.randn(1,3,32),{'minimax_keyframes':kfs}]]
    counts=u.conditioning_token_counts(source,continuity=plan)
    effective=align_continuity_conditioning(source,video,audio,plan)
    actual=u.conditioning_token_counts(effective)
    assert counts[0]==actual[0]
    assert counts[1]>=actual[1] and counts[2]>=actual[2]
    prepared=helpers.prepare_conditioning(effective,5,12,(8,8),previous_video=video)
    window=u.conditioning_token_counts(prepared)
    assert counts[1]>=window[1] and counts[2]+1>=window[2]
    assert kfs[0]['latent'].shape[-2:]==(2,2)


@pytest.mark.parametrize('gib,policy',[(64,'preferred'),(2,'memory_fallback')])
@pytest.mark.parametrize('soft,strength',[(False,1.0),(True,0.0),(True,1.0)])
def test_real_planner_continuity_prefix_audio_and_local_fallback(setup_node,monkeypatch,gib,policy,soft,strength):
    # Real planner and tail alignment, CPU sampling double; no checkpoint render.
    from _dasiwa_h3_integration.h3_upscale_continuity import align_continuity_conditioning
    continuity=dict(refine_start_token=5,source_tokens=7,frame_offset=17,
                    audio_start=28,source_audio_tokens=37,explanation='TEST active continuity')
    module=types.ModuleType(PACKAGE+'.h3_upscale_continuity')
    module.continuity_plan=lambda *args:continuity
    module.align_continuity_conditioning=align_continuity_conditioning
    monkeypatch.setitem(sys.modules,module.__name__,module)
    # setup_node replaced this module's planner, so load the source independently.
    spec=importlib.util.spec_from_file_location('_actual_h3_planner',ROOT/'nodes/h3_tiled_sampling.py')
    actual=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(actual)
    plans=[]
    def plan(*args,**kw):
        result=actual.plan_tiles(*args,**kw)
        plans.append(result)
        return result
    monkeypatch.setattr(helpers,'plan_tiles',plan)
    monkeypatch.setattr(u,'refinement_memory_budget',lambda *args:(gib*1024**3,False,0))
    calls=[]
    def sample(model,pc,nc,cfg,samples,mask,*args):
        assert model.model_options['model_function_wrapper'].coordinates=='local'
        calls.append(mask.tensors[0].clone())
        v,a=samples.tensors
        return NestedTensor((v+mask.tensors[0]*0.125,a+999))
    monkeypatch.setattr(u,'sample_chunk',sample)
    video=torch.randn(1,24,35,94,68,dtype=torch.float16)
    audio=torch.randn(1,32,2,199,dtype=torch.float16)
    positive=[[torch.zeros(1,3,4),{}]]
    original=video.clone()
    output,report=u.DaSiWaH3TiledUpscale().upscale(setup_node,positive,
        {'samples':NestedTensor((video,audio))},scale=2,continuity_context={'operation':'continue'},
        continuity_soft_refine=soft,continuity_mask_strength=strength,sampler=object())
    result=output['samples'].tensors[0]
    assert plans[0]['layout_policy']==policy
    assert policy in report and 'coordinates=local' in report
    assert result.shape==(1,24,35,188,136)
    assert torch.equal(result[:,:,:5],u.resize_video(video[:,:,:5],188,136))
    assert output['samples'].tensors[1] is audio
    assert torch.equal(video,original) and positive[0][1]=={}
    assert calls and len(calls)>1
    assert all('model_function_wrapper' not in c.model_options for c in setup_node.clones)


def test_continuity_orchestration_freezes_prefix_and_refines_joint_seam(setup_node,monkeypatch):
    # Explicit helper double tests node wiring/masking. Real ClipStore and native
    # timeline math are covered separately by test_h3_upscale_continuity.
    module=types.ModuleType(PACKAGE+'.h3_upscale_continuity')
    plan=dict(refine_start_token=20,source_tokens=22,frame_offset=68,
              audio_start=113,source_audio_tokens=122,explanation='TEST continuity seam')
    module.continuity_plan=lambda video,audio,context: plan
    def budget_plan(*args,**kw):
        assert kw['keyframe_tokens'] == 2
        assert kw['ref_tokens'] == 18
        return dict(tile_width=64,tile_height=64,overlap=32,
                    chunk_tokens=10,temporal_overlap_tokens=5,explanation='TEST continuity reserve')
    monkeypatch.setattr(helpers,'plan_tiles',budget_plan)
    aligned=[]
    def align(conditioning,video,audio,actual_plan,*,inject_tail=True):
        assert actual_plan is plan
        aligned.append(inject_tail)
        if not inject_tail:
            return conditioning
        return [[tokens,{**md,'minimax_keyframes':[
            dict(resolved_frame_index=68,latent=video[:,:,20:22],audio_latent=audio[...,113:122])]}]
            for tokens,md in conditioning]
    module.align_continuity_conditioning=align
    monkeypatch.setitem(sys.modules,module.__name__,module)
    calls=[]
    def sample(model,positive,negative,cfg,samples,mask,noise_tensor,sampler,sigmas,seed):
        v,a=samples.tensors
        calls.append(mask.tensors[0].clone())
        assert not torch.count_nonzero(mask.tensors[1])
        return NestedTensor((v+mask.tensors[0]*0.125,a+999))
    monkeypatch.setattr(u,'sample_chunk',sample)
    video=torch.randn(1,24,27,2,2)
    audio=torch.randn(1,32,2,150)
    positive=[[torch.zeros(1,3,4),{}]]
    output,report=u.DaSiWaH3TiledUpscale().upscale(setup_node,positive,
        {'samples':NestedTensor((video,audio))},upscale_model='interpolation',
        continuity_context={'operation':'continue'},negative=positive,sampler=object(),
        director_guide={'first_frame':torch.ones(1,32,32,3)})
    result=output['samples'].tensors[0]
    upscaled=u.resize_video(video,4,4)
    assert result.shape==(1,24,27,4,4)
    assert torch.equal(result[:,:,:20],upscaled[:,:,:20])
    assert not torch.equal(result[:,:,22:],upscaled[:,:,22:])
    assert output['samples'].tensors[1] is audio
    assert len(calls)==2
    assert not torch.count_nonzero(calls[0][:,:,:5])
    assert torch.all(calls[0][:,:,5:]==1)
    assert torch.all(calls[1]==1)
    assert aligned==[True,False]
    assert 'TEST continuity seam' in report
    assert 'no endpoint re-encode' in report
    assert 'minimax_keyframes' not in positive[0][1]
