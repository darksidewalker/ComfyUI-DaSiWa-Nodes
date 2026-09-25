import uuid
import pytest
import torch
from comfy.nested_tensor import NestedTensor

def latent(frames=124):
    t = ((frames - 5)//17)*5+2
    return {'samples':NestedTensor((torch.randn(1,24,t,2,3),torch.randn(1,32,2,round(frames*40/24))))}

def context(**kwargs):
    return {'operation':'new','session':'test','run_id':uuid.uuid4().hex,'mode':'REF2VA',
            'resolved_prompt':'A rider moves forward in a forest.',**kwargs}

def ready(store, value=None):
    value=latent() if value is None else value
    ticket=store.stage(value,context())
    store.publish(ticket,'movie.mp4')
    return ticket

def config(core, **kwargs):
    return core.parse_settings({'session':'test','operation':'new',**kwargs})

@pytest.mark.parametrize('frames',[5,22,124,141,243,362])
def test_safetensors_round_trip_and_prefix_exactness(modules,store,frames):
    core,_,_=modules
    original=latent(frames)
    ticket=ready(store,original)
    loaded,meta=store.load('test',ticket['clip_id'])
    assert meta['frames']==frames
    for a,b in zip(original['samples'].tensors,loaded['samples'].tensors): assert torch.equal(a,b)

def test_failed_or_cancelled_export_never_becomes_latest(modules,store):
    ticket=ready(store)
    incomplete=store.stage(latent(),context())
    assert store.list_clips('test')['latest_id']==ticket['clip_id']
    with pytest.raises(ValueError,match='not finished'):store.load('test',incomplete['clip_id'])

def test_twenty_continuations_have_no_audio_rounding_drift(modules):
    core,_,_=modules
    original=latent()
    result=original
    guide={'mode':'REF2VA','width':48,'height':32,'resolved_prompt':'old'}
    cfg=config(core,operation='continue',source_id='parent')
    for i in range(20):
        previous=result
        updated,target,layout=core.prepare_continuation(previous,{'mode_family':'ref2va'},guide,cfg)
        assert updated['length']==141
        assert target['samples'].tensors[0].shape[2]==42
        sampled={'samples':NestedTensor(tuple(torch.randn_like(t) for t in target['samples'].tensors))}
        result=core.append_tail(previous,sampled,layout)
        v,a,frames=core.validate_h3_av_latent(result,name='chain')
        assert frames==124+(i+1)*119
        assert a.shape[-1]==round(frames*40/24)
        ov,oa=original['samples'].tensors
        assert torch.equal(v[:,:,:ov.shape[2]],ov)
        assert torch.equal(a[...,:oa.shape[-1]],oa)

def test_tail_is_on_active_conditioning_and_preserves_references(modules):
    core,_,_=modules
    previous=latent()
    cfg=config(core,operation='continue',source_id='source')
    guide={'mode':'REF2VA','width':48,'height':32,'ref_images':{'ref_image_1':object()}}
    updated,target,layout=core.prepare_continuation(previous,{'mode_family':'ref2va'},guide,cfg)
    positive=[[torch.zeros(1,2,3),{'marker':17,'minimax_refs':[{'kind':'image'}]}]]
    guided=core.add_tail(positive,previous,target,layout)
    assert positive[0][1].get('minimax_keyframes') is None
    assert guided[0][1]['marker']==17
    assert guided[0][1]['minimax_refs']==positive[0][1]['minimax_refs']
    kf=guided[0][1]['minimax_keyframes'][0]
    assert kf['resolved_frame_index']==0
    assert torch.equal(kf['latent'],previous['samples'].tensors[0][:,:,-7:])
    assert torch.equal(kf['audio_latent'],previous['samples'].tensors[1][...,-layout['overlap_audio_tokens']:])
    assert updated['ref_images'] is guide['ref_images']

@pytest.mark.parametrize('change,message',[
    ({'width':64},'Source canvas'),({'mode':'I2VA'},'model families'),({'mode':'Image Inpaint'},'video mode')])
def test_incompatible_source_rejected_before_sampling(modules,change,message):
    core,_,_=modules
    guide={'mode':'REF2VA','width':48,'height':32,**change}
    with pytest.raises(ValueError,match=message):
        core.prepare_continuation(latent(),{'mode_family':'ref2va'},guide,config(core,operation='continue',source_id='x'))

def test_source_metadata_fps_must_be_native_24(modules):
    core, _, _ = modules
    with pytest.raises(ValueError, match="24 fps"):
        core.prepare_continuation(latent(), {'mode_family': 'ref2va', 'fps': 30},
                                  {'mode': 'REF2VA', 'width': 48, 'height': 32},
                                  config(core, operation='continue', source_id='parent'))


@pytest.mark.parametrize('value',[0,1,118,120,357])
def test_invalid_extension_fails_early(modules,value):
    core,_,_=modules
    with pytest.raises(ValueError): config(core,extension_frames=value)

@pytest.mark.parametrize('path',['../other','/etc','x/y','x\\y','', 'x'*81])
def test_store_rejects_path_escape(modules,store,path):
    with pytest.raises(ValueError):store.session_dir(path)
