"""Actual Core tiny H3 sampling with local segment masks and CFG, CPU only."""
from test_h3_segment_backend import core,torch,latent,context,importlib
import pytest


def tiny_model():
    import comfy.supported_models
    import comfy.model_patcher
    config = comfy.supported_models.MiniMaxH3(dict(hidden_size=96,num_layers=1,token_refiner_num_layers=1,num_attention_heads=1,
        attention_head_dim=96,ffn_hidden_size=128,text_dim=96,timestep_input_dim=16,time_embed_hidden_size=96,time_embed_dim=48,
        rope_inv_freq_len=16,dtype=torch.float32))
    model = config.get_model({},device=torch.device('cpu'))
    with torch.no_grad():
        for param in model.parameters(): param.uniform_(-.01,.01)
        model.diffusion_model.rope.inv_freq.fill_(.01)
    return comfy.model_patcher.ModelPatcher(model,load_device=torch.device('cpu'),offload_device=torch.device('cpu'))


@pytest.mark.parametrize('denoise,cfg,learned',[(0,1,False),(.1,1,False),(.1,2,False),(0,1,True),(.1,2,True)])
def test_segment_upscale_native_sampling_prefix_and_original_audio(tmp_path,monkeypatch,denoise,cfg,learned):
    mod = importlib.import_module('h3_segment_test.nodes.h3_continuity.segment_nodes')
    up = importlib.import_module('h3_segment_test.nodes.h3_upscale_continuity')
    node = importlib.import_module('h3_segment_test.nodes.nodes_minimax_h3_tiled_upscale')
    store = core.ClipStore(tmp_path)
    store.publish(store.stage(latent(243,.03),dict(session='s',run_id='root',mode='REF2VA',resolved_prompt='root')),'root.mp4')
    monkeypatch.setattr(mod,'ClipStore',lambda:store)
    monkeypatch.setattr(up,'ClipStore',lambda:store)
    _,sampled,layout = core.prepare_continuation(latent(243,.03),store.inspect('s','root'),dict(mode='REF2VA',width=64,height=64),core.parse_settings({'version':3,'source_id':'root'},1))
    for tensor in sampled['samples'].tensors: tensor.fill_(.07)
    ctx = {**context(),'extension_frames':17,'layout':layout,'operation':'continue','output_mode':'segment'}
    render,ctx,ticket = mod.DaSiWaH3ContinuitySegmentPrepare().prepare(sampled,ctx)
    v,a = render['samples'].tensors
    model = tiny_model()
    positive = [[torch.ones(1,2,96)*.01,{}]]
    negative = [[torch.zeros(1,2,96),{}]] if cfg != 1 else None
    kwargs = {}
    if learned:
        backend = importlib.import_module('h3_segment_test.nodes.h3_latent_upscale')
        monkeypatch.setattr(backend,'upscale_model_names',lambda:['synthetic'])
        class Scale:
            temporal_halo = 1
            sharpness_tiers = False
            dtype = torch.float32
            def __init__(self,*a,**k): pass
            def upscale(self,x,h,w): return node.resize_video(x,h,w)
            def close(self): pass
        monkeypatch.setattr(backend,'H3LatentUpscaler',Scale)
        kwargs['upscale_model'] = 'synthetic'
    result,report = node.DaSiWaH3TiledUpscale().upscale(model,positive,render,scale=1,denoise=denoise,steps=1,cfg=cfg,negative=negative,
        continuity_context=ctx,spatial_tiling=False,temporal_chunking=False,**kwargs)
    rv,ra = result['samples'].tensors
    plan = up.continuity_plan(v,a,ctx)
    assert torch.equal(rv[:,:,:plan['refine_start_token']],v[:,:,:plan['refine_start_token']])
    assert ra is a and torch.equal(ra,a)
    assert model.model_options.get('model_function_wrapper') is None
    assert 'Segment continuity' in report
