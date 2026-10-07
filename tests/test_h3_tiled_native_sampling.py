"""Full native sampler + tiny randomly initialized H3 model; no pretrained quality test."""

import pytest


@pytest.mark.parametrize('automatic_plan',[False,True])
@pytest.mark.parametrize('with_continuity',[False,True])
@pytest.mark.parametrize('spatial,temporal',[(True,True),(False,True),(True,False),(False,False)])
def test_complete_native_sampler_tiled_temporal_upscale(automatic_plan,with_continuity,spatial,temporal,tmp_path,monkeypatch):
    """Complete native CPU sampler + tiny random H3 model; no pretrained render."""
    import importlib.util
    import sys
    import types
    from pathlib import Path
    import torch
    root=Path(__file__).resolve().parents[3]
    sys.path.insert(0,str(root))
    if '--cpu' not in sys.argv:
        sys.argv.append('--cpu')
    import comfy.cli_args
    comfy.cli_args.args.cpu=True
    import comfy.model_base
    import comfy.model_management as mm
    monkeypatch.setattr(mm,'get_torch_device',lambda:torch.device('cpu'))
    import comfy.model_patcher
    import comfy.supported_models
    from comfy.nested_tensor import NestedTensor
    config=comfy.supported_models.MiniMaxH3(dict(image_model='minimax_h3',hidden_size=32,num_layers=1,
        token_refiner_num_layers=0,num_attention_heads=4,attention_head_dim=8,ffn_hidden_size=64,
        text_dim=32,timestep_input_dim=8,time_embed_hidden_size=32,time_embed_dim=16,rope_inv_freq_len=1))
    config.set_inference_dtype(torch.float32,None)
    base=comfy.model_base.MiniMaxH3(config,device=torch.device('cpu'))
    base.eval().requires_grad_(False)
    torch.manual_seed(13)
    with torch.no_grad():
        for n,p in base.diffusion_model.named_parameters():
            p.normal_(0,0.02)
        base.diffusion_model.rope.inv_freq.fill_(1)
    model=comfy.model_patcher.ModelPatcher(base,load_device=torch.device('cpu'),offload_device=torch.device('cpu'))
    package=types.ModuleType('_h3_native_integration')
    package.__path__=[str(root/'custom_nodes/ComfyUI-DaSiWa-Nodes/nodes')]
    sys.modules[package.__name__]=package
    s=importlib.util.spec_from_file_location(package.__name__+'.node',root/'custom_nodes/ComfyUI-DaSiWa-Nodes/nodes/nodes_minimax_h3_tiled_upscale.py')
    m=importlib.util.module_from_spec(s)
    sys.modules[s.name]=m
    s.loader.exec_module(m)
    # Exercise both the real planner and forced multiwindow geometry for this
    # tiny random model. In both paths the model and sampler remain native.
    if not automatic_plan:
        helpers=__import__(package.__name__+'.h3_tiled_sampling',fromlist=['*'])
        monkeypatch.setattr(helpers,'plan_tiles',lambda *a,**k:dict(
            tile_width=64 if k['spatial_tiling'] else 128,
            tile_height=64 if k['spatial_tiling'] else 128,
            overlap=32 if k['spatial_tiling'] else 0,
            chunk_tokens=10 if k['temporal_chunking'] else a[0][2],
            temporal_overlap_tokens=5 if k['temporal_chunking'] else 0,
            explanation='TEST forced geometry honoring switches'))
    video=torch.randn(1,24,12,2,2)
    audio=torch.randn(1,32,2,65)
    conditioning=[[torch.randn(1,3,32),{'minimax_keyframes':[{'resolved_frame_index':0,'latent':torch.randn(1,24,1,2,2)}]}]]
    context=None
    if with_continuity:
        import folder_paths
        monkeypatch.setattr(folder_paths,'get_output_directory',lambda:str(tmp_path))
        from _h3_native_integration.h3_continuity.core import ClipStore
        from _h3_native_integration.h3_continuity.nodes import DaSiWaH3ContinuityAppend
        from _h3_native_integration.h3_continuity.vendor.continuation_nodes import MiniMaxH3LatentTailGuide
        store=ClipStore()
        previous={'samples':NestedTensor((video[:,:,:7].clone(),audio[...,:37].clone()))}
        ticket=store.stage(previous,dict(session='native_test',run_id='source',mode='I2VA',resolved_prompt='test'))
        # Metadata publication is a test fixture, not a rendered/exported video.
        store.publish(ticket,tmp_path/'source.webm')
        sampled={'samples':NestedTensor((torch.randn(1,24,7,2,2),torch.randn(1,32,2,37)))}
        conditioning=MiniMaxH3LatentTailGuide.execute(
            [[torch.randn(1,3,32),{}]],previous,sampled,2,9)[0]
        context=dict(operation='continue',session='native_test',source_id='source',
                     mode='I2VA',resolved_prompt='test',extension_frames=17,
                     layout=dict(overlap_video_tokens=2,overlap_audio_tokens=9))
        cumulative,_=DaSiWaH3ContinuityAppend().commit(sampled,context)
        video,audio=cumulative['samples'].tensors
    output,report=m.DaSiWaH3TiledUpscale().upscale(model,conditioning,{'samples':NestedTensor((video,audio))},
        scale=4,upscale_model='interpolation',steps=1,denoise=0.2,seed=23,memory_budget_mb=128 if automatic_plan else 4,continuity_context=context,
        continuity_soft_refine=with_continuity,continuity_mask_strength=1.0,
        spatial_tiling=spatial,temporal_chunking=temporal)
    assert output['samples'].tensors[0].shape==(1,24,12,8,8)
    assert torch.isfinite(output['samples'].tensors[0]).all()
    assert output['samples'].tensors[1] is audio
    assert 'model_function_wrapper' not in model.model_options
    assert f"spatial_tiling={'on' if spatial else 'off'}" in report
    assert f"temporal_chunking={'on' if temporal else 'off'}" in report
    if automatic_plan:
        assert 'Grid search minimizes calls' in report
    if with_continuity:
        expected_prefix=m.resize_video(video,8,8)[:,:,:5]
        assert torch.equal(output['samples'].tensors[0][:,:,:5],expected_prefix)
        assert 'continuity' in report.lower()
    print('PASS COMPLETE NATIVE CPU SAMPLER + H3 MODEL + SINGLE NODE:',report)
    import comfy.model_management as mm
    mm.unload_model_and_clones(model,unload_additional_models=False)
    del model,base
    import gc
    gc.collect()
