"""Tiny randomly initialized native H3 transformer; not a pretrained quality test."""

def test_real_native_h3_tiled_forward():
    import sys
    from pathlib import Path
    import importlib.util
    import torch
    root=Path(__file__).resolve().parents[3]
    sys.path.insert(0,str(root))
    if '--cpu' not in sys.argv:
        sys.argv.append('--cpu')
    import comfy.cli_args
    comfy.cli_args.args.cpu=True
    import comfy.ops
    import comfy.utils
    from comfy.ldm.minimax.model import MiniMaxH3Model
    spec=importlib.util.spec_from_file_location('h3_tiled',root/'custom_nodes/ComfyUI-DaSiWa-Nodes/nodes/h3_tiled_sampling.py')
    h3=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h3)
    torch.manual_seed(1)
    model=MiniMaxH3Model(hidden_size=32,num_layers=1,token_refiner_num_layers=0,
        num_attention_heads=4,attention_head_dim=8,ffn_hidden_size=64,text_dim=32,
        timestep_input_dim=8,time_embed_hidden_size=32,time_embed_dim=16,
        rope_inv_freq_len=1,dtype=torch.float32,device=torch.device('cpu'),operations=comfy.ops.manual_cast)
    model.eval().requires_grad_(False)
    with torch.no_grad():
        for n,p in model.named_parameters():
            p.normal_(0,0.02)
        model.rope.inv_freq.fill_(1)
    video=torch.randn(1,24,2,6,8)
    audio=torch.randn(1,32,2,8)
    packed,shapes=comfy.utils.pack_latents([video,audio])
    keyframe={'resolved_frame_index':0,'latent':torch.randn(1,24,1,6,8)}
    ct={'latent_shapes':shapes,'minimax_payload':{'keyframes':[keyframe],'cond_video_latents':[keyframe['latent']], 'cond_audio_latents':[],'audio_scale':1.0},
        'c_crossattn':torch.randn(1,3,32),'transformer_options':{},'audio_denoise_mask':torch.zeros_like(audio)}
    calls=[]
    def apply_model(x,t,**c):
        streams=comfy.utils.unpack_latents(x,c['latent_shapes'])
        out=model(streams,t,context=c['c_crossattn'],transformer_options=c['transformer_options'],
                  minimax_payload=c['minimax_payload'],audio_denoise_mask=c['audio_denoise_mask'])
        calls.append(streams[0].shape)
        # Flow x0 conversion, matching the wrapper's apply_model contract.
        return comfy.utils.pack_latents([streams[0]-0.5*out[0],streams[1]-0.5*out[1]])[0]
    with torch.inference_mode():
        output=h3.H3TiledDiffusion(64,64,32)(apply_model,{'input':packed,'timestep':torch.tensor([500.]),'c':ct})
        result=comfy.utils.unpack_latents(output,shapes)
    assert result[0].shape==video.shape and torch.isfinite(result[0]).all()
    assert torch.equal(result[1],audio)
    assert len(calls)>1
    print('PASS real native MiniMaxH3Model (tiny random weights):',len(calls),'tile forwards, finite packed AV output, audio frozen exactly, global-position endpoint conditioning')
