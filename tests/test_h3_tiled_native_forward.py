"""Tiny randomly initialized native H3 transformer; not a pretrained quality test."""

import pytest

@pytest.mark.parametrize('coordinates',['global','local'])
@pytest.mark.parametrize('hw',[(6,8),(7,9)])
def test_real_native_h3_tiled_forward(coordinates,hw,monkeypatch):
    import sys
    from pathlib import Path
    import importlib.util
    import torch
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root.parent/'ComfyUI'))
    if '--cpu' not in sys.argv:
        sys.argv.append('--cpu')
    import comfy.cli_args
    comfy.cli_args.args.cpu=True
    import comfy.model_management as mm
    monkeypatch.setattr(mm,'get_torch_device',lambda:torch.device('cpu'))
    import comfy.ops
    import comfy.utils
    from comfy.ldm.minimax.model import MiniMaxH3Model
    spec=importlib.util.spec_from_file_location('h3_tiled',root/'nodes/h3_tiled_sampling.py')
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
    video=torch.randn(1,24,2,*hw)
    audio=torch.randn(1,32,2,8)
    packed,shapes=comfy.utils.pack_latents([video,audio])
    keyframe={'resolved_frame_index':17,'latent':torch.randn(1,24,1,*hw),'audio_latent':torch.randn(1,32,2,3)}
    refs=[dict(kind='image',latent=torch.randn(1,24,1,2,4),latent_h=2,latent_w=4),
          dict(kind='video_audio',latent=torch.randn(1,24,2,4,2),latent_t=2,latent_h=4,latent_w=2,
               ref_audio_t=2,audio_latent=torch.randn(1,32,2,2)),
          dict(kind='audio',ref_audio_t=3,audio_latent=torch.randn(1,32,2,3))]
    cond_video=[keyframe['latent']]+[r['latent'] for r in refs if 'latent' in r]
    cond_audio=[keyframe['audio_latent']]+[r['audio_latent'] for r in refs if 'audio_latent' in r]
    ct={'latent_shapes':shapes,'minimax_payload':{'keyframes':[keyframe], 'refs':refs,
        'cond_video_latents':cond_video, 'cond_audio_latents':cond_audio,'audio_scale':1.0,
        'text_token_tags':torch.tensor([1,0,2]), 'seed':31,
        'visual_cond_noise_aug':0.9,'audio_cond_noise_aug':0.8},
        'c_crossattn':torch.randn(1,3,32),'transformer_options':{},'audio_denoise_mask':torch.zeros_like(audio),
        'denoise_mask':torch.linspace(0,1,video.numel()).reshape_as(video)}
    calls=[]
    def apply_model(x,t,**c):
        streams=comfy.utils.unpack_latents(x,c['latent_shapes'])
        out=model(streams,t,context=c['c_crossattn'],transformer_options=c['transformer_options'],
                  minimax_payload=c['minimax_payload'],audio_denoise_mask=c['audio_denoise_mask'],denoise_mask=c['denoise_mask'])
        calls.append(streams[0].shape)
        # Flow x0 conversion, matching the wrapper's apply_model contract.
        return comfy.utils.pack_latents([streams[0]-0.5*out[0],streams[1]-0.5*out[1]])[0]
    with torch.inference_mode():
        output=h3.H3TiledDiffusion(64,64,32,coordinates=coordinates)(apply_model,{'input':packed,'timestep':torch.tensor([500.]),'c':ct})
        result=comfy.utils.unpack_latents(output,shapes)
    assert result[0].shape==video.shape and torch.isfinite(result[0]).all()
    assert torch.equal(result[1],audio)
    assert len(calls)>1
    with torch.inference_mode():
        direct_payload=dict(ct['minimax_payload'])
        padded_keyframe=dict(keyframe,latent=h3.pad_to_patch_size(keyframe['latent'],(1,2,2)))
        direct_payload.update(keyframes=[padded_keyframe],cond_video_latents=[padded_keyframe['latent']]+cond_video[1:])
        direct=apply_model(packed,torch.tensor([500.]),**dict(ct,minimax_payload=direct_payload))
        one=h3.H3TiledDiffusion(h3._ceil2(hw[1])*16,h3._ceil2(hw[0])*16,0,coordinates=coordinates)(
            apply_model,{'input':packed,'timestep':torch.tensor([500.]),'c':ct})
    torch.testing.assert_close(one,direct,atol=2e-6,rtol=2e-5)
    print('PASS native MiniMaxH3Model:',coordinates,hw,len(calls),'forwards; AV guides/references, soft mask, one-tile equivalence')
