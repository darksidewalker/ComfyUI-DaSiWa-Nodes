"""CPU contract tests; a fake VAE tests wiring, not visual model quality."""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
import comfy.cli_args
comfy.cli_args.args.cpu = True
spec = importlib.util.spec_from_file_location('dasiwa_h3_upscale_test', ROOT / 'nodes/nodes_minimax_h3_tiled_upscale.py')
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)


def test_resolution_defaults_to_input_latent_not_director_dimensions():
    video = torch.zeros(1, 24, 7, 48, 64)
    assert u.target_size(video, 2) == (2048, 1536)
    assert u.target_size(video, 1.5) == (1536, 1152)


@pytest.mark.parametrize('scale', [0.5, float('nan'), float('inf'), -1, 0])
def test_resolution_rejects_invalid_or_shrinking_inputs(scale):
    with pytest.raises(ValueError):
        u.target_size(torch.zeros(1,24,2,48,64),scale)


def test_resize_preserves_channel_time_order_and_dtype():
    value = torch.arange(3 * 24).reshape(1,24,3,1,1).expand(-1,-1,-1,2,4).to(torch.float16)
    result = u.resize_video(value,4,8)
    assert result.shape == (1,24,3,4,8)
    assert result.dtype == torch.float16
    assert torch.equal(result[:,:,0,0,0], value[:,:,0,0,0])
    assert torch.equal(result[:,:,2,3,7], value[:,:,2,1,3])
    assert u.resize_video(value,2,4) is value


def test_audio_windows_cover_native_phase_aligned_overlaps():
    assert u.audio_window(12,65,0,7) == (0,37)
    assert u.audio_window(12,65,5,12) == (28,65)


class RecordingVAE:
    def __init__(self):
        self.images = []
    def encode(self,pixels):
        self.images.append(pixels)
        return torch.full((1,24,1,pixels.shape[1]//16,pixels.shape[2]//16),float(len(self.images)))


def test_endpoint_reencoding_does_not_modify_source_conditioning():
    first = torch.ones(1,32,32,3)
    last = torch.zeros(1,32,32,3)
    oldlatent = torch.zeros(1,24,1,2,2)
    tail = torch.full((1,24,2,2,2),3.0)
    kfs = [{'resolved_frame_index':0,'latent':oldlatent},
           {'resolved_frame_index':5,'latent':tail},
           {'resolved_frame_index':21,'latent':oldlatent}]
    original = [[torch.zeros(1,3,2),{'minimax_keyframes':kfs,'minimax_refs':[{'name':'keep'}]}]]
    vae=RecordingVAE()
    result,report=u.encode_endpoints(original,vae,first,last,64,96)
    assert len(vae.images)==2
    assert vae.images[0].shape==(1,96,64,3)
    assert result[0][1]['minimax_keyframes'][0]['latent'].shape==(1,24,1,6,4)
    assert torch.all(result[0][1]['minimax_keyframes'][2]['latent']==2)
    assert result[0][1]['minimax_keyframes'][1]['latent'] is tail
    assert kfs[0]['latent'] is oldlatent
    assert result[0][1]['minimax_refs']==original[0][1]['minimax_refs']
    assert '2 endpoint keyframes' in report


def test_cond_only_mode_does_not_require_director_or_vae():
    cond=[[torch.zeros(1,3,2),{}]]
    result,report=u.encode_endpoints(cond,None,None,None,64,64)
    assert result is cond
    assert 'conditioning only' in report
    assert u.director_endpoint(None,'first_frame') is None


def test_pixels_require_matching_vae_and_no_widget_introspection():
    with pytest.raises(ValueError,match='H3 video VAE'):
        u.encode_endpoints([],None,torch.ones(1,32,32,3),None,64,64)
    with pytest.raises(ValueError):
        u.director_endpoint('not-a-guide','first_frame')


def test_attention_budget_counts_refs_and_keyframes():
    cond=[[torch.zeros(1,12,8),{'minimax_refs':[{'latent':torch.zeros(1,24,2,4,6)}],
                               'minimax_keyframes':[{'latent':torch.zeros(1,24,1,4,6)}]}]]
    # Native references have 12 rows; one keyframe is sized per tile by the planner.
    assert u.conditioning_token_counts(cond)==(12,12,1)


def test_keyframe_budget_separates_cropped_video_from_native_refs_and_audio():
    cond=[[torch.zeros(1,12,8),{
        'minimax_refs':[{'latent':torch.zeros(1,24,2,4,6),
                         'audio_latent':torch.zeros(1,32,2,7)}],
        'minimax_keyframes':[{'latent':torch.zeros(1,24,3,90,70),
                              'audio_latent':torch.zeros(1,32,2,5)}]}]]
    assert u.conditioning_token_counts(cond)==(12,36,3)


def test_budget_recovers_managed_residency_once_and_respects_cap(monkeypatch):
    import comfy.model_management as mm
    device = torch.device('cuda:0')
    class Patcher:
        def __init__(self, size, dev=device, model=None, dynamic=True):
            self.model = model if model is not None else object()
            self.size, self.dev, self.dynamic = size, dev, dynamic
        def loaded_size(self): return self.size
        def current_loaded_device(self): return self.dev
        def is_dynamic(self): return self.dynamic
        def model_patches_models(self): return []
    model = Patcher(20 * 1024**3)
    clone = Patcher(20 * 1024**3, model=model.model)
    vae = Patcher(3 * 1024**3)
    other_gpu = Patcher(8 * 1024**3, torch.device('cuda:1'))
    monkeypatch.setattr(mm,'get_free_memory',lambda d: 5 * 1024**3)
    monkeypatch.setattr(mm,'loaded_models',lambda: [model, clone, vae, other_gpu])
    monkeypatch.setattr(mm,'extra_reserved_memory',lambda: 400 * 1024**2)
    pool, streamed, reserved = u.refinement_memory_budget(model,device,0)
    assert (pool, streamed, reserved) == (28 * 1024**3, True, 400 * 1024**2)
    assert u.refinement_memory_budget(model,device,8192)[0] == 8 * 1024**3
    monkeypatch.setattr(mm,'vram_state',mm.VRAMState.HIGH_VRAM)
    model.dynamic = False
    assert not u.refinement_memory_budget(model,device,0)[1]
    monkeypatch.setattr(mm,'vram_state',mm.VRAMState.NORMAL_VRAM)
    assert u.refinement_memory_budget(model,device,0)[1]
    assert u.refinement_memory_budget(model,torch.device('cpu'),0) == (5 * 1024**3,False,0)


def test_closing_image_does_not_replace_intermediate_single_frame_guides():
    old=torch.full((1,24,1,2,2),9.0)
    cond=[[torch.zeros(1,3,4),{'minimax_keyframes':[
        {'resolved_frame_index':0,'latent':old},
        {'resolved_frame_index':5,'latent':old},
        {'resolved_frame_index':21,'latent':old}]}]]
    result,_=u.encode_endpoints(cond,RecordingVAE(),torch.zeros(1,32,32,3),
                               torch.ones(1,32,32,3),64,64,last_frame_index=21)
    keyframes=result[0][1]['minimax_keyframes']
    assert keyframes[1]['latent'] is old
    assert torch.all(keyframes[2]['latent']==2)


@pytest.mark.parametrize('strength',[0,0.5,1])
def test_continuity_soft_mask_is_global_and_opt_in(strength):
    part=torch.zeros(1,24,12,2,2)
    plan={'refine_start_token':5,'source_tokens':7}
    full=u.continuity_refine_mask(part,0,plan,soft=True,strength=strength)
    chunk=u.continuity_refine_mask(part[:,:,:7],5,plan,soft=True,strength=strength)
    assert torch.equal(full[:,:,5:],chunk)
    assert not torch.count_nonzero(full[:,:,:5])
    assert torch.all(full[:,:,7:]==1)
    assert torch.all(full[:,:,5]==1-strength)
    assert torch.all(full[:,:,6]==1-strength*0.5)
    assert torch.all(u.continuity_refine_mask(part,0,None,soft=True,strength=strength)==1)
    hard=u.continuity_refine_mask(part,0,plan,soft=False,strength=strength)
    assert torch.all(hard[:,:,5:]==1)


@pytest.mark.parametrize('strength',[-0.1,1.1,float('nan')])
def test_continuity_mask_rejects_invalid_strength(strength):
    with pytest.raises(ValueError,match='continuity_mask_strength'):
        u.continuity_refine_mask(torch.zeros(1,24,2,2,2),0,None,strength=strength)
