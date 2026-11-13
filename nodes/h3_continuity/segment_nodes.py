"""Bounded source-tail rendering; checkpoints remain at the source resolution."""
import uuid

import torch
from comfy.nested_tensor import NestedTensor
import comfy.model_management

from .core import ClipStore
from .segment_policy import render_layout
from .vendor.continuation_nodes import _calculate_guided_continuation_layout, video_latent_t, validate_h3_av_latent


class DaSiWaH3ContinuitySegmentPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'sampled':('LATENT',{'tooltip':'Source-resolution H3 sampler output; hidden sampled overlap is discarded, not rendered.'}),
                            'context':('DF_H3_CONTINUITY_CONTEXT',{'tooltip':'This sampler’s Director Guide continuity_context. Works in both Director output modes: Full timeline behaves like Append & Stage, New segment only renders the bounded segment.'})}}

    RETURN_TYPES = ('LATENT','DF_H3_CONTINUITY_CONTEXT','DF_H3_CONTINUITY_TICKET')
    RETURN_NAMES = ('render_latent','segment_context','ticket')
    FUNCTION = 'prepare'
    CATEGORY = 'DaSiWa/MiniMax H3'

    def prepare(self, sampled, context):
        if context.get('disabled'):
            return sampled,context,{'disabled':True}
        if context.get('output_mode') != 'segment':
            # Full timeline through the same wiring: identical to Append & Stage.
            from .nodes import DaSiWaH3ContinuityAppend
            cumulative, ticket = DaSiWaH3ContinuityAppend().commit(sampled, context)
            return cumulative, context, ticket
        comfy.model_management.throw_exception_if_processing_interrupted()
        store = ClipStore()
        if context['operation'] != 'continue':
            _,_,frames = validate_h3_av_latent(sampled,name='first take')
            ticket = store.stage(sampled,{**context,'run_id':uuid.uuid4().hex})
            plan = dict(frame_offset=0,video_start=0,audio_start=0,render_frames=frames,
                        crop_start_frame=0,crop_frames=frames,crop_start_sample=0,crop_samples=round(frames*32000/24))
            return sampled,{**context,'render_kind':'segment','render_plan':plan,'source_frames':0},ticket
        metadata = store.inspect(context['session'],context['source_id'])
        from .segment_store import file_stamp
        source_path = store.member(context['session'],context['source_id'],'delta.safetensors' if metadata.get('schema') == 2 else 'latent.safetensors')
        if 'source_file_stamp' in context and context['source_file_stamp'] != file_stamp(source_path):
            raise ValueError('Continuity source changed after sampling preparation; run the Guide again.')
        plan = render_layout(metadata['frames'],context['extension_frames'])
        native = _calculate_guided_continuation_layout(metadata['frames'],context['overlap_frames'],context['extension_frames'])
        tail = store.load_range(context['session'],context['source_id'],plan['video_start'],video_latent_t(metadata['frames']),plan['audio_start'],round(metadata['frames']*40/24))
        pv,pa = tail['samples'].tensors
        sv,sa = sampled['samples'].tensors
        # Keep the immutable original tail, not the sampler's stochastic overlap.
        render = {'samples':NestedTensor((torch.cat((pv,sv[:,:,native.overlap_video_t:].to(pv)),2),
                                          torch.cat((pa,sa[...,native.overlap_audio_t:].to(pa)),-1)))}
        ticket = store.stage_delta(sampled,context,metadata)
        layout = dict(overlap_video_tokens=native.overlap_video_t,overlap_audio_tokens=native.overlap_audio_t)
        return render,{**context,'layout':layout,'render_kind':'segment','render_plan':plan,'source_frames':metadata['frames']},ticket


def segment_render_plan(video, audio, context):
    if context.get('render_kind') != 'segment':
        raise ValueError('Segment Decode/Upscale requires Segment Prepare.segment_context.')
    _,_,frames = validate_h3_av_latent({'samples':NestedTensor((video,audio))},name='segment render window')
    plan = context.get('render_plan')
    if context.get('operation') == 'continue':
        expected = render_layout(context.get('source_frames'),context.get('extension_frames'))
        if plan != expected:
            raise ValueError('Segment render offsets differ from the global timeline.')
    elif isinstance(plan,dict):
        expected = dict(frame_offset=0,video_start=0,audio_start=0,render_frames=frames,
                        crop_start_frame=0,crop_frames=frames,crop_start_sample=0,crop_samples=round(frames*32000/24))
        if plan != expected:
            raise ValueError('Invalid first-take segment render offsets.')
    else:
        raise ValueError('Missing segment render plan.')
    if frames != plan['render_frames']:
        raise ValueError('Segment render latent length differs from Segment Prepare.')
    return plan


def reject_swapped_vaes(vae,audio_vae):
    # Classified by Core's model classes, not filenames/keys, so new H3 VAE variants keep passing.
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
    from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
    if isinstance(getattr(vae,'first_stage_model',None),MiniMaxH3AudioVAE):
        raise ValueError("Segment Decode: the H3 audio VAE is connected to 'vae'. Connect an H3 video VAE (e.g. minimax_h3_video_vae_* or minimax_h3_sol_vae_*) to 'vae' and minimax_h3_audio_vae_* to 'audio_vae'.")
    if isinstance(getattr(audio_vae,'first_stage_model',None),MiniMaxH3VideoVAE):
        raise ValueError("Segment Decode: an H3 video VAE is connected to 'audio_vae'. Connect minimax_h3_audio_vae_* to 'audio_vae' and the video VAE (e.g. minimax_h3_video_vae_* or minimax_h3_sol_vae_*) to 'vae'.")


def validate_segment_vaes(vae,audio_vae):
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
    from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE, BigVGAN, AMPBlock1, Activation1d
    vm,am = getattr(vae,'first_stage_model',None),getattr(audio_vae,'first_stage_model',None)
    if not isinstance(vm,MiniMaxH3VideoVAE) or any(getattr(vm,k,None) != v for k,v in dict(clip_length=17,tokens_chunk_size=5,token_drop=3,token_overlap=2,frame_pre_padding=3,vae_ratio_t=4).items()):
        raise ValueError('Segment decoding requires the validated native H3 video temporal layout (17 frames / 5 tokens).')
    if not isinstance(am,MiniMaxH3AudioVAE) or getattr(am,'samples_per_latent',None) != 800 or getattr(audio_vae,'audio_sample_rate_output',getattr(audio_vae,'audio_sample_rate',None)) != 32000:
        raise ValueError('Segment decoding requires the native 40 Hz / 32 kHz H3 audio VAE.')
    decoder = am.decoder
    valid = (type(decoder) is BigVGAN and decoder.num_upsamples == 7 and decoder.num_kernels == 3
             and decoder.conv_pre.kernel_size == (7,) and decoder.conv_post.kernel_size == (7,))
    if valid:
        for stage,(rate,kernel) in enumerate(zip((5,5,2,2,2,2,2),(9,9,4,4,4,4,4))):
            up = decoder.ups[stage]
            valid &= len(up) == 1 and up[0].stride == (rate,) and up[0].kernel_size == (kernel,)
            for j,k in enumerate((3,7,11)):
                block = decoder.resblocks[stage*3+j]
                valid &= type(block) is AMPBlock1 and len(block.convs1) == len(block.convs2) == 3
                valid &= all(c.kernel_size == (k,) and c.dilation == (d,) for c,d in zip(block.convs1,(1,3,5)))
                valid &= all(c.kernel_size == (k,) and c.dilation == (1,) for c in block.convs2)
        for activation in decoder.modules():
            if isinstance(activation,Activation1d):
                valid &= (activation.upsample.ratio == activation.downsample.ratio == 2
                          and activation.upsample.filter.shape[-1] == activation.downsample.lowpass.filter.shape[-1] == 12)
    if not valid:
        raise ValueError('H3 audio decoder topology changed; bounded-context decoding must be revalidated before segment export.')


class DaSiWaH3ContinuitySegmentDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'latent':('LATENT',{'tooltip':'Segment Prepare.render_latent, optionally spatially upscaled; do not connect the raw sampler or stored delta.'}),
            'vae':('VAE',{'tooltip':'Native MiniMax H3 video VAE. Other temporal layouts are rejected.'}),
            'audio_vae':('VAE',{'tooltip':'Native MiniMax H3 32 kHz audio VAE. In New segment only, audio is cropped at cumulative sample boundaries without whole-clip loudness normalization; Full timeline decodes exactly like VAE Decode Audio.'}),
            'context':('DF_H3_CONTINUITY_CONTEXT',{'tooltip':'The matching Segment Prepare.segment_context; retain it unchanged through optional Upscale.'})}}

    RETURN_TYPES = ('IMAGE','AUDIO')
    RETURN_NAMES = ('images','audio')
    FUNCTION = 'decode'
    CATEGORY = 'DaSiWa/MiniMax H3'

    def decode(self, latent, vae, audio_vae, context):
        reject_swapped_vaes(vae,audio_vae)
        if context.get('render_kind') != 'segment':
            # Full timeline: exactly the core VAE Decode / VAE Decode Audio results.
            from comfy_extras.nodes_audio import vae_decode_audio
            # Match Core VAEDecode without importing the ambiguous top-level 'nodes'.
            video = latent['samples']
            if video.is_nested:
                video = video.unbind()[0]
            images = vae.decode(video)
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            return images, vae_decode_audio(audio_vae, latent)
        video,audio = latent['samples'].tensors
        plan = segment_render_plan(video,audio,context)
        validate_segment_vaes(vae,audio_vae)
        comfy.model_management.throw_exception_if_processing_interrupted()
        pixels = vae.decode(video)
        if pixels.ndim == 5 and pixels.shape[0] == 1:
            pixels = pixels[0]
        if pixels.ndim != 4 or pixels.shape[0] != plan['render_frames'] or pixels.shape[-1] != 3:
            raise ValueError('Video VAE output differs from the validated H3 render frame plan.')
        start = plan['crop_start_frame']
        images = pixels[start:start+plan['crop_frames']].clone()
        del pixels
        comfy.model_management.throw_exception_if_processing_interrupted()
        waveform = audio_vae.decode(audio).movedim(-1,1)
        if waveform.ndim != 3 or waveform.shape[:2] != (1,2) or waveform.shape[-1] != audio.shape[-1]*800:
            raise ValueError('Audio VAE output differs from the validated H3 sample plan.')
        start,end = plan['crop_start_sample'],plan['crop_start_sample']+plan['crop_samples']
        # The native 40-Hz endpoint can undershoot the exact video endpoint by <=267
        # samples; pad only that quantization remainder, never repeat source audio.
        if end > waveform.shape[-1]:
            if end-waveform.shape[-1] > 267:
                raise ValueError('Audio decoder ended before the global segment boundary.')
            waveform = torch.nn.functional.pad(waveform,(0,end-waveform.shape[-1]))
        result = waveform[...,start:end].clone()
        global_start = plan['frame_offset'] + plan['crop_start_frame']
        return images,{'waveform':result,'sample_rate':32000,
                       'h3_segment_interval':{'start':(global_start,24),'end':(global_start+plan['crop_frames'],24)}}


NODE_CLASS_MAPPINGS = {cls.__name__:cls for cls in (DaSiWaH3ContinuitySegmentPrepare,DaSiWaH3ContinuitySegmentDecode)}
NODE_DISPLAY_NAME_MAPPINGS = {'DaSiWaH3ContinuitySegmentPrepare':'MiniMaxH3 Continuity • Segment Prepare',
                              'DaSiWaH3ContinuitySegmentDecode':'MiniMaxH3 Continuity • Segment Decode'}
