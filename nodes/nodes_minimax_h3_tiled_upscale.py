"""Single-node, model-scoped MiniMax H3 upscale/refine orchestration.

Spatial diffusion is internal; no dependency on bbaudio's node registrations.
Audio is frozen during refinement and the original audio stream is returned.
"""
import logging
import math

import torch
import torch.nn.functional as F

LOG = logging.getLogger(__name__)


def target_size(video, scale):
    if not math.isfinite(float(scale)) or scale < 1:
        raise ValueError('H3 upscale scale must be finite and at least 1.')
    w = int(round(video.shape[-1] * 16 * scale / 32)) * 32
    h = int(round(video.shape[-2] * 16 * scale / 32)) * 32
    if w < 32 or h < 32 or w % 32 or h % 32:
        raise ValueError('H3 target dimensions must be positive multiples of 32 pixels.')
    if w < video.shape[-1] * 16 or h < video.shape[-2] * 16:
        raise ValueError('This node upscales only; neither target dimension may shrink.')
    return w, h


def continuity_refine_mask(part, start, continuity, soft=False, strength=1.0):
    """Global-token smoothstep ramp; identical weights across temporal windows."""
    if not math.isfinite(float(strength)) or not 0 <= strength <= 1:
        raise ValueError('continuity_mask_strength must be finite and between 0 and 1.')
    mask = torch.ones_like(part)
    if continuity is None:
        return mask
    head, seam = continuity['refine_start_token'], continuity['source_tokens']
    if seam <= head:
        raise ValueError('Continuity source-tail overlap must be positive.')
    indices = torch.arange(start, start + part.shape[2], device=part.device)
    weights = (indices >= head).float()
    if soft and strength > 0:
        phase = ((indices.float() - head) / (seam - head)).clamp(0, 1)
        ramp = phase.square() * (3 - 2 * phase)
        weights = weights * ((1 - strength) + strength * ramp)
    mask *= weights.to(part.dtype).view(1, 1, -1, 1, 1)
    return mask


def resize_video(video, height, width):
    """Spatial interpolation only: retain the temporal axis and channel order."""
    if video.shape[-2:] == (height, width):
        return video
    b, c, t, h, w = video.shape
    frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    result = F.interpolate(frames.float(), size=(height, width), mode='bilinear', align_corners=False)
    return result.reshape(b, t, c, height, width).permute(0, 2, 1, 3, 4).to(video.dtype)


def audio_window(total_video_tokens, total_audio_tokens, start, end):
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN
    def frames(n):
        return sum(FRAME_PER_TOKEN[i % len(FRAME_PER_TOKEN)] for i in range(n))
    total = frames(total_video_tokens)
    a = round(frames(start) * total_audio_tokens / total)
    z = total_audio_tokens if end == total_video_tokens else round(frames(end) * total_audio_tokens / total)
    if z <= a:
        raise ValueError('Audio stream is too short for the selected temporal window.')
    return a, z


def director_endpoint(guide, name):
    # Use the connected runtime guide, not hidden widget lookups, workflow JSON,
    # downstream introspection, or an independent second Director.
    if guide is None:
        return None
    if not isinstance(guide, dict):
        raise ValueError('director_guide must be a MiniMax H3 Director guide.')
    return guide.get(name)


def encode_endpoints(conditioning, vae, first, last, width, height, last_frame_index=None):
    """Re-encode available endpoint pixels, never stretch their latent grid."""
    if first is None and last is None:
        return conditioning, 'conditioning only (no endpoint pixels)'
    if vae is None:
        raise ValueError('Connect the H3 video VAE when start_image or Director endpoint images are supplied.')
    replacements = {}
    for name, pixels in [('first', first), ('last', last)]:
        if pixels is None:
            continue
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4 or pixels.shape[-1] < 3:
            raise ValueError('H3 endpoint images must be an IMAGE tensor [N,H,W,C].')
        image = pixels[:1, :, :, :3].movedim(-1, 1)
        image = F.interpolate(image.float(), size=(height, width), mode='bicubic', align_corners=False, antialias=True)
        encoded = vae.encode(image.movedim(1, -1))
        if encoded.ndim != 5 or encoded.shape[1] != 24 or encoded.shape[-2:] != (height // 16, width // 16):
            raise ValueError('Endpoint encoding is not an H3 video latent. Connect the MiniMax H3 video VAE.')
        replacements[name] = encoded
    if last_frame_index is None:
        last_frame_index = max((kf.get('resolved_frame_index', kf.get('frame_index', 0))
                                for _, md in conditioning for kf in md.get('minimax_keyframes', [])), default=0)
    result = []
    count = 0
    for tokens, metadata in conditioning:
        md = dict(metadata)
        kfs = []
        for keyframe in metadata.get('minimax_keyframes', []):
            kf = dict(keyframe)
            frame = kf.get('resolved_frame_index', kf.get('frame_index', 0))
            name = 'first' if frame == 0 else ('last' if frame == last_frame_index else None)
            if name in replacements and kf.get('latent') is not None:
                # Endpoint substitution only; multi-frame continuation tails
                # must keep their own temporal conditioning, not become a still.
                if kf['latent'].shape[2] == 1:
                    kf['latent'] = replacements[name]
                    count += 1
            kfs.append(kf)
        if 'minimax_keyframes' in metadata:
            md['minimax_keyframes'] = kfs
        result.append([tokens, md])
    if count == 0:
        LOG.warning('[DaSiWa H3 Upscale] Endpoint pixels supplied, but conditioning has no matching single-frame keyframe.')
    return result, f'{count} endpoint keyframes VAE-encoded at {width}x{height}'


def conditioning_token_counts(conditioning):
    """Fixed native reference/audio rows and tile-cropped keyframe frames."""
    text, refs, keyframes = 0, 0, 0
    for tokens, md in conditioning:
        if isinstance(tokens, torch.Tensor) and tokens.ndim >= 2:
            text = max(text, tokens.shape[1])
        current = 0
        for ref in md.get('minimax_refs', []):
            value = ref.get('latent')
            if isinstance(value, torch.Tensor):
                t = value.shape[2] if value.ndim == 5 else 1
                current += t * math.ceil(value.shape[-2] / 2) * math.ceil(value.shape[-1] / 2)
            audio = ref.get('audio_latent')
            if isinstance(audio, torch.Tensor):
                current += audio.shape[-1] * audio.shape[-2]
        current_keyframes = 0
        for kf in md.get('minimax_keyframes', []):
            value = kf.get('latent')
            if isinstance(value, torch.Tensor):
                current_keyframes += value.shape[2]
            audio = kf.get('audio_latent')
            if isinstance(audio, torch.Tensor):
                current += audio.shape[-1] * audio.shape[-2]
        refs = max(refs, current)
        keyframes = max(keyframes, current_keyframes)
    return text, refs, keyframes


def refinement_memory_budget(model, device, memory_budget_mb):
    import comfy.model_management as mm
    available = int(mm.get_free_memory(device))
    if device.type == 'cpu':
        # Host weights are not reclaimable: CPU inference still needs them.
        return (min(available, memory_budget_mb * 1024**2) if memory_budget_mb else available), False, 0
    seen = set()
    # ComfyUI may evict VAE/CLIP/other managed weights, and aimdo can reclaim
    # dynamic residency on demand. Never count unrelated CUDA allocations.
    dependencies = {id(p.model) for p in model.model_patches_models()}
    for patcher in [model, *mm.loaded_models()]:
        identity = id(patcher.model)
        if identity not in seen and identity not in dependencies and patcher.current_loaded_device() == device:
            available += int(patcher.loaded_size())
            seen.add(identity)
    budget = min(available, memory_budget_mb * 1024**2) if memory_budget_mb else available
    streaming = model.is_dynamic() or mm.vram_state in (mm.VRAMState.NORMAL_VRAM, mm.VRAMState.LOW_VRAM, mm.VRAMState.NO_VRAM)
    return budget, streaming, int(mm.extra_reserved_memory())


def sample_chunk(model, positive, negative, cfg, samples, mask, noise_tensor, sampler, sigmas, seed):
    import comfy.samplers
    import comfy.model_management as mm
    import comfy.utils
    import latent_preview
    guider = comfy.samplers.CFGGuider(model)
    if negative is not None:
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
    elif cfg != 1:
        raise ValueError('CFG other than 1 requires negative conditioning.')
    else:
        guider.inner_set_conds({'positive': positive})
    callback = latent_preview.prepare_callback(model, len(sigmas) - 1)
    output = guider.sample(noise_tensor, samples, sampler, sigmas, denoise_mask=mask,
                           callback=callback, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
    return output.to(mm.intermediate_device())


class DaSiWaH3TiledUpscale:
    CATEGORY = 'DaSiWa/MiniMax H3'
    FUNCTION = 'upscale'
    RETURN_TYPES = ('LATENT', 'STRING')
    RETURN_NAMES = ('latent', 'plan')
    DESCRIPTION = ('Single-node H3 latent upscale and per-step Tiled Diffusion. Automatic conservative '
                   'tile/temporal planning; optional Director guide + video VAE re-encode endpoint images. '
                   'Audio is preserved. No bbaudio dependency; no downstream widget introspection.')

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        from .h3_latent_upscale import upscale_model_names
        return {'required': {
            'model': ('MODEL',), 'conditioning': ('CONDITIONING',), 'latent': ('LATENT',),
            'scale': ('FLOAT', {'default': 2.0, 'min': 1.0, 'max': 8.0, 'step': 0.1}),
            'upscale_model': (['interpolation'] + upscale_model_names(),),
            'steps': ('INT', {'default': 1, 'min': 1, 'max': 100}),
            'denoise': ('FLOAT', {'default': 0.2, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
            'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff, 'control_after_generate': True}),
        }, 'optional': {
            'director_guide': ('MINIMAX_H3_DIRECTOR_GUIDE',), 'vae': ('VAE',),
            'start_image': ('IMAGE',), 'negative': ('CONDITIONING',),
            'cfg': ('FLOAT', {'default': 1.0, 'min': 0, 'max': 20, 'step': 0.1}),
            'sampler_name': (comfy.samplers.SAMPLER_NAMES, {'default': 'euler'}),
            'scheduler': (comfy.samplers.SCHEDULER_NAMES, {'default': 'simple'}),
            'memory_budget_mb': ('INT', {'default': 0, 'min': 0, 'max': 131072,
                                       'tooltip': '0: automatic device-memory estimate. Nonzero caps the planning budget; not a hard allocator limit.'}),
            'sampler': ('SAMPLER',), 'noise': ('NOISE',),
            'continuity_context': ('DF_H3_CONTINUITY_CONTEXT',),
            'upscale_precision': (['auto', 'bf16', 'fp16', 'fp32'], {'default': 'auto',
                'tooltip': 'Learned latent-upscaler compute precision only. Auto uses ComfyUI device/backend policy. Does not change the incoming diffusion model or VAE. Interpolation does not use this setting.'}),
            'continuity_soft_refine': ('BOOLEAN', {'default': False,
                'tooltip': 'Active Continuity only: smoothly introduce video refinement over the source-tail overlap. Not RGB color matching.'}),
            'continuity_mask_strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.05,
                'tooltip': '0: original hard mask. 1: full smoothstep ramp over the existing Continuity overlap. Ignored unless soft refine is enabled and Continuity is active.'}),
            'spatial_tiling': ('BOOLEAN', {'default': True,
                'tooltip': 'Split diffusion into spatial tiles. Off uses the full target canvas; never re-enabled automatically.'}),
            'temporal_chunking': ('BOOLEAN', {'default': True,
                'tooltip': 'Split learned upscale and diffusion into temporal windows. Off processes the complete video in each pass.'}),
        }}

    @torch.inference_mode()
    def upscale(self, model, conditioning, latent, scale=2.0, upscale_model='interpolation', steps=1,
                denoise=0.2, seed=0, director_guide=None, vae=None, start_image=None,
                negative=None, cfg=1.0,
                sampler_name='euler', scheduler='simple', memory_budget_mb=0,
                sampler=None, noise=None, upscale_precision='auto', continuity_context=None,
                continuity_soft_refine=False, continuity_mask_strength=1.0,
                spatial_tiling=True, temporal_chunking=True):
        import comfy.model_management as mm
        import comfy.model_base
        import comfy.samplers
        from comfy.nested_tensor import NestedTensor
        from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, BasicScheduler
        from .h3_latent_upscale import H3LatentUpscaler, upscale_model_names
        from .h3_tiled_sampling import (plan_tiles, temporal_ranges, prepare_conditioning,
                                        append_video, H3TiledDiffusion, _frames)
        if not isinstance(model.model, comfy.model_base.MiniMaxH3):
            raise ValueError('DaSiWa H3 Tiled Upscale requires a native MiniMax H3 model.')
        if model.model_options.get('model_function_wrapper') is not None:
            raise ValueError('This node installs Tiled Diffusion itself. Connect the unwrapped H3 model, not bbaudio Tiled Diffusion or another model-function wrapper.')
        if not 0 <= denoise <= 1 or steps < 1:
            raise ValueError('Denoise must be 0–1 and steps must be positive.')
        samples = latent.get('samples')
        if not getattr(samples, 'is_nested', False) or len(samples.tensors) != 2:
            raise ValueError('Expected native H3 packed video+audio LATENT, not an IMAGE or video-only latent.')
        video, audio = samples.tensors
        if video.ndim != 5 or video.shape[:2] != (1, 24) or audio.ndim != 4 or audio.shape[:3] != (1, 32, 2):
            raise ValueError('Expected H3 video [1,24,T,H,W] and audio [1,32,2,A].')
        if video.shape[2] < 1 or audio.shape[-1] < 1:
            raise ValueError('Video and audio latent streams must be nonempty.')
        if latent.get('noise_mask') is not None:
            raise ValueError('Masked/inpaint source latents are not supported by this upscale node. Finish that pass before upscaling.')
        for _, md in conditioning:
            if md.get('control') is not None:
                raise ValueError('ControlNet/Fun Control conditioning is not supported in H3 Tiled Upscale.')
        if upscale_precision not in ('auto', 'bf16', 'fp16', 'fp32'):
            raise ValueError(f'Unknown H3 upscale precision: {upscale_precision}')
        continuity = None
        if continuity_context is not None:
            from .h3_upscale_continuity import continuity_plan
            continuity = continuity_plan(video, audio, continuity_context)
        width, height = target_size(video, scale)
        if director_guide is not None and not isinstance(director_guide, dict):
            raise ValueError('director_guide must be a MiniMax H3 Director guide.')
        if director_guide is not None and director_guide.get('mode') == 'Image Inpaint':
            raise ValueError('H3 Tiled Upscale handles video modes, not Image Inpaint.')
        device = mm.get_torch_device()
        budget, streaming_weights, reserved = refinement_memory_budget(model, device, memory_budget_mb)
        model_bytes = int(model.model_size()) if device.type != 'cpu' else 0
        text_tokens, ref_tokens, keyframe_tokens = conditioning_token_counts(conditioning)
        if negative is not None:
            nt, nr, nk = conditioning_token_counts(negative)
            text_tokens, ref_tokens = max(text_tokens, nt), max(ref_tokens, nr)
            keyframe_tokens = max(keyframe_tokens, nk)
        plan = plan_tiles(video.shape, width, height, budget, model_bytes=model_bytes,
                          text_tokens=text_tokens, ref_tokens=ref_tokens, keyframe_tokens=keyframe_tokens,
                          streaming_weights=streaming_weights,
                          reserved_bytes=reserved + audio.numel() * (6 * audio.element_size() + 8),
                          audio_tokens=audio.shape[-2] * audio.shape[-1],
                          latent_element_size=video.element_size(), enforce_budget=denoise > 0,
                          spatial_tiling=spatial_tiling, temporal_chunking=temporal_chunking)
        ranges = temporal_ranges(video.shape[2], plan['chunk_tokens'], plan['temporal_overlap_tokens'])
        # CPU staging is bounded by the complete output+noise, not constant-memory
        # streaming. Refuse an unsafe host allocation instead of relying on OOM.
        output_shape = (*video.shape[:3], height // 16, width // 16)
        bytes_needed = math.prod(output_shape) * video.element_size() * 4 + audio.numel() * audio.element_size() * 3
        if bytes_needed > int(mm.get_free_memory(torch.device('cpu'))) * 0.6:
            raise MemoryError('H3 upscale output/noise staging exceeds the conservative available-RAM budget. Reduce target size or video length.')
        first = start_image if start_image is not None else director_endpoint(director_guide, 'first_frame')
        last = director_endpoint(director_guide, 'last_frame')
        if continuity is not None:
            positive, endpoint_info = conditioning, 'continuity source-tail conditioning (no endpoint re-encode)'
        elif denoise > 0:
            positive, endpoint_info = encode_endpoints(conditioning, vae, first, last, width, height,
                                                        last_frame_index=_frames(video.shape[2]) - 1)
        else:
            positive, endpoint_info = conditioning, 'endpoint encoding skipped (no diffusion)'
        names = upscale_model_names()
        selected = upscale_model
        if selected != 'interpolation' and selected not in names:
            raise ValueError(f'H3 latent upscale checkpoint is not available: {selected}')
        # Release only this supplied diffusion model and its clones, never all
        # loaded models/services. The sampler reloads it via native management.
        if selected != 'interpolation':
            mm.unload_model_and_clones(model, unload_additional_models=False)
        upscaled = torch.empty(output_shape, device='cpu', dtype=video.dtype)
        backend = None
        try:
            if selected != 'interpolation':
                backend = H3LatentUpscaler(selected, device, precision=upscale_precision)
            halo = backend.temporal_halo if backend is not None else 0
            actual_precision = str(backend.dtype).removeprefix('torch.') if backend is not None else 'not used (interpolation)'
            for start, end in ranges:
                mm.throw_exception_if_processing_interrupted()
                a, z = max(0, start - halo), min(video.shape[2], end + halo)
                part = video[:, :, a:z]
                result = (backend.upscale(part, height // 16, width // 16) if backend is not None
                          else resize_video(part, height // 16, width // 16))
                upscaled[:, :, start:end] = result[:, :, start - a:end - a].to('cpu')
                del result
        finally:
            if backend is not None:
                backend.close()
        audio_cpu = audio.to('cpu')
        if continuity is not None and denoise > 0:
            from .h3_upscale_continuity import align_continuity_conditioning
            positive = align_continuity_conditioning(positive, upscaled, audio_cpu, continuity)
            if negative is not None:
                negative = align_continuity_conditioning(negative, upscaled, audio_cpu, continuity, inject_tail=False)
        report = (f"{video.shape[-1]*16}x{video.shape[-2]*16} → {width}x{height}; "
                  f"upscale={selected}; precision={actual_precision}; tile={plan['tile_width']}x{plan['tile_height']}px; "
                  f"overlap={plan['overlap']}px; temporal={len(ranges)} chunks; "
                  f"spatial_tiling={'on' if spatial_tiling else 'off'}; "
                  f"temporal_chunking={'on' if temporal_chunking else 'off'}; "
                  f"audio=original; {endpoint_info}; {plan['explanation']}")
        if continuity is not None:
            report += '; ' + continuity['explanation']
            report += (f'; soft_refine_mask={continuity_mask_strength:g}' if continuity_soft_refine and denoise > 0
                       else '; soft_refine_mask=off')
        if denoise == 0:
            report += '; diffusion skipped (denoise=0)'
        from .helper_logging import log_dasiwa
        log_dasiwa('MiniMaxH3 Enhanced Upscale', report)
        if denoise == 0:
            result = dict(latent)
            result['samples'] = NestedTensor((upscaled, audio))
            return result, report
        sigmas = BasicScheduler.execute(model, scheduler, steps, denoise)[0]
        if sigmas.ndim != 1 or len(sigmas) < 2 or not torch.isfinite(sigmas).all() or (sigmas[:-1] < sigmas[1:]).any():
            raise ValueError('SIGMAS must be a finite, nonincreasing schedule with at least two entries.')
        sampler = sampler if sampler is not None else comfy.samplers.sampler_object(sampler_name)
        noise = noise if noise is not None else Noise_RandomNoise(seed)
        # One full noise field makes temporal overlaps deterministic: never
        # regenerate a different field independently for adjacent chunks.
        full_noise = noise.generate_noise({'samples': NestedTensor((upscaled, audio_cpu)),
                                          **({'batch_index': latent['batch_index']} if 'batch_index' in latent else {})})
        if not getattr(full_noise, 'is_nested', False) or len(full_noise.tensors) != 2:
            raise ValueError('The supplied NOISE must generate native video+audio nested tensors.')
        working = model.clone()
        if spatial_tiling:
            working.set_model_unet_function_wrapper(H3TiledDiffusion(plan['tile_width'], plan['tile_height'], plan['overlap']))
        completed = None
        try:
            for start, end in ranges:
                mm.throw_exception_if_processing_interrupted()
                a, z = audio_window(video.shape[2], audio.shape[-1], start, end)
                part = upscaled[:, :, start:end]
                if continuity is not None and end <= continuity['refine_start_token']:
                    completed = append_video(completed, part, start)
                    continue
                ac = audio_cpu[..., a:z]
                chunk_samples = NestedTensor((part, ac))
                video_mask = continuity_refine_mask(part, start, continuity,
                    soft=continuity_soft_refine, strength=continuity_mask_strength)
                mask = NestedTensor((video_mask, torch.zeros_like(ac)))
                previous = completed
                pc = prepare_conditioning(positive, start, end, (height // 16, width // 16), previous_video=previous)
                nc = (prepare_conditioning(negative, start, end, (height // 16, width // 16)) if negative is not None else None)
                chunk_noise = NestedTensor((full_noise.tensors[0][:, :, start:end], full_noise.tensors[1][..., a:z]))
                result = sample_chunk(working, pc, nc, cfg, chunk_samples, mask, chunk_noise,
                                      sampler, sigmas, getattr(noise, 'seed', seed))
                completed = append_video(completed, result.tensors[0].to('cpu'), start)
        finally:
            # No wrapper escapes into the upstream model; drop execution-local
            # conditioning/layout/tensor references after success or cancellation.
            working.model_options.pop('model_function_wrapper', None)
        result = dict(latent)
        if completed is None:
            raise RuntimeError('H3 temporal sampling produced no video chunks.')
        if continuity is not None:
            # Preserve the already-upscaled old prefix exactly, even if native
            # normalization or overlap blending introduced rounding drift.
            completed[:, :, :continuity['refine_start_token']] = upscaled[:, :, :continuity['refine_start_token']]
        result['samples'] = NestedTensor((completed.to(video.dtype), audio))
        return result, report
