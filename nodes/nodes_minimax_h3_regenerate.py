"""MiniMax H3 Hi-Res Regenerate: the Open-H3-Regenerate-2K refine recipe on the Enhanced Upscale core.

Reference: NVlabs/Sana sol-engine models/Open-H3-Regenerate-2K (open_h3/pipeline.py, sigmas.py).
Learned latent upscale to the target grid (4K via 2K), then three Euler steps at CFG 1 on pinned
sigmas with the merged refiner LoRA. Tiling, temporal windows, Continuity, Director endpoints,
reference tokens and draft-audio passthrough are the unchanged DaSiWaH3TiledUpscale path.
"""
from .nodes_minimax_h3_tiled_upscale import DaSiWaH3TiledUpscale
from .h3_regenerate_telemetry import make_telemetry, phase

# open_h3/sigmas.py SCHEDULES[3]; unshifted, so the model's shift setting does not apply.
SIGMAS = (0.909375, 0.8571428571428571, 0.6666666666666666, 0.0)
# Landscape (width, height) as upstream RESOLUTIONS: 1080P/4K on whole latent patches (1088/2176 rows).
TARGETS = {'1080P': (1920, 1088), '2K': (2560, 1440), '4K': (3840, 2176)}


def regenerate_size(width, height, target):
    """Pixel target for a draft: upstream's 16:9 grid for near-16:9 drafts, else the short side kept."""
    long_side, short_side = TARGETS[target]
    portrait = height > width
    aspect = max(width, height) / min(width, height)
    # Upstream itself maps the 1.75 draft (1344x768) onto 16:9; larger mismatches would stretch.
    if abs(aspect / (long_side / short_side) - 1) <= 0.05:
        return (short_side, long_side) if portrait else (long_side, short_side)
    scale = short_side / min(width, height)
    return int(round(width * scale / 32)) * 32, int(round(height * scale / 32)) * 32


def regenerate_stages(width, height, target):
    """Upscale stages in pixels; above 2K the upscaler first goes to 2K (upstream trained range)."""
    final = regenerate_size(width, height, target)
    stages = [final]
    if target == '4K':
        via = regenerate_size(width, height, '2K')
        if via[0] > width or via[1] > height:
            stages.insert(0, via)
    return stages


def is_diffusers_lora(state_dict):
    return any(k.startswith('transformer.') or k.startswith('transformer_blocks.') for k in state_dict)


class DaSiWaH3HiResRegenerate:
    CATEGORY = 'DaSiWa/MiniMax H3'
    FUNCTION = 'regenerate'
    RETURN_TYPES = ('LATENT', 'STRING')
    RETURN_NAMES = ('latent', 'plan')
    DESCRIPTION = ('Open-H3-Regenerate-2K refine preset: learned latent upscale to 1080P/2K/4K, then three Euler '
                   'steps at CFG 1 on the published sigmas with the refiner LoRA patched onto a clone of the '
                   'model only. Needs the LoRA converted to native H3 keys (Quant Station "Diffusers → H3"). '
                   'Connect the plain H3 model without Turbo/Sharp LoRAs: the merged LoRA already contains them.')

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        from .h3_latent_upscale import upscale_model_names
        names = upscale_model_names()
        # The reference recipe uses the 3D upscaler; fall back to the Enhanced Upscale preference.
        base = lambda n: n.replace('\\', '/').rsplit('/', 1)[-1].lower()
        preferred = next((n for n in names if '3d_conv' in base(n)),
                         next((n for n in names if '2d_conv' in base(n)), 'interpolation'))
        return {'required': {
            'model': ('MODEL', {'tooltip': 'Native MiniMax H3 model without Turbo/Sharp LoRAs.'}),
            'conditioning': ('CONDITIONING', {'tooltip': 'Positive H3 conditioning for refinement. CFG is fixed at 1; no negative conditioning is needed.'}),
            'latent': ('LATENT', {'tooltip': 'Completed native H3 packed video+audio latent. Original audio passes through unchanged.'}),
            'lora_name': (folder_paths.get_filename_list('loras'), {
                'tooltip': 'Open-H3-Regenerate-2K LoRA converted to native H3 keys (exact or SVD).'}),
            'strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 2.0, 'step': 0.05,
                'tooltip': 'Refiner LoRA strength, default 1 as in the reference recipe. Zero disables the LoRA, not the three-step refinement.'}),
            'upscale_model': (['interpolation'] + names, {'default': preferred,
                'tooltip': 'Learned latent upscaler from latent_upscale_models. Prefers 3D, then 2D when installed; interpolation uses no learned checkpoint and differs from the reference recipe.'}),
            'target': (list(TARGETS), {'default': '2K',
                'tooltip': '1080P/4K are computed at 1088/2176 rows like the reference; crop after decode if needed. '
                           'Near-16:9 drafts get the exact 16:9 grid, other aspects keep their aspect at this short side.'}),
            'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff, 'control_after_generate': True,
                'tooltip': 'Refinement noise seed. A connected noise input supplies the noise instead.'}),
        }, 'optional': {
            'director_guide': ('MINIMAX_H3_DIRECTOR_GUIDE', {'tooltip': 'Optional matching Director Guide for endpoint re-encoding at the target size; also connect the H3 video VAE.'}),
            'vae': ('VAE', {'tooltip': 'H3 video VAE for optional endpoint image re-encoding, not the audio VAE.'}),
            'start_image': ('IMAGE', {'tooltip': 'Optional explicit first-frame image for target-size re-encoding; requires the H3 video VAE.'}),
            'noise': ('NOISE', {'tooltip': 'Optional ComfyUI noise source replacing the seed-generated refinement noise.'}),
            'continuity_context': ('DF_H3_CONTINUITY_CONTEXT', {'tooltip': 'Matching Guide continuity context. Follow the Continuity assembly order; protects the old prefix from diffusion refinement.'}),
            'spatial_tiling': ('BOOLEAN', {'default': True,
                'tooltip': 'Allow automatic spatial tiling. Off samples the full canvas like the reference (more VRAM).'}),
        }}

    def load_lora(self, lora_name):
        import comfy.utils
        import folder_paths
        path = folder_paths.get_full_path_or_raise('loras', lora_name)
        if self.loaded_lora is None or self.loaded_lora[0] != path:
            self.loaded_lora = None
            self.loaded_lora = (path, comfy.utils.load_torch_file(path, safe_load=True))
        return self.loaded_lora[1]

    def regenerate(self, model, conditioning, latent, lora_name, strength=1.0, upscale_model='interpolation',
                   target='2K', seed=0, director_guide=None, vae=None, start_image=None, noise=None,
                   continuity_context=None, spatial_tiling=True):
        import comfy.lora
        import comfy.lora_convert
        import comfy.model_base
        if not isinstance(model.model, comfy.model_base.MiniMaxH3):
            raise ValueError('DaSiWa H3 Hi-Res Regenerate requires a native MiniMax H3 model.')
        if target not in TARGETS:
            raise ValueError(f'Unknown H3 Hi-Res Regenerate target: {target}')
        samples = latent.get('samples')
        if not getattr(samples, 'is_nested', False) or len(samples.tensors) != 2:
            raise ValueError('Expected native H3 packed video+audio LATENT, not an IMAGE or video-only latent.')
        video = samples.tensors[0]
        stages = regenerate_stages(video.shape[-1] * 16, video.shape[-2] * 16, target)
        telemetry = make_telemetry()
        with phase(telemetry, 'lora'):
            lora = self.load_lora(lora_name)
            if is_diffusers_lora(lora):
                # ComfyUI would skip every Diffusers key silently and run without the refiner.
                raise ValueError(f'{lora_name} has Diffusers keys. Convert it to native H3 keys first '
                                 '(Quant Station "Diffusers → H3" or DEAS tools/convert_open_h3_regenerate_lora.py).')
            patches = comfy.lora.load_lora(comfy.lora_convert.convert_lora(lora),
                                           comfy.lora.model_lora_keys_unet(model.model, {}))
            if not patches:
                raise ValueError(f'{lora_name} matches no layer of this H3 model; it is not an H3 refiner LoRA.')
            patched = model.clone()  # the caller's model and its generation chain stay unpatched
            patched.add_patches(patches, strength)
        output, report = DaSiWaH3TiledUpscale().upscale(
            patched, conditioning, latent, upscale_model=upscale_model, steps=len(SIGMAS) - 1, denoise=1.0,
            seed=seed, director_guide=director_guide, vae=vae, start_image=start_image, cfg=1.0,
            sampler_name='euler', noise=noise, continuity_context=continuity_context,
            spatial_tiling=spatial_tiling, sigmas=SIGMAS, sizes=stages, telemetry=telemetry)
        return output, f'H3 Hi-Res Regenerate {target}; lora={lora_name} ({len(patches)} layers, strength {strength:g}); {report}'
