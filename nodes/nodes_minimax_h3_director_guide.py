"""MiniMax H3 Director Guide with optional RefMod integration."""

import math

from .helper_minimax_h3_director import normalize_guide
from .helper_logging import log_dasiwa


def _describe_output(value) -> str:
    if isinstance(value, dict):
        details = []
        for key, item in value.items():
            shape = getattr(item, "shape", None)
            details.append(f"{key}:{tuple(shape) if shape is not None else type(item).__name__}")
        return "{" + ", ".join(details) + "}"
    shape = getattr(value, "shape", None)
    if shape is not None:
        return f"{type(value).__name__}{tuple(shape)}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}[{len(value)}]"
    return type(value).__name__


def _native_node(name):
    """Resolve at execution so installed ComfyUI updates are used automatically."""
    try:
        from comfy_extras import nodes_minimax_h3
        return getattr(nodes_minimax_h3, name)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            f"MiniMax H3 Director requires ComfyUI's native {name} node. "
            "Update ComfyUI to a version that includes MiniMax H3 support."
        ) from exc

def _h3_vae_kind(value):
    first_stage = getattr(value, "first_stage_model", None)
    class_name = type(first_stage).__name__ if first_stage is not None else ""

    if class_name == "MiniMaxH3VideoVAE":
        return "video"
    if class_name == "MiniMaxH3AudioVAE":
        return "audio"
    return "unknown"


def _validate_h3_vaes(vae, audio_vae, mode, *, audio_required=True):
    vae_kind = _h3_vae_kind(vae)
    audio_vae_kind = _h3_vae_kind(audio_vae)

    if vae_kind == "audio":
        raise ValueError(
            "MiniMax H3 Director: Audio VAE is connected to the 'vae' "
            "(Video VAE) slot. Connect minimax_h3_video_vae_* to 'vae'."
        )

    if mode == "REF2VA":
        if audio_required and audio_vae is None:
            raise ValueError("audio_vae is required for REF2VA")

        if audio_vae_kind == "video":
            raise ValueError(
                "MiniMax H3 Director: Video VAE is connected to the "
                "'audio_vae' slot. Connect minimax_h3_audio_vae_* to 'audio_vae'."
            )

def execute_director_guide(clip, vae, guide, audio_vae=None,
                           retention=1.0, reference_fps=24.0, max_total_tokens=0):
    from .helper_director_refmods import active_mods, mappings, translate
    state = normalize_guide(guide)
    mods = state.ref_mods
    active = active_mods(mods)
    if active and state.mode != "REF2VA":
        raise ValueError("Director RefMods requires REF2VA mode.")
    if not math.isfinite(retention) or not 0 < retention <= 1:
        raise ValueError("RefMod retention must be greater than 0 and at most 1; disable a slot in Director instead.")
    if not math.isfinite(reference_fps) or not 1 <= reference_fps <= 120:
        raise ValueError("reference_fps must be between 1 and 120.")
    if max_total_tokens < 0:
        raise ValueError("max_total_tokens must be nonnegative.")
    audio_required = state.mode == "REF2VA" and (
        not active or state.ref_audios or state.ref_video_audios
    )
    _validate_h3_vaes(
        vae, audio_vae, state.mode, audio_required=audio_required
    )
    if state.mode in {"T2VA", "I2VA", "FL2VA", "L2VA", "Image Inpaint"}:
        if state.mode == "Image Inpaint" and state.first_frame is None:
            raise ValueError("Image Inpaint requires one image keyframe")
        log_dasiwa(
            "MiniMax H3 Director Guide",
            f"mode={state.mode}; upstream=MiniMaxH3ImageToVideo; "
            f"clip={type(clip).__name__}; vae={type(vae).__name__}; "
            f"audio_vae=not-used; "
            f"frames={5 if state.mode == 'Image Inpaint' else state.length}; "
            f"first_frame={state.first_frame is not None}; "
            f"last_frame={state.last_frame is not None and state.mode != 'Image Inpaint'}",
        )
        positive, latent = _native_node("MiniMaxH3ImageToVideo").execute(
            clip, vae, state.resolved_prompt, state.width, state.height,
            5 if state.mode == "Image Inpaint" else state.length,
            state.first_frame, None if state.mode == "Image Inpaint" else state.last_frame)
        log_dasiwa("MiniMax H3 Director Guide",
                   f"passed forward from MiniMaxH3ImageToVideo: conditioning={_describe_output(positive)}; latent={_describe_output(latent)}")
        return positive, latent, "No RefMods connected."

    tags, descriptions, names, reference_map = mappings(mods, state.ref_images, state.ref_videos)
    prompt = translate(state.resolved_prompt, tags)
    blocks = []
    for mod, strength, slot in active:
        block = mod.ref_block(strength * retention)
        block["refmod"] = True
        blocks.append(block)

    def token_count(block):
        if block["kind"] == "audio":
            return block["ref_audio_t"] * 2
        return (block.get("latent_t", 1) * (block["latent_h"] // 2) *
                (block["latent_w"] // 2) + block.get("ref_audio_t", 0) * 2)

    if max_total_tokens and sum(map(token_count, blocks)) > max_total_tokens:
        raise ValueError("RefMods exceed the total reference token budget.")

    items = []
    for block in blocks:
        pixels = vae.decode(block["latent"])
        if pixels.ndim == 5 and pixels.shape[0] == 1:
            pixels = pixels[0]
        if pixels.ndim != 4 or pixels.shape[-1] != 3 or pixels.shape[0] < 1:
            raise ValueError("Connect the MiniMax H3 video VAE: invalid decoded RefMod shape.")
        if block["kind"] == "image":
            item = {"type": "image", "data": pixels[:1].cpu().clone()}
        else:
            times = [i / 2 for i in range(math.ceil(pixels.shape[0] * 2 / reference_fps))]
            indices = [min(round(t * reference_fps), pixels.shape[0] - 1) for t in times]
            item = {"type": "video", "data": pixels[indices].cpu(), "timestamps": times}
        items.append(item)
        del pixels

    class ReferenceClip:
        def tokenize(self, text, **kwargs):
            kwargs["minimax_ref_items"] = list(kwargs.get("minimax_ref_items") or []) + items
            return clip.tokenize(text, **kwargs)

        def encode_from_tokens_scheduled(self, tokens):
            return clip.encode_from_tokens_scheduled(tokens)

    if audio_required and audio_vae is None:
        raise ValueError("audio_vae is required for REF2VA audio references.")
    log_dasiwa(
        "MiniMax H3 Director Guide",
        f"mode=REF2VA; upstream=MiniMaxH3ReferenceToVideo; "
        f"clip={type(clip).__name__}; vae={type(vae).__name__}; "
        f"audio_vae={type(audio_vae).__name__ if audio_vae is not None else 'not-used'}; "
        f"frames={state.length}; refs=images:{len(state.ref_images)},"
        f"videos:{len(state.ref_videos)},video_audio:{len(state.ref_video_audios)},"
        f"audio:{len(state.ref_audios)},refmods:{len(active)}",
    )
    positive, latent = _native_node("MiniMaxH3ReferenceToVideo").execute(
        clip=ReferenceClip() if active else clip, vae=vae, audio_vae=audio_vae,
        prompt=prompt, width=state.width, height=state.height, length=state.length,
        ref_image_size=state.ref_image_size, ref_images=state.ref_images,
        ref_videos=state.ref_videos, ref_video_audios=state.ref_video_audios,
        ref_audios=state.ref_audios)
    if active:
        merged = []
        for embedding, metadata in positive:
            if "minimax_token_tags" not in metadata:
                raise ValueError("Use a MiniMax H3 compatible text/vision encoder for RefMods.")
            refs = list(metadata.get("minimax_refs", [])) + blocks
            total = sum(map(token_count, refs))
            if max_total_tokens and total > max_total_tokens:
                raise ValueError(f"Combined reference tokens {total} exceed budget {max_total_tokens}.")
            merged.append([embedding, {**metadata, "minimax_refs": refs}])
        positive = merged
        log_dasiwa("MiniMax H3 Director Guide", reference_map)
    return positive, latent, reference_map or "No RefMods connected."


class MiniMaxH3DirectorGuide:
    """Turn one Director guide socket into the exact native MiniMax H3 call, with optional RefMod support."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "guide": ("MINIMAX_H3_DIRECTOR_GUIDE",),
            },
            "optional": {
                "audio_vae": ("VAE",),
                "retention": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reference_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.5}),
                "max_total_tokens": ("INT", {"default": 0, "min": 0, "max": 1048576, "step": 64}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "apply"
    CATEGORY = "DaSiWa/MiniMax H3"

    def apply(self, clip, vae, guide, audio_vae=None,
              retention=1.0, reference_fps=24.0, max_total_tokens=0):
        positive, latent, _ = execute_director_guide(
            clip=clip, vae=vae, guide=guide, audio_vae=audio_vae,
            retention=retention, reference_fps=reference_fps, max_total_tokens=max_total_tokens
        )
        return positive, latent


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3DirectorGuide": MiniMaxH3DirectorGuide,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3DirectorGuide": "MiniMax H3 Director Guide",
}
