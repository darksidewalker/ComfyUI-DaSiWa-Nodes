"""Thin Director adapter for ComfyUI's native MiniMax H3 nodes."""

from .helper_minimax_h3_director import normalize_guide
from .helper_refmod_format import refmod_fingerprint
from .helper_logging import log_dasiwa
import uuid


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

class MiniMaxH3DirectorGuide:
    """Turn one Director guide socket into the exact native MiniMax H3 call."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "guide": ("MINIMAX_H3_DIRECTOR_GUIDE",),
            },
            "optional": {"audio_vae": ("VAE",)},
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "DF_H3_CONTINUITY_CONTEXT")
    RETURN_NAMES = ("positive", "latent", "continuity_context")
    FUNCTION = "apply"
    CATEGORY = "DaSiWa/MiniMax H3"

    @classmethod
    def IS_CHANGED(cls, clip, vae, guide, audio_vae=None, **kwargs):
        if isinstance(guide, dict) and "continuity" in guide:
            from .h3_continuity.core import parse_settings
            settings = parse_settings(guide["continuity"])
            if settings["capture"] or settings["operation"] == "continue":
                return float("nan")
        items = guide.get("minimax_ref_items", []) if isinstance(guide, dict) else []
        return tuple((item["name"], refmod_fingerprint(item["name"])) for item in items if item.get("name"))

    def apply(self, clip, vae, guide, audio_vae=None, prompt=None, unique_id=None):
        raw = guide.get("continuity") if isinstance(guide, dict) else None
        if raw is None:
            return (*self._apply_native(clip, vae, guide, audio_vae), {"disabled": True})
        from .h3_continuity.core import ClipStore, parse_settings, prepare_continuation, add_tail
        from .h3_continuity.vendor.continuation_nodes import _require_native_arbitrary_guides
        settings = parse_settings(raw)
        continuing = settings["operation"] == "continue"
        if not continuing and (not settings["capture"] or guide["mode"] == "Image Inpaint"):
            return (*self._apply_native(clip, vae, guide, audio_vae), {"disabled": True})
        if guide["mode"] == "Image Inpaint":
            raise ValueError("Continuity requires a video mode, not Image Inpaint.")
        if abs(float(guide.get("frame_rate", 24)) - 24) > 1e-6:
            raise ValueError("H3 continuity uses native 24 fps. Set Director frame_rate to 24.")
        from .h3_continuity.validation import validate_capture_graph
        validate_capture_graph(prompt, unique_id)
        _require_native_arbitrary_guides()
        context = {**settings, "run_id": uuid.uuid4().hex, "mode": guide["mode"],
                   "resolved_prompt": guide.get("resolved_prompt", guide.get("prompt", ""))}
        if not continuing:
            positive, latent = self._apply_native(clip, vae, guide, audio_vae)
            context.update(source_id="", overlap_frames=0, extension_frames=0)
            return positive, latent, context
        if settings["source_kind"] == "video":
            from .h3_continuity.video_source import import_checkpoint
            _validate_h3_vaes(vae, audio_vae, "REF2VA")
            previous, metadata, source_id = import_checkpoint(settings, guide, vae, audio_vae)
            context["source_id"] = source_id
        else:
            previous, metadata = ClipStore().load(settings["session"], settings["source_id"])
        updated, target, layout = prepare_continuation(previous, metadata, guide, settings)
        positive, _ = self._apply_native(clip, vae, updated, audio_vae)
        context.update(layout=layout, resolved_prompt=updated["resolved_prompt"],
                       provenance={"source_kind": settings["source_kind"],
                                   "source_video_id": settings.get("source_video_id", "")})
        return add_tail(positive, previous, target, layout), target, context

    def _apply_native(self, clip, vae, guide, audio_vae=None):
        state = normalize_guide(guide)
        pure_preencoded = bool(state.minimax_ref_items) and not any((
            state.ref_images, state.ref_videos, state.ref_video_audios, state.ref_audios))
        _validate_h3_vaes(vae, audio_vae, state.mode, audio_required=not pure_preencoded)

        if state.mode in {"T2VA", "I2VA", "FL2VA", "L2VA"}:
            native = _native_node("MiniMaxH3ImageToVideo")
            log_dasiwa(
                "MiniMax H3 Director Guide",
                f"mode=FL2VA; upstream=MiniMaxH3ImageToVideo; clip={type(clip).__name__}; "
                f"vae={type(vae).__name__}; audio_vae=not-used; frames={state.length}; "
                f"first_frame={state.first_frame is not None}; last_frame={state.last_frame is not None}",
            )
            positive, latent = native.execute(
                clip, vae, state.resolved_prompt, state.width, state.height, state.length,
                state.first_frame, state.last_frame,
            )
            log_dasiwa(
                "MiniMax H3 Director Guide",
                f"passed forward from MiniMaxH3ImageToVideo: conditioning={_describe_output(positive)}; "
                f"latent={_describe_output(latent)}",
            )
            return positive, latent

        if state.mode == "Image Inpaint":
            if state.first_frame is None:
                raise ValueError("Image Inpaint requires one image keyframe")
            native = _native_node("MiniMaxH3ImageToVideo")
            log_dasiwa(
                "MiniMax H3 Director Guide",
                f"mode=Image Inpaint; upstream=MiniMaxH3ImageToVideo; clip={type(clip).__name__}; "
                f"vae={type(vae).__name__}; audio_vae=not-used; frames=5; "
                f"first_frame={state.first_frame is not None}; last_frame=None",
            )
            positive, latent = native.execute(
                clip, vae, state.resolved_prompt, state.width, state.height, 5,
                state.first_frame, None,
            )
            log_dasiwa(
                "MiniMax H3 Director Guide",
                f"passed forward from MiniMaxH3ImageToVideo: conditioning={_describe_output(positive)}; "
                f"latent={_describe_output(latent)}",
            )
            return positive, latent

        ref_blocks = list(state.minimax_ref_items or [])
        if audio_vae is None and not pure_preencoded:
            raise ValueError("audio_vae is required for REF2VA")
        native = _native_node("MiniMaxH3ReferenceToVideo")
        ref_clip = clip
        native_blocks = []
        if ref_blocks:
            decoded_items = []
            for block in ref_blocks:
                if block.get("kind") == "audio":
                    decoded_items.append({"type": "audio"})
                    native_blocks.append({"kind": "audio", "ref_audio_t": int(block.get("latent_t", 0)),
                                          "audio_latent": block["latent"]})
                    continue
                latent = block["latent"]
                pixels = vae.decode(latent)
                if getattr(pixels, "ndim", 0) == 5 and pixels.shape[0] == 1:
                    pixels = pixels[0]
                if getattr(pixels, "ndim", 0) != 4 or pixels.shape[-1] != 3:
                    raise ValueError("Connect the MiniMax H3 video VAE: invalid decoded RefMod shape.")
                # Video refs need ref_audio_t/latent_t for ComfyUI's PackedLayout; images don't.
                is_video = block["kind"] == "video" or (getattr(latent, "ndim", 0) >= 5 and latent.shape[2] > 1)
                decoded_items.append({"type": "image" if not is_video else "video", "data": pixels.cpu().clone()})
                if is_video:
                    native_block = {"kind": block["kind"], "latent": latent,
                                    "latent_t": latent.shape[2], "latent_h": latent.shape[3], "latent_w": latent.shape[4],
                                    "ref_audio_t": 0, "audio_latent": None}
                else:
                    # Image refs are still 5D (B,C,T=1,H,W); skip the temporal dim.
                    native_block = {"kind": "image", "latent": latent,
                                    "latent_h": latent.shape[3], "latent_w": latent.shape[4]}
                native_blocks.append(native_block)

            class ReferenceClip:
                def tokenize(self, text, **kwargs):
                    # Native H3 contract: reference media are supplied only at tokenize time.
                    kwargs["minimax_ref_items"] = list(kwargs.get("minimax_ref_items") or []) + decoded_items
                    return clip.tokenize(text, **kwargs)

                def encode_from_tokens_scheduled(self, tokens):
                    return clip.encode_from_tokens_scheduled(tokens)
            ref_clip = ReferenceClip()
        log_dasiwa(
            "MiniMax H3 Director Guide",
            f"mode=REF2VA; upstream=MiniMaxH3ReferenceToVideo; clip={type(clip).__name__}; "
            f"vae={type(vae).__name__}; audio_vae={type(audio_vae).__name__}; frames={state.length}; "
            f"refs=images:{len(state.ref_images)},videos:{len(state.ref_videos)},"
            f"video_audio:{len(state.ref_video_audios)},audio:{len(state.ref_audios)}",
        )
        # Current native signature puts prompt before the optional VAEs. Bind every
        # argument by name so this adapter remains safe if ComfyUI reorders inputs.
        positive, latent = native.execute(
            clip=ref_clip, prompt=state.resolved_prompt,
            width=state.width, height=state.height, length=state.length,
            ref_image_size=state.ref_image_size, vae=vae, audio_vae=audio_vae,
            ref_images=state.ref_images, ref_videos=state.ref_videos,
            ref_video_audios=state.ref_video_audios, ref_audios=state.ref_audios,
        )
        if ref_blocks:
            positive = [[embedding, {**metadata, "minimax_refs": list(metadata.get("minimax_refs", [])) + native_blocks}]
                        for embedding, metadata in positive]
        log_dasiwa(
            "MiniMax H3 Director Guide",
            f"passed forward from MiniMaxH3ReferenceToVideo: conditioning={_describe_output(positive)}; "
            f"latent={_describe_output(latent)}",
        )
        return positive, latent


NODE_CLASS_MAPPINGS = {"MiniMaxH3DirectorGuide": MiniMaxH3DirectorGuide}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3DirectorGuide": "MiniMax H3 Director Guide"}
