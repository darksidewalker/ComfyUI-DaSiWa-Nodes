"""Read-only continuity planning and cumulative-timeline H3 upscale guides."""
from collections.abc import Mapping
import math

from comfy.nested_tensor import NestedTensor
from .h3_continuity.core import ClipStore
from .h3_continuity.vendor.continuation_nodes import (
    audio_t_at_frame_boundary,
    frame_count_from_video_latent_t,
    validate_h3_av_latent,
    video_latent_t,
)


def _integer(mapping, key):
    value = mapping.get(key)
    if type(value) is not int:
        raise ValueError(f"Continuity {key} must be an integer.")
    return value


def continuity_plan(video, audio, context):
    """Validate an appended cumulative AV latent without loading its source."""
    if context is None:
        return None
    if not isinstance(context, Mapping):
        raise TypeError("continuity_context must be a mapping.")
    if context.get("disabled") or context.get("operation") != "continue":
        return None
    _, _, total_frames = validate_h3_av_latent(
        {"samples": NestedTensor((video, audio))}, name="cumulative upscale input")
    metadata = ClipStore().inspect(context["session"], context["source_id"])
    source_frames = metadata["frames"]
    source_t = video_latent_t(source_frames)
    if (video.shape[4] * 16, video.shape[3] * 16) != (metadata["width"], metadata["height"]):
        raise ValueError("Cumulative upscale input canvas differs from the continuity source canvas.")
    extension = _integer(context, "extension_frames")
    if extension < 17 or extension % 17:
        raise ValueError("Continuity extension_frames must be a positive multiple of 17.")
    if total_frames != source_frames + extension:
        raise ValueError("Continuity upscale requires the appended cumulative latent: frame count must equal source_frames + extension_frames.")
    layout = context.get("layout")
    if not isinstance(layout, Mapping):
        raise ValueError("Continuity layout must contain native overlap tokens.")
    overlap_t = _integer(layout, "overlap_video_tokens")
    overlap_frames = frame_count_from_video_latent_t(overlap_t)
    if overlap_t > source_t:
        raise ValueError("Continuity overlap exceeds the source video length.")
    if "overlap_frames" in context and _integer(context, "overlap_frames") != overlap_frames:
        raise ValueError("Continuity overlap_frames differs from its native token layout.")
    refine_start = source_t - overlap_t
    if refine_start % 5:
        raise ValueError("Continuity refine start must be native period-aligned (multiple of 5).")
    # A complete five-token native period represents 17 frames, not 17k+5.
    frame_offset = source_frames - overlap_frames
    if frame_offset != refine_start // 5 * 17:
        raise ValueError("Continuity overlap does not preserve the native temporal phase.")
    audio_start = audio_t_at_frame_boundary(frame_offset)
    source_audio_t = audio_t_at_frame_boundary(source_frames)
    if _integer(layout, "overlap_audio_tokens") != source_audio_t - audio_start:
        raise ValueError("Continuity overlap_audio_tokens differs from the global audio boundaries.")
    return dict(refine_start_token=refine_start, source_tokens=source_t,
                frame_offset=frame_offset, audio_start=audio_start,
                source_audio_tokens=source_audio_t, source_frames=source_frames,
                overlap_tokens=overlap_t, total_frames=total_frames,
                explanation=f"Continuity: freeze {refine_start} prefix video tokens; refine from frame {frame_offset} with the learned-upscaled source tail. Preserve original cumulative audio.")


def align_continuity_conditioning(conditioning, upscaled_video, original_audio, plan, *, inject_tail=True):
    """Copy each entry, shift local guides, and optionally replace the native AV tail.

    Tensor slices are cloned for the new anchor; text, refs and unknown metadata
    keep their identities. No source tensor or incoming conditioning is mutated.
    """
    if plan is None:
        return conditioning
    if not isinstance(conditioning, (list, tuple)):
        raise TypeError("conditioning must be a ComfyUI conditioning sequence.")
    start, end = plan["refine_start_token"], plan["source_tokens"]
    offset = plan["frame_offset"]
    result = []
    for entry in conditioning:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2 or not isinstance(entry[1], Mapping):
            raise TypeError("conditioning entries must contain text and a metadata mapping.")
        metadata = dict(entry[1])
        incoming = metadata.get("minimax_keyframes", [])
        if not isinstance(incoming, (list, tuple)):
            raise TypeError("minimax_keyframes must be a sequence of mappings.")
        keyframes = []
        replaced = False
        for guide in incoming:
            if not isinstance(guide, Mapping):
                raise TypeError("Every MiniMax keyframe must be a mapping.")
            shifted = dict(guide)
            position = guide.get("resolved_frame_index")
            if isinstance(position, bool) or not isinstance(position, (int, float)) or not math.isfinite(position):
                raise ValueError("MiniMax keyframes require a finite resolved_frame_index.")
            shifted["resolved_frame_index"] = position + offset
            latent, audio = guide.get("latent"), guide.get("audio_latent")
            native_tail = (position == 0 and getattr(latent, "ndim", None) == 5
                           and latent.shape[2] == end - start
                           and getattr(audio, "ndim", None) == 4)
            if inject_tail and native_tail:
                shifted.update(latent=upscaled_video[:, :, start:end].clone(),
                               audio_latent=original_audio[..., plan["audio_start"]:plan["source_audio_tokens"]].clone())
                replaced = True
            keyframes.append(shifted)
        if inject_tail and not replaced:
            keyframes.append(dict(resolved_frame_index=offset,
                                  latent=upscaled_video[:, :, start:end].clone(),
                                  audio_latent=original_audio[..., plan["audio_start"]:plan["source_audio_tokens"]].clone()))
        if incoming or inject_tail or "minimax_keyframes" in metadata:
            metadata["minimax_keyframes"] = keyframes
        copied = list(entry)
        copied[1] = metadata
        result.append(tuple(copied) if isinstance(entry, tuple) else copied)
    return result
