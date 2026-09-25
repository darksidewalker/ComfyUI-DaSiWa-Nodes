"""Native latent-tail continuation nodes for MiniMax H3 video and audio."""

import inspect
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import node_helpers
import torch
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io

logger = logging.getLogger(__name__)

FPS = 24
AUDIO_LATENT_FPS = 40
VIDEO_LATENT_CHANNELS = 24
AUDIO_LATENT_CHANNELS = 32
AUDIO_CHANNELS = 2


def video_latent_t(frame_count: int) -> int:
    """Return the MiniMax H3 video latent length for an aligned frame count."""
    if frame_count < 5:
        raise ValueError("frame_count must be at least 5")
    if frame_count % 17 != 5:
        raise ValueError("frame_count must satisfy the MiniMax H3 17k + 5 grid")
    if frame_count == 5:
        return 2
    return ((frame_count - 5) // 17) * 5 + 2


def frame_count_from_video_latent_t(video_t: int) -> int:
    """Return the pixel-frame count represented by a MiniMax H3 video latent."""
    if video_t < 2:
        raise ValueError("video temporal length must be at least 2")
    if (video_t - 2) % 5 != 0:
        raise ValueError("video temporal length must satisfy the MiniMax H3 5k + 2 grid")
    groups = (video_t - 2) // 5
    return groups * 17 + 5


def audio_t_at_frame_boundary(frame_index: int) -> int:
    """Return the cumulative 40 Hz audio-latent index at a 24 fps frame boundary."""
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    return round(frame_index * AUDIO_LATENT_FPS / FPS)


def _extract_h3_av_streams(
    latent: Mapping[str, Any],
    *,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and extract the structural video/audio streams of an H3 latent."""
    if not isinstance(latent, Mapping):
        raise TypeError(f"{name} must be a mapping containing a 'samples' entry")
    if "samples" not in latent:
        raise ValueError(f"{name} is missing the required 'samples' entry")

    samples = latent["samples"]
    if not isinstance(samples, NestedTensor):
        raise TypeError(f"{name}['samples'] must be a comfy.nested_tensor.NestedTensor")

    streams = samples.tensors
    if len(streams) != 2:
        raise ValueError(f"{name}['samples'] must contain exactly two streams: video then audio")
    video, audio = streams
    if not isinstance(video, torch.Tensor):
        raise TypeError(f"{name} video stream must be a torch.Tensor")
    if not isinstance(audio, torch.Tensor):
        raise TypeError(f"{name} audio stream must be a torch.Tensor")
    if video.ndim != 5:
        raise ValueError(f"{name} video must have shape [1, 24, T, H, W], got {tuple(video.shape)}")
    if audio.ndim != 4:
        raise ValueError(
            f"{name} audio must have shape [1, 32, 2, T_audio], got {tuple(audio.shape)}"
        )
    if video.shape[0] != 1 or audio.shape[0] != 1:
        raise ValueError(f"{name} video and audio batch sizes must both be 1")
    if video.shape[1] != VIDEO_LATENT_CHANNELS:
        raise ValueError(f"{name} video must have {VIDEO_LATENT_CHANNELS} channels")
    if audio.shape[1] != AUDIO_LATENT_CHANNELS:
        raise ValueError(f"{name} audio must have {AUDIO_LATENT_CHANNELS} latent channels")
    if audio.shape[2] != AUDIO_CHANNELS:
        raise ValueError(f"{name} audio must have {AUDIO_CHANNELS} channels")
    if video.shape[3] <= 0 or video.shape[4] <= 0:
        raise ValueError(f"{name} video spatial dimensions must be positive")
    if audio.shape[-1] <= 0:
        raise ValueError(f"{name} audio temporal length must be positive")

    frame_count_from_video_latent_t(video.shape[2])
    return video, audio


def validate_h3_av_latent(
    latent: Mapping[str, Any],
    *,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Validate and extract a cumulative MiniMax H3 video/audio latent pair."""
    video, audio = _extract_h3_av_streams(latent, name=name)
    frame_count = frame_count_from_video_latent_t(video.shape[2])
    expected_audio_t = audio_t_at_frame_boundary(frame_count)
    if audio.shape[-1] != expected_audio_t:
        raise ValueError(
            f"{name} audio temporal length must be {expected_audio_t} for "
            f"{frame_count} frames, got {audio.shape[-1]}"
        )
    return video, audio, frame_count


def _valid_window_audio_lengths(frame_count: int) -> tuple[int, ...]:
    """Return possible audio lengths for a frame-aligned slice of the global timeline."""
    numerator = frame_count * AUDIO_LATENT_FPS
    lower, remainder = divmod(numerator, FPS)
    if remainder == 0:
        return (lower,)
    return (lower, lower + 1)


def validate_h3_av_window(
    latent: Mapping[str, Any],
    *,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Validate an H3 AV window while allowing global-boundary audio rounding."""
    video, audio = _extract_h3_av_streams(latent, name=name)
    frame_count = frame_count_from_video_latent_t(video.shape[2])
    valid_audio_lengths = _valid_window_audio_lengths(frame_count)
    if audio.shape[-1] not in valid_audio_lengths:
        expected = " or ".join(str(length) for length in valid_audio_lengths)
        raise ValueError(
            f"{name} audio temporal length must be {expected} for an aligned "
            f"{frame_count}-frame window, got {audio.shape[-1]}"
        )
    return video, audio, frame_count


@dataclass(frozen=True)
class _ContinuationLayout:
    """Resolved video/audio token lengths for one continuation window."""

    window_frame_count: int
    overlap_video_t: int
    new_video_t: int
    overlap_audio_t: int
    new_audio_t: int
    transition_video_t: int
    transition_audio_t: int


def _validate_guided_continuation_lengths(
    overlap_frames: int,
    extension_frames: int,
    previous_frame_count: int,
) -> None:
    """Validate a hidden-overlap continuation against the native H3 grids."""
    if overlap_frames < 5 or overlap_frames % 17 != 5:
        raise ValueError("overlap_frames must be at least 5 and satisfy 17k + 5")
    if overlap_frames > previous_frame_count:
        raise ValueError("overlap_frames cannot exceed the previous latent's frame count")
    if extension_frames < 17 or extension_frames % 17 != 0:
        raise ValueError("extension_frames must be at least 17 and be a multiple of 17")


def _calculate_guided_continuation_layout(
    previous_frame_count: int,
    overlap_frames: int,
    extension_frames: int,
) -> _ContinuationLayout:
    """Resolve a fresh target whose leading span is hidden motion context."""
    overlap_video_t = video_latent_t(overlap_frames)
    new_video_t = extension_frames // 17 * 5
    window_frame_count = overlap_frames + extension_frames
    if overlap_video_t + new_video_t != video_latent_t(window_frame_count):
        raise ValueError("guided continuation window does not satisfy the MiniMax H3 video grid")

    overlap_start_frame = previous_frame_count - overlap_frames
    extended_frame_count = previous_frame_count + extension_frames
    previous_audio_end = audio_t_at_frame_boundary(previous_frame_count)
    overlap_audio_start = audio_t_at_frame_boundary(overlap_start_frame)
    extended_audio_end = audio_t_at_frame_boundary(extended_frame_count)
    return _ContinuationLayout(
        window_frame_count=window_frame_count,
        overlap_video_t=overlap_video_t,
        new_video_t=new_video_t,
        overlap_audio_t=previous_audio_end - overlap_audio_start,
        new_audio_t=extended_audio_end - previous_audio_end,
        transition_video_t=0,
        transition_audio_t=0,
    )


def _assemble_guided_target(
    previous_video: torch.Tensor,
    previous_audio: torch.Tensor,
    layout: _ContinuationLayout,
) -> dict[str, NestedTensor]:
    """Allocate an entirely fresh AV target for motion-context continuation."""
    video = previous_video.new_zeros(
        (
            1,
            VIDEO_LATENT_CHANNELS,
            layout.overlap_video_t + layout.new_video_t,
            previous_video.shape[3],
            previous_video.shape[4],
        )
    )
    audio = previous_audio.new_zeros(
        (
            1,
            AUDIO_LATENT_CHANNELS,
            AUDIO_CHANNELS,
            layout.overlap_audio_t + layout.new_audio_t,
        )
    )
    return {"samples": NestedTensor((video, audio))}


class MiniMaxH3GuidedContinuationWindow(io.ComfyNode):
    """Prepare a fresh H3 target with a hidden overlap for native guide conditioning."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Define the V3 schema for a native-guide continuation target."""
        return io.Schema(
            node_id="MiniMaxH3GuidedContinuationWindow",
            display_name="MiniMax H3 Guided Continuation Window",
            category="MiniMax H3/continuation",
            description=(
                "Allocates a fresh AV target whose leading overlap is generated as hidden "
                "context under a MiniMax H3 Latent Tail Guide."
            ),
            inputs=[
                io.Latent.Input(
                    "previous_av_latent",
                    tooltip="The completed cumulative H3 AV latent to continue.",
                ),
                io.Int.Input(
                    "overlap_frames",
                    default=22,
                    min=5,
                    max=3600,
                    step=17,
                    tooltip="Hidden motion-context span; must satisfy 17k + 5.",
                ),
                io.Int.Input(
                    "extension_frames",
                    default=119,
                    min=17,
                    max=3570,
                    step=17,
                    tooltip="New visible frame count; must be a multiple of 17.",
                ),
            ],
            outputs=[
                io.Latent.Output(
                    "latent",
                    tooltip="Fresh unmasked AV target for the continuation sampler.",
                ),
                io.Int.Output(
                    "window_length",
                    tooltip="Connect to MiniMax H3 Image to Video's length input.",
                ),
                io.Int.Output("overlap_video_tokens"),
                io.Int.Output("overlap_audio_tokens"),
                io.Int.Output(
                    "transition_video_tokens",
                    tooltip="Always zero: the complete hidden overlap is discarded after sampling.",
                ),
                io.Int.Output(
                    "transition_audio_tokens",
                    tooltip="Always zero: the complete hidden overlap is discarded after sampling.",
                ),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        previous_av_latent: Mapping[str, Any],
        overlap_frames: int,
        extension_frames: int,
    ) -> io.NodeOutput:
        """Return an unmasked zero target and its globally aligned overlap metadata."""
        previous_video, previous_audio, previous_frame_count = validate_h3_av_latent(
            previous_av_latent,
            name="previous_av_latent",
        )
        _validate_guided_continuation_lengths(
            overlap_frames,
            extension_frames,
            previous_frame_count,
        )
        layout = _calculate_guided_continuation_layout(
            previous_frame_count,
            overlap_frames,
            extension_frames,
        )
        target = _assemble_guided_target(previous_video, previous_audio, layout)
        target_video, target_audio = target["samples"].tensors
        logger.info(
            "Prepared MiniMax H3 guided continuation target: previous_frames=%d, "
            "overlap_frames=%d, extension_frames=%d, window_frames=%d, "
            "video_tokens=%d, overlap_video_tokens=%d, audio_tokens=%d, "
            "overlap_audio_tokens=%d",
            previous_frame_count,
            overlap_frames,
            extension_frames,
            layout.window_frame_count,
            target_video.shape[2],
            layout.overlap_video_t,
            target_audio.shape[-1],
            layout.overlap_audio_t,
        )
        return io.NodeOutput(
            target,
            layout.window_frame_count,
            layout.overlap_video_t,
            layout.overlap_audio_t,
            0,
            0,
        )


def _validate_append_compatibility(
    previous_video: torch.Tensor,
    previous_audio: torch.Tensor,
    sampled_video: torch.Tensor,
    sampled_audio: torch.Tensor,
) -> None:
    """Validate shape, dtype, and device compatibility before concatenation."""
    if previous_video.shape[3:] != sampled_video.shape[3:]:
        raise ValueError("sampled_window video spatial dimensions must match the previous latent")
    if previous_video.dtype != sampled_video.dtype:
        raise TypeError("sampled_window video dtype must match the previous latent")
    if previous_video.device != sampled_video.device:
        raise ValueError("sampled_window video device must match the previous latent")
    if previous_audio.dtype != sampled_audio.dtype:
        raise TypeError("sampled_window audio dtype must match the previous latent")
    if previous_audio.device != sampled_audio.device:
        raise ValueError("sampled_window audio device must match the previous latent")


def _validate_overlap_tokens(
    overlap_video_tokens: int,
    overlap_audio_tokens: int,
    sampled_video_t: int,
    sampled_audio_t: int,
    previous_frame_count: int,
) -> int:
    """Validate overlap token counts and return the new video-token count."""
    if overlap_video_tokens <= 0 or overlap_video_tokens >= sampled_video_t:
        raise ValueError(
            "overlap_video_tokens must be positive and smaller than sampled video length"
        )
    if overlap_audio_tokens <= 0 or overlap_audio_tokens >= sampled_audio_t:
        raise ValueError(
            "overlap_audio_tokens must be positive and smaller than sampled audio length"
        )
    new_video_t = sampled_video_t - overlap_video_tokens
    if new_video_t <= 0 or new_video_t % 5 != 0:
        raise ValueError("sampled_window new video-token count must be positive and divisible by 5")
    overlap_frames = frame_count_from_video_latent_t(overlap_video_tokens)
    if overlap_frames > previous_frame_count:
        raise ValueError("overlap token counts cannot exceed the previous latent")
    expected_audio_tokens = audio_t_at_frame_boundary(
        previous_frame_count
    ) - audio_t_at_frame_boundary(previous_frame_count - overlap_frames)
    if overlap_audio_tokens != expected_audio_tokens:
        raise ValueError(
            "overlap_audio_tokens does not match overlap_video_tokens at the current "
            "global frame boundary"
        )
    return new_video_t


def _validate_transition_tokens(
    transition_video_tokens: int,
    transition_audio_tokens: int,
    overlap_video_tokens: int,
    overlap_audio_tokens: int,
    previous_frame_count: int,
) -> None:
    """Validate transition token counts within the sampled overlap."""
    if transition_video_tokens == 0 and transition_audio_tokens == 0:
        return
    if transition_video_tokens == 0 or transition_audio_tokens == 0:
        raise ValueError("video and audio transition token counts must both be zero or nonzero")
    if transition_video_tokens <= 0 or transition_video_tokens >= overlap_video_tokens:
        raise ValueError(
            "transition_video_tokens must be positive and smaller than overlap_video_tokens"
        )
    if transition_video_tokens % 5 != 0:
        raise ValueError("transition_video_tokens must be divisible by 5")
    if transition_audio_tokens <= 0 or transition_audio_tokens >= overlap_audio_tokens:
        raise ValueError(
            "transition_audio_tokens must be positive and smaller than overlap_audio_tokens"
        )
    transition_frames = transition_video_tokens // 5 * 17
    expected_audio_tokens = audio_t_at_frame_boundary(
        previous_frame_count
    ) - audio_t_at_frame_boundary(previous_frame_count - transition_frames)
    if transition_audio_tokens != expected_audio_tokens:
        raise ValueError(
            "transition_audio_tokens does not match transition_video_tokens at the current "
            "global frame boundary"
        )


def _require_native_arbitrary_guides() -> None:
    """Require the upstream H3 layout that supports arbitrary guide positions."""
    from comfy.ldm.minimax.model import PackedLayout

    parameters = inspect.signature(PackedLayout.__init__).parameters
    if "frame_count" in parameters:
        raise RuntimeError(
            "MiniMax H3 Latent Tail Guide requires ComfyUI commit e01fb4c or newer, "
            "which supports native guides at arbitrary frames."
        )


def _existing_keyframes(positive: Any) -> list[dict[str, Any]]:
    """Return a copied canonical MiniMax keyframe list from CONDITIONING."""
    if not isinstance(positive, list) or not positive:
        raise TypeError("positive must be a non-empty ComfyUI CONDITIONING list")
    first = positive[0]
    if not isinstance(first, (list, tuple)) or len(first) < 2:
        raise TypeError("positive entries must contain a tensor and metadata mapping")
    metadata = first[1]
    if not isinstance(metadata, Mapping):
        raise TypeError("positive conditioning metadata must be a mapping")
    keyframes = metadata.get("minimax_keyframes", [])
    if not isinstance(keyframes, (list, tuple)):
        raise TypeError("positive minimax_keyframes metadata must be a list or tuple")
    if not all(isinstance(keyframe, Mapping) for keyframe in keyframes):
        raise TypeError("every positive minimax_keyframes entry must be a mapping")
    return [dict(keyframe) for keyframe in keyframes]


def _reject_conflicting_head_guides(
    keyframes: list[dict[str, Any]],
    guide_frames: int,
) -> None:
    """Reject an existing guide that competes with the latent tail at the target head."""
    for keyframe in keyframes:
        position = keyframe.get("resolved_frame_index")
        if not isinstance(position, (int, float)):
            raise ValueError("existing MiniMax H3 guides must have resolved_frame_index")
        if 0 <= float(position) < guide_frames:
            raise ValueError(
                "positive already contains a MiniMax H3 guide inside the latent-tail "
                "overlap; disconnect first_frame and other opening guides"
            )


class MiniMaxH3LatentTailGuide(io.ComfyNode):
    """Condition an H3 target directly on a synchronized tail of a prior AV latent."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Define the V3 schema for direct latent-tail conditioning."""
        return io.Schema(
            node_id="MiniMaxH3LatentTailGuide",
            display_name="MiniMax H3 Latent Tail Guide",
            category="MiniMax H3/continuation",
            description=(
                "Adds the previous generated AV latent tail as a native frame-zero H3 guide "
                "without a VAE round trip."
            ),
            inputs=[
                io.Conditioning.Input(
                    "positive",
                    tooltip="Positive conditioning for the complete continuation window.",
                ),
                io.Latent.Input(
                    "previous_av_latent",
                    tooltip="The completed cumulative H3 AV latent whose tail provides context.",
                ),
                io.Latent.Input(
                    "target_av_latent",
                    tooltip="The fresh guided continuation target that will be sampled.",
                ),
                io.Int.Input("overlap_video_tokens", min=1, force_input=True),
                io.Int.Input("overlap_audio_tokens", min=1, force_input=True),
            ],
            outputs=[io.Conditioning.Output(display_name="positive")],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        positive: Any,
        previous_av_latent: Mapping[str, Any],
        target_av_latent: Mapping[str, Any],
        overlap_video_tokens: int,
        overlap_audio_tokens: int,
    ) -> io.NodeOutput:
        """Append one native synchronized AV guide built from the previous latent tail."""
        _require_native_arbitrary_guides()
        previous_video, previous_audio, previous_frame_count = validate_h3_av_latent(
            previous_av_latent,
            name="previous_av_latent",
        )
        target_video, target_audio, target_frame_count = validate_h3_av_window(
            target_av_latent,
            name="target_av_latent",
        )
        _validate_append_compatibility(
            previous_video,
            previous_audio,
            target_video,
            target_audio,
        )
        _validate_overlap_tokens(
            overlap_video_tokens,
            overlap_audio_tokens,
            target_video.shape[2],
            target_audio.shape[-1],
            previous_frame_count,
        )
        guide_frames = frame_count_from_video_latent_t(overlap_video_tokens)
        keyframes = _existing_keyframes(positive)
        _reject_conflicting_head_guides(keyframes, guide_frames)
        keyframes.append(
            {
                "resolved_frame_index": 0,
                "latent": previous_video[:, :, -overlap_video_tokens:].clone(),
                "audio_latent": previous_audio[..., -overlap_audio_tokens:].clone(),
            }
        )
        keyframes.sort(key=lambda keyframe: float(keyframe["resolved_frame_index"]))
        guided_positive = node_helpers.conditioning_set_values(
            positive,
            {"minimax_keyframes": keyframes},
        )
        logger.info(
            "Added native MiniMax H3 latent-tail guide: previous_frames=%d, "
            "target_frames=%d, guide_frames=%d, video_tokens=%d, audio_tokens=%d",
            previous_frame_count,
            target_frame_count,
            guide_frames,
            overlap_video_tokens,
            overlap_audio_tokens,
        )
        return io.NodeOutput(guided_positive)


class MiniMaxH3AppendContinuation(io.ComfyNode):
    """Append new H3 AV tokens, optionally retaining a resampled transition."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Define the V3 schema for cumulative continuation assembly."""
        return io.Schema(
            node_id="MiniMaxH3AppendContinuation",
            display_name="MiniMax H3 Append Continuation",
            category="MiniMax H3/continuation",
            description=(
                "Trims the hidden overlap and appends new AV tokens. With nonzero transition "
                "counts it also replaces the corresponding previous tail."
            ),
            inputs=[
                io.Latent.Input(
                    "previous_av_latent",
                    tooltip=(
                        "The completed cumulative MiniMax H3 AV latent used to prepare the window."
                    ),
                ),
                io.Latent.Input(
                    "sampled_window",
                    tooltip=(
                        "The continuation window after sampling with KSampler or KSampler Advanced."
                    ),
                ),
                io.Int.Input("overlap_video_tokens", min=1, force_input=True),
                io.Int.Input("overlap_audio_tokens", min=1, force_input=True),
                io.Int.Input("transition_video_tokens", min=0, force_input=True),
                io.Int.Input("transition_audio_tokens", min=0, force_input=True),
            ],
            outputs=[
                io.Latent.Output("latent"),
                io.Int.Output("total_length"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        previous_av_latent: Mapping[str, Any],
        sampled_window: Mapping[str, Any],
        overlap_video_tokens: int,
        overlap_audio_tokens: int,
        transition_video_tokens: int,
        transition_audio_tokens: int,
    ) -> io.NodeOutput:
        """Trim hidden context, optionally replace a transition, and append new tokens."""
        previous_video, previous_audio, previous_frame_count = validate_h3_av_latent(
            previous_av_latent,
            name="previous_av_latent",
        )
        sampled_video, sampled_audio = _extract_h3_av_streams(
            sampled_window,
            name="sampled_window",
        )
        _validate_append_compatibility(
            previous_video,
            previous_audio,
            sampled_video,
            sampled_audio,
        )
        new_video_t = _validate_overlap_tokens(
            overlap_video_tokens,
            overlap_audio_tokens,
            sampled_video.shape[2],
            sampled_audio.shape[-1],
            previous_frame_count,
        )
        _validate_transition_tokens(
            transition_video_tokens,
            transition_audio_tokens,
            overlap_video_tokens,
            overlap_audio_tokens,
            previous_frame_count,
        )

        new_video = sampled_video[:, :, overlap_video_tokens:]
        new_audio = sampled_audio[..., overlap_audio_tokens:]
        if transition_video_tokens:
            sampled_video_transition = sampled_video[
                :, :, overlap_video_tokens - transition_video_tokens : overlap_video_tokens
            ]
            sampled_audio_transition = sampled_audio[
                ..., overlap_audio_tokens - transition_audio_tokens : overlap_audio_tokens
            ]
            previous_video_prefix = previous_video[:, :, :-transition_video_tokens]
            previous_audio_prefix = previous_audio[..., :-transition_audio_tokens]
        else:
            sampled_video_transition = sampled_video[:, :, :0]
            sampled_audio_transition = sampled_audio[..., :0]
            previous_video_prefix = previous_video
            previous_audio_prefix = previous_audio
        extended_video = torch.cat(
            (previous_video_prefix, sampled_video_transition, new_video),
            dim=2,
        )
        extended_audio = torch.cat(
            (previous_audio_prefix, sampled_audio_transition, new_audio),
            dim=-1,
        )
        total_length = frame_count_from_video_latent_t(extended_video.shape[2])
        expected_audio_t = audio_t_at_frame_boundary(total_length)
        if extended_audio.shape[-1] != expected_audio_t:
            raise ValueError(
                "appended audio length does not match the global MiniMax H3 frame boundary: "
                f"expected {expected_audio_t}, got {extended_audio.shape[-1]}"
            )

        extended_latent = {"samples": NestedTensor((extended_video, extended_audio))}
        logger.info(
            "Appended MiniMax H3 continuation: previous_frames=%d, new_video_tokens=%d, "
            "new_audio_tokens=%d, transition_video_tokens=%d, "
            "transition_audio_tokens=%d, total_frames=%d",
            previous_frame_count,
            new_video_t,
            new_audio.shape[-1],
            transition_video_tokens,
            transition_audio_tokens,
            total_length,
        )
        return io.NodeOutput(extended_latent, total_length)
