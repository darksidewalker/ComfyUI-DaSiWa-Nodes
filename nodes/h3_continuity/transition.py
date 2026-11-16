"""H3 transition bridge between two finished clips.

The Guide anchors the trimmed end of clip A and start of clip B as native H3
guides at both ends of one short window. The user's own sampler renders it;
Render keeps only the unguided middle, so A + bridge + B joins model-free.
"""
import json
import os
import tempfile
import uuid
from collections import deque
from contextlib import closing
from pathlib import Path

import numpy as np
import torch

from .. import helper_batch_output
from ..nodes_minimax_h3_director_guide import MiniMaxH3DirectorGuide, _native_node
from . import assembly
from .vendor.continuation_nodes import validate_h3_av_window

MAX_PROMPT = 4000


def _roots():
    return assembly._roots(None, None)


def transition_spec(data):
    if isinstance(data, str):
        if len(data.encode("utf-8")) > assembly.MAX_JSON:
            raise ValueError("Transition request exceeds 64 KiB.")
        data = json.loads(data)
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("Unsupported transition request version.")
    rows = data.get("rows")
    if not isinstance(rows, list) or len(rows) != 2 or any(not isinstance(r, dict) or r.get("kind") != "media" for r in rows):
        raise ValueError("A transition needs exactly two uploaded video clips.")
    prefix = data.get("filename_prefix", "H3_Transition")
    assembly._request(dict(version=1, rows=rows, filename_prefix=prefix))
    context, gap = data.get("context_frames"), data.get("transition_frames")
    if type(context) is not int or context < 5 or context % 17 != 5:
        raise ValueError("Context per side must be 5, 22, 39 … frames (17k + 5).")
    if type(gap) is not int or gap < 12 or (2 * context + gap) % 17 != 5:
        raise ValueError("Transition length must keep the H3 window on its 17k + 5 grid (12, 29, 46 … frames).")
    prompt = data.get("prompt", "")
    if not isinstance(prompt, str) or len(prompt) > MAX_PROMPT:
        raise ValueError("Transition description must be text of at most 4000 characters.")
    return dict(version=1, rows=rows, context_frames=context, transition_frames=gap, prompt=prompt.strip(),
                filename_prefix=prefix, window_frames=2 * context + gap, guide_b_frame=context + gap)


def transition_prompt(data):
    head, gap = data["context_frames"] / 24, data["transition_frames"] / 24
    text = (f"This is a transition window. Its opening {head:.3f} seconds are hidden context from the end of the "
            f"previous shot and its final {head:.3f} seconds are hidden context from the start of the next shot. "
            f"Generate the {gap:.3f} seconds between them as one continuous bridge: match subjects, space, "
            "camera motion and sound at both seams, without a cut, fade or freeze.")
    return text + ("\n\n" + data["prompt"] if data["prompt"] else "")


def _guide_frame(frame, width, height):
    """Match Core's center-crop + PIL Lanczos without source-resolution FP32 clips."""
    from PIL import Image
    image = frame.to_image()
    old_width, old_height = image.size
    old_aspect, new_aspect = old_width / old_height, width / height
    x = round((old_width - old_width * (new_aspect / old_aspect)) / 2) if old_aspect > new_aspect else 0
    y = round((old_height - old_height * (old_aspect / new_aspect)) / 2) if old_aspect < new_aspect else 0
    return np.array(image.crop((x, y, old_width - x, old_height - y)).resize(
        (width, height), resample=Image.Resampling.LANCZOS))


def clip_edges(data, input_root=None, output_root=None):
    """Bounded decode: keep only the trimmed tail of A and head of B, frame-locked audio."""
    from .pyav_media import guard
    check = guard(interrupt=True)
    request = dict(version=1, rows=data["rows"])
    plan, size, _ = assembly._plan(request, input_root=input_root, output_root=output_root, check=check)
    count = data["context_frames"]
    if any(row["end"] - row["start"] < count for row in plan):
        raise ValueError(f"A selected clip is shorter than the {count}-frame transition context; trim less or reduce context.")
    width, height = generation_canvas(size)
    # Peak: both FP32 edges, retained uint8 frames + stack, audio blocks + concat, RGB decode/resize workspace.
    required = count * width * height * 3 * 10 + count * 2 * 2000 * 4 * 3 + size[0] * size[1] * 12 + width * height * 16
    edges = []
    for row, tail in zip(plan, (True, False)):
        retained = sum(p.numel() * p.element_size() + a["waveform"].numel() * a["waveform"].element_size()
                       for p, a in edges)
        if not helper_batch_output.can_allocate_in_ram(required - retained):
            _release_cached_ram(required - retained + helper_batch_output.ram_safety_reserve_bytes())
            if not helper_batch_output.can_allocate_in_ram(required - retained):
                raise ValueError("Transition context exceeds available host RAM (including safety reserve). Reduce context frames or free RAM.")
        with closing(assembly._video_frames(row, check)) as frames:
            images = deque((_guide_frame(f, width, height) for f in frames), maxlen=count) if tail else [
                _guide_frame(f, width, height) for _, f in zip(range(count), frames)]
        with closing(assembly._audio_frames(row, check)) as blocks:
            audio = deque(blocks, maxlen=count) if tail else [b for _, b in zip(range(count), blocks)]
        pixels = torch.from_numpy(np.stack(images)).float().div_(255)
        waveform = torch.from_numpy(np.concatenate(list(audio), axis=1))[None]
        edges.append((pixels, {"waveform": waveform, "sample_rate": 48000}))
        del images, audio
    (tail, tail_audio), (head, head_audio) = edges
    return dict(tail=tail, tail_audio=tail_audio, head=head, head_audio=head_audio, canvas=size, colors=plan[0]["colors"])


def _release_cached_ram(target):
    """Free this executor's cached RAM via Core's pressure hooks, cheapest first; never unloads models."""
    import gc
    import inspect
    try:
        from comfy import memory_management, model_management
    except ImportError:
        return
    release = getattr(memory_management, "extra_ram_release", None)
    free_pins = getattr(model_management, "free_pins", None)
    for stage in ("inactive", "pins", "active"):
        available = helper_batch_output.available_ram_bytes()
        if available >= target:
            return
        if stage == "pins":
            if free_pins is not None:
                options = {"loaded": True} if "loaded" in inspect.signature(free_pins).parameters else {}
                free_pins(target - available, **options)
        elif release is not None:
            release(target, free_active=stage == "active")
        gc.collect()


def generation_canvas(size):
    """The clips' own framing on H3's 32-pixel grid, capped at H3's native pixel budget."""
    scale = min(1.0, (768 * 1344 / (size[0] * size[1])) ** 0.5)
    return tuple(max(32, round(value * scale / 32) * 32) for value in size)


class DaSiWaH3TransitionGuide:
    """Director Guide drop-in for one transition window; the dialog swaps it into the Director's sampler graph."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "vae": ("VAE", {"tooltip": "Native MiniMax H3 video VAE; encodes both edge clips as guides."}),
            "guide": ("MINIMAX_H3_DIRECTOR_GUIDE", {"tooltip": "Director settings: mode, canvas and references. Its prompt and Continuity are replaced for the transition."}),
            "transition_data": ("STRING", {"multiline": True, "default": "",
                                           "tooltip": "Set by Director → Join clips. Two uploaded clips, context frames per side, bridge length and optional description."}),
            "audio_vae": ("VAE", {"tooltip": "Native MiniMax H3 audio VAE; encodes both edge soundtracks as guides."}),
        }}

    RETURN_TYPES = ("CONDITIONING", "LATENT", "DF_H3_TRANSITION")
    RETURN_NAMES = ("positive", "latent", "transition")
    FUNCTION = "apply"
    CATEGORY = "DaSiWa/MiniMax H3/Continuity"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # uploaded clips are external mutable files

    def apply(self, clip, vae, guide, transition_data, audio_vae):
        if guide.get("mode") == "Image Inpaint":
            raise ValueError("H3 transitions require a video mode, not Image Inpaint.")
        data = transition_spec(transition_data)
        input_root, output_root = _roots()
        edges = clip_edges(data, input_root=input_root, output_root=output_root)
        text = transition_prompt(data)
        width, height = generation_canvas(edges["canvas"])
        updated = {**guide, "width": width, "height": height, "length": data["window_frames"], "prompt": text, "resolved_prompt": text,
                   "prompt_blocks": [], "first_frame": None, "last_frame": None}
        updated.pop("continuity", None)
        positive, latent = MiniMaxH3DirectorGuide()._apply_native(clip, vae, updated, audio_vae)
        add_guide = _native_node("MiniMaxH3AddGuide")
        for frame, images, audio in ((0, edges["tail"], edges["tail_audio"]),
                                     (data["guide_b_frame"], edges["head"], edges["head_audio"])):
            positive = add_guide.execute(positive=positive, latent=latent, frame_idx=frame, vae=vae,
                                         audio_vae=audio_vae, image=images, audio=audio)[0]
        return positive, latent, {**data, "canvas": edges["canvas"], "colors": edges["colors"]}


def _cover(images, width, height):
    """Inverse of the guide's centre cover-crop: return to the source clip canvas."""
    if images.shape[1:3] == (height, width):
        return images
    pixels = images.movedim(-1, 1)
    scale = max(width / pixels.shape[3], height / pixels.shape[2])
    size = (max(height, round(pixels.shape[2] * scale)), max(width, round(pixels.shape[3] * scale)))
    pixels = torch.nn.functional.interpolate(pixels, size=size, mode="bicubic", align_corners=False).clamp_(0, 1)
    top, left = (size[0] - height) // 2, (size[1] - width) // 2
    return pixels[:, :, top:top + height, left:left + width].movedim(1, -1)


class DaSiWaH3TransitionRender:
    """Decode the sampled window and save only its unguided middle as a joinable clip."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT", {"tooltip": "Sampler output for the Transition Guide window."}),
            "vae": ("VAE", {"tooltip": "Native MiniMax H3 video VAE."}),
            "audio_vae": ("VAE", {"tooltip": "Native MiniMax H3 32 kHz audio VAE."}),
            "transition": ("DF_H3_TRANSITION", {"tooltip": "Transition Guide plan for this window."}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "render"
    CATEGORY = "DaSiWa/MiniMax H3/Continuity"
    OUTPUT_NODE = True

    def render(self, latent, vae, audio_vae, transition):
        import av
        import comfy.model_management
        from ..helper_pyav_video import _configure_video_stream, _configure_audio_stream, _encode_audio
        from .pyav_media import guard
        video, audio, frames = validate_h3_av_window(latent, name="transition window")
        if frames != transition["window_frames"]:
            raise ValueError("Sampled latent is not the window prepared by this Transition Guide.")
        start, count = transition["context_frames"], transition["transition_frames"]
        comfy.model_management.throw_exception_if_processing_interrupted()
        pixels = vae.decode(video)
        if pixels.ndim == 5 and pixels.shape[0] == 1:
            pixels = pixels[0]
        if pixels.ndim != 4 or pixels.shape[0] != frames or pixels.shape[-1] != 3:
            raise ValueError("Video VAE output differs from the transition window.")
        width, height = transition["canvas"]
        images = _cover(pixels[start:start + count].float(), width, height).clone()
        del pixels
        comfy.model_management.throw_exception_if_processing_interrupted()
        waveform = audio_vae.decode(audio).movedim(-1, 1)
        first, last = round(start * 32000 / 24), round((start + count) * 32000 / 24)
        if waveform.ndim != 3 or waveform.shape[1] != 2 or last - waveform.shape[-1] > 267:
            raise ValueError("Audio VAE output differs from the transition window.")
        waveform = torch.nn.functional.pad(waveform, (0, max(0, last - waveform.shape[-1])))[0, :, first:last]

        input_root, _ = _roots()
        folder = input_root / "h3_transitions"
        folder.mkdir(parents=True, exist_ok=True)
        final = folder / f"{transition['filename_prefix']}_{uuid.uuid4().hex}.mp4"
        handle = tempfile.NamedTemporaryFile(prefix=".h3-transition-", suffix=".partial.mp4", dir=folder, delete=False)
        temporary = Path(handle.name)
        handle.close()
        colors = transition["colors"]
        check = guard(interrupt=True)
        try:
            with av.open(str(temporary), "w", format="mp4", options={"movflags": "+faststart"}) as output:
                stream = _configure_video_stream(output, "libx264", width, height, 24, 8, 18)
                for key, value in colors.items():
                    setattr(stream.codec_context, key, value)
                sound, rate = _configure_audio_stream(output, "aac", 48000, 2, "192k")
                full = colors.get("color_range") == 2
                for index, image in enumerate(images):
                    check()
                    frame = av.VideoFrame.from_ndarray(image.mul(255).round_().clamp_(0, 255).byte().cpu().numpy(), format="rgb24")
                    frame = frame.reformat(format="yuv420p", dst_colorspace="itu709" if colors.get("colorspace") == 1 else None,
                                           dst_color_range="JPEG" if full else "MPEG")
                    frame.pts = index
                    for packet in stream.encode(frame):
                        output.mux(packet)
                for packet in stream.encode(None):
                    output.mux(packet)
                _encode_audio(output, sound, (waveform.float().cpu().numpy(), 32000), rate)
            assembly._verify(temporary, count, check)
            with temporary.open("rb+") as file:
                os.fsync(file.fileno())
            os.link(temporary, final)
        finally:
            temporary.unlink(missing_ok=True)
        item = dict(filename=f"h3_transitions/{final.name}", frames=count, seconds=count / 24)
        preview = dict(filename=final.name, subfolder="h3_transitions", type="input", format="video/mp4")
        return {"ui": {"h3_transition": [item], "gifs": [preview]}, "result": (item["filename"],)}
