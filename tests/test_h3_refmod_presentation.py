"""RefMod presentation uses native Qwen timing without changing DiT references."""
import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "ComfyUI"))
import comfy.cli_args
comfy.cli_args.args.cpu = True

package = types.ModuleType("dasiwa_refmod_test_nodes")
package.__path__ = [str(ROOT / "nodes")]
sys.modules[package.__name__] = package
guide_module = importlib.import_module("dasiwa_refmod_test_nodes.nodes_minimax_h3_director_guide")


class Clip:
    def __init__(self):
        self.calls = []
        self.embedding = torch.zeros(1, 2, 8)

    def tokenize(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return text

    def encode_from_tokens_scheduled(self, tokens):
        return [[self.embedding, {"untouched": "keep"}]]


class VAE:
    def __init__(self, frames=39, batched=False):
        self.pixels = torch.arange(frames, dtype=torch.float32)[:, None, None, None].expand(-1, 32, 32, 3).clone()
        self.batched = batched
        self.decodes = []

    def decode(self, latent):
        self.decodes.append(latent)
        return self.pixels.unsqueeze(0) if self.batched else self.pixels

    def encode(self, pixels):
        return torch.zeros(1, 24, 1, pixels.shape[1] // 16, pixels.shape[2] // 16)


def ref_guide(blocks, **kwargs):
    return dict(mode="REF2VA", resolved_prompt="<Picture 1> and <Video 1> continue.",
                width=32, height=32, length=39, ref_image_size="match",
                minimax_ref_items=blocks, **kwargs)


@pytest.mark.parametrize("frames,batched", [(5, False), (22, False), (39, True), (124, False)])
def test_video_refmod_uses_native_two_fps_timestamps(frames, batched):
    latent = torch.zeros(1, 24, (frames - 5) // 17 * 5 + 2, 2, 2)
    clip, vae = Clip(), VAE(frames, batched)
    positive, target, context = guide_module.MiniMaxH3DirectorGuide().apply(
        clip, vae, ref_guide([{"kind": "video", "latent": latent}]))
    media = clip.calls[0][1]["minimax_ref_items"][0]
    expected = list(range(0, frames, 12))
    assert media["data"].shape[0] == len(expected)
    assert media["data"][:, 0, 0, 0].tolist() == expected
    assert media["timestamps"] == [i / 2.0 for i in range(len(expected))]
    ref = positive[0][1]["minimax_refs"][0]
    assert ref["latent"] is latent
    assert (ref["latent_t"], ref["latent_h"], ref["latent_w"]) == tuple(latent.shape[2:])
    assert positive[0][0] is clip.embedding
    assert positive[0][1]["untouched"] == "keep"
    assert len(target["samples"].tensors) == 2
    assert context == {"disabled": True}
    assert media["data"].untyped_storage().nbytes() == media["data"].numel() * media["data"].element_size()


def test_refmod_video_matches_native_reference_presentation():
    native = guide_module._native_node("MiniMaxH3ReferenceToVideo")
    refmod_clip, native_clip, vae = Clip(), Clip(), VAE(39)
    vae.pixels /= 255  # Native Lanczos resizing round-trips through 8-bit pixels.
    video = torch.zeros(1, 24, 12, 2, 2)
    guide_module.MiniMaxH3DirectorGuide().apply(
        refmod_clip, vae, ref_guide([{"kind": "video", "latent": video}]))
    native.execute(clip=native_clip, prompt="same", width=32, height=32, length=39,
                   ref_videos={"ref_video_1": vae.pixels})
    actual = refmod_clip.calls[0][1]["minimax_ref_items"][0]
    expected = native_clip.calls[0][1]["minimax_ref_items"][0]
    assert actual["type"] == expected["type"]
    assert actual["timestamps"] == expected["timestamps"]
    assert torch.equal(actual["data"], expected["data"])


def test_image_and_audio_refmods_preserve_reference_order_and_latents():
    image, video, audio = torch.zeros(1, 24, 1, 2, 2), torch.zeros(1, 24, 12, 2, 2), torch.zeros(1, 32, 2, 65)
    clip, vae = Clip(), VAE(39)
    def decode(latent):
        return vae.pixels[:1] if latent is image else vae.pixels
    vae.decode = decode
    blocks = [{"kind": "image", "latent": image}, {"kind": "video", "latent": video},
              {"kind": "audio", "latent": audio, "latent_t": 65}]
    timeline_image = torch.zeros(1, 32, 32, 3)
    positive, _, _ = guide_module.MiniMaxH3DirectorGuide().apply(
        clip, vae, ref_guide(blocks, ref_images={"ref_image_1": timeline_image}), audio_vae=VAE())
    media = clip.calls[0][1]["minimax_ref_items"]
    assert [item["type"] for item in media] == ["image", "image", "video", "audio"]
    assert torch.equal(media[1]["data"], vae.pixels[:1])
    assert "timestamps" not in media[1]
    refs = positive[0][1]["minimax_refs"]
    assert [r["kind"] for r in refs] == ["image", "image", "video", "audio"]
    assert refs[1]["latent"] is image and refs[2]["latent"] is video and refs[3]["audio_latent"] is audio


@pytest.mark.parametrize("mode", ["T2VA", "I2VA", "FL2VA", "L2VA", "Image Inpaint"])
def test_non_reference_modes_keep_native_keyframe_path(mode):
    clip, vae = Clip(), VAE(1)
    first, last = torch.zeros(1, 32, 32, 3), torch.ones(1, 32, 32, 3)
    g = dict(mode=mode, resolved_prompt="continue", width=32, height=32, length=22,
             first_frame=first if mode in {"I2VA", "FL2VA", "Image Inpaint"} else None,
             last_frame=last if mode in {"L2VA", "FL2VA"} else None)
    positive, _, _ = guide_module.MiniMaxH3DirectorGuide().apply(clip, vae, g)
    assert vae.decodes == []
    assert "minimax_ref_items" not in clip.calls[0][1]
    assert len(clip.calls[0][1]["images"]) == (2 if mode == "FL2VA" else 0 if mode == "T2VA" else 1)
    assert "minimax_refs" not in positive[0][1]


def test_continuation_tail_keeps_full_refmod_latents():
    core = importlib.import_module("dasiwa_refmod_test_nodes.h3_continuity.core")
    from comfy.nested_tensor import NestedTensor
    source_video, source_audio = torch.zeros(1, 24, 12, 2, 2), torch.zeros(1, 32, 2, 65)
    previous = {"samples": NestedTensor((source_video, source_audio))}
    ref = torch.ones(1, 24, 12, 2, 2)
    g = ref_guide([{"kind": "video", "latent": ref}])
    settings = dict(version=3, overlap_frames=22, extension_frames=17,
                    continuation_prompt="Continue walking.", idea="")
    updated, target, layout = core.prepare_continuation(
        previous, {"fps": 24, "mode_family": "ref2va"}, g, settings)
    clip, vae = Clip(), VAE(39)
    positive, _ = guide_module.MiniMaxH3DirectorGuide()._apply_native(clip, vae, updated)
    positive = core.add_tail(positive, previous, target, layout)
    assert positive[0][1]["minimax_refs"][0]["latent"] is ref
    tail = positive[0][1]["minimax_keyframes"][-1]
    assert torch.equal(tail["latent"], source_video[:, :, :layout["overlap_video_tokens"]])
    assert torch.equal(tail["audio_latent"], source_audio[..., :layout["overlap_audio_tokens"]])
    assert clip.calls[0][1]["minimax_ref_items"][0]["data"].shape[0] == 4
