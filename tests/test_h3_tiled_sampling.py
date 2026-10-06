"""CPU tensor execution; only the expensive apply_model is a test double."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
COMFY = ROOT.parents[1]
sys.path.insert(0, str(COMFY))
if "--cpu" not in sys.argv:
    sys.argv.append("--cpu")
spec = importlib.util.spec_from_file_location("dasiwa_h3_tiled_sampling", ROOT / "nodes/h3_tiled_sampling.py")
h3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h3)


@pytest.mark.parametrize("total", [1, 4, 5, 6, 19, 31, 62])
def test_temporal_ranges_cover_native_phases(total):
    ranges = h3.temporal_ranges(total, 17, 3)
    covered = set()
    for start, end in ranges:
        assert start % len(h3.FRAME_PER_TOKEN) == 0
        for k in range(end - start):
            assert h3.FRAME_PER_TOKEN[k % 5] == h3.FRAME_PER_TOKEN[(start + k) % 5]
        covered.update(range(start, end))
    assert covered == set(range(total))
    assert ranges[-1][1] == total
    assert all(b <= 15 or b == total for _, b in ranges[:1])


def test_planner_validation_and_budget_response():
    generous = h3.plan_tiles((1, 24, 60, 32, 32), 1024, 768, 24 * 1024**3)
    small = h3.plan_tiles((1, 24, 60, 32, 32), 1024, 768, 1024**3, model_bytes=900 * 1024**2, text_tokens=100)
    assert small["tile_width"] * small["tile_height"] <= generous["tile_width"] * generous["tile_height"]
    for plan in (small, generous):
        assert plan["tile_width"] % 32 == plan["tile_height"] % 32 == plan["overlap"] % 32 == 0
        assert plan["overlap"] < min(plan["tile_width"], plan["tile_height"])
        h3.temporal_ranges(60, plan["chunk_tokens"], plan["temporal_overlap_tokens"])
        assert "Not a VRAM guarantee" in plan["explanation"]
    with pytest.raises(ValueError):
        h3.plan_tiles((1, 24, 5, 4, 4), 99, 128, 10000)
    with pytest.raises(ValueError):
        h3.temporal_ranges(20, 5, 1)


def test_resize_retains_channel_time_order():
    video = torch.empty(1, 2, 3, 2, 2, dtype=torch.float16)
    for c in range(2):
        for t in range(3):
            video[:, c, t] = 10 * c + t
    resized = h3._resize_video(video, (5, 7))
    assert resized.dtype == video.dtype
    for c in range(2):
        for t in range(3):
            assert torch.all(resized[:, c, t] == 10 * c + t)


def test_conditioning_reanchors_and_keeps_refs_audio_metadata():
    video = torch.arange(12.).reshape(1, 1, 12, 1, 1)
    audio = torch.arange(100.).reshape(1, 1, 1, 100)
    refs, layout = [{"kind": "image", "latent": torch.ones(1, 1, 1, 2, 2)}], {"user": "metadata"}
    keyframe = dict(resolved_frame_index=0, latent=video, audio_latent=audio, label="keep")
    metadata = dict(minimax_keyframes=[keyframe], minimax_refs=refs, layout=layout, untouched=audio)
    source = [[torch.zeros(1, 2, 3), metadata]]
    result = h3.prepare_conditioning(source, 5, 10, (3, 4))
    md = result[0][1]
    visual = [k for k in md["minimax_keyframes"] if "latent" in k]
    audible = [k for k in md["minimax_keyframes"] if "audio_latent" in k]
    assert [k["resolved_frame_index"] for k in visual] == [0, 1, 5, 9, 13]
    assert [k["latent"].flatten()[0].item() for k in visual] == [5, 6, 7, 8, 9]
    assert all(k["latent"].shape[-2:] == (3, 4) for k in visual)
    assert torch.equal(audible[0]["audio_latent"], audio[..., 29:56])
    assert md["layout"] is layout and md["minimax_refs"] is refs and md["untouched"] is audio
    assert keyframe["latent"] is video and keyframe["audio_latent"] is audio
    anchored = h3.prepare_conditioning(source, 5, 10, (3, 4), previous_video=video + 100)
    anchor = anchored[0][1]["minimax_keyframes"][0]["latent"]
    torch.testing.assert_close(anchor, torch.full_like(anchor, 105))
    assert torch.equal(video.flatten(), torch.arange(12.))


def test_append_blends_without_mutation():
    old = torch.ones(1, 2, 10, 3, 4, dtype=torch.float16)
    new = torch.full((1, 2, 10, 3, 4), 3., dtype=torch.float16)
    result = h3.append_video(old, new, 5)
    assert result.shape[2] == 15 and result.dtype == old.dtype
    assert torch.all(result[:, :, :5] == 1) and torch.all(result[:, :, 10:] == 3)
    assert torch.all(result[:, :, 5:10] > 1) and torch.all(result[:, :, 5:10] < 3)
    assert torch.all(old == 1) and torch.all(new == 3)
    assert h3.append_video(None, old, 0).data_ptr() != old.data_ptr()
    with pytest.raises(ValueError):
        h3.append_video(old, new, 11)


def packed_case(dtype=torch.float32, h=7, w=9):
    video = torch.arange(2 * h * w, dtype=dtype).reshape(1, 1, 2, h, w)
    audio = torch.full((1, 2, 2, 3), 2., dtype=dtype)
    packed, shapes = h3.comfy.utils.pack_latents([video, audio])
    keyframe = {"resolved_frame_index": 0, "latent": torch.ones(1, 1, 1, 3, 4, dtype=dtype)}
    ref = dict(kind="image", latent=torch.ones(1, 1, 1, 2, 2), latent_h=2, latent_w=2)
    payload = {"keyframes": [keyframe], "refs": [ref]}
    options = {"sentinel": object(), "patches_replace": {}}
    mask = torch.ones(1, 1, 2, h, w)
    mask[..., :2, :2] = 0
    c = dict(minimax_payload=payload, c_crossattn=torch.ones(1, 2, 3), latent_shapes=shapes,
             transformer_options=options, denoise_mask=mask, audio_denoise_mask=torch.zeros_like(audio))
    return video, audio, dict(input=packed, timestep=torch.ones(1), c=c)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_tiling_global_positions_masks_odd_edges_and_no_cache(dtype):
    video, audio, args = packed_case(dtype)
    original = args["input"].clone()
    c = args["c"]
    full = h3.PackedLayout(2, 2, 8, 10, 3, keyframes=[dict(resolved_frame_index=0, latent=torch.ones(1, 1, 1, 8, 10))], refs=c["minimax_payload"]["refs"])
    expected_positions = set(map(tuple, full.position_ids[full.img_pos[full.img_update]].tolist()))
    seen_positions, calls = set(), []

    def apply_model_test_double(packed, timestep, **ct):
        v, a = h3.comfy.utils.unpack_latents(packed, ct["latent_shapes"])
        layout = ct["minimax_payload"]["layout"]
        positions = layout.position_ids[layout.img_pos[layout.img_update]]
        seen_positions.update(map(tuple, positions.tolist()))
        calls.append(tuple(v.shape))
        assert layout.signature[2:4] == (h3._ceil2(v.shape[3]), h3._ceil2(v.shape[4]))
        # All reference/audio segments retain full-frame anchors, not tile-local
        # width-dependent stereo positions.
        for a0, b0, kind in layout.segments:
            if kind in ("audio", "ref_img", "ref_audio", "cond_audio"):
                fa, fb, _ = next(s for s in full.segments if s[2] == kind)
                torch.testing.assert_close(layout.position_ids[a0:b0], full.position_ids[fa:fb])
        assert ct["transformer_options"]["sentinel"] is c["transformer_options"]["sentinel"]
        ct["transformer_options"]["block_index"] = 123  # native forward writes here
        assert ct["denoise_mask"].shape == v.shape
        assert not torch.count_nonzero(ct["audio_denoise_mask"])
        assert ct["minimax_payload"]["refs"] is c["minimax_payload"]["refs"]
        assert ct["minimax_payload"]["keyframes"][0]["latent"].shape[-2:] == layout.signature[2:4]
        # Native apply_model zero-mask semantics are x0=input, not x0=0.
        return h3.comfy.utils.pack_latents([v + ct["denoise_mask"].to(v.dtype), a])[0]

    wrapper = h3.H3TiledDiffusion(64, 64, 32)
    output = wrapper(apply_model_test_double, args)
    vo, ao = h3.comfy.utils.unpack_latents(output, c["latent_shapes"])
    torch.testing.assert_close(vo, video + c["denoise_mask"].to(dtype))
    torch.testing.assert_close(ao, audio, rtol=0, atol=0)
    assert output.dtype == dtype and output.device == video.device
    assert expected_positions == seen_positions
    assert len(calls) > 1
    assert "block_index" not in c["transformer_options"]
    assert torch.equal(original, args["input"])
    assert c["minimax_payload"]["keyframes"][0]["latent"].shape[-2:] == (3, 4)
    assert all(not isinstance(v, torch.Tensor) for v in vars(wrapper).values())


def test_packed_mask_and_cancellation(monkeypatch):
    video, audio, args = packed_case()
    c = args["c"]
    c["denoise_mask"] = h3.comfy.utils.pack_latents([c["denoise_mask"], c.pop("audio_denoise_mask")])[0]
    calls = []
    def apply_model_test_double(x, t, **ct):
        assert ct["denoise_mask"].ndim == 5
        assert ct["audio_denoise_mask"].shape == audio.shape
        calls.append(x.shape)
        return x
    wrapper = h3.H3TiledDiffusion(64, 64, 32)
    torch.testing.assert_close(wrapper(apply_model_test_double, args), args["input"])
    checks = []
    def interrupt():
        checks.append(True)
        if len(checks) == 2:
            raise RuntimeError("test cancellation")
    monkeypatch.setattr(h3.comfy.model_management, "throw_exception_if_processing_interrupted", interrupt)
    calls.clear()
    with pytest.raises(RuntimeError, match="test cancellation"):
        wrapper(apply_model_test_double, args)
    assert len(calls) == 1


def test_partial_audio_mask_retains_native_x0_not_carried_input():
    video, audio, args = packed_case()
    audio = torch.linspace(0.123, 0.987, audio.numel()).reshape_as(audio)
    args["input"] = h3.comfy.utils.pack_latents([video, audio])[0]
    mask = args["c"]["audio_denoise_mask"]
    mask[..., -1] = 1
    calls = []
    frozen = audio * 1.125  # emulate native AV schedule's carried-audio x0
    def apply_model_test_double(x, t, **ct):
        v, a = h3.comfy.utils.unpack_latents(x, ct["latent_shapes"])
        calls.append(True)
        prediction = torch.where(mask == 0, frozen, a + len(calls))
        return h3.comfy.utils.pack_latents([v, prediction])[0]
    output = h3.H3TiledDiffusion(64, 64, 32)(apply_model_test_double, args)
    _, result = h3.comfy.utils.unpack_latents(output, args["c"]["latent_shapes"])
    assert torch.equal(result[mask == 0], frozen[mask == 0])
    torch.testing.assert_close(result[mask == 1], audio[mask == 1] + (len(calls) + 1) / 2)
