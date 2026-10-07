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
    small = h3.plan_tiles((1, 24, 60, 32, 32), 1024, 768, 2 * 1024**3, model_bytes=900 * 1024**2, text_tokens=100)
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


def test_streamed_bf16_model_larger_than_vram_gets_usable_tiles():
    plan = h3.plan_tiles((1, 24, 35, 90, 70), 2240, 2880, 30 * 1024**3,
                         model_bytes=38444 * 1024**2, text_tokens=512, ref_tokens=6300,
                         streaming_weights=True, reserved_bytes=400 * 1024**2,
                         audio_tokens=400)
    assert min(plan['tile_width'], plan['tile_height']) >= 512
    assert plan['tile_width'] * plan['tile_height'] * plan['chunk_tokens'] // 1024 <= plan['target_rows']
    assert 'streamed' in plan['explanation']
    capped = h3.plan_tiles((1, 24, 35, 90, 70), 2240, 2880, 8 * 1024**3,
                           model_bytes=38444 * 1024**2, text_tokens=512, ref_tokens=6300,
                           streaming_weights=True, reserved_bytes=400 * 1024**2,
                           audio_tokens=400)
    assert capped['tile_width'] * capped['tile_height'] <= plan['tile_width'] * plan['tile_height']


def test_target_grid_search_avoids_redundant_third_strip():
    plan = h3.plan_tiles((1,24,35,90,70),2240,2880,30*1024**3,
                         model_bytes=38444*1024**2, text_tokens=512,ref_tokens=6300,
                         streaming_weights=True,reserved_bytes=400*1024**2,audio_tokens=400)
    windows = h3.temporal_ranges(35,plan['chunk_tokens'],plan['temporal_overlap_tokens'])
    nx = len(h3._starts(140,plan['tile_width']//16,plan['overlap']//16))
    ny = len(h3._starts(180,plan['tile_height']//16,plan['overlap']//16))
    assert nx*ny*len(windows) <= 4  # previous halving planner scheduled six
    assert nx*ny*plan['tile_width']*plan['tile_height'] < 3*2240*1440
    assert plan['forwards_per_step'] == nx*ny*len(windows)


def test_short_clip_can_reduce_time_when_full_duration_does_not_fit():
    plan = h3.plan_tiles((1,24,20,32,32),1024,1024,640*1024**2)
    assert plan['chunk_tokens'] < 20
    assert min(plan['tile_width'],plan['tile_height']) >= 512
    h3.temporal_ranges(20,plan['chunk_tokens'],plan['temporal_overlap_tokens'])


@pytest.mark.parametrize('spatial,temporal',[(True,True),(False,True),(True,False),(False,False)])
def test_keyframes_scale_with_selected_tile_and_chunk_anchor(spatial,temporal):
    plan = h3.plan_tiles((1,24,35,90,70),2240,2880,64*1024**3,
                         streaming_weights=True,keyframe_tokens=3,
                         spatial_tiling=spatial,temporal_chunking=temporal)
    frames = 3 + (plan['chunk_tokens'] < 35)
    area = plan['tile_width']//32 * (plan['tile_height']//32)
    assert area * (plan['chunk_tokens'] + frames) <= plan['target_rows']
    assert plan['keyframe_rows_per_forward'] == area * frames


def test_impossible_budget_does_not_silently_plan_thousands_of_tiny_tiles():
    with pytest.raises(MemoryError, match='H3 refinement'):
        h3.plan_tiles((1, 24, 35, 90, 70), 2240, 2880, 32 * 1024**3,
                      model_bytes=38444 * 1024**2)


def test_audio_and_full_canvas_buffers_compete_with_attention_rows():
    kw = dict(model_bytes=38 * 1024**3, streaming_weights=True)
    quiet = h3.plan_tiles((1, 24, 35, 90, 70), 2240, 2880, 30 * 1024**3, **kw)
    audio = h3.plan_tiles((1, 24, 35, 90, 70), 2240, 2880, 30 * 1024**3,
                         audio_tokens=20000, **kw)
    assert audio['target_rows'] < quiet['target_rows']
    assert audio['tile_width'] * audio['tile_height'] <= quiet['tile_width'] * quiet['tile_height']


@pytest.mark.parametrize('width,height', [(2240,2880), (3840,2176), (7680,4320), (256,1024), (32,32)])
def test_minimum_tiles_cover_large_and_small_canvases(width, height):
    plan = h3.plan_tiles((1,24,35,16,16), width, height, 12 * 1024**3,
                         model_bytes=38 * 1024**3, streaming_weights=True)
    assert plan['tile_width'] >= min(width,512)
    assert plan['tile_height'] >= min(height,512)
    rows = plan['tile_width'] // 32 * (plan['tile_height'] // 32) * plan['chunk_tokens']
    assert rows <= plan['target_rows']


def test_minimum_tile_reduces_temporal_window_before_failing():
    plan = h3.plan_tiles((1,24,35,16,16), 2240, 2880, 1024**3, streaming_weights=True)
    assert min(plan['tile_width'],plan['tile_height']) >= 512
    assert plan['chunk_tokens'] < 30
    h3.temporal_ranges(35,plan['chunk_tokens'],plan['temporal_overlap_tokens'])
    with pytest.raises(MemoryError):
        h3.plan_tiles((1,24,35,16,16), 2240, 2880, 1024**2)
    h3.plan_tiles((1,24,35,16,16), 2240, 2880, 1024**2, enforce_budget=False)


@pytest.mark.parametrize('spatial,temporal', [(True,True),(False,True),(True,False),(False,False)])
def test_planner_switches_are_independent_and_never_reenabled(spatial,temporal):
    plan=h3.plan_tiles((1,24,60,32,32),1024,1024,16*1024**3,
                       spatial_tiling=spatial,temporal_chunking=temporal)
    if not spatial:
        assert (plan['tile_width'],plan['tile_height'],plan['overlap'])==(1024,1024,0)
    if not temporal:
        assert plan['chunk_tokens']==60
        assert plan['temporal_overlap_tokens']==0
    with pytest.raises(MemoryError):
        h3.plan_tiles((1,24,60,32,32),1024,1024,1024,
                      spatial_tiling=spatial,temporal_chunking=temporal)


@pytest.mark.parametrize('width,height,total,mib',[
    (768,1024,35,1024),(1024,768,20,640),(512,768,19,1024),
    (32,256,6,64),(768,768,62,2048),(1024,1024,1,256),
    (1024,1024,35,1)])
@pytest.mark.parametrize('spatial,temporal',[(True,True),(False,True),(True,False),(False,False)])
def test_plan_matches_exhaustive_actual_window_oracle(width,height,total,mib,spatial,temporal):
    import math
    shape=(1,24,total,16,16)
    budget=mib*1024**2*0.65
    chunks=[total] if not temporal else [c for c in range(10,min(total,30)+1,5)]
    if temporal and total<=30 and total not in chunks:
        chunks.append(total)
    widths=[width] if not spatial else range(min(width,512),width+1,32)
    heights=[height] if not spatial else range(min(height,512),height+1,32)
    valid=[]
    for chunk in chunks:
        frames=2+int(chunk<total)
        buffers=24*(height//16)*(width//16)*(chunk*20+frames*6)
        rows=int((budget-buffers)/(96*1024))-148
        windows=h3.temporal_ranges(total,chunk,5 if chunk<total else 0)
        for tw in widths:
            for th in heights:
                area=tw//32*(th//32)
                if area*(chunk+frames)>rows:
                    continue
                overlap=min(max(32,math.ceil(min(tw,th)/128)*32),min(tw,th)-32) if spatial else 0
                n=len(h3._starts(width//16,tw//16,overlap//16))*len(h3._starts(height//16,th//16,overlap//16))
                # Actual slice extents, not the planner's count shortcut.
                work=n*area*sum(b-a for a,b in windows)
                valid.append((n*len(windows),work,-chunk,abs(tw*height-th*width)))
    kwargs=dict(text_tokens=8,ref_tokens=128,audio_tokens=12,keyframe_tokens=2,
                spatial_tiling=spatial,temporal_chunking=temporal)
    if not valid:
        with pytest.raises(MemoryError):
            h3.plan_tiles(shape,width,height,mib*1024**2,**kwargs)
        return
    plan=h3.plan_tiles(shape,width,height,mib*1024**2,**kwargs)
    actual=(plan['forwards_per_step'],plan['processed_video_rows'],-plan['chunk_tokens'],
            abs(plan['tile_width']*height-plan['tile_height']*width))
    assert actual==min(valid)
    assert f'target={width}x{height}px' in plan['explanation']


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
