"""Model-free recovery contracts; no GPU/Core import is needed for mask tests."""
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nodes import nodes_minimax_h3_derope as derope


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(derope, "_recover_interrupt", lambda: None, raising=False)


def recover(images, baseline=None, enabled=True):
    return derope.DaSiWaH3DeropeRecover().recover(
        images, json.dumps({"holds": [1] * len(images)}), baseline, enabled)[0]


def scene(n=5, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(80), torch.arange(120), indexing="ij")
    texture = .3 + .15 * ((x + y) % 2).float()
    baseline = texture[None, :, :, None].expand(n, -1, -1, 3).clone().to(dtype)
    softened = F.avg_pool2d(baseline.permute(0, 3, 1, 2).float(), 5, 1, 2).permute(0, 2, 3, 1).to(dtype)
    return baseline, softened


def test_abi_and_off_is_old_byte_exact_without_touching_baseline():
    node = derope.DaSiWaH3DeropeRecover()
    schema = node.INPUT_TYPES()
    assert list(schema["required"]) == ["images", "hold_map"]
    assert list(schema["optional"]) == ["baseline", "keep_background"]
    assert schema["optional"]["baseline"] == ("IMAGE",)
    assert schema["optional"]["keep_background"][1]["default"] is False
    assert node.RETURN_TYPES == ("IMAGE",)
    images = torch.randn(6, 2, 3, 3)
    images[0, 0, 0, 0] = float("nan")
    hold_map = json.dumps({"holds": [2, 1, 3]})
    old = images[torch.tensor([0, 2, 3])].cpu()
    for baseline in [None, object(), torch.tensor(float("nan"))]:
        out = node.recover(images, hold_map, baseline=baseline)[0]
        assert torch.equal(out.view(torch.uint8), old.view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_static_texture_is_exactly_restored_and_inputs_unchanged(dtype):
    b, r = scene(dtype=dtype)
    saved_b, saved_r = b.clone(), r.clone()
    out = recover(r, b)
    assert torch.equal(out[:, 20:60, 20:100], b[:, 20:60, 20:100])
    assert out.shape == r.shape and out.dtype == r.dtype and out.device.type == "cpu"
    assert torch.equal(b, saved_b) and torch.equal(r, saved_r)


def test_shifted_limb_union_removes_old_limb_and_keeps_new_one():
    b, r = scene()
    b[:, 25:55, 35:43] = torch.tensor([.9, .05, .05])
    r[:, 25:55, 60:68] = torch.tensor([.9, .05, .05])
    out = recover(r, b)
    assert torch.equal(out[:, 25:55, 35:43], r[:, 25:55, 35:43])
    assert torch.equal(out[:, 25:55, 60:68], r[:, 25:55, 60:68])
    assert torch.equal(out[:, 20:60, 95:110], b[:, 20:60, 95:110])


def test_baseline_neighbor_motion_protects_low_contrast_moving_limb():
    b = torch.full((5, 80, 120, 3), .4)
    for i in range(5):
        b[i, 25:55, 20+i*12:28+i*12] = .45
    r = torch.full_like(b, .4)
    out = recover(r, b)
    assert torch.equal(out, r)  # .05 is below structural-change threshold, but motion must remove it


def test_temporal_union_holds_a_one_frame_threshold_dip():
    b = torch.full((5, 80, 120, 3), .4)
    r = b.clone()
    r[:, 25:55, 45:65] = .5
    r[2, 25:55, 45:65] = .45
    out = recover(r, b)
    assert torch.equal(out[:, 25:55, 45:65], r[:, 25:55, 45:65])


def test_global_pan_falls_back_to_full_derope(caplog):
    torch.manual_seed(12)
    coarse = torch.rand(1, 3, 10, 15)
    texture = F.interpolate(coarse, size=(80, 120), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    b = torch.stack([torch.roll(texture, i*8, 1) for i in range(5)])
    r = (b * .8 + .05).clamp(0, 1)
    with caplog.at_level("INFO"):
        out = recover(r, b)
    assert torch.equal(out, r)
    assert "coverage" in caplog.text and "De-rope" in caplog.text


@pytest.mark.parametrize("n", [1, 2])
def test_short_clips(n):
    b, r = scene(n)
    assert torch.equal(recover(r, b)[:, 20:60, 20:100], b[:, 20:60, 20:100])


@pytest.mark.parametrize("bad", [None, object(), torch.zeros(4, 80, 120, 3), torch.zeros(5, 80, 119, 3),
                                  torch.zeros(5, 80, 120, 1), torch.zeros(5, 80, 120, 3, dtype=torch.uint8)])
def test_enabled_rejects_missing_or_malformed_baseline(bad):
    _, r = scene()
    with pytest.raises(ValueError, match="De-rope.*baseline"):
        recover(r, bad)


@pytest.mark.parametrize("target,value", [("b", float("nan")), ("b", float("inf")), ("r", -.01), ("r", 1.01)])
def test_invalid_pixels_rejected(target, value):
    b, r = scene()
    (b if target == "b" else r)[-1, 0, 0, 0] = value
    with pytest.raises(ValueError, match="De-rope.*finite.*0.*1"):
        recover(r, b)


def test_interrupt_checked_inside_validation_and_mask_loops(monkeypatch):
    b, r = scene()
    calls = []
    def interrupted():
        calls.append(1)
        if len(calls) == len(b) + 2:
            raise RuntimeError("cancelled")
    monkeypatch.setattr(derope, "_recover_interrupt", interrupted)
    with pytest.raises(RuntimeError, match="cancelled"):
        recover(r, b)
    assert len(calls) == len(b) + 2


def test_mask_release_is_gradual_not_a_one_frame_flash():
    b = torch.full((9, 80, 120, 3), .4)
    r = torch.full_like(b, .42)
    r[:2, 25:55, 45:65] = .8
    out = recover(r, b)
    alpha = (out[2:, 40, 55, 0] - .4) / .02
    assert alpha[0] == pytest.approx(1, abs=1e-5)
    assert bool((alpha[1:] <= alpha[:-1]).all())
    assert float((alpha[1:] - alpha[:-1]).abs().max()) <= .251


def test_broad_scene_change_uses_full_second_pass():
    torch.manual_seed(20)
    b = torch.full((3, 80, 120, 3), .4)
    r = F.interpolate(torch.rand(3, 3, 6, 9), size=(80, 120), mode="bilinear",
                      align_corners=False).permute(0, 2, 3, 1)
    assert torch.equal(recover(r, b), r)


def test_small_exposure_and_texture_changes_are_not_motion():
    b, r = scene()
    r = r + .02
    out = recover(r, b)
    assert torch.equal(out[:, 20:60, 20:100], b[:, 20:60, 20:100])


def test_rgba_noncontiguous_and_mixed_dtype_baseline():
    b, r = scene(n=2)
    b = torch.cat((b, torch.ones_like(b[..., :1])), -1).double().transpose(1, 2)
    r = torch.cat((r, torch.ones_like(r[..., :1])), -1).transpose(1, 2)
    out = recover(r, b)
    assert out.dtype == r.dtype and out.shape == r.shape
    assert torch.equal(out[:, 20:100, 20:60], b[:, 20:100, 20:60].float())


@pytest.mark.parametrize("shape", [(1, 1, 1, 3), (2, 3, 2, 3)])
def test_tiny_valid_images(shape):
    b = torch.full(shape, .4)
    r = torch.full(shape, .42)
    assert torch.equal(recover(r, b), b)


@pytest.mark.parametrize("images", [None, [], torch.empty(0, 4, 4, 3), torch.empty(1, 0, 4, 3),
                                      torch.zeros(2, 4, 3), torch.zeros(1, 4, 4, 3, dtype=torch.int32)])
def test_malformed_recovered_images(images):
    with pytest.raises(ValueError, match="De-rope"):
        derope.DaSiWaH3DeropeRecover().recover(images, '{"holds": [1]}', images, True)


def test_off_never_enters_background_code(monkeypatch):
    def forbidden(*args):
        raise AssertionError("OFF inspected optional preservation")
    monkeypatch.setattr(derope, "_preserve_background", forbidden)
    b, r = scene(2)
    hold_map = '{"holds": [1, 1]}'
    node = derope.DaSiWaH3DeropeRecover()
    assert torch.equal(node.recover(r, hold_map)[0], r)
    assert torch.equal(node.recover(r, hold_map, object(), False)[0], r)


def test_on_recovers_held_frames_before_baseline_shape_validation():
    b, r = scene(2)
    held = r[torch.tensor([0, 0, 1, 1, 1])]
    result = derope.DaSiWaH3DeropeRecover().recover(held, '{"holds": [2, 3]}', b, True)[0]
    assert torch.equal(result, recover(r, b))
