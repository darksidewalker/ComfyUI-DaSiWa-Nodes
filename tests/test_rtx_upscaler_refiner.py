import torch

from nodes.rtx_upscaler_refiner import (
    _fit_frame_to_target_aspect,
    _same_aspect,
)


def test_center_crop_matches_target_aspect_exactly():
    frame = torch.ones((3, 120, 160), dtype=torch.float32)

    fitted = _fit_frame_to_target_aspect(frame, 1920, 1080, "Center Crop (Fill)")

    _, height, width = fitted.shape
    assert _same_aspect(width, height, 1920, 1080)
    assert width <= 160
    assert height <= 120


def test_letterbox_matches_target_aspect_exactly_for_common_ratio():
    frame = torch.ones((3, 120, 160), dtype=torch.float32)

    fitted = _fit_frame_to_target_aspect(frame, 1920, 1080, "Letterbox (Fit)")

    _, height, width = fitted.shape
    assert _same_aspect(width, height, 1920, 1080)
    assert width >= 160
    assert height >= 120
    assert torch.count_nonzero(fitted == 0) > 0


def test_matching_aspect_returns_contiguous_copy_without_resize():
    frame = torch.ones((3, 90, 160), dtype=torch.float32).transpose(1, 2)

    fitted = _fit_frame_to_target_aspect(frame, 1080, 1920, "Center Crop (Fill)")

    assert fitted.shape == frame.shape
    assert fitted.is_contiguous()
