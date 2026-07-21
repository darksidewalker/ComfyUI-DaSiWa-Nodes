import os
import gc
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


folder_paths = types.ModuleType("folder_paths")
folder_paths.get_temp_directory = lambda: "/tmp"
sys.modules.setdefault("folder_paths", folder_paths)
MODULE_PATH = Path(__file__).parents[1] / "nodes" / "rtx_upscaler_refiner.py"
spec = importlib.util.spec_from_file_location("rtx_upscaler_refiner", MODULE_PATH)
assert spec is not None and spec.loader is not None
rtx_upscaler_refiner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rtx_upscaler_refiner)

DaSiWa_RTX_UpscalerRefiner = rtx_upscaler_refiner.DaSiWa_RTX_UpscalerRefiner
_fit_frame_to_target_aspect = rtx_upscaler_refiner._fit_frame_to_target_aspect
_same_aspect = rtx_upscaler_refiner._same_aspect


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


def test_validate_inputs_accepts_new_comfyui_positional_signature():
    node = DaSiWa_RTX_UpscalerRefiner()

    assert node.validate_inputs("images", "IMAGE", object(), object()) is True


def test_validate_inputs_accepts_class_level_positional_signature():
    validate_inputs = DaSiWa_RTX_UpscalerRefiner.__dict__["validate_inputs"]

    assert validate_inputs(
        DaSiWa_RTX_UpscalerRefiner, "images", "IMAGE", object(), object()
    ) is True


def test_projected_output_bytes_reports_full_rgb_batch_size():
    assert rtx_upscaler_refiner._projected_output_bytes(480, 7680, 4320, torch.float32) == 191_102_976_000


def test_large_cpu_output_uses_comfy_temp_mmap_with_stable_frame_indexes(tmp_path, monkeypatch):
    monkeypatch.setattr(rtx_upscaler_refiner, "MAX_IN_MEMORY_OUTPUT_BYTES", 1)
    monkeypatch.setattr(rtx_upscaler_refiner, "_temporary_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(rtx_upscaler_refiner, "_has_free_disk_space", lambda *_: True)

    output, storage_path = rtx_upscaler_refiner._allocate_output_tensor((3, 2, 2, 3), torch.float32, torch.device("cpu"))
    output[0].fill_(1)
    output[1].fill_(2)
    output[2].fill_(3)

    assert storage_path is not None
    assert os.path.dirname(storage_path) == str(tmp_path)
    assert [output[index, 0, 0, 0].item() for index in range(3)] == [1, 2, 3]


def test_mmap_output_fails_cleanly_when_comfy_temp_has_insufficient_space(tmp_path, monkeypatch):
    monkeypatch.setattr(rtx_upscaler_refiner, "MAX_IN_MEMORY_OUTPUT_BYTES", 1)
    monkeypatch.setattr(rtx_upscaler_refiner, "_temporary_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(rtx_upscaler_refiner, "_has_free_disk_space", lambda *_: False)

    with pytest.raises(RuntimeError, match="ComfyUI temporary directory"):
        rtx_upscaler_refiner._allocate_output_tensor((3, 2, 2, 3), torch.float32, torch.device("cpu"))


def test_mmap_output_removes_its_temporary_file_when_tensor_is_released(tmp_path, monkeypatch):
    monkeypatch.setattr(rtx_upscaler_refiner, "MAX_IN_MEMORY_OUTPUT_BYTES", 1)
    monkeypatch.setattr(rtx_upscaler_refiner, "_temporary_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(rtx_upscaler_refiner, "_has_free_disk_space", lambda *_: True)

    output, storage_path = rtx_upscaler_refiner._allocate_output_tensor((1, 2, 2, 3), torch.float32, torch.device("cpu"))

    assert storage_path is not None
    assert os.path.exists(storage_path)
    del output
    gc.collect()
    assert not os.path.exists(storage_path)
