"""Aspect choices preserve each calculator's existing area and alignment rules."""
import sys
from pathlib import Path

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "ComfyUI"),
               str(Path(__file__).resolve().parents[1] / "nodes")]
from nodes_scaling import DaSiWa_ResolutionScaleCalculator as Calculator
import nodes_rtx_upscaler_refiner as rtx


@pytest.mark.parametrize("aspect", ["7:9", "9:7"])
@pytest.mark.parametrize("alignment,expected", [
    (8, (888, 1144)), (16, (896, 1152)), (32, (896, 1152)),
    (64, (896, 1152)), (128, (896, 1152)),
])
def test_rtx_aspect_presets_keep_decimal_mp_and_ratio_first_alignment(aspect, alignment, expected):
    assert aspect in rtx.DaSiWa_RTX_UpscalerRefiner.INPUT_TYPES()["required"]["ratio_preset"][0]
    if aspect == "9:7":
        expected = expected[::-1]
    assert rtx.compute_aligned_ratio_dims(aspect, 1, alignment) == expected



@pytest.mark.parametrize("aspect", ["7:9", "9:7"])
@pytest.mark.parametrize("mode,divisor,expected", [
    ("Standard", 8, (903, 1161)),
    ("WAN/LTX (Div32)", 8, (896, 1152)),
    ("LTX 2-Stage (Div64)", 8, (896, 1152)),
    ("CUSTOM", 8, (904, 1160)),
    ("CUSTOM", 16, (896, 1168)),
])
def test_calculator_aspect_presets(aspect, mode, divisor, expected):
    assert aspect in Calculator.INPUT_TYPES()["required"]["aspect_preset_when_not_image"][0]
    if aspect == "9:7":
        expected = expected[::-1]
    calculator = Calculator()
    kwargs = dict(resolution_preset="1024p", scale_from_image=False,
                  aspect_preset_when_not_image=aspect, mode=mode, custom_divisor=divisor)
    assert calculator.calculate(**kwargs) == (*expected, *map(float, expected))
    assert calculator.calculate(**kwargs, swap_aspect_when_not_image=True)[:2] == expected[::-1]
