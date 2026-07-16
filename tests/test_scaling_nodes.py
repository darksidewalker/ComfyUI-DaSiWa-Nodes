import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "nodes" / "scaling_nodes.py"
spec = importlib.util.spec_from_file_location("scaling_nodes", MODULE_PATH)
assert spec is not None and spec.loader is not None
scaling_nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scaling_nodes)

Calculator = scaling_nodes.DaSiWa_ResolutionScaleCalculator


def _fake_tensor(w, h):
    """Return a minimal (1, h, w, 4) tensor dict that mimics IMAGE shape."""
    import numpy as np
    return np.zeros((1, h, w, 4), dtype=np.float32)


class TestComfyUINativeCompatibility:
    """Verify DaSiWa matches ComfyUI native Scale Image to Total Pixels behavior."""

    def test_issue_example_065mp_div32(self):
        """Issue #4 example: 2304x1536 @ 0.65 MP Div32 -> 1024x672"""
        img = _fake_tensor(2304, 1536)
        w, h, _, _ = Calculator().calculate(
            resolution_preset="0.65 MP - Balanced",
            image=img,
            mode="WAN/LTX (Div32)",
        )
        assert (w, h) == (1024, 672)

    def test_mp_multiplier_is_1024_squared(self):
        """1 MP must equal exactly 1024*1024 pixels, not 1_000_000."""
        img = _fake_tensor(1024, 1024)
        w, h, _, _ = Calculator().calculate(
            resolution_preset="1.05 MP - HD+",
            image=img,
            mode="Standard",
        )
        total = w * h
        expected = 1.05 * 1024 * 1024
        assert abs(total - expected) < 2048  # within rounding tolerance


class TestResolutionPresetsRealWorld:
    """###p presets must resolve to their real-world pixel count."""

    def _check_preset(self, preset_name, expected_w, expected_h):
        img = _fake_tensor(expected_w, expected_h)
        ar = expected_w / expected_h
        w, h, _, _ = Calculator().calculate(
            resolution_preset=preset_name,
            image=img,
            mode="Standard",
        )
        assert (w, h) == (expected_w, expected_h), f"{preset_name}: got {w}x{h}"

    def test_720p(self):
        self._check_preset("720p", 1280, 720)

    def test_1080p(self):
        self._check_preset("1080p", 1920, 1080)

    def test_4k(self):
        self._check_preset("4K", 3840, 2160)

    def test_1440p(self):
        self._check_preset("1440p", 2560, 1440)


class TestNoScalePassthrough:
    def test_passes_source_dimensions(self):
        img = _fake_tensor(2048, 1152)
        w, h, wf, hf = Calculator().calculate(
            resolution_preset="0.52 MP - SD",
            no_scale=True,
            image=img,
        )
        assert (w, h) == (2048, 1152)
        assert (wf, hf) == (2048.0, 1152.0)


class TestAspectFromManual:
    def test_16x9_aspect_manual(self):
        w, h, _, _ = Calculator().calculate(
            resolution_preset="0.65 MP - Balanced",
            scale_from_image=False,
            aspect_preset_when_not_image="CUSTOM",
            custom_aspect_width=16,
            custom_aspect_height=9,
            mode="WAN/LTX (Div32)",
        )
        actual_ar = w / h
        assert abs(actual_ar - 16 / 9) < 0.05
