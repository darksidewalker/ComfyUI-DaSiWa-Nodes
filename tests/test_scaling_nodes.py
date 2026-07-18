import importlib.util
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).parents[1] / "nodes" / "scaling_nodes.py"
spec = importlib.util.spec_from_file_location("scaling_nodes", MODULE_PATH)
assert spec is not None and spec.loader is not None
scaling_nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scaling_nodes)

Calculator = scaling_nodes.DaSiWa_ResolutionScaleCalculator


def _image(values):
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)


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
        assert abs(total - expected) < 2048

    def test_1mp_presets_produce_1024_square_at_1x1(self):
        img = _fake_tensor(1024, 1024)
        calculator = Calculator()
        for preset in ("1024p", "1.00 MP - 1024p"):
            w, h, _, _ = calculator.calculate(
                resolution_preset=preset,
                image=img,
                mode="Standard",
            )
            assert (w, h) == (1024, 1024)

    def test_divisor_rounding_matches_comfyui_resolution_selector(self):
        calculator = Calculator()
        for mode, custom_divisor, expected in (
            ("WAN/LTX (Div32)", 8, (1376, 768)),
            ("LTX 2-Stage (Div64)", 8, (1344, 768)),
            ("CUSTOM", 8, (1368, 768)),
        ):
            w, h, _, _ = calculator.calculate(
                resolution_preset="1.00 MP - 1024p",
                scale_from_image=False,
                aspect_preset_when_not_image="CUSTOM",
                custom_aspect_width=16,
                custom_aspect_height=9,
                mode=mode,
                custom_divisor=custom_divisor,
            )
            assert (w, h) == expected


class TestResolutionPresetsRealWorld:
    """###p presets must resolve to their real-world pixel count."""

    def _check_preset(self, preset_name, expected_w, expected_h):
        img = _fake_tensor(expected_w, expected_h)
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
        assert abs(w / h - 16 / 9) < 0.05


class TestNativeTorchResize:
    @staticmethod
    def _resize(image, size_mode, aspect_mode, target_width, target_height, **kwargs):
        return scaling_nodes.DaSiWa_TorchResize().resize(
            image,
            size_mode,
            aspect_mode,
            target_width,
            target_height,
            kwargs.get("scale_multiplier", 1.0),
            kwargs.get("interpolation", "Nearest"),
            kwargs.get("gamma_correct", False),
            kwargs.get("divisible_by", 1),
            kwargs.get("pad_color", "0, 0, 0"),
            kwargs.get("crop_position", "center"),
            kwargs.get("batch_size", 0),
            kwargs.get("max_batch_megapixels", 16.0),
            kwargs.get("cache_size", 16),
        )[0]

    def test_node_is_registered_by_package(self):
        package_source = (Path(__file__).parents[1] / "__init__.py").read_text(encoding="utf-8")
        assert "DaSiWa_TorchResize" in package_source

    def test_input_schema_separates_size_and_aspect_modes(self):
        controls = scaling_nodes.DaSiWa_TorchResize.INPUT_TYPES()["required"]
        assert "resize_mode" not in controls
        assert "keep_aspect" not in controls
        assert controls["size_mode"][0] == ["Multiplier", "Target resolution"]
        assert controls["aspect_mode"][0] == ["Stretch", "Fit", "Fill and crop", "Fit and pad", "Long side with divisible crop"]

    def test_validation_accepts_comfyui_positional_metadata(self):
        node = scaling_nodes.DaSiWa_TorchResize()
        assert node.VALIDATE_INPUTS(image=object()) is True
        assert node.validate_inputs("image", "IMAGE", object(), object()) is True

    def test_resize_multiplier_sets_output_dimensions(self):
        output = self._resize(
            _image([[0.0, 1.0], [0.25, 0.75]]), "Multiplier", "Fit", 1, 1,
            scale_multiplier=2.0, interpolation="Bilinear",
        )
        assert output.shape == (1, 4, 4, 1)

    def test_nearest_is_pixel_perfect(self):
        output = self._resize(
            _image([[0.0, 1.0], [0.25, 0.75]]), "Target resolution", "Stretch", 4, 4,
        )
        assert torch.equal(output[0, :, :, 0], torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.25, 0.25, 0.75, 0.75],
            [0.25, 0.25, 0.75, 0.75],
        ]))

    def test_gamma_correct_bilinear_interpolates_in_linear_light(self):
        output = self._resize(
            _image([[0.0, 1.0]]), "Target resolution", "Stretch", 3, 1,
            interpolation="Bilinear", gamma_correct=True,
        )
        assert output[0, 0, 1, 0] > 0.7

    def test_lanczos_cache_reuses_matching_resize_weights(self):
        node = scaling_nodes.DaSiWa_TorchResize()
        image = torch.rand((1, 8, 8, 3), dtype=torch.float32)
        args = (image, "Target resolution", "Stretch", 16, 16, 1.0, "Lanczos", False, 1, "0, 0, 0", "center", 0, 16.0, 16)
        node.resize(*args)
        first = scaling_nodes._LANCZOS_CACHE.stats()
        node.resize(*args)
        second = scaling_nodes._LANCZOS_CACHE.stats()
        assert first["entries"] >= 1
        assert second["hits"] >= first["hits"] + 2

    def test_stretch_floors_dimensions_to_divisible_by(self):
        output = self._resize(_image([[0.0, 1.0]]), "Target resolution", "Stretch", 13, 9, divisible_by=4)
        assert output.shape == (1, 8, 12, 1)

    def test_fit_preserves_aspect_at_divisible_dimensions(self):
        output = self._resize(_image([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]), "Target resolution", "Fit", 13, 9, divisible_by=4)
        assert output.shape == (1, 4, 8, 1)

    def test_fill_and_crop_returns_requested_divisible_size(self):
        output = self._resize(_image([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]), "Target resolution", "Fill and crop", 13, 9, divisible_by=4)
        assert output.shape == (1, 8, 12, 1)

    def test_fit_and_pad_uses_color_and_bottom_right_position(self):
        image = torch.ones((1, 2, 4, 3), dtype=torch.float32)
        output = self._resize(image, "Target resolution", "Fit and pad", 8, 8, pad_color="255, 0, 0", crop_position="bottom-right")
        assert torch.equal(output[0, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
        assert torch.equal(output[0, -1, -1], torch.tensor([1.0, 1.0, 1.0]))

    def test_long_side_divisible_crop_locks_long_side_and_crops_short_side(self):
        output = self._resize(torch.ones((1, 9, 16, 3)), "Target resolution", "Long side with divisible crop", 15, 11, divisible_by=4)
        assert output.shape == (1, 4, 12, 3)

    def test_auto_batch_size_limits_frames_by_target_megapixels(self):
        assert scaling_nodes._auto_batch_size(100, 512, 512, 1024, 1024, 4.0) == 4

    def test_auto_chunking_preserves_a_hundred_video_frames(self):
        image = torch.rand((100, 8, 8, 3), dtype=torch.float32)
        output = self._resize(
            image, "Target resolution", "Stretch", 16, 16,
            max_batch_megapixels=0.001,
        )
        assert output.shape == (100, 16, 16, 3)
        assert torch.equal(output[::2, ::2, ::2], image[::2])
