import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ComfyUI"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nodes"))
import nodes_rtx_upscaler_refiner as rtx


@pytest.mark.skipif(not torch.cuda.is_available(), reason="RTX effect requires CUDA")
def test_regular_node_keeps_single_image_batch_with_default_chunking(monkeypatch):
    class FakeEffect:
        def run(self, frame):
            return type("Result", (), {"image": frame.clone()})()

    monkeypatch.setattr(rtx, "_import_vfx", lambda: (object(), object()))
    monkeypatch.setattr(rtx, "_maybe_vfx_effect", lambda *args, **kwargs: __import__("contextlib").nullcontext(FakeEffect() if args[1] else None))
    monkeypatch.setattr(rtx, "_cleanup_cuda", lambda **kwargs: None)
    monkeypatch.setattr(rtx, "_can_fit_in_vram", lambda *args: True)
    monkeypatch.setattr(rtx, "_dasiwa_rtx_free_memory", lambda *args: pytest.fail("synthetic test must not offload live models"))
    options = rtx.DaSiWa_RTX_UpscalerRefiner.INPUT_TYPES()["optional"]
    assert options["chunking"][1]["default"] is True
    assert options["lossless_fp16"][1]["default"] is True
    images = torch.full((5, 8, 8, 3), 0.1, dtype=torch.float32)
    kwargs = dict(images=images, denoise=True, denoise_quality="High", deblur=False,
                  deblur_quality="High", upscale="Off", upscale_quality="High",
                  resize_type="Same Size", scale=1.0, megapixels=1.0,
                  width=8, height=8, divisible_by="8", ratio_preset="1:1",
                  resize_method="Center Crop (Fill)", device_id=0, chunk_frames=2)
    chunked, = rtx.DaSiWa_RTX_UpscalerRefiner().execute(**kwargs)
    unchunked, = rtx.DaSiWa_RTX_UpscalerRefiner().execute(**kwargs, chunking=False)
    assert chunked.shape == images.shape
    assert chunked.dtype == torch.float32
    assert torch.equal(chunked, unchunked)


def test_regular_batch_lossless_fp16_only_when_exact(monkeypatch):
    monkeypatch.setattr(rtx, "can_allocate_in_ram", lambda *args: True)
    exact = torch.full((5, 8, 8, 3), 0.25, dtype=torch.float32)
    result = rtx._lossless_fp16_image_batch(exact, 2)
    assert result.dtype == torch.float16 and torch.equal(result.float(), exact)
    exact[4, 0, 0, 0] = 0.1
    result = rtx._lossless_fp16_image_batch(exact, 2)
    assert result is exact and result.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="RTX effect requires CUDA")
def test_direct_sdk_copy_keeps_pixels_after_sdk_reuses_buffer():
    class ReusingEffect:
        def __init__(self):
            self.buffer = torch.empty((3, 8, 8), device="cuda")

        def run(self, frame):
            self.buffer.copy_(frame)
            return type("Result", (), {"image": self.buffer})()

    effect = ReusingEffect()
    row = torch.empty((8, 8, 3), dtype=torch.float32)
    frame = torch.full((3, 8, 8), 0.25, device="cuda")
    rtx._run_vfx_effect_into(effect, frame, row, torch.device("cuda:0"))
    effect.run(torch.ones_like(frame))
    assert torch.equal(row, torch.full_like(row, 0.25))

# All feature tests below use concrete SDK signatures and CPU tensors. They are
# contract tests, NOT NVIDIA SDK inference/quality validation.
from contextlib import nullcontext
from types import SimpleNamespace
from enum import IntEnum
import inspect


class Encoding(IntEnum):
    RGB8 = 0
    RGB10A2 = 1


class Levels(IntEnum):
    HIGH = 3
    ULTRA = 4
    DENOISE_HIGH = 10
    DENOISE_ULTRA = 11
    DEBLUR_HIGH = 14
    DEBLUR_ULTRA = 15
    HIGHBITRATE_HIGH = 18
    STREAMING_MEDIUM = 21
    STREAMING_ULTRA = 23


class ModernVSR:
    calls = []
    def __init__(self, quality=Levels.HIGH, strength=1.0, device=0, image_encoding=Encoding.RGB8):
        self.quality, self.strength, self.device, self.image_encoding = quality, strength, device, image_encoding
        self.calls.append((quality, strength, device, image_encoding))
        self.buffer = None
        self.closed = False
        self.output_height = self.output_width = 8
    def load(self):
        pass
    def close(self):
        self.closed = True
    def run(self, frame):
        rgb = rtx._dasiwa_rtx_unpack_rgb10a2(frame) if self.image_encoding == Encoding.RGB10A2 else frame
        rgb = torch.nn.functional.interpolate(rgb[None], size=(self.output_height, self.output_width),
            mode="bilinear", align_corners=False)[0]
        result = rtx._dasiwa_rtx_pack_rgb10a2(rgb) if self.image_encoding == Encoding.RGB10A2 else rgb
        if self.buffer is None or self.buffer.shape != result.shape:
            self.buffer = torch.empty_like(result)
        self.buffer.copy_(result)
        return SimpleNamespace(image=self.buffer)


class ModernHDR:
    calls = []
    def __init__(self, *, contrast=100, saturation=100, middle_gray=50, luminance=650, debanding_off=0, device=0):
        self.calls.append((contrast, saturation, middle_gray, luminance, debanding_off, device))
        self.closed = False
        self.buffer = None
    def close(self):
        self.closed = True
    def run(self, frame):
        # Deliberately distinct normalized signal, not simulated HDR science.
        result = rtx._dasiwa_rtx_pack_rgb10a2(frame * 0.5)
        if self.buffer is None or self.buffer.shape != result.shape:
            self.buffer = torch.empty_like(result)
        self.buffer.copy_(result)
        return SimpleNamespace(image=self.buffer)


class LegacyLevels(IntEnum):
    HIGH = 3
    DENOISE_HIGH = 10
    DEBLUR_HIGH = 14


class LegacyVSR:
    QualityLevel = LegacyLevels
    def __init__(self, quality=Levels.HIGH, device=0):
        raise AssertionError("schema must not instantiate any effect")


@pytest.fixture
def modern_sdk(monkeypatch):
    sdk = SimpleNamespace(VideoSuperRes=ModernVSR, TrueHDR=ModernHDR,
        ImageEncoding=Encoding, effects=SimpleNamespace(QualityLevel=Levels))
    monkeypatch.setitem(sys.modules, "nvvfx", sdk)
    ModernVSR.calls.clear()
    ModernHDR.calls.clear()
    return sdk


@pytest.fixture
def cpu_execution(monkeypatch, modern_sdk):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: SimpleNamespace(synchronize=lambda: None))
    original_to = torch.Tensor.to
    def to_cpu(self, *args, **kwargs):
        if "device" in kwargs and str(kwargs["device"]).startswith("cuda"):
            kwargs["device"] = torch.device("cpu")
        return original_to(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", to_cpu)
    monkeypatch.setattr(rtx, "_allocate_output_tensor", lambda shape, dtype, device, **kw: (torch.empty(shape, dtype=dtype), None))
    monkeypatch.setattr(rtx, "_model_management", None)


def options(images, **extra):
    values = dict(images=images, denoise=False, denoise_quality="High", deblur=False,
        deblur_quality="High", upscale="Off", upscale_quality="High", resize_type="Same Size",
        scale=1., megapixels=1., width=8, height=8, divisible_by="8", ratio_preset="1:1",
        resize_method="Center Crop (Fill)", device_id=0, lossless_fp16=False, auto_unload_models=False)
    values.update(extra)
    return values


def test_workflow_abi(modern_sdk):
    schema = rtx.DaSiWa_RTX_UpscalerRefiner.INPUT_TYPES()
    required = ["images", "denoise", "denoise_quality", "deblur", "deblur_quality", "upscale",
        "upscale_quality", "resize_type", "scale", "megapixels", "width", "height", "divisible_by",
        "ratio_preset", "resize_method", "device_id"]
    old_optional = ["empty_cache", "use_mmap", "auto_unload_models", "chunking", "lossless_fp16", "chunk_frames"]
    assert list(schema["required"]) == required
    assert list(schema["optional"])[:6] == old_optional
    assert list(inspect.signature(rtx.DaSiWa_RTX_UpscalerRefiner.execute).parameters)[1:23] == required + old_optional
    assert rtx.DaSiWa_RTX_UpscalerRefiner.RETURN_TYPES == ("IMAGE",)
    assert rtx.DaSiWa_RTX_UpscalerRefiner.RETURN_NAMES == ("images",)
    assert rtx.DaSiWa_RTX_UpscalerRefiner.FUNCTION == "execute"
    assert set(rtx.NODE_CLASS_MAPPINGS) == {"DaSiWa_RTX_UpscalerRefiner"}


@pytest.mark.parametrize("kind", ["modern", "legacy", "missing"])
def test_capability_probe_no_allocation(monkeypatch, modern_sdk, kind):
    if kind == "legacy":
        monkeypatch.setitem(sys.modules, "nvvfx", SimpleNamespace(VideoSuperRes=LegacyVSR))
    elif kind == "missing":
        monkeypatch.setitem(sys.modules, "nvvfx", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(AssertionError("GPU probe")))
    capabilities = rtx._dasiwa_rtx_capabilities()
    schema = rtx.DaSiWa_RTX_UpscalerRefiner.INPUT_TYPES()
    assert capabilities["hdr"] == (kind == "modern")
    assert capabilities["encoding"] == (kind == "modern")
    assert "hdr_enabled" in schema["optional"]
    assert "image_encoding" in schema["optional"]
    assert schema["optional"]["hdr_enabled"][1]["dasiwa_unavailable"] == (kind != "modern")
    assert schema["optional"]["image_encoding"][1]["dasiwa_unavailable"] == (kind != "modern")
    assert schema["required"]["upscale"][0][:3] == ["Off", "VSR", "High Bitrate"]


def test_old_sdk_strength_is_detected_by_signature(monkeypatch):
    class OldStrength:
        def __init__(self, quality=3, strength=1.0, device=0):
            raise AssertionError("schema allocation")
    monkeypatch.setitem(sys.modules, "nvvfx", SimpleNamespace(VideoSuperRes=OldStrength))
    assert rtx._dasiwa_rtx_capabilities()["strength"]
    assert "denoise_strength" in rtx.DaSiWa_RTX_UpscalerRefiner.INPUT_TYPES()["optional"]


@pytest.mark.parametrize("extra, message", [
    ({"image_encoding": "10-bit RGB (RGB10A2)"}, "RGB10A2"),
    ({"hdr_enabled": True}, "TrueHDR"),
    ({"upscale": "Streaming Ultra"}, "Streaming Ultra"),
    ({"denoise_strength": .5}, "strength")])
def test_explicit_unsupported_requests_fail(monkeypatch, extra, message):
    monkeypatch.setitem(sys.modules, "nvvfx", SimpleNamespace(VideoSuperRes=LegacyVSR))
    with pytest.raises(RuntimeError, match=message):
        rtx.DaSiWa_RTX_UpscalerRefiner().execute(**options(torch.zeros(1, 8, 8, 3), **extra))


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_rgb10a2_roundtrip(noncontiguous):
    rgb = torch.stack([torch.linspace(0, 1, 35).reshape(5, 7), torch.ones(5, 7), torch.zeros(5, 7)])
    if noncontiguous:
        rgb = rgb.transpose(1, 2)
    packed = rtx._dasiwa_rtx_pack_rgb10a2(rgb)
    assert packed.dtype == torch.uint32 and packed.is_contiguous()
    assert torch.equal(rtx._dasiwa_rtx_unpack_rgb10a2(packed), (rgb * 1023).round() / 1023)
    for channel in range(3):
        color = torch.zeros(3, 1, 1)
        color[channel] = 1
        assert int(rtx._dasiwa_rtx_pack_rgb10a2(color).item()) == (1023 << (channel * 10)) | (3 << 30)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("passes", [(True, False, False), (False, True, False), (False, False, True), (True, True, True)])
def test_pipeline_and_reused_buffers(cpu_execution, packed, passes):
    images = torch.stack([torch.full((8, 8, 3), .25), torch.full((8, 8, 3), .75)])
    denoise, deblur, upscale = passes
    result, = rtx.DaSiWa_RTX_UpscalerRefiner().execute(**options(images, denoise=denoise, deblur=deblur,
        upscale="VSR" if upscale else "Off", image_encoding="10-bit RGB (RGB10A2)" if packed else "8-bit RGB",
        denoise_strength=.2, deblur_strength=.4, upscale_strength=.6))
    expected = (images * 1023).round() / 1023 if packed else images
    assert result.dtype == torch.float32 and torch.equal(result, expected)
    expected_strengths = [s for active, s in zip(passes, [.2, .4, .6]) if active]
    assert [call[1] for call in ModernVSR.calls] == expected_strengths
    assert all(call[3] == (Encoding.RGB10A2 if packed else Encoding.RGB8) for call in ModernVSR.calls)


@pytest.mark.parametrize("mode, level", [("Streaming Medium", Levels.STREAMING_MEDIUM), ("Streaming Ultra", Levels.STREAMING_ULTRA)])
def test_streaming_exact_levels(cpu_execution, mode, level):
    rtx.DaSiWa_RTX_UpscalerRefiner().execute(**options(torch.ones(1, 8, 8, 3), upscale=mode, upscale_quality="Low"))
    assert ModernVSR.calls[0][0] == level


@pytest.mark.parametrize("multipass", [False, True])
@pytest.mark.parametrize("alpha", [False, True])
def test_hdr_only_and_final_pass(cpu_execution, multipass, alpha):
    images = torch.stack([torch.full((8, 8, 4), .25), torch.full((8, 8, 4), .75)])
    result, = rtx.DaSiWa_RTX_UpscalerRefiner().execute(**options(images, hdr_enabled=True,
        denoise=multipass, deblur=multipass, upscale="VSR" if multipass else "Off",
        preserve_alpha=alpha, channel_integrity_check=True, hdr_contrast=120, hdr_saturation=110,
        hdr_middle_gray=60, hdr_luminance=1000, hdr_debanding=False))
    assert result.dtype == torch.float32
    assert result.shape[-1] == (4 if alpha else 3)
    assert torch.equal(result[..., :3], (images[..., :3] * .5 * 1023).round() / 1023)
    if alpha:
        assert torch.equal(result[..., 3], images[..., 3])
    assert ModernHDR.calls == [(120, 110, 60, 1000, 1, 0)]


@pytest.mark.parametrize("method", ["Center Crop (Fill)", "Letterbox (Fit)"])
def test_alpha_spatial_alignment(cpu_execution, method):
    alpha = torch.linspace(0, 1, 128).reshape(8, 16, 1)
    images = torch.cat([torch.full((8, 16, 3), .5), alpha], dim=-1)[None]
    result, = rtx.DaSiWa_RTX_UpscalerRefiner().execute(**options(images, upscale="VSR", resize_type="Manual",
        width=16, height=16, preserve_alpha=True, resize_method=method, image_encoding="10-bit RGB (RGB10A2)"))
    aligned = rtx._fit_frame_to_target_aspect(alpha.permute(2, 0, 1), 16, 16, method)
    expected = torch.nn.functional.interpolate(aligned[None], size=(16, 16), mode="bilinear", align_corners=False)[0, 0]
    assert result.shape == (1, 16, 16, 4)
    assert torch.equal(result[0, :, :, 3], expected)


def test_legacy_no_effect_call_and_alpha_default(cpu_execution):
    images = torch.full((1, 8, 8, 4), .25)
    result, = rtx.DaSiWa_RTX_UpscalerRefiner().execute(**options(images))
    assert result.shape == (1, 8, 8, 3)
    assert torch.equal(result, images[..., :3])
    assert not ModernHDR.calls and not ModernVSR.calls
    assert rtx._projected_output_bytes(2, 8, 8, torch.float32, 4) == 2048


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_integrity_nonfinite(value):
    frame = torch.ones(3, 8, 8)
    result = frame.clone()
    result[0, 0, 0] = value
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        rtx._dasiwa_rtx_integrity(frame, result, "VSR")


def test_integrity_collapse_and_legitimate_frames():
    frame = torch.ones(3, 8, 8)
    result = frame.clone()
    result[0] = 0
    with pytest.raises(RuntimeError, match="suspected channel collapse"):
        rtx._dasiwa_rtx_integrity(frame, result, "VSR")
    for legitimate in (torch.zeros_like(frame), result, torch.full_like(frame, .001)):
        rtx._dasiwa_rtx_integrity(legitimate, legitimate, "VSR")


def test_integrity_preclamp_and_hdr_not_sdr(cpu_execution, monkeypatch):
    original_run = ModernVSR.run
    def run(self, frame):
        result = original_run(self, frame)
        result.image[0, 0, 0] = float("inf")
        return result
    monkeypatch.setattr(ModernVSR, "run", run)
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        rtx.DaSiWa_RTX_UpscalerRefiner().execute(**options(torch.ones(1, 8, 8, 3), denoise=True, channel_integrity_check=True))


def test_new_settings_constructor_defaults_match_installed_contract(modern_sdk):
    instance = rtx._create_vfx_effect(ModernVSR, Levels.ULTRA, 2)
    assert instance.strength == 1.0 and instance.device == 2 and instance.image_encoding == Encoding.RGB8


def test_legacy_sdk_execution_with_old_prompt(cpu_execution, monkeypatch):
    class WorkingLegacy(ModernVSR):
        QualityLevel = LegacyLevels
        def __init__(self, quality=LegacyLevels.HIGH, device=0):
            super().__init__(quality=quality, device=device)
    monkeypatch.setitem(sys.modules, "nvvfx", SimpleNamespace(VideoSuperRes=WorkingLegacy))
    images = torch.full((2, 8, 8, 3), .25)
    result, = rtx.DaSiWa_RTX_UpscalerRefiner().execute(**options(images, denoise=True))
    assert torch.equal(result, images)
    assert ModernVSR.calls == [(LegacyLevels.DENOISE_HIGH, 1.0, 0, Encoding.RGB8)]


def test_fp16_hdr_input_keeps_pq_precision(cpu_execution):
    images = torch.full((1, 8, 8, 3), .25, dtype=torch.float16)
    result, = rtx.DaSiWa_RTX_UpscalerRefiner().execute(**options(images, hdr_enabled=True, lossless_fp16=True))
    expected = torch.full((1, 8, 8, 3), 128 / 1023)
    assert torch.equal(result, expected) and result.dtype == torch.float32


@pytest.mark.parametrize("extra, message", [
    ({"image_encoding": "10-bit RGB (RGB10A2)"}, "RGB10A2"),
    ({"hdr_enabled": True}, "TrueHDR"),
    ({"denoise_strength": .5}, "strength")])
def test_core_input_preparation_preserves_unsupported_requests(monkeypatch, extra, message):
    import execution
    monkeypatch.setitem(sys.modules, "nvvfx", SimpleNamespace(VideoSuperRes=LegacyVSR))
    prepared, *_ = execution.get_input_data(extra, rtx.DaSiWa_RTX_UpscalerRefiner, "rtx-test")
    assert {name: value[0] for name, value in prepared.items()} == extra
    with pytest.raises(RuntimeError, match=message):
        rtx.DaSiWa_RTX_UpscalerRefiner().execute(**options(
            torch.zeros(1, 8, 8, 3), **{name: value[0] for name, value in prepared.items()}))


# Memory allocation, staging and stream ownership regression contracts.
def test_gpu_budget_includes_working_reserve_and_targeted_reclaim(monkeypatch):
    device = torch.device("cuda:0")
    budgets, requests = [], []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev: (0, 8 * 1024 ** 3))
    monkeypatch.setattr(rtx, "_can_fit_in_vram", lambda size, dev: budgets.append((size, dev)) or len(budgets) > 1)
    monkeypatch.setattr(torch.cuda, "device", lambda dev: nullcontext())
    monkeypatch.setattr(rtx, "_model_management", SimpleNamespace(free_memory=lambda size, dev: requests.append((size, dev))))
    original_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: original_empty(*a, **{k: v for k, v in kw.items() if k != "device"}))
    rtx._allocate_output_tensor((1, 8, 8, 3), torch.float32, device, working_reserve_bytes=1000)
    assert budgets == [(1768, device), (1768, device)]
    assert requests == [(2080, device)]


@pytest.mark.parametrize("workspace_fits", [True, False])
def test_below_cap_but_over_device_capacity_skips_output_eviction(monkeypatch, workspace_fits):
    device = torch.device("cuda:0")
    requests = []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev: (0, 8 * 1024 ** 3))
    monkeypatch.setattr(rtx, "_can_fit_in_vram", lambda size, dev: size == 1000 and workspace_fits)
    monkeypatch.setattr(rtx, "_dasiwa_rtx_free_memory", lambda size, dev: requests.append(size))
    monkeypatch.setattr(rtx, "can_allocate_in_ram", lambda size: True)
    marker = object()
    original_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda shape, **kw: original_empty(shape, **kw) if shape == () else marker)
    shape = (1, 32768, 32768, 3)  # 12 GiB, below fixed 24 GiB cap but impossible on 8 GiB GPU.
    monkeypatch.setattr(rtx, "_dasiwa_rtx_ram_budget", lambda *args: (64 * 1024 ** 3, 26 * 1024 ** 3, 38 * 1024 ** 3))
    if workspace_fits:
        assert rtx._allocate_output_tensor(shape, torch.float32, device, working_reserve_bytes=1000) == (marker, None)
        assert requests == []
    else:
        with pytest.raises(RuntimeError, match="working reserve"):
            rtx._allocate_output_tensor(shape, torch.float32, device, working_reserve_bytes=1000)
        assert requests == [1000]  # Only the small compute workspace may request reclamation.


def test_cap_impossible_does_not_evict_models(monkeypatch):
    monkeypatch.setattr(rtx, "MAX_GPU_OUTPUT_BYTES", 1)
    monkeypatch.setattr(rtx, "can_allocate_in_ram", lambda size: True)
    monkeypatch.setattr(rtx, "_can_fit_in_vram", lambda size, dev: True)
    monkeypatch.setattr(rtx, "_dasiwa_rtx_free_memory", lambda *a: pytest.fail("above-cap eviction"))
    result, _ = rtx._allocate_output_tensor((1, 8, 8, 3), torch.float32, torch.device("cuda:0"), working_reserve_bytes=1000)
    assert result.device.type == "cpu"


def test_cap_impossible_workspace_abort_without_eviction(monkeypatch):
    monkeypatch.setattr(rtx, "MAX_GPU_OUTPUT_BYTES", 1)
    monkeypatch.setattr(rtx, "_can_fit_in_vram", lambda *a: False)
    monkeypatch.setattr(rtx, "_dasiwa_rtx_free_memory", lambda *a: pytest.fail("above-cap eviction"))
    with pytest.raises(RuntimeError, match="working reserve"):
        rtx._allocate_output_tensor((1, 8, 8, 3), torch.float32, torch.device("cuda:0"), working_reserve_bytes=1000)


def test_cpu_output_requires_compute_workspace(monkeypatch):
    monkeypatch.setattr(rtx, "_can_fit_in_vram", lambda *a: False)
    requests = []
    monkeypatch.setattr(rtx, "_dasiwa_rtx_free_memory", lambda size, dev: requests.append((size, dev)))
    with pytest.raises(RuntimeError, match="working reserve"):
        rtx._allocate_output_tensor((1, 8, 8, 3), torch.float32, torch.device("cpu"),
                                    compute_device=torch.device("cuda:1"), working_reserve_bytes=1000)
    assert requests == [(1000, torch.device("cuda:1"))]


def test_separate_gpu_output_and_compute_budgets(monkeypatch):
    checked = []
    monkeypatch.setattr(rtx, "_can_fit_in_vram", lambda size, dev: checked.append((size, dev)) or True)
    original_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: original_empty(*a, **{k: v for k, v in kw.items() if k != "device"}))
    rtx._allocate_output_tensor((1, 8, 8, 3), torch.float32, torch.device("cuda:1"),
                                compute_device=torch.device("cuda:0"), working_reserve_bytes=1000)
    assert checked == [(1000, torch.device("cuda:0")), (768, torch.device("cuda:1"))]


@pytest.mark.parametrize("allow_mmap", [False, True])
def test_ram_short_controlled_abort_or_explicit_mmap(monkeypatch, allow_mmap):
    monkeypatch.setattr(rtx, "available_ram_bytes", lambda: 512 * 1024 ** 2)
    marker = object()
    monkeypatch.setattr(rtx, "allocate_cpu_output", lambda *a, **kw: (marker, "mapped") if kw["force_mmap"] else pytest.fail("not forced"))
    if allow_mmap:
        assert rtx._allocate_output_tensor((1, 8, 8, 3), torch.float32, torch.device("cpu"), allow_mmap=True) == (marker, "mapped")
    else:
        with pytest.raises(RuntimeError, match="enable use_mmap"):
            rtx._allocate_output_tensor((1, 8, 8, 3), torch.float32, torch.device("cpu"))


def test_rtx_ram_budget_accepts_reported_output_without_allocating(monkeypatch):
    gib = 1024 ** 3
    monkeypatch.setattr(rtx, "available_ram_bytes", lambda: int(7.3 * gib))
    shape = (200, 1280, 640, 3)  # 1.83 GiB; never allocate the full batch in this regression test.
    available, reserve, usable = rtx._dasiwa_rtx_ram_budget(shape, torch.float32)
    assert available == int(7.3 * gib)
    assert gib <= reserve < 2 * gib
    assert int(1.83 * gib) <= usable
    marker = object()
    original_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda s, **kw: original_empty(s, **kw) if s == () else marker)
    assert rtx._allocate_output_tensor(shape, torch.float32, torch.device("cpu")) == (marker, None)


@pytest.mark.parametrize("available_gib", [0.5, 7.3, 40])
def test_ram_budget_preserves_bounded_os_and_frame_headroom(monkeypatch, available_gib):
    gib = 1024 ** 3
    available = int(available_gib * gib)
    monkeypatch.setattr(rtx, "available_ram_bytes", lambda: available)
    shape = (200, 2160, 3840, 4)
    reported, reserve, usable = rtx._dasiwa_rtx_ram_budget(shape, torch.float32)
    staging = 2 * 2160 * 3840 * 4 * 4
    assert reported == available
    assert reserve == max(gib, min(2 * gib, available // 10)) + staging
    assert usable == max(0, available - reserve)


def test_no_routine_cleanup_and_explicit_selected_device(monkeypatch):
    called = []
    monkeypatch.setattr(rtx, "_cleanup_cuda", lambda soft, dev: called.append((soft, dev)))
    monkeypatch.setattr(rtx, "can_allocate_in_ram", lambda size: True)
    for explicit in (False, True):
        rtx._allocate_output_tensor((1, 8, 8, 3), torch.float32, torch.device("cpu"),
                                    soft_empty_cache=explicit, compute_device=torch.device("cuda:1"))
    assert called == [(True, torch.device("cuda:1"))]


def test_reserve_scales_for_source_padding_and_effect_residency():
    reserve = rtx._dasiwa_rtx_working_reserve
    base = reserve(64, 64, 128, 128, "Center Crop (Fill)", 1)
    assert reserve(512, 512, 128, 128, "Center Crop (Fill)", 1) > base
    assert reserve(64, 64, 128, 128, "Center Crop (Fill)", 3) > base
    assert reserve(64, 64, 128, 128, "Center Crop (Fill)", 1, True) > base
    assert reserve(512, 64, 64, 512, "Letterbox (Fit)", 1) > reserve(512, 64, 64, 512, "Center Crop (Fill)", 1)


@pytest.mark.parametrize("modern", [False, True])
def test_stream_signature_blocking_default_and_ownership(monkeypatch, modern):
    events = []
    stream = SimpleNamespace(cuda_stream=123, synchronize=lambda: events.append("stream-sync"))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: stream)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: events.append("device-sync"))
    buffer = torch.empty(3, 8, 8)
    def perform(frame):
        buffer.copy_(frame)
        events.append("run")
        return SimpleNamespace(image=buffer)
    class Modern:
        def run(self, frame, *, stream_ptr=0, non_blocking=False):
            assert stream_ptr == 123 and non_blocking is False
            return perform(frame)
    class Legacy:
        def run(self, frame):
            return perform(frame)
    effect = Modern() if modern else Legacy()
    source = torch.full_like(buffer, .25)
    owned = rtx._run_vfx_effect(effect, source, torch.device("cuda:0"))
    row = torch.empty(8, 8, 3)
    rtx._run_vfx_effect_into(effect, source, row, torch.device("cuda:0"))
    buffer.fill_(1)
    assert torch.all(owned == .25) and torch.all(row == .25) and torch.all(source == .25)
    assert ("device-sync" not in events) if modern else (events[:3] == ["stream-sync", "run", "device-sync"])


def test_fp16_capacity_checked_before_temporaries(monkeypatch):
    images = torch.full((1, 8, 8, 3), .25)
    monkeypatch.setattr(rtx, "can_allocate_in_ram", lambda size: False)
    monkeypatch.setattr(torch.Tensor, "to", lambda *a, **kw: pytest.fail("temporary before budget"))
    assert rtx._lossless_fp16_image_batch(images, 16) is images


def test_fp16_bounded_spatial_checks_noncontiguous_and_last_pixel(monkeypatch):
    images = torch.full((2, 64, 4096, 4), .25)[..., :3]
    seen = []
    original_to = torch.Tensor.to
    def track(self, *a, **kw):
        if a and a[0] == torch.float16 and self.ndim == 3:
            seen.append(self.numel())
        return original_to(self, *a, **kw)
    monkeypatch.setattr(torch.Tensor, "to", track)
    monkeypatch.setattr(rtx, "can_allocate_in_ram", lambda size: True)
    exact = rtx._lossless_fp16_image_batch(images, 1024)
    assert exact.dtype == torch.float16 and torch.equal(exact.float(), images)
    assert max(seen) * 8 <= 1024 * 1024
    images[-1, -1, -1, -1] = .1
    assert rtx._lossless_fp16_image_batch(images, 1024) is images


def test_stages_single_source_frame_and_terminal_avoids_clone(cpu_execution, monkeypatch):
    shapes = []
    original_to = torch.Tensor.to
    def track(self, *a, **kw):
        if str(kw.get("device", "")).startswith("cuda"):
            shapes.append(tuple(self.shape))
        return original_to(self, *a, **kw)
    monkeypatch.setattr(torch.Tensor, "to", track)
    monkeypatch.setattr(rtx, "_run_vfx_effect", lambda *a: pytest.fail("terminal clone"))
    images = torch.stack([torch.full((8, 8, 4), .25), torch.full((8, 8, 4), .75)])
    original = images.clone()
    result, = rtx.DaSiWa_RTX_UpscalerRefiner().execute(**options(images, denoise=True, preserve_alpha=True, chunk_frames=16))
    assert torch.equal(result, original) and torch.equal(images, original)
    assert all(len(shape) == 3 for shape in shapes) and len(shapes) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_modern_stream_cuda_no_global_barrier_and_buffer_reuse(monkeypatch):
    class Effect:
        def __init__(self):
            self.buffer = torch.empty(3, 8, 8, device="cuda")
        def run(self, frame, *, stream_ptr=0, non_blocking=False):
            assert stream_ptr == torch.cuda.current_stream().cuda_stream
            assert not non_blocking
            self.buffer.copy_(frame)
            return SimpleNamespace(image=self.buffer)
    effect = Effect()
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: pytest.fail("global barrier"))
    with torch.cuda.stream(torch.cuda.Stream()):
        source = torch.full((3, 8, 8), .25, device="cuda")
        owned = rtx._run_vfx_effect(effect, source, torch.device("cuda:0"))
        row = torch.empty(8, 8, 3, device="cuda")
        rtx._run_vfx_effect_into(effect, source, row, torch.device("cuda:0"))
        effect.run(torch.ones_like(source), stream_ptr=torch.cuda.current_stream().cuda_stream)
        assert torch.all(owned == .25) and torch.all(row == .25)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_packed_helpers_match_int64_reference_exactly_cuda():
    rgb = torch.rand(3, 37, 53, device="cuda") * 1.2 - .1
    values = (rgb.clamp(0, 1) * 1023).round().to(torch.int64)
    expected = (values[0] | (values[1] << 10) | (values[2] << 20) | (3 << 30)).to(torch.uint32)
    packed = rtx._dasiwa_rtx_pack_rgb10a2(rgb)
    assert torch.equal(packed, expected)
    words = packed.to(torch.int64)
    expected_rgb = torch.stack([(words >> shift) & 1023 for shift in (0, 10, 20)]).float() / 1023
    assert torch.equal(rtx._dasiwa_rtx_unpack_rgb10a2(packed), expected_rgb)


@pytest.mark.parametrize("native_version", [True, False])
def test_known_native_stream_compatibility_handoff(monkeypatch, native_version):
    events = []
    caller = SimpleNamespace(synchronize=lambda: events.append("caller-sync"))
    default = SimpleNamespace(synchronize=lambda: events.append("default-sync"))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda dev: caller)
    monkeypatch.setattr(torch.cuda, "default_stream", lambda dev: default)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    class Native:
        __module__ = "nvvfx.effects.video_super_res"
        def run(self, frame, *, stream_ptr=0):
            pass
    with rtx._dasiwa_rtx_execution_stream((Native, Levels),
            {"version": "0.2.0.0" if native_version else "future"}, torch.device("cuda:0")):
        assert events == (["caller-sync"] if native_version else [])
    assert events == (["caller-sync", "default-sync"] if native_version else [])
