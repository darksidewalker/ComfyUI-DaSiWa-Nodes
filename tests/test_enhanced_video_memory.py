"""Bounded encoder staging and source-storage lifetime regressions."""
import importlib.util
import sys
import types
from pathlib import Path

import av
import numpy as np
import pytest
import torch


@pytest.fixture
def video(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(PromptServer=None))
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(
        get_output_directory=lambda: str(tmp_path),
        get_temp_directory=lambda: str(tmp_path),
        get_save_image_path=lambda *args: (str(tmp_path), "test", 1, "", "test"),
    ))
    path = Path(__file__).resolve().parents[1] / "nodes" / "nodes_enhanced_video_combine.py"
    spec = importlib.util.spec_from_file_location("nodes.video_memory_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("count", [1, 2, 3, 5])
@pytest.mark.parametrize("bit_depth", [8, 10])
def test_pingpong_frame_iteration_without_batch_copy(video, monkeypatch, count, bit_depth):
    images = torch.arange(count, dtype=torch.float32)[:, None, None, None].expand(count, 4, 4, 3) / max(1, count - 1)
    original = images.clone()
    def no_batch_copy(*args, **kwargs):
        raise AssertionError("encoding must not materialize a pingpong batch")
    monkeypatch.setattr(video, "_pingpong_frames", no_batch_copy)
    arrays = list(video._frame_arrays(images, bit_depth, True))
    indices = list(range(count)) + list(range(count - 2, 0, -1))
    assert len(arrays) == video._encoded_frame_count(images, True)
    for array, index in zip(arrays, indices):
        expected = round(float(images[index, 0, 0, 0]) * (1023 if bit_depth == 10 else 255))
        if bit_depth == 10:
            expected *= 64
        assert np.all(array == expected)
    assert torch.equal(images, original)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_bit_depth_samples_before_cpu_cast(video, monkeypatch, noncontiguous, device):
    images = torch.zeros((12, 128, 128, 3), dtype=torch.float16, device=device)
    if noncontiguous:
        images = images.transpose(1, 2)
    sizes = []
    original_to = torch.Tensor.to
    def record_to(self, *args, **kwargs):
        sizes.append(self.numel())
        return original_to(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", record_to)
    assert video.detect_bit_depth(images) == 8
    assert sizes and max(sizes) <= 250_000


@pytest.mark.parametrize("depth", [8, 10])
def test_bit_depth_quantization(video, depth):
    values = torch.arange(1, 200, dtype=torch.float32) / (255 if depth == 8 else 1023)
    assert video.detect_bit_depth(values.reshape(1, 1, -1, 1)) == depth


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("pingpong", [False, True])
def test_disabled_pass_frames_owns_no_source_storage(video, monkeypatch, pingpong, device):
    images = torch.zeros((5, 16, 16, 3), device=device)
    monkeypatch.setattr(video, "_encode_with_available_encoder", lambda *args, **kwargs: "test")
    result = video.DaSiWa_EnhancedVideoCombine().combine(
        images, 24, "H.264", "MP4", "8-bit", 20, pingpong,
        False, "test", True, False,
    )["result"][0]
    assert tuple(result.shape) == (0, 16, 16, 3)
    assert result.dtype == images.dtype and result.device == images.device
    assert result.untyped_storage().nbytes() == 0
    assert result._base is None


@pytest.mark.parametrize("pingpong", [False, True])
def test_enabled_pass_frames_preserves_sequence(video, monkeypatch, pingpong):
    images = torch.rand((5, 16, 16, 3))
    monkeypatch.setattr(video, "_encode_with_available_encoder", lambda *args, **kwargs: "test")
    result = video.DaSiWa_EnhancedVideoCombine().combine(
        images, 24, "H.264", "MP4", "8-bit", 20, pingpong,
        False, "test", True, True,
    )["result"][0]
    expected = torch.cat((images, images[1:-1].flip(0))) if pingpong else images
    assert torch.equal(result, expected)


def test_real_pyav_pingpong_encode(video, tmp_path):
    images = torch.linspace(0, 1, 5)[:, None, None, None].expand(5, 32, 32, 3)
    path = tmp_path / "pingpong.mp4"
    video.helper_pyav_video.encode_attempt(
        str(path), "MP4", "libx264", 32, 32, 24, 8, 20,
        lambda: video._frame_arrays(images, 8, True),
    )
    with av.open(str(path)) as container:
        decoded = list(container.decode(video=0))
    assert len(decoded) == 8
    means = [frame.to_ndarray(format="rgb24").mean() for frame in decoded]
    assert np.allclose(means, [0, 64, 128, 191, 255, 191, 128, 64], atol=5)



def _rgba_clip():
    images = torch.ones((2, 32, 32, 4))
    images[1, ..., :3] = 0.5  # distinct frames, or libwebp merges them
    images[:, :, :16, 3] = 0  # left half transparent
    return images


def _assert_left_transparent(alphas):
    assert len(alphas) == 2
    for alpha in alphas:
        assert alpha[:, :12].max() < 20 and alpha[:, 20:].min() > 235


def test_animated_webp_keeps_rgba_alpha_and_loops(video, tmp_path):
    from PIL import Image, ImageSequence
    video.DaSiWa_EnhancedVideoCombine().combine(
        _rgba_clip(), 12, "Auto", "Animated WebP", "Auto", 90, False, False, "test", True, False,
    )
    [path] = tmp_path.glob("*.webp")
    with Image.open(path) as webp:
        assert webp.info["loop"] == 0
        _assert_left_transparent([np.array(frame.convert("RGBA"))[..., 3] for frame in ImageSequence.Iterator(webp)])


def test_vp9_webm_keeps_rgba_alpha(video, tmp_path):
    video.DaSiWa_EnhancedVideoCombine().combine(
        _rgba_clip(), 12, "VP9", "WebM", "10-bit", 20, False, False, "test", True, False,
    )
    [path] = tmp_path.glob("*.webm")
    decoder = av.CodecContext.create("libvpx-vp9", "r")  # FFmpeg's native vp9 decoder ignores alpha
    with av.open(str(path)) as container:
        frames = [frame for packet in container.demux(video=0) for frame in decoder.decode(packet)]
    _assert_left_transparent([frame.to_ndarray(format="rgba")[..., 3] for frame in frames])


def test_frame_exports_embed_workflow_metadata_like_save_image(video, tmp_path):
    import json
    from PIL import Image
    images = torch.rand(3, 4, 4, 3)
    metadata = video._metadata_file({"1": {"class_type": "X"}}, {"workflow": {"nodes": []}})
    paths = video._save_frame_exports(images, str(tmp_path / "clip.mp4"), True, True, metadata=metadata)
    assert len(paths) == 2
    for path in paths:
        text = Image.open(path).text
        assert json.loads(text["prompt"]) == {"1": {"class_type": "X"}}
        assert json.loads(text["workflow"]) == {"nodes": []}
    # save_metadata off passes None: plain PNGs as before.
    plain = video._save_frame_exports(images, str(tmp_path / "plain.mp4"), True, False, metadata=None)
    assert "workflow" not in Image.open(plain[0]).text
