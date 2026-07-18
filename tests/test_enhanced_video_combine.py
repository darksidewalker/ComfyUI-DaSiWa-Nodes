import importlib.util
import os
import sys
import types
from pathlib import Path

import torch
import pytest


class _FolderPaths:
    @staticmethod
    def get_output_directory():
        return "/tmp"

    @staticmethod
    def get_temp_directory():
        return "/tmp"

    @staticmethod
    def get_save_image_path(prefix, output_dir, width, height):
        return output_dir, prefix.replace("/", "_"), 1, "", prefix


sys.modules.setdefault("folder_paths", _FolderPaths())
MODULE_PATH = Path(__file__).parents[1] / "nodes" / "enhanced_video_combine.py"
spec = importlib.util.spec_from_file_location("enhanced_video_combine", MODULE_PATH)
assert spec is not None and spec.loader is not None
enhanced_video_combine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enhanced_video_combine)


def test_node_schema_and_registration():
    controls = enhanced_video_combine.DaSiWa_EnhancedVideoCombine.INPUT_TYPES()["required"]
    package_source = (Path(__file__).parents[1] / "__init__.py").read_text(encoding="utf-8")
    preview_source = (Path(__file__).parents[1] / "js" / "enhanced_video_combine_preview.js").read_text(encoding="utf-8")

    assert {"images", "bit_depth", "pass_frames", "crop_to_audio", "audio_codec", "audio_bitrate", "filename_prefix", "quality", "pingpong", "save_metadata"} <= controls.keys()
    assert enhanced_video_combine.DaSiWa_EnhancedVideoCombine.INPUT_TYPES()["optional"]["audio"][0] == "AUDIO"
    assert controls["codec"][0] == ["AV1", "VP9", "H.265 (HEVC)", "H.264"]
    assert controls["container"][0] == ["Auto", "WebM", "MKV", "MP4"]
    assert controls["quality"][1]["default"] == 20
    assert controls["pingpong"][1]["default"] is False
    assert controls["pass_frames"][1]["default"] is False
    assert controls["filename_prefix"][1]["default"] == "video_%date:hhmmss%"
    assert enhanced_video_combine._output_filename("video_130405", 1, ".mp4", False) == "video_130405_00001.mp4"
    assert enhanced_video_combine._output_filename("video_130405", 1, ".mp4", True) == "video_130405_00001-audio.mp4"

    assert controls["audio_codec"][0] == ["Auto", "AAC", "Opus", "MP3"]
    assert controls["audio_bitrate"][1]["default"] == "192k"
    assert "DaSiWa_EnhancedVideoCombine" in package_source
    assert 'name: "DaSiWa.EnhancedVideoCombinePreview"' in preview_source
    assert "this.addDOMWidget" in preview_source
    assert "message?.gifs?.[0] ?? message?.videos?.[0]" in preview_source
    assert 'saveFirstFrame.type = "checkbox"' in preview_source
    assert 'saveLastFrame.type = "checkbox"' in preview_source
    assert "this.setSize" not in preview_source
    assert "video.fps" in preview_source
    assert '"Video preview"' not in preview_source
    assert "preview.controls = true" not in preview_source
    assert "previewWidget.aspectRatio = preview.videoWidth / preview.videoHeight" in preview_source
    assert "fitPreviewHeight(previewNode)" in preview_source
    assert "actions.append(saveFirstFrameLabel, saveLastFrameLabel)" in preview_source
    assert 'preview.addEventListener("click", togglePlayback)' in preview_source
    assert "if (saveFirstFrame.checked) saveFrame(preview, false);" in preview_source
    assert "if (saveLastFrame.checked) saveFrame(preview, true);" in preview_source
    assert 'preview.dataset.filename = video.filename' in preview_source
    assert 'link.download = `${filename}-${lastFrame ? "last" : "first"}-frame.png`;' in preview_source



def test_auto_bit_depth_distinguishes_8_and_10_bit_quantization():
    eight_bit = torch.tensor([0, 64, 127, 255], dtype=torch.float32).reshape(1, 2, 2, 1) / 255
    ten_bit = torch.tensor([0, 256, 511, 1023], dtype=torch.float32).reshape(1, 2, 2, 1) / 1023

    assert enhanced_video_combine.detect_bit_depth(eight_bit) == 8
    assert enhanced_video_combine.detect_bit_depth(ten_bit) == 10


def test_validate_inputs_accepts_comfyui_positional_signature():
    node = enhanced_video_combine.DaSiWa_EnhancedVideoCombine()
    assert node.VALIDATE_INPUTS(images=object()) is True
    assert node.validate_inputs("images", "IMAGE", object(), object()) is True
    assert enhanced_video_combine.DaSiWa_EnhancedVideoCombine.__dict__["validate_inputs"](
        node, "images", "IMAGE", object(), object()
    ) is True


def test_10_bit_frame_data_uses_rgb48le_values():
    images = torch.tensor([[[[0.0, 0.5, 1.0]]]], dtype=torch.float32)
    payload = enhanced_video_combine._frame_bytes(images, 10)

    assert len(payload) == 6
    assert torch.frombuffer(bytearray(payload), dtype=torch.uint16).tolist() == [0, 32768, 65472]


def test_encoder_priority_prefers_nvenc_then_other_hardware_then_software():
    assert enhanced_video_combine._ENCODER_NAMES["H.264"][:2] == ("h264_nvenc", "h264_qsv")
    assert enhanced_video_combine._ENCODER_NAMES["H.265 (HEVC)"][:2] == ("hevc_nvenc", "hevc_qsv")
    assert enhanced_video_combine._ENCODER_NAMES["AV1"][:2] == ("av1_nvenc", "av1_qsv")
    assert enhanced_video_combine._ENCODER_NAMES["VP9"] == ("vp9_qsv", "vp9_vaapi", "libvpx-vp9")


def test_auto_container_prioritizes_webm_then_mkv_then_mp4_for_av1_and_vp9():
    assert enhanced_video_combine._container_candidates("AV1", "Auto") == ("WebM", "MKV", "MP4")
    assert enhanced_video_combine._container_candidates("VP9", "Auto") == ("WebM", "MKV", "MP4")
    assert enhanced_video_combine._container_candidates("H.264", "Auto") == ("MP4", "MKV")
    assert enhanced_video_combine._container_candidates("H.265 (HEVC)", "MKV") == ("MKV",)


def test_pingpong_appends_reverse_interior_frames():
    images = torch.arange(5, dtype=torch.float32).reshape(5, 1, 1, 1)

    assert enhanced_video_combine._pingpong_frames(images, False).flatten().tolist() == [0, 1, 2, 3, 4]
    assert enhanced_video_combine._pingpong_frames(images, True).flatten().tolist() == [0, 1, 2, 3, 4, 3, 2, 1]


def test_filename_prefix_expands_comfyui_date_format(monkeypatch):
    real_datetime = enhanced_video_combine.datetime.datetime

    class FixedDatetime:
        @classmethod
        def now(cls):
            return real_datetime(2026, 7, 18, 13, 4, 5)

    monkeypatch.setattr(enhanced_video_combine.datetime, "datetime", FixedDatetime)

    assert enhanced_video_combine._format_filename_prefix(
        "video/%date:yyyy-MM-dd%/%date:hhmmss%"
    ) == "video/2026-07-18/130405"


def test_audio_file_converts_comfyui_audio_to_interleaved_float32():
    audio_path, duration = enhanced_video_combine._audio_file({
        "waveform": torch.tensor([[[0.0, 0.5], [-0.5, 1.0]]]),
        "sample_rate": 2,
    })
    try:
        assert audio_path[1:] == (2, 2)
        assert duration == 1.0
        assert torch.frombuffer(bytearray(Path(audio_path[0]).read_bytes()), dtype=torch.float32).tolist() == [0.0, -0.5, 0.5, 1.0]
    finally:
        os.unlink(audio_path[0])


def test_audio_encode_maps_audio_and_crops_video(monkeypatch):
    captured = []

    class Result:
        returncode = 0
        stderr = b""

    monkeypatch.setattr(enhanced_video_combine, "_available_encoders", lambda _ffmpeg: {"libx264"})
    monkeypatch.setattr(enhanced_video_combine.subprocess, "run", lambda command, **kwargs: captured.append(command) or Result())

    assert enhanced_video_combine._encode_with_available_encoder(
        "ffmpeg", "H.264", 8, 2, 2, 24, b"frames", "output.mp4", "MP4", 20, 20,
        None, ("audio.f32le", 48000, 2), 1.25, True, "MP3", "128k",
    ) == "libx264"
    assert "-map" in captured[0]
    assert "1:a:0" in captured[0]
    assert ["-c:a", "libmp3lame", "-b:a", "128k"] == captured[0][captured[0].index("-c:a"):captured[0].index("-c:a") + 4]
    assert ["-t", "1.250000000"] == captured[0][captured[0].index("-t"):captured[0].index("-t") + 2]


def test_audio_fallbacks_are_container_compatible():
    assert enhanced_video_combine._audio_encoder_candidates("AAC", "WebM") == ("aac", "libopus")
    assert enhanced_video_combine._audio_encoder_candidates("MP3", "MP4") == ("libmp3lame", "aac")
    assert enhanced_video_combine._audio_encoder_candidates("Auto", "MKV") == ("aac", "libopus", "libmp3lame", "pcm_s16le")


def test_audio_encode_falls_back_when_requested_encoder_fails(monkeypatch):
    captured = []

    class FailedResult:
        returncode = 1
        stderr = b"requested audio encoder is unavailable"

    class SuccessResult:
        returncode = 0
        stderr = b""

    monkeypatch.setattr(enhanced_video_combine, "_available_encoders", lambda _ffmpeg: {"libx264"})
    monkeypatch.setattr(
        enhanced_video_combine.subprocess,
        "run",
        lambda command, **kwargs: captured.append(command) or (FailedResult() if len(captured) == 1 else SuccessResult()),
    )

    assert enhanced_video_combine._encode_with_available_encoder(
        "ffmpeg", "H.264", 8, 2, 2, 24, b"frames", "output.mp4", "MP4", 20, 20,
        None, ("audio.f32le", 48000, 2), 1.25, False, "MP3", "128k",
    ) == "libx264"
    assert "libmp3lame" in captured[0]
    assert "aac" in captured[1]


def test_encoder_listing_extracts_encoder_names(monkeypatch):
    class Result:
        returncode = 0
        stdout = " V....D h264_nvenc NVIDIA NVENC h264 encoder (codec h264)\n V....D libx264 H.264 encoder (codec h264)\n"

    monkeypatch.setattr(enhanced_video_combine.subprocess, "run", lambda *args, **kwargs: Result())

    assert enhanced_video_combine._available_encoders("ffmpeg") == {"h264_nvenc", "libx264"}


def test_missing_ffmpeg_reports_required_mp4_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(enhanced_video_combine, "find_ffmpeg", lambda: None)
    images = torch.rand((2, 4, 6, 3), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="H.264/MP4 fallback"):
        enhanced_video_combine.DaSiWa_EnhancedVideoCombine().combine(
            images, 24.0, "H.264", "Auto", "Auto", 10, False, True,
            "video", True, False,
        )
