"""Capture wiring, including the native 0.4.74 audio-lock path; no models needed."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("h3_capture_validation", ROOT / "nodes/h3_continuity/validation.py")
assert spec is not None and spec.loader is not None
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def graph():
    return {
        "guide": {"class_type": "MiniMaxH3DirectorGuide", "inputs": {}},
        "split": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["guide", 1]}},
        "lock": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["external-audio", 0]}},
        "join": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["split", 0], "audio_latent": ["lock", 0]}},
        "sampler": {"class_type": "SamplerCustomAdvanced", "inputs": {"latent_image": ["join", 0]}},
        "append": {"class_type": "DaSiWaH3ContinuityAppend", "inputs": {"sampled": ["sampler", 0], "context": ["guide", 2]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["append", 0]}},
        "export": {"class_type": "DaSiWa_EnhancedVideoCombine", "inputs": {"images": ["decode", 0], "container": "MP4", "pingpong": False, "crop_to_audio": False}},
        "publish": {"class_type": "DaSiWaH3ContinuityPublish", "inputs": {"filename": ["export", 1], "ticket": ["append", 1]}},
    }


class CaptureValidationTests(unittest.TestCase):
    def test_direct_and_locked_audio_paths_keep_the_guide_video(self):
        p = graph()
        validation.validate_capture_graph(p, "guide")
        p["sampler"]["inputs"]["latent_image"] = ["guide", 1]
        validation.validate_capture_graph(p, "guide")
        p["lock"]["inputs"]["samples"] = ["guide", 1]
        p["sampler"]["inputs"]["latent_image"] = ["lock", 0]
        validation.validate_capture_graph(p, "guide")

    def test_audio_only_or_mask_only_guide_link_does_not_authorize_video(self):
        for video in (["split", 1], ["unrelated-video", 0]):
            with self.subTest(video=video):
                p = graph()
                p["join"]["inputs"].update(video_latent=video, audio_latent=["split", 1])
                p["lock"]["inputs"]["mask"] = ["guide", 1]
                with self.assertRaisesRegex(ValueError, "sampler driven by this Director Guide"):
                    validation.validate_capture_graph(p, "guide")

    def test_cyclic_latent_chain_fails_without_recursion_overflow(self):
        p = graph()
        p["split"]["inputs"]["av_latent"] = ["join", 0]
        with self.assertRaises(ValueError):
            validation.validate_capture_graph(p, "guide")

    def test_export_audio_alone_cannot_satisfy_capture(self):
        p = graph()
        p["export"]["inputs"].update(images=["unrelated-images", 0], audio=["append", 0])
        with self.assertRaisesRegex(ValueError, "images must come from this cumulative latent"):
            validation.validate_capture_graph(p, "guide")


if __name__ == "__main__":
    unittest.main()
