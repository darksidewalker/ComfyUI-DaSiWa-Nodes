"""H3 de-rope: hold planning pinned to MAINodes Motion Lab output (db1f62d), exact recovery, audio clock."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nodes.nodes_minimax_h3_derope import (
    DaSiWaH3DeropePrepare, DaSiWaH3DeropeRecover, _runs, final_holds, remap_conditioning, smear_audio,
)

# 73 frames = 22 tokens; a burst at tokens 8-10 and 13. Expected runs are H3JerkOracle + H3TimeSmear's output.
PROFILE = np.array([1.0 + 0.01 * i for i in range(22)])
PROFILE[[8, 9, 10, 13]] = [5, 6, 5, 4]


def test_holds_match_motion_lab():
    assert [tuple(r) for r in _runs(final_holds(PROFILE, 73, "balanced"))] == [(1, 18), (2, 4), (3, 4), (4, 47)]
    assert [tuple(r) for r in _runs(final_holds(PROFILE, 73, "economy"))] == [
        (1, 22), (2, 4), (3, 21), (2, 4), (1, 21), (2, 1)]
    for preset in ("balanced", "max quality", "economy"):
        assert (sum(final_holds(PROFILE, 73, preset)) - 5) % 17 == 0


def test_recover_inverts_the_smear():
    holds = final_holds(PROFILE, 73, "balanced")
    frames = torch.arange(73).float().view(73, 1, 1, 1).expand(73, 2, 2, 3)
    smeared = frames[torch.tensor([i for i, h in enumerate(holds) for _ in range(h)])]
    (out,) = DaSiWaH3DeropeRecover().recover(smeared, json.dumps({"holds": holds, "world_len": 73}))
    assert torch.equal(out, frames)
    with pytest.raises(ValueError):
        DaSiWaH3DeropeRecover().recover(smeared[:-1], json.dumps({"holds": holds}))


def test_audio_lands_on_the_dilated_clock():
    pytest.importorskip("torchaudio")
    holds = final_holds(PROFILE, 73, "balanced")
    audio = {"waveform": torch.randn(1, 2, 48000 * 73 // 24), "sample_rate": 48000}
    out = smear_audio(audio, holds, 24)
    assert out["waveform"].shape[-1] == sum(round(h * c * 2000) for h, c in _runs(holds))


class _VAE:
    def encode(self, pixels):  # H3 video VAE shape: 24 channels, 17k+5 frames -> 5k+2 tokens, /16 spatial
        n = pixels.shape[0]
        return torch.zeros(1, 24, (n - 5) // 17 * 5 + 2, pixels.shape[1] // 16, pixels.shape[2] // 16)


def test_prepare_builds_the_dilated_av_latent():
    root = os.environ.get("COMFY_ROOT")
    if root:
        sys.path.insert(0, root)
    pytest.importorskip("comfy_extras.nodes_minimax_h3")
    video = torch.randn(1, 24, 22, 2, 2)
    video[:, :, 9:12] *= 8  # a burst the oracle must find
    latent, hold_map, report, _ = DaSiWaH3DeropePrepare().prepare(
        torch.zeros(73, 32, 32, 3), {"samples": video}, _VAE(), "balanced", 24.0)
    holds = json.loads(hold_map)["holds"]
    v, a = latent["samples"].tensors
    assert len(holds) == 73 and v.shape[2] == (sum(holds) - 5) // 17 * 5 + 2 and a.shape[1] == 32
    assert "noise_mask" not in latent and "frames held" in report
    with pytest.raises(ValueError):
        DaSiWaH3DeropePrepare().prepare(torch.zeros(70, 32, 32, 3), {"samples": video}, _VAE(), "balanced", 24.0)


def test_keyframes_follow_the_slowed_clock():
    holds = [1, 1, 3, 3, 1]  # world 5 frames -> starts 0,1,2,5,8 (9 frames)
    one = torch.zeros(1, 24, 1, 2, 2)
    cond = [[torch.zeros(1), {"minimax_keyframes": [
        {"resolved_frame_index": 0, "latent": one}, {"resolved_frame_index": 4, "latent": one},
        {"resolved_frame_index": 3, "latent": torch.zeros(1, 24, 3, 2, 2)},
        {"resolved_frame_index": 1, "audio_latent": torch.zeros(1, 32, 2, 4)}], "minimax_refs": ["r"]}]]
    out, dropped = remap_conditioning(cond, holds)
    assert [k["resolved_frame_index"] for k in out[0][1]["minimax_keyframes"]] == [0, 8]
    assert dropped == 2 and out[0][1]["minimax_refs"] == ["r"]
    assert cond[0][1]["minimax_keyframes"][1]["resolved_frame_index"] == 4  # input untouched
    out, dropped = remap_conditioning(cond, holds, keep=False)
    assert out[0][1]["minimax_keyframes"] == [] and dropped == 4
