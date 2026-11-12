"""beta57 scheduler (nodes/extra_samplers.py).

Run: PYTHONPATH=<ComfyUI> <venv>/bin/python -m pytest tests/test_extra_samplers.py
"""
import importlib.util
from pathlib import Path

import torch

import comfy.samplers as samplers

spec = importlib.util.spec_from_file_location("extra_samplers", Path(__file__).resolve().parents[1] / "nodes" / "extra_samplers.py")
extra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extra)
extra.register()
extra.register()  # idempotent


class _Sampling:
    sigmas = torch.linspace(0.03, 14.6, 1000)


def test_beta57_is_registered_once_in_the_ksampler_list():
    assert samplers.KSampler.SCHEDULERS.count("beta57") == 1


def test_beta57_is_core_beta_with_alpha_05_beta_07():
    got = samplers.calculate_sigmas(_Sampling(), "beta57", 10)
    want = samplers.beta_scheduler(_Sampling(), 10, alpha=0.5, beta=0.7)
    assert torch.equal(got, want)
    assert not torch.equal(got, samplers.calculate_sigmas(_Sampling(), "beta", 10))
