"""CPU contract check for V26's Append/upscale ordering; no neural upscaler."""
import argparse
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--comfy", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.comfy.resolve()))
import comfy.cli_args
comfy.cli_args.args.cpu = True
import torch
from comfy.nested_tensor import NestedTensor
package = ModuleType("h3_order_test")
package.__path__ = [str(args.repo.resolve())]
sys.modules[package.__name__] = package
core = importlib.import_module("h3_order_test.nodes.h3_continuity.core")

frames = 124
previous = {"samples": NestedTensor((
    torch.zeros(1, 24, (frames - 5) // 17 * 5 + 2, 4, 4),
    torch.zeros(1, 32, 2, round(frames * 40 / 24))))}
settings = core.parse_settings({"version": 3, "source_id": "parent"}, 5)
_, target, layout = core.prepare_continuation(previous, {"mode_family": "ref2va", "fps": 24},
                                            {"mode": "REF2VA", "width": 64, "height": 64}, settings)
video, audio = target["samples"].tensors
resized = {"samples": NestedTensor((video.repeat_interleave(2, -1).repeat_interleave(2, -2), audio))}
try:
    core.append_tail(previous, resized, layout)
except ValueError as exc:
    assert "spatial dimensions" in str(exc)
    rejected = str(exc)
else:
    raise AssertionError("A resized sample should not be appended to the unscaled parent")
combined = core.append_tail(previous, target, layout)
cv, ca, result_frames = core.validate_h3_av_latent(combined, name="combined")
assert result_frames == 243 and cv.shape[-2:] == (4, 4)
assert torch.equal(cv[:, :, :previous["samples"].tensors[0].shape[2]], previous["samples"].tensors[0])
assert ca.shape[-1] == round(result_frames * 40 / 24)
print(json.dumps({"upscale_before_append": rejected, "append_before_upscale": "passed",
                  "source_frames": frames, "added_frames": settings["extension_frames"],
                  "combined_frames": result_frames, "checkpoint_canvas": [64, 64],
                  "note": "Synthetic latent dimensions only; no GPU or third-party upscaler inference."}, indent=2))
