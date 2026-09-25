import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
import pytest

comfy_root = os.environ.get("COMFYUI_ROOT")
if not comfy_root:
    raise pytest.UsageError("Set COMFYUI_ROOT to the ComfyUI checkout")
sys.path.append(comfy_root)
import comfy.cli_args
comfy.cli_args.args.cpu = True
package = ModuleType("dasiwa_continuity_test")
package.__path__ = [str(Path(__file__).resolve().parents[2] / "nodes" / "h3_continuity")]
sys.modules[package.__name__] = package
repo_package = ModuleType("dasiwa_nodes_test")
repo_package.__path__ = [str(Path(__file__).resolve().parents[2] / "nodes")]
sys.modules[repo_package.__name__] = repo_package

@pytest.fixture
def modules():
    return importlib.import_module("dasiwa_continuity_test.core"), None, None

@pytest.fixture
def store(tmp_path, modules):
    core, _, _ = modules
    return core.ClipStore(tmp_path / "continuity")
