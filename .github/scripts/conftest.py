"""Pin this package's node module before pytest's rootdir (the ComfyUI checkout):
the core nodes.py module wins the sys.path scan, so nodes.* imports are registered
explicitly from this package's file."""
import importlib.util
import sys
import types
from pathlib import Path

root = Path(__file__).resolve().parents[2]
name = "nodes.nodes_wildcard_preset_prompt_builder"

if "nodes" not in sys.modules:
    package = types.ModuleType("nodes")
    package.__path__ = [str(root / "nodes")]
    sys.modules["nodes"] = package

if name not in sys.modules:
    from aiohttp import web

    sys.modules.setdefault("server", types.SimpleNamespace(
        PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(routes=web.RouteTableDef()))))
    spec = importlib.util.spec_from_file_location(name, root / "nodes" / "nodes_wildcard_preset_prompt_builder.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
