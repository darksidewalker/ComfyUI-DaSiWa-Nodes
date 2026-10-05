"""Run pack LLM/Forge tests with Core available, without its nodes.py collision."""
import argparse
import sys
import types
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui-root", type=Path, required=True)
    args, pytest_args = parser.parse_known_args()
    core = args.comfyui_root.resolve()
    if not (core / "folder_paths.py").is_file():
        parser.error("--comfyui-root must contain ComfyUI's folder_paths.py")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(core))
    sys.path.insert(0, str(root))
    # Historical pack tests import nodes.*; production keeps Core's nodes.py.
    package = types.ModuleType("nodes")
    package.__path__ = [str(root / "nodes")]
    sys.modules["nodes"] = package
    import pytest

    if not pytest_args:
        pytest_args = [str(path) for pattern in ("test_llm*.py", "test_h3_forge*.py")
                       for path in sorted((root / "tests").glob(pattern))]
        pytest_args.append("-q")
    return pytest.main(pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
