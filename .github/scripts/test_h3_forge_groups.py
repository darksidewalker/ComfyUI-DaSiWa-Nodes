"""REF2VA Forge explicit same-subject reference groups."""
import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# Forge and its siblings use relative imports, so load it through the pack's "nodes" package
# instead of a bare file spec; a spec-less module has no package and breaks "from .x import".
package = sys.modules.get("nodes") or types.ModuleType("nodes")
package.__path__ = [str(ROOT / "nodes")]
sys.modules["nodes"] = package
forge = importlib.import_module("nodes.h3_forge")


def image(role="subject", group=None, **extra):
    return {"kind": "image", "role": role, "subject_group": group, **extra}


def test_explicit_group_combines_only_selected_subjects():
    refs = [image(group="A", keep="face"), image(), image(group="A", drop="hat"), image(group="B"), image(group="B")]
    lines, pics = forge.format_references(refs, "REF2VA")
    assert len(pics) == 5
    assert len(lines) == 3
    assert "<Picture 1>, <Picture 3>" in lines[0]
    assert "keep (<Picture 1>): face" in lines[0]
    assert "drop (<Picture 3>): hat" in lines[0]
    assert "<Picture 2>" in lines[1]
    assert "<Picture 4>, <Picture 5>" in lines[2]


def test_no_explicit_group_never_guesses_from_brief_or_old_toggle():
    refs = [image(), image()]
    message = forge.build_user_message(forge.load_bundle(), "Rin in picture 1 and 2", "REF2VA", 5, 3, "faithful", refs, False)
    assert "<Picture 1> · subject" in message
    assert "<Picture 2> · subject" in message
    assert "ONE subject shown" not in message


@pytest.mark.parametrize("refs, mode", [
    ([image(group="A"), image("keyframe", "A")], "REF2VA"),
    ([image(group="A"), image("style", "A")], "REF2VA"),
    ([image(group="A"), image(group="A")], "I2VA"),
])
def test_group_never_absorbs_keyframe_style_or_base_mode(refs, mode):
    lines, _ = forge.format_references(refs, mode)
    assert len(lines) == 2


def test_single_member_group_stays_individual():
    lines, _ = forge.format_references([image(group="A"), image()], "REF2VA")
    assert len(lines) == 2


def test_warning_when_model_splits_declared_group():
    refs = [image(group="A"), image(group="A")]
    split = "<Subject 1> Rin from <Picture 1>.\n<Subject 2> Rin from <Picture 2>."
    together = "<Subject 1> Rin from <Picture 1> and <Picture 2>."
    assert forge.group_warnings(split, refs)
    assert forge.group_warnings(together, refs) == []


def test_warning_is_cautious_when_model_omits_provenance():
    assert forge.group_warnings("<Subject 1> Rin, same in both references.", [image(group="A"), image(group="A")]) == []
