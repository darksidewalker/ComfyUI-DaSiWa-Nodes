"""Exercise the real Forge orchestration with only the external chat mocked."""
import sys
import types
from pathlib import Path
from urllib.error import HTTPError

import pytest

ROOT = Path(__file__).resolve().parents[1]
# ComfyUI also has a top-level nodes.py; load this pack's namespace explicitly.
if "nodes" not in sys.modules:
    package = types.ModuleType("nodes")
    package.__path__ = [str(ROOT / "nodes")]
    sys.modules["nodes"] = package
from nodes import h3_forge as forge
from nodes import h3_prompting as hp


def picture(role, path=None):
    ref = {"kind": "image", "role": "subject", "easy_role": role}
    if path:
        ref["path"] = path
    return ref


def draft(descriptions=None):
    segments = {
        "Subject definitions": "<Subject 1>: waves.",
        "Summary": "[reference generation] A short scene.",
        "Detailed description": "Watercolour with thin ink lines.\n[Shot 1] <Subject 1> waves in <Subject 2>.",
        "Soundscape": "Room tone.", "Music": "N/A",
    }
    if descriptions is not None:
        segments["Descriptions"] = descriptions
    return "\n".join(f"===SEGMENT: {label}===\n{body}" for label, body in segments.items())


def run(monkeypatch, tmp_path, *, descriptions=None, sees=True, flag=True,
        references=None, fallback=False, input_available=True, raw=None):
    sent = []

    class Backend:
        base = "https://fixture.invalid"

        def can_see(self, name):
            return sees

        def chat(self, name, system, user, images, *rest):
            sent.append({"system": system, "user": user, "images": list(images)})
            if fallback and len(sent) == 1:
                raise HTTPError(self.base, 400, "Images unsupported", {}, None)
            return raw if raw is not None else draft(descriptions), {}

        def unload(self, name):
            return True

    refs = references if references is not None else [picture("character-1", "a.png"), picture("place", "b.png")]
    for ref in refs:
        if ref.get("path"):
            (tmp_path / ref["path"]).write_bytes(b"fixture")
    monkeypatch.setattr(forge, "_image_b64", lambda path: Path(path).name)
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": Backend()})
    body = {"mode": "REF2VA", "model": "openai:fixture", "brief": "Character 1 waves in the place.",
            "duration": 5, "easy": True, "see_pictures": flag, "references": refs}
    result = forge._generate(body, str(tmp_path) if input_available else None, None, None)
    return result, sent


def test_actual_chat_receives_coherent_vision_contract(monkeypatch, tmp_path):
    result, sent = run(monkeypatch, tmp_path, descriptions="<Subject 1>: Silver hair and a blue coat.\n<Subject 2>: A stone courtyard.")
    system, user = sent[0]["system"], sent[0]["user"]
    assert "You have not seen the pictures" not in system
    assert "Never name a medium yourself" not in system
    assert "6. `Descriptions`" in system.split("## Segments to emit")[-1]
    assert "never describe how anyone or anything looks" not in user
    assert "Silver hair and a blue coat." in result["fields"]["ref"]["retention_analysis"]
    assert "A stone courtyard." in result["fields"]["ref"]["retention_analysis"]
    assert "Watercolour" in result["fields"]["ref"]["detailed_description"]
    assert set(result["fields"]["ref"]) == {key for _, key in hp._REF_FIELDS}
    assert "Descriptions" not in result["simple_prompt"]
    assert not result["warnings"]


@pytest.mark.parametrize("descriptions, missing", [
    (None, ["<Subject 1>", "<Subject 2>"]),
    ("<Subject 1>: Silver hair.", ["<Subject 2>"]),
    ("<Subject 1>: N/A\n<Subject 2>:", ["<Subject 1>", "<Subject 2>"]),
])
def test_incomplete_descriptions_warn_without_losing_base_fields(monkeypatch, tmp_path, descriptions, missing):
    result, _ = run(monkeypatch, tmp_path, descriptions=descriptions)
    warning = " ".join(result["warnings"])
    assert "Descriptions" in warning
    for tag in missing:
        assert tag in warning
    assert len(result["fields"]["ref"]) == 6
    assert result["fields"]["ref"]["summary"]
    if descriptions and "Silver hair" in descriptions:
        assert "Silver hair." in result["fields"]["ref"]["retention_analysis"]


def test_partial_attachments_request_only_visible_targets(monkeypatch, tmp_path):
    refs = [picture("character-1"), picture("place", "b.png"), picture("pose"), picture("last-frame")]
    result, sent = run(monkeypatch, tmp_path, references=refs,
                       descriptions="<Subject 1>: Invented hair.\n<Subject 2>: A stone courtyard.\n<Picture 3>: Invented pose.")
    user = sent[0]["user"]
    assert sent[0]["images"] == ["b.png"]
    assert "- <Subject 1> (a character):" not in user
    assert "- <Subject 2> (the place):" in user
    assert "- <Picture 3> (a pose):" not in user
    assert "- <Picture 4> (the last frame):" not in user
    assert "<Picture 1>, <Picture 3>, <Picture 4> are not visible" in user
    assert "Invented" not in result["simple_prompt"]
    assert not result["warnings"]


@pytest.mark.parametrize("options", [
    {"flag": False}, {"flag": None}, {"sees": False}, {"input_available": False},
    {"references": [picture("character-1"), picture("place")]},
    {"sees": None, "fallback": True},
])
def test_no_images_restore_original_blind_contract(monkeypatch, tmp_path, options):
    result, sent = run(monkeypatch, tmp_path, **options)
    blind = hp.load_bundle()["modes"][hp.EASY_MODE]["system"]
    assert sent[-1]["system"] == blind
    assert sent[-1]["images"] == []
    assert "Descriptions" not in sent[-1]["user"]
    assert "are attached" not in sent[-1]["user"]
    assert "never describe how anyone or anything looks" in sent[-1]["user"]
    assert result["saw_images"] == 0
    assert not result["warnings"]
    assert "As the pictures show" not in result["simple_prompt"]
    if options.get("fallback"):
        assert len(sent) == 2 and sent[0]["images"]
        assert "6. `Descriptions`" in sent[0]["system"]


def test_required_base_segment_still_fails_before_optional_validation(monkeypatch, tmp_path):
    with pytest.raises(forge.ForgeError) as error:
        run(monkeypatch, tmp_path, raw=draft().replace("===SEGMENT: Summary===", "===SEGMENT: Wrong===") )
    assert error.value.code == "missing_segments"
    assert "Summary" in error.value.message


def test_group_picture_supplies_both_visible_characters(monkeypatch, tmp_path):
    refs = [picture("character-1"), picture("group-21", "group.png")]
    result, sent = run(monkeypatch, tmp_path, references=refs,
                       descriptions="<Subject 1>: Blue coat on the right.\n<Subject 2>: Red coat on the left.")
    assert "- <Subject 1> (a character):" in sent[0]["user"]
    assert "- <Subject 2> (a character):" in sent[0]["user"]
    assert "on the right in <Picture 2>" in sent[0]["user"]
    assert "on the left in <Picture 2>" in sent[0]["user"]
    assert "Blue coat on the right." in result["simple_prompt"]
    assert "Red coat on the left." in result["simple_prompt"]


def test_frame_style_pose_descriptions_fold_into_six_fields(monkeypatch, tmp_path):
    refs = [picture("character-1", "a.png"), picture("place", "b.png"),
            picture("style", "style.png"), picture("first-frame", "first.png"),
            picture("last-frame", "last.png"), picture("pose", "pose.png")]
    descriptions = ("<Subject 1>: Silver hair.\n<Subject 2>: Stone courtyard.\n"
                    "<Subject 3>: Watercolour with ink lines.\n<Picture 4>: A wide courtyard opening.\n"
                    "<Picture 5>: A close view of the doorway.\n<Picture 6>: One arm raised, facing left.")
    result, _ = run(monkeypatch, tmp_path, references=refs, descriptions=descriptions)
    ref = result["fields"]["ref"]
    for phrase in ("Watercolour with ink lines.", "A wide courtyard opening.", "A close view of the doorway."):
        assert phrase in ref["retention_analysis"]
    assert "One arm raised, facing left." in ref["subject_definitions"]
    assert not result["warnings"]
    assert len(ref) == 6


def test_dynamic_expected_segments_do_not_mutate_bundle():
    base = hp.load_bundle()["modes"][hp.EASY_MODE]
    original = dict(base, segments=list(base["segments"]))
    vision = hp.easy_vision_spec(base, ["<Picture 1>"])
    assert vision["segments"] == [*base["segments"], hp.DESCRIBE_SEGMENT]
    assert base == original
    assert hp.easy_vision_spec(base, []) is base


def test_blind_draft_does_not_retain_unsolicited_descriptions(monkeypatch, tmp_path):
    result, _ = run(monkeypatch, tmp_path, flag=False,
                    descriptions="<Subject 1>: Invented hair.\n<Subject 2>: Invented room.")
    assert "Invented" not in result["simple_prompt"]
    assert not result["warnings"]
