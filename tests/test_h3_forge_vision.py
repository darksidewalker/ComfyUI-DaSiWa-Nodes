"""Labelled REF2VA writes blind by default; the writer sees the pictures only when asked."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nodes import h3_forge as forge


def pic(easy_role, path):
    return {"kind": "image", "role": "subject", "easy_role": easy_role, "path": path}


def _backend(sent, sees=True):
    class Backend:
        base = "https://fixture.invalid"
        def can_see(self, name):
            return sees
        def chat(self, name, system, user, images, *rest):
            sent.append({"user": user, "images": list(images)})
            segments = {
                "Subject definitions": "<Subject 1>: waves",
                "Summary": "A short scene.",
                "Detailed description": "integrated_multimodal_description: Keeps the look of the reference pictures.\n\n[Shot 1] <Subject 1> waves in <Subject 2>.",
                "Soundscape": "Room tone.", "Music": "N/A",
            }
            return "\n".join(f"===SEGMENT: {label}===\n{body}" for label, body in segments.items()), {}
        def unload(self, name):
            return True
    return Backend()


def _run(monkeypatch, tmp_path, see_pictures, sees=True):
    for name in ("a.png", "b.png"):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr(forge, "_image_b64", lambda path: Path(path).name)
    sent = []
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": _backend(sent, sees)})
    body = {"mode": "REF2VA", "model": "openai:fixture", "brief": "Character 1 waves in the place.", "duration": 5,
            "easy": True, "references": [pic("character-1", "a.png"), pic("place", "b.png")]}
    if see_pictures is not None:
        body["see_pictures"] = see_pictures
    return forge._generate(body, str(tmp_path), None, None), sent


def test_labelled_pictures_stay_blind_by_default(monkeypatch, tmp_path):
    for flag in (None, False):
        result, sent = _run(monkeypatch, tmp_path, flag)
        assert result["easy"] is True and result["saw_images"] == 0
        assert sent[0]["images"] == []
        assert "attached" not in sent[0]["user"]


def test_see_pictures_sends_them_in_label_order(monkeypatch, tmp_path):
    result, sent = _run(monkeypatch, tmp_path, True)
    assert result["easy"] is True and result["saw_images"] == 2
    assert sent[0]["images"] == ["a.png", "b.png"]
    user = sent[0]["user"]
    assert "The pictures <Picture 1>, <Picture 2> are attached to this message, in that order" in user
    assert "never describe how anyone looks" in user
    assert user.index("Cast (fixed)") < user.index("are attached")


ALL_KINDS = [pic("character-1", "a.png"), pic("place", "b.png"), pic("style", "c.png"),
             pic("first-frame", "d.png"), pic("pose", "e.png")]


def test_seeing_writer_is_asked_for_a_description_of_each_by_kind(monkeypatch, tmp_path):
    from nodes import h3_prompting as hp
    message = hp.build_user_message(hp.load_bundle(), "Character 1 waves", "REF2VA", 8, 5, "balanced", ALL_KINDS, True,
                                    [f"<Picture {n}>" for n in range(1, 6)], cast=hp.easy_cast(ALL_KINDS))
    assert "===SEGMENT: Descriptions===" in message
    assert "- <Subject 1> (a character): what they look like" in message
    assert "- <Subject 2> (the place): the background" in message
    assert "- <Subject 3> (the style): how it is drawn" in message
    assert "- <Picture 4> (the first frame): a general description of the whole picture" in message
    assert "- <Picture 5> (a pose): the pose only" in message
    result, sent = _run(monkeypatch, tmp_path, False)
    assert "Descriptions" not in sent[0]["user"]


def test_descriptions_join_retention_and_the_pose_line():
    from nodes import h3_prompting as hp
    cast = hp.easy_cast(ALL_KINDS)
    segments = {
        "Subject definitions": "<Subject 1>: waves",
        "Detailed description": "[Shot 1] <Subject 1> waves in <Subject 2>.",
        "Descriptions": ("<Subject 1>: A woman with short silver hair in a lavender scarf.\n"
                         "<Subject 2>: A lavender field under a low evening sun\n"
                         "<Subject 3>: Soft watercolour with thin ink lines.\n"
                         "<Picture 4>: <Subject 1> stands left of a stone bridge, seen from the waist up.\n"
                         "<Picture 5>: Standing with one hand raised to shade the eyes, facing left."),
    }
    hp.easy_segments(cast, segments)
    ret = segments["Retention analysis"].splitlines()
    assert ret[0].endswith("in every shot. As the pictures show: A woman with short silver hair in a lavender scarf.")
    assert ret[1].endswith("time of day. As the pictures show: A lavender field under a low evening sun.")
    assert ret[2].endswith("throughout. As the pictures show: Soft watercolour with thin ink lines.")
    assert ret[3].endswith("subject positions. As the pictures show: <Subject 1> stands left of a stone bridge, seen from the waist up.")
    assert segments["Subject definitions"].endswith("As the pictures show: Standing with one hand raised to shade the eyes, facing left.")
    assert "Descriptions" not in segments
    # Blind runs write none, and every line is exactly as before.
    segments = {"Subject definitions": "<Subject 1>: waves", "Detailed description": "[Shot 1] <Subject 1> waves."}
    hp.easy_segments(cast, segments)
    assert "As the pictures show" not in segments["Retention analysis"] + segments["Subject definitions"]


def test_acting_lines_run_into_one_paragraph_are_split():
    from nodes import h3_prompting as hp
    body = ("<Subject 1>: walks with a gentle gait, bending down to smell the flowers, her expression animated by delight. "
            "<Subject 2>: smiles warmly while observing <Subject 1>, maintaining a relaxed posture.")
    acting = hp._acting(body)
    assert acting["<Subject 1>"].endswith("animated by delight.")
    assert acting["<Subject 2>"] == "smiles warmly while observing <Subject 1>, maintaining a relaxed posture."
    assert hp._acting("<Subject 1>: tired, slow blinks\n<Subject 2> — stiff and formal") == {
        "<Subject 1>": "tired, slow blinks", "<Subject 2>": "stiff and formal"}


def test_see_pictures_on_a_blind_model_falls_back_to_labels(monkeypatch, tmp_path):
    result, sent = _run(monkeypatch, tmp_path, True, sees=False)
    assert result["saw_images"] == 0 and result["vision"] is False
    assert sent[0]["images"] == [] and "attached" not in sent[0]["user"]
