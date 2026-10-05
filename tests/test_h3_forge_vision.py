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


def test_see_pictures_on_a_blind_model_falls_back_to_labels(monkeypatch, tmp_path):
    result, sent = _run(monkeypatch, tmp_path, True, sees=False)
    assert result["saw_images"] == 0 and result["vision"] is False
    assert sent[0]["images"] == [] and "attached" not in sent[0]["user"]
