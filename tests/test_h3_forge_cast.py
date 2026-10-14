"""The Cast list: a card per character, read from their pictures and used as written."""
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nodes import h3_cast as hc
from nodes import h3_forge as forge
from nodes import h3_prompting as hp
from nodes.llm_backends import ForgeError

BAND = [{"n": n, "box": [0.2 * (n - 1) + 0.01, 0.1, 0.2 * n - 0.01, 0.95]} for n in range(1, 6)]
PIGGYBACK = [{"n": 1, "box": [0.30, 0.25, 0.70, 1.0]}, {"n": 2, "box": [0.35, 0.05, 0.75, 0.70]}]


def pic(path="p.png", easy_role="character-1", **extra):
    return {"kind": "image", "role": "subject", "easy_role": easy_role, "path": path, **extra}


def group(path, who, **extra):
    return pic(path, picture_kinds=["character"], who=who, **extra)


# ── the card's text ──────────────────────────────────────────────────────

def test_cards_are_checked():
    long = "word " * 200
    assert hc.given_cast({"1": "Red hair.", "2": "  ", "x": "a", "40": "b", 3: long}) == {1: "Red hair.", 3: long.strip()[:600]}
    assert hc.given_cast(["Red hair."]) == {}


def test_a_card_becomes_the_kept_list():
    assert hc.kept_from_look("A young woman with long green hair. She wears a red dress.") == "long green hair, a red dress"
    assert hc.kept_from_look("An adult woman with white braids, blue eyes, and wears a white dress.") == "white braids, blue eyes, a white dress"


def test_a_card_never_opens_with_the_name_h3_cannot_place():
    assert hc.unnamed("Character 2 is a young woman with long white hair.") == "A young woman with long white hair."
    assert hc.unnamed("Character 1 has shoulder-length auburn hair.") == "A character with shoulder-length auburn hair."
    assert hc.unnamed("The character on the left has short black hair.") == "A character with short black hair."
    assert hc.unnamed("A young man with short black hair.") == "A young man with short black hair."
    unit = {"kind": "character"}
    assert hc.cast_look("Character 2 is an adult woman in a red coat.\nKept: red coat", unit) == "An adult woman in a red coat."
    assert hc.written_look("green eyes, not blue") == "green eyes, not blue."


def test_the_reader_answer_is_cleaned():
    assert hc.usable_reading("<think>hmm</think>An adult woman with red hair.") == "An adult woman with red hair."
    assert hc.usable_reading("planning...\n</think>\nAn adult woman.") == "An adult woman."
    assert hc.usable_reading("An adult woman with red hair.\nUNAVAILABLE") == "An adult woman with red hair."
    assert hc.usable_reading("UNAVAILABLE") is None
    assert hc.split_description("An adult woman with red hair. She wears blue. She holds a cup.\nKept: red hair, blue dress") == "An adult woman with red hair. She wears blue."
    assert hc.group_lines("**1:** A woman.\nCharacter 2 - A man.\n3) A child.", 3) == ["A woman.", "A man.", "A child."]
    assert hc.group_lines("1: A woman.", 2) is None


# ── what the reader reads ────────────────────────────────────────────────

def test_a_boxed_band_member_is_read_from_her_crop():
    cast = hc.cast_for([group("band.png", [1, 2, 3, 4, 5], boxes=BAND)], "REF2VA")
    unit = hc.cast_unit(cast, 3)
    assert [p["crop"] for p in unit["pictures"]] == [BAND[2]["box"]]
    msg = hc.describe_message(unit, "They play a gig.")
    assert "The first picture is cut from a picture of several people, around this one person." in msg
    assert "Describe ONLY" not in msg  # a crop shows them alone
    assert 'where this is "Character 3"' in msg and "They play a gig." in msg


def test_overlapping_boxes_are_read_whole_by_position():
    cast = hc.cast_for([group("pig.png", [1, 2], boxes=PIGGYBACK)], "REF2VA")
    unit = hc.cast_unit(cast, 2)
    assert unit["pictures"][0]["crop"] is None
    msg = hc.describe_message(unit)
    assert "The first picture shows 2 people. Describe ONLY the one on the right, not the one on the left." in msg


def test_nobody_placed_is_read_as_a_group_and_each_gets_their_line():
    cast = hc.cast_for([group("two.png", [1, 2], who_axis="none")], "REF2VA")
    unit = hc.cast_unit(cast, 2)
    assert unit["kind"] == "group" and unit["people"] == 2 and unit["index"] == 1
    assert "It shows 2 characters. Describe each of them in turn, left to right" in hc.describe_message(unit)
    assert hc.cast_look("1: A tall man in a coat.\n2: An adult woman with a red bow.\nKept: coat, bow", unit) == "An adult woman with a red bow."


def test_a_note_on_one_of_two_pictures_is_said_for_that_picture():
    cast = hc.cast_for([pic("full.png"), pic("face.png", keep="her face")], "REF2VA")
    msg = hc.describe_message(hc.cast_unit(cast, 1))
    assert "These 2 pictures show the same character; describe it once." in msg
    assert "From the second picture, take only: her face." in msg and "Take everything else from the other pictures." in msg


def test_nobody_on_their_own_is_not_read():
    assert hc.cast_unit(hc.cast_for([pic("p.png")], "REF2VA"), 2) is None


def test_frames_are_read_from_the_boxes_in_them():
    frame = {"kind": "image", "role": "keyframe", "path": "f.png", "boxes": BAND}
    cast = hc.cast_for([frame], "I2VA")
    unit = hc.cast_unit(cast, 5)
    assert unit["frames"] == 1 and unit["pictures"][0]["crop"] == BAND[4]["box"]
    assert "frame of the video, cut around the one person to describe" in hc.describe_message(unit)
    # FL2VA: the second picture is the last frame.
    fl = hc.cast_for([{**frame, "boxes": BAND[:2]}, {**frame, "path": "g.png", "boxes": BAND[:2]}], "FL2VA")
    assert [f["which"] for f in fl["frames"]] == ["first", "last"] and hc.cast_unit(fl, 1)["frames"] == 2


# ── cards at Generate ────────────────────────────────────────────────────

LONG = ("A young woman with long green hair in a high ponytail and green eyes. She wears a red jacket over a white "
        "shirt, black shorts and tall boots. She has a small scar on her chin. A silver ring on her left hand.")


def test_a_card_is_used_as_written_and_never_cut():
    cast = hp.easy_cast([pic("a.png"), pic("b.png", "character-2")])
    cards = hc.carded_tags(cast, {1: LONG})
    lines = "\n".join(hp.easy_lines(cast, cards=cards))
    assert f"· looks: {LONG}" in lines and '"looks:" is what was seen in the pictures' in lines
    assert "never describe how anyone or anything looks" not in lines
    segments = {"Subject definitions": "N/A", "Summary": "s", "Detailed description": "[Shot 1] <Subject 1> waves at <Subject 2>.",
                "Soundscape": "x", "Music": "N/A"}
    hp.easy_segments(cast, segments, None, cards)
    assert f"<Subject 1> is the character in <Picture 1>. {LONG}" in segments["Subject definitions"]
    assert "<Subject 2> is the character in <Picture 2>; keep their appearance exactly as the pictures show." in segments["Subject definitions"]
    assert "fully_preserved — long green hair in a high ponytail and green eyes, a red jacket over a white shirt" in segments["Retention analysis"]


def test_a_seeing_writer_is_not_asked_to_describe_a_carded_character():
    cast = hp.easy_cast([pic("a.png"), pic("b.png", "character-2")])
    carded = hc.carded_tags(cast, {2: "Blue hair."})
    targets = [t for t, _, _ in hp.describe_targets(cast, ["<Picture 1>", "<Picture 2>"], carded)]
    assert targets == ["<Subject 1>"]
    lines = "\n".join(hp.easy_lines(cast, sees_pictures=True, cards=carded))
    assert "Record visible reference appearance in Descriptions" in lines and 'A "looks:" line supplies reference appearance' in lines
    # The writer's own Description of a carded character does not replace the card.
    segments = {"Subject definitions": "N/A", "Summary": "s", "Detailed description": "[Shot 1] <Subject 1> and <Subject 2>.",
                "Soundscape": "x", "Music": "N/A", "Descriptions": "<Subject 1>: Red hair.\n<Subject 2>: Green hair."}
    hp.easy_segments(cast, segments, ["<Subject 1>"], carded)
    assert "<Subject 2> is the character in <Picture 2>. Blue hair." in segments["Subject definitions"]


def test_frame_modes_put_the_card_beside_where_they_stand():
    frame = {"kind": "image", "role": "keyframe", "path": "f.png", "boxes": [{"n": 1, "box": [0.05, 0, 0.45, 1]}, {"n": 2, "box": [0.55, 0, 0.95, 1]}]}
    bundle = hp.load_bundle()
    user = hp.build_user_message(bundle, "Character 2 waves at Character 1.", "I2VA", 8, 5, "balanced", [frame], True, ["<Picture 1>"],
                                 cards={2: "An adult woman with a red bow"})
    assert "the character on the right in the first frame waves at the character on the left in the first frame." in user
    assert "Cast, as the person describes them." in user
    assert "- The character on the right in the first frame: An adult woman with a red bow." in user
    assert "Character 2" not in user.split("Cast, as the person")[1]


# ── the reader route ─────────────────────────────────────────────────────

class Seeing:
    def __init__(self, raw="An adult woman with red hair. She wears a white dress.\nKept: red hair, white dress", sees=None):
        self.raw, self.sees, self.calls, self.unloads, self.base = raw, sees, [], [], "http://127.0.0.1:8099"

    def can_see(self, name):
        return self.sees

    def unload(self, name):
        self.unloads.append(name)
        return True

    def chat(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.raw, {}


@pytest.fixture
def pictures(tmp_path):
    from PIL import Image
    Image.new("RGB", (1000, 500), "white").save(tmp_path / "band.png")
    return tmp_path


@pytest.fixture(autouse=True)
def no_warm_left():
    yield
    forge.release_warm()


def body(**extra):
    return {"model": "openai:r4", "mode": "REF2VA", "brief": "A gig.", "character": 2,
            "references": [group("band.png", [1, 2, 3, 4, 5], boxes=BAND)], **extra}


def test_a_card_is_read_from_the_crop_with_the_reader_prompt(pictures, monkeypatch):
    model = Seeing()
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    out = forge.read_cast(body(), str(pictures))
    assert out["text"] == "An adult woman with red hair. She wears a white dress." and out["cropped"] == 1
    (name, system, user, images, sampling, *_), kwargs = model.calls[0]
    assert system.startswith(hp.load_bundle()["picture_description"])
    assert "## Age\n" in system and "Every human you describe is an adult." in system
    assert user == hc.describe_message(hc.cast_unit(hc.cast_for(body()["references"], "REF2VA"), 2), "A gig.")
    assert len(images) == 1 and sampling["temperature"] == 0.2 and kwargs["max_tokens"] == 240
    assert model.unloads == ["r4"]


def test_reads_keep_the_model_warm_until_the_panel_lets_go(pictures, monkeypatch):
    model = Seeing()
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    monkeypatch.setattr(forge, "WARM_SECONDS", 60)
    first = forge.read_cast(body(keep_warm=True), str(pictures))
    forge.read_cast(body(keep_warm=True, character=3), str(pictures))
    assert not first["unloaded"] and model.unloads == []
    forge.release_warm()
    assert model.unloads == ["r4"]


def test_a_warm_model_goes_when_its_time_is_up(pictures, monkeypatch):
    model = Seeing()
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    monkeypatch.setattr(forge, "WARM_SECONDS", 0.05)
    forge.read_cast(body(keep_warm=True), str(pictures))
    timer = next(iter(forge._WARM.values()))[2]
    timer.join(2)
    assert model.unloads == ["r4"] and not forge._WARM


def test_a_reader_that_cannot_see_or_nobody_to_read_says_so(pictures, monkeypatch):
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": Seeing(sees=False)})
    with pytest.raises(ForgeError) as err:
        forge.read_cast(body(), str(pictures))
    assert err.value.code == "no_vision"
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": Seeing()})
    with pytest.raises(ForgeError) as err:
        forge.read_cast(body(character=9), str(pictures))
    assert err.value.code == "not_shown"
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": Seeing(raw="UNAVAILABLE")})
    with pytest.raises(ForgeError) as err:
        forge.read_cast(body(), str(pictures))
    assert err.value.code == "no_reading"


def test_the_draft_uses_the_cards_and_rejects_bad_ones(pictures, monkeypatch):
    raw = "\n".join(f"===SEGMENT: {label}===\n{text}" for label, text in [
        ("Subject definitions", "N/A"), ("Summary", "[reference generation] A gig."),
        ("Detailed description", "Keeps the look of the reference pictures.\n\n[Shot 1] <Subject 2> sings while <Subject 1> plays."),
        ("Soundscape", "Guitar."), ("Music", "N/A")])
    model = Seeing(raw=raw)
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    refs = [group("band.png", [1, 2], boxes=BAND[:2])]
    out = forge.generate({"model": "openai:w", "mode": "REF2VA", "easy": True, "brief": "Character 2 sings.", "duration": 5,
                          "references": refs, "cast": {"2": LONG}}, str(pictures))
    user = model.calls[0][0][2]
    assert f"looks: {LONG}" in user
    assert f"<Subject 2> is the character on the right in <Picture 1>. {LONG}" in out["fields"]["ref"]["subject_definitions"]
    assert out["cards_used"] == ["2"]
    with pytest.raises(ForgeError):
        forge.generate({"model": "openai:w", "mode": "REF2VA", "easy": True, "brief": "x", "references": refs, "cast": ["no"]}, str(pictures))



def _hold_lock(seconds):
    """Another call holding the Forge lock for a while: one thread takes and releases it."""
    taken = threading.Event()

    def hold():
        with forge._REQUEST_LOCK:
            taken.set()
            threading.Event().wait(seconds)
    t = threading.Thread(target=hold)
    t.start()
    taken.wait(2)
    return t


def test_a_read_waits_for_a_call_already_running(pictures, monkeypatch):
    model = Seeing()
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    holder = _hold_lock(0.3)
    out = forge.read_cast(body(), str(pictures))
    holder.join()
    assert out["text"].startswith("An adult woman with red hair")
    monkeypatch.setattr(forge, "READ_WAIT", 0.1)
    holder = _hold_lock(0.5)
    with pytest.raises(ForgeError) as err:
        forge.read_cast(body(), str(pictures))
    holder.join()
    assert err.value.code == "busy"



# ── the place, the style and each pose ────────────────────────────────────

def scene_refs():
    return [pic("girl.png"), pic("room.png", "place"), pic("ink.png", "style"),
            pic("pose.png", "pose", role="pose", picture_kinds=["pose"], who=[1])]


def test_scene_cards_are_keyed_by_what_they_are():
    given = hc.given_cast({"1": "Red hair.", "place": "A library.", "style": "Ink wash.", "pose@4": "Arms raised.", "pose@x": "no", "scene": "no"})
    assert given == {1: "Red hair.", "place": "A library.", "style": "Ink wash.", "pose@4": "Arms raised."}
    assert hc.character_cards(given) == {1: "Red hair."}


def test_the_place_style_and_pose_are_read_with_their_own_message():
    cast = hc.cast_for(scene_refs(), "REF2VA")
    place = hc.scene_unit(cast, "place")
    assert place["tag"] == "<Subject 2>" and "This shows the place." in hc.describe_message(place)
    style = hc.describe_message(hc.scene_unit(cast, "style"))
    assert "This shows a style." in style and "Start with the medium" in style and "Never what is in the picture" in style
    assert "Start with the medium" not in hc.describe_message(place)
    pose = hc.scene_unit(cast, "pose@4")
    msg = hc.describe_message(pose, "She waves.")
    assert "Describe only the pose" in msg and "other characters take this pose" in msg and "this pose" in msg
    assert hc.scene_unit(cast, "pose@1") is None and hc.scene_unit(cast, "place@1") is None


def test_a_place_behind_the_characters_is_the_setting_only():
    cast = hc.cast_for([pic("both.png", picture_kinds=["character", "place"], who=[1])], "REF2VA")
    assert "Describe only the setting" in hc.describe_message(hc.scene_unit(cast, "place"))


def test_a_pose_card_keeps_only_the_stance():
    reading = ("She stands with her weight on one hip and her left hand raised to her chin, her long lavender braid over one shoulder. "
               "She wears a red dress.\nKept: weight on one hip, hand to chin")
    look = hc.cast_look(reading, {"kind": "pose"})
    assert look == "She stands with her weight on one hip and her left hand raised to her chin."
    cast = hc.cast_for(scene_refs(), "REF2VA")
    assert hc.carded_tags(cast, {"pose@4": "Arms crossed, head tilted. Her blue jacket is open."}) == {"<Picture 4>": "Arms crossed, head tilted."}


def test_scene_cards_go_in_as_written():
    cast = hp.easy_cast(scene_refs())
    cards = hc.carded_tags(cast, {"place": "A sunlit library with tall arched windows and dark wooden shelves.",
                                  "style": "Soft anime cel shading, thin clean linework.", "pose@4": "Arms raised high, head tipped back."})
    lines = "\n".join(hp.easy_lines(cast, cards=cards))
    assert "is the place, shown in <Picture 2>" in lines and "looks: A sunlit library with tall arched windows" in lines
    assert "looks: Soft anime cel shading, thin clean linework." in lines
    assert "gives a pose only, taken by <Subject 1>" in lines and "· pose: Arms raised high, head tipped back." in lines
    segments = {"Subject definitions": "N/A", "Summary": "s", "Detailed description": "[Shot 1] <Subject 1> reads in <Subject 2>.",
                "Soundscape": "x", "Music": "N/A"}
    hp.easy_segments(cast, segments, None, cards)
    defs, kept = segments["Subject definitions"], segments["Retention analysis"]
    assert "<Subject 2> is the place in <Picture 2>, where the video happens. A sunlit library with tall arched windows and dark wooden shelves." in defs
    assert "<Subject 3> is the rendering style of <Picture 3>, applied to the whole video and not its content. Soft anime cel shading" in defs
    assert "As the pictures show: Arms raised high, head tipped back." in defs
    assert "<Subject 2> (appears in [Shot 1]): fully_preserved — its layout, landmarks, light and time of day are retained." in kept
    # A seeing writer is not asked to describe what has a card.
    targets = [t for t, _, _ in hp.describe_targets(cast, ["<Picture 1>", "<Picture 2>", "<Picture 3>", "<Picture 4>"], cards)]
    assert targets == ["<Subject 1>"]


def test_the_reader_route_reads_the_place_and_a_pose(pictures, monkeypatch):
    from PIL import Image
    for name in ("girl.png", "room.png", "ink.png", "pose.png"):
        Image.new("RGB", (64, 64), "white").save(pictures / name)
    model = Seeing(raw="A library with tall windows. Warm afternoon light.\nKept: tall windows, warm light")
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": model})
    base = {"model": "openai:r4", "mode": "REF2VA", "brief": "", "references": scene_refs()}
    out = forge.read_cast({**base, "subject": "place"}, str(pictures))
    assert out["subject"] == "place" and out["text"] == "A library with tall windows. Warm afternoon light."
    assert "This shows the place." in model.calls[0][0][2]
    model.raw = "Arms raised high, her red sleeves falling back. Head tipped back."
    out = forge.read_cast({**base, "subject": "pose@4"}, str(pictures))
    assert out["text"] == "Arms raised high. Head tipped back."
    with pytest.raises(ForgeError) as err:
        forge.read_cast({**base, "subject": "pose@2"}, str(pictures))
    assert err.value.code == "not_shown"
