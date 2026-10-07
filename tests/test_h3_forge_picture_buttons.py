"""The Forge's picture buttons: kinds that combine, who is in a picture in tap order."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nodes import h3_prompting as hp


def pic(easy_role, **extra):
    return {"kind": "image", "role": "subject", "easy_role": easy_role, "path": f"{easy_role}.png", **extra}


def tags(cast):
    return [(s["tag"], s["kind"], s["pictures"], [p["position"] for p in s.get("placements", [])]) for s in cast["subjects"]]


def test_old_single_labels_read_exactly_as_before():
    cast = hp.easy_cast([pic("character-1"), pic("group-21"), pic("place"), pic("first-frame")])
    assert tags(cast) == [("<Subject 1>", "character", [1], ["right"]), ("<Subject 2>", "character", [], ["left"]),
                          ("<Subject 3>", "place", [3], [])]
    assert [(f["picture"], f["which"]) for f in cast["frames"]] == [(4, "first")]
    assert "on the right in <Picture 2>" in hp._shown_in(cast["subjects"][0])


def test_characters_and_the_place_from_one_picture():
    cast = hp.easy_cast([pic("group-12", picture_kinds=["character", "place"], who=[1, 2], keep="their outfits")])
    assert tags(cast) == [("<Subject 1>", "character", [], ["left"]), ("<Subject 2>", "character", [], ["right"]),
                          ("<Subject 3>", "place", [1], [])]
    # The note belongs to the first kind, the people; the place takes the picture whole.
    assert all(not r.get("keep") for r in cast["subjects"][2]["refs"])
    segments = {"Subject definitions": "N/A", "Detailed description": "[Shot 1] <Subject 1> and <Subject 2> talk in <Subject 3>."}
    hp.easy_segments(cast, segments)
    assert "<Subject 3> (appears in [Shot 1]): fully_preserved — its layout" in segments["Retention analysis"]
    assert "<Subject 1> (appears in [Shot 1]): partially_preserved" in segments["Retention analysis"]


def test_any_order_of_any_characters_and_four_in_a_row():
    cast = hp.easy_cast([pic("character-1", picture_kinds=["character"], who=[3, 1]), pic("character-4", picture_kinds=["character"], who=[4, 3, 2, 1])])
    by = {s["number"]: [(p["picture"], p["position"]) for p in s["placements"]] for s in cast["subjects"]}
    assert by[3] == [(1, "left"), (2, "middle left")]
    assert by[1] == [(1, "right"), (2, "far right")]


def test_top_to_bottom_and_front_to_back():
    stacked = hp.easy_cast([pic("character-2", picture_kinds=["character"], who=[2, 1], who_axis="y")])
    assert hp._shown_in(stacked["subjects"][1]) == "at the top in <Picture 1>"
    piggy = hp.easy_cast([pic("character-2", picture_kinds=["character"], who=[2, 1], who_axis="z")])
    assert hp._shown_in(piggy["subjects"][0]) == "behind in <Picture 1>"
    assert hp._shown_in(piggy["subjects"][1]) == "in front in <Picture 1>"


def test_whos_in_a_frame_places_them_in_it():
    cast = hp.easy_cast([pic("first-frame", picture_kinds=["first-frame"], who=[2, 1]), pic("character-1"), pic("character-2")])
    lines = "\n".join(hp.easy_lines(cast))
    assert '<Subject 1> is "Character 1" in the brief, a character, shown in <Picture 2>, and on the right in the first frame <Picture 1>' in lines
    assert '<Subject 2> is "Character 2" in the brief, a character, shown in <Picture 3>, and on the left in the first frame <Picture 1>' in lines


def test_a_character_who_is_also_the_first_frame():
    cast = hp.easy_cast([pic("character-1", picture_kinds=["character", "first-frame"], who=[1], keep="her red coat")])
    assert cast["subjects"][0]["pictures"] == [1] and cast["subjects"][0]["in_frames"] == []
    assert [(f["picture"], f["which"], f["ref"].get("keep")) for f in cast["frames"]] == [(1, "first", "")]


def test_pose_and_custom_stand_alone():
    assert hp.picture_kinds(pic("pose", picture_kinds=["pose", "place"])) == ["pose"]
    cast = hp.easy_cast([pic("pose", picture_kinds=["pose"], instructions="arms raised")])
    assert cast["subjects"] == [] and [u["which"] for u in cast["uses"]] == ["pose"]


def test_who_is_bounded_and_deduplicated():
    assert hp.picture_who(pic("character-1", picture_kinds=["character"], who=[2, 2, 0, 40, True, "3", 5])) == [2, 5]
    assert hp.picture_who(pic("group-123")) == [1, 2, 3]


# -- a frame's people, by name (I2VA / FL2VA / L2VA) --------------------------

def frame(path, who=None, axis=None):
    return {"kind": "image", "role": "keyframe", "path": path, **({"who": who} if who else {}), **({"who_axis": axis} if axis else {})}


def test_i2va_piggyback_names_who_is_in_front():
    # The test that found it: a man carrying a woman on his back, tapped
    # front to back as 1 then 2.
    brief = "Character 1 jogs down the beach while Character 2 laughs and holds on."
    out = hp.framed_brief(brief, [frame("f.png", [1, 2], "z")], "I2VA")
    assert out == "the character in front in the first frame jogs down the beach while the character behind in the first frame laughs and holds on."
    assert out in hp.build_user_message(hp.load_bundle(), brief, "I2VA", 5, 5, "Balanced", [frame("f.png", [1, 2], "z")], False)


def test_fl2va_and_l2va_name_the_right_frame():
    assert hp.framed_brief("Character 1 waves.", [frame("a.png"), frame("b.png", [1])], "FL2VA") == "the character in the last frame waves."
    assert hp.framed_brief("Character 2 sits.", [frame("a.png", [1, 2])], "L2VA") == "the character on the right in the last frame sits."


def test_unplaced_names_and_other_modes_are_left_alone():
    assert hp.framed_brief("Character 3 waits.", [frame("a.png", [1, 2])], "I2VA") == "Character 3 waits."
    assert hp.framed_brief("Character 1 waits.", [frame("a.png")], "I2VA") == "Character 1 waits."
    assert hp.framed_brief("Character 1 waits.", [frame("a.png", [1])], "T2VA") == "Character 1 waits."


def test_a_seeing_writer_is_told_which_person_to_describe():
    cast = hp.easy_cast([pic("character-1", picture_kinds=["character"], who=[2, 1], who_axis="z")])
    rules = dict((tag, rule) for tag, _, rule in hp.describe_targets(cast))
    assert "it is ONLY the one behind, further from the camera than the other, not the one in front, nearest the camera" in rules["<Subject 1>"]
    assert "it is ONLY the one in front, nearest the camera, not the one behind" in rules["<Subject 2>"]
