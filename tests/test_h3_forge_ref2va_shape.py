"""Labelled REF2VA in the reference guide's shape, and the code-side fixes ported from PromptForge."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nodes import h3_prompting as hp


def pic(easy_role, **extra):
    return {"kind": "image", "role": "subject", "easy_role": easy_role, "path": f"{easy_role}.png", **extra}


def segments_for(description, **extra):
    return {"Subject definitions": "N/A", "Summary": "[reference generation] A scene.",
            "Detailed description": description, "Soundscape": "Room tone.", "Music": "N/A", **extra}


# -- the shipped bundle still fits the vision contract ------------------------

def test_the_shipped_bundle_takes_the_vision_contract():
    spec = hp.load_bundle()["modes"][hp.EASY_MODE]
    vision = hp.easy_vision_spec(spec, ["<Picture 1>"])
    assert "You have not seen the pictures" not in vision["system"]
    assert "keep everyone looking as those Descriptions say" in vision["system"]
    assert "The app folds these lines into Subject definitions" in vision["system"]
    assert vision["segments"][-1] == hp.DESCRIBE_SEGMENT


def test_the_shipped_bundle_carries_the_guides_description_rule():
    # The opening of MiniMax's reference guide: its one rule for
    # detailed_description. PromptForge's export left it out until 6 Oct 2026.
    for mode in ("REF2VA", hp.EASY_MODE):
        assert "Make `detailed_description` as detailed and explicit as possible" in hp.load_bundle()["modes"][mode]["system"]


def test_an_older_bundle_wording_still_works():
    old = {"system": "You have not seen the pictures, so you do not know how they are drawn.\n"
                     "Never name a medium yourself — anime.\n\n"
                     "Do not write Retention analysis. The app writes it from your shots.\n\n## Segments to emit\n\n1. `Summary`",
           "segments": ["Summary"]}
    assert "attached pictures" in hp.easy_vision_spec(old, ["<Picture 1>"])["system"]


# -- typed notes and Keep lines ------------------------------------------------

def test_a_typed_note_makes_the_subject_partial_in_the_writers_reading():
    cast = hp.easy_cast([pic("first-frame"), pic("character-1", instructions="Reference for her haircut")])
    segments = segments_for("[Shot 1] <Subject 1> turns.",
                            **{"Subject definitions": "Keep <Subject 1>: the haircut from <Picture 2>."})
    hp.easy_segments(cast, segments)
    assert "<Subject 1> is the character in <Picture 2>. The haircut from <Picture 2>." in segments["Subject definitions"]
    assert "<Subject 1> (appears in [Shot 1]): partially_preserved — the haircut from <Picture 2>." in segments["Retention analysis"]
    assert "<Picture 1> ([Shot 1] first frame): fully_preserved" in segments["Retention analysis"]


def test_without_a_keep_line_the_note_is_used_as_typed():
    cast = hp.easy_cast([pic("character-1", keep="the red scarf.", drop="the hat")])
    segments = segments_for("[Shot 1] <Subject 1> waves.")
    hp.easy_segments(cast, segments)
    assert "partially_preserved — from <Picture 1>, keep: the red scarf; from <Picture 1>, leave out: the hat." in segments["Retention analysis"]


def test_a_keep_line_where_nothing_was_typed_is_ignored():
    cast = hp.easy_cast([pic("character-1")])
    segments = segments_for("[Shot 1] <Subject 1> waves.", **{"Subject definitions": "Keep <Subject 1>: only the hat"})
    hp.easy_segments(cast, segments)
    assert "fully_preserved" in segments["Retention analysis"] and "hat" not in segments["Retention analysis"]


# -- <Picture N> is for a frame ------------------------------------------------

def test_the_place_picture_cited_as_an_opening_becomes_the_place():
    # A 9B's own Shot 1 (PromptForge, 6 Oct 2026).
    cast = hp.easy_cast([pic("character-1"), pic("character-2"), pic("place")])
    segments = segments_for("[Shot 1] The shot begins from <Picture 3>, establishing the grand dining hall. <Subject 1> sits left.",
                            Summary="[reference generation] <Subject 1> and <Subject 2> drink tea in <Picture 3>.")
    warnings = hp.easy_segments(cast, segments)
    assert "The shot begins from <Subject 3>, establishing the grand dining hall." in segments["Detailed description"]
    assert "drink tea in <Subject 3>." in segments["Summary"]
    assert "<Subject 3> is the place in <Picture 3>" in segments["Subject definitions"]
    assert any("<Picture 3>" in w and "<Subject 3>" in w for w in warnings)


def test_frames_and_pose_pictures_stay_pictures():
    cast = hp.easy_cast([pic("first-frame"), pic("character-1"), pic("pose")])
    segments = segments_for("[Shot 1] The shot begins from <Picture 1>. <Picture 2> raises a hand as in <Picture 3>.")
    hp.easy_segments(cast, segments)
    assert segments["Detailed description"] == "[Shot 1] The shot begins from <Picture 1>. <Subject 1> raises a hand as in <Picture 3>."


def test_a_picture_of_two_characters_is_left_and_warned():
    cast = hp.easy_cast([pic("group-12")])
    segments = segments_for("[Shot 1] Opens on <Picture 1>.")
    warnings = hp.easy_segments(cast, segments)
    assert "Opens on <Picture 1>." in segments["Detailed description"]
    assert any("more than one subject" in w for w in warnings)


# -- the style line keeps its lighting -----------------------------------------

def test_a_guessed_medium_drops_only_its_clause():
    shots = "[Shot 1] <Subject 1> reads."
    assert hp.keep_reference_look(f"Cinematic 2D animation, soft twilight lighting with warm lantern glow.\n\n{shots}", "x") \
        == f"Keeps the look of the reference pictures, soft twilight lighting with warm lantern glow.\n\n{shots}"
    assert hp.keep_reference_look(f"Keeps the look of the reference pictures, anime, golden light.\n\n{shots}", "x") \
        == f"Keeps the look of the reference pictures, golden light.\n\n{shots}"
    assert hp.keep_reference_look(f"Anime, cinematic.\n\n{shots}", "x") == f"Keeps the look of the reference pictures.\n\n{shots}"
    asked = f"Live-action, cinematic, warm light.\n\n{shots}"
    assert hp.keep_reference_look(asked, "live-action, cinematic please") == asked
