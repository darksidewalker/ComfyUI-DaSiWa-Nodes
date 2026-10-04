"""REF2VA Forge easy mode: labelled pictures, code-written cast, music only when asked."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "nodes"))
spec = importlib.util.spec_from_file_location("h3_forge_test", ROOT / "nodes" / "h3_forge.py")
forge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forge)


def pic(easy_role, **extra):
    return {"kind": "image", "role": "subject", "easy_role": easy_role, **extra}


REFS = [pic("character-2"), pic("place"), pic("character-1", keep="tired"), pic("character-2"), pic("first-frame")]


def test_cast_numbers_characters_then_place_and_lists_frames_apart():
    cast = forge.easy_cast(REFS)
    assert [(s["tag"], s["name"], s["pictures"]) for s in cast["subjects"]] == [
        ("<Subject 1>", "Character 1", [3]), ("<Subject 2>", "Character 2", [1, 4]), ("<Subject 3>", "the place", [2])]
    assert [(f["picture"], f["which"]) for f in cast["frames"]] == [(5, "first")]


def test_group_picture_places_each_character():
    refs = [pic("character-1"), pic("group-21"), pic("place"), pic("group-123")]
    cast = forge.easy_cast(refs)
    by = {s["name"]: s for s in cast["subjects"]}
    assert by["Character 1"]["pictures"] == [1]
    assert by["Character 1"]["placements"] == [{"picture": 2, "position": "right"}, {"picture": 4, "position": "left"}]
    assert by["Character 3"]["tag"] == "<Subject 3>" and by["Character 3"]["pictures"] == []
    assert by["the place"]["tag"] == "<Subject 4>"
    lines = "\n".join(forge.easy_lines(cast))
    assert '<Subject 1> is "Character 1" in the brief, a character, shown in <Picture 1>, and on the right in <Picture 2> and on the left in <Picture 4>' in lines
    assert '<Subject 3> is "Character 3" in the brief, a character, shown on the right in <Picture 4>' in lines
    segments = {"Detailed description": "[Shot 1] <Subject 1>, <Subject 2> and <Subject 3> in <Subject 4>."}
    forge.easy_segments(cast, segments)
    assert "<Subject 2> is the character on the left in <Picture 2> and in the middle in <Picture 4>;" in segments["Subject definitions"]
    assert forge.easy_brief("Character 2 waves", cast) == "<Subject 2> waves"


def test_separate_characters_beyond_four_stay_separate():
    cast = forge.easy_cast([pic(f"character-{n}") for n in range(1, 10)])
    assert len(cast["subjects"]) == 9
    assert cast["subjects"][-1]["name"] == "Character 9"
    assert forge.easy_brief("Character 9 waves", cast) == "<Subject 9> waves"


def test_unknown_label_is_character_1():
    assert forge.easy_cast([pic("dragon")])["subjects"][0]["name"] == "Character 1"


def test_brief_names_become_tags():
    cast = forge.easy_cast(REFS)
    assert forge.easy_brief("Character 1 pours tea for character 2 in the place while Character 3 waits", cast) == \
        "<Subject 1> pours tea for <Subject 2> in <Subject 3> while Character 3 waits"
    assert forge.easy_brief("It takes place at night, in place of the party.", cast) == "It takes place at night, in place of the party."
    assert forge.easy_brief("outfit from picture 4, not image 9, and <Picture 2>", cast) == "outfit from <Picture 4>, not image 9, and <Picture 2>"


def test_user_message_has_cast_not_picture_lines_and_no_attached_claim():
    message = forge.build_user_message(forge.load_bundle(), "Character 1 waves", "REF2VA", 10, 5, "balanced",
                                       REFS + [{"kind": "audio", "duration_seconds": 4}], False, cast=forge.easy_cast(REFS))
    assert 'Brief: "<Subject 1> waves"' in message
    assert '<Subject 2> is "Character 2" in the brief, a character, shown in <Picture 1> and <Picture 4>' in message
    assert "keep: tired" in message
    assert "<Picture 5> is the first frame" in message
    assert "<Audio 1>" in message
    assert "- <Picture 1> ·" not in message
    assert "is attached" not in message
    assert "Director output canvas is unknown" in message


def test_pose_custom_and_instructions_in_easy_mode():
    refs = [pic("character-1", instructions="the red coat matters"), pic("pose", instructions="arms crossed"),
            pic("custom", role="custom", instructions="use only the lighting"),
            {"kind": "image", "role": "subject", "instructions": "a saved reference", "saved_reference": True}]
    cast = forge.easy_cast(refs)
    assert [s["name"] for s in cast["subjects"]] == ["Character 1"]
    assert [(u["picture"], u["which"]) for u in cast["uses"]] == [(2, "pose"), (3, "custom")]
    message = forge.build_user_message(forge.load_bundle(), "Character 1 in the pose of picture 2", "REF2VA", 10, 5, "balanced",
                                       refs, False, output_canvas={"width": 1280, "height": 720}, cast=cast)
    assert 'Brief: "<Subject 1> in the pose of <Picture 2>"' in message
    assert "instructions: the red coat matters" in message
    assert "<Picture 2> gives a pose only" in message and "instructions: arms crossed" in message
    assert "<Picture 3> is used only as its instructions say · instructions: use only the lighting" in message
    # The saved reference has no label, so it keeps its own reference line.
    assert "- <Picture 4> · subject" in message and "a saved reference" in message
    assert "aspect ratio 16:9" in message
    segments = {"Detailed description": "[Shot 1] <Subject 1> stands."}
    forge.easy_segments(cast, segments)
    assert "<Picture 2> gives the pose only" in segments["Subject definitions"]
    assert "not identity, clothing or background. Arms crossed." in segments["Subject definitions"]
    assert "<Picture 3>: Use only the lighting." in segments["Subject definitions"]


def test_shots_line_and_cut_repair():
    assert forge.shots_line("Auto") is None and forge.shots_line(None) is None
    assert "exactly 3 shots" in forge.shots_line("3")
    segments = {"Detailed description": "[Shot 1] A.\n[Shot 2] At 00:04.000, B.\n[Shot 3] At 00:10.000, C."}
    assert forge.repair_cut_times(10, segments) == [("00:10.000", "00:07.000")]
    assert "[Shot 3] At 00:07.000" in segments["Detailed description"]
    assert forge.shot_count_warning("2", segments) and forge.shot_count_warning("3", segments) is None


def test_shot_boxes_join_the_idea():
    assert forge.fold_shot_briefs("Rain.", "2", ["She runs.", " He turns. "]) == "Rain.\nShot 1: She runs.\nShot 2: He turns."
    assert forge.fold_shot_briefs("x", "3", ["a", "", "c"]) == "x\nShot 1: a\nShot 3: c"
    assert forge.fold_shot_briefs("x", "1", ["a", "b"]) == "x\nShot 1: a"
    assert forge.fold_shot_briefs("x", "Auto", ["a"]) == "x"
    assert forge.fold_shot_briefs("", "2", ["a", "b"]) == "Shot 1: a\nShot 2: b"
    assert forge.fold_shot_briefs(" x ", None, None) == "x"


def test_segments_written_in_code():
    cast = forge.easy_cast(REFS)
    segments = {
        "Subject definitions": "<Subject 1>: tired, slow blinks\n<Subject 2> — stiff and formal",
        "Summary": "S.",
        "Detailed description": "integrated_multimodal_description: Style.\n\n[Shot 1] <Subject 3>, <Subject 1> sits.\n\n[Shot 2] At 00:03.000, <Subject 1> pours.",
        "Soundscape": "Rain.", "Music": "Koto.",
    }
    warnings = forge.easy_segments(cast, segments)
    defs, ret = segments["Subject definitions"], segments["Retention analysis"]
    assert defs.startswith("<Subject 1> is the character in <Picture 3>; keep their appearance exactly as the pictures show. In this scene: tired, slow blinks.")
    assert "<Subject 2> is the character in <Picture 1> and <Picture 4>;" in defs
    assert "<Subject 3> is the place in <Picture 2>, where the video happens" in defs
    assert "<Picture 5> is the first frame of [Shot 1]." in defs
    assert "<Subject 1> (appears in [Shot 1]-[Shot 2])" in ret
    assert "<Subject 3> (appears in [Shot 1]-[Shot 2])" in ret
    assert "tired" not in ret
    assert len(warnings) == 1 and "Character 2" in warnings[0]
    fields = forge.builder_fields(segments, "REF2VA")
    assert fields["ref"]["retention_analysis"] == ret


def test_bundle_has_easy_mode_without_retention():
    bundle = forge.load_bundle()
    assert "Retention analysis" not in bundle["modes"][forge.EASY_MODE]["segments"]
    assert "Subject definitions" in bundle["modes"][forge.EASY_MODE]["segments"]


def test_style_line_never_guesses_a_medium():
    shots = "[Shot 1] <Subject 1> reads."
    guessed = "Live-action, cinematic, warm interior light.\n\n" + shots
    assert forge.keep_reference_look(guessed, "Character 1 reads") == "Keeps the look of the reference pictures.\n\n" + shots
    assert forge.keep_reference_look(guessed, "live-action, cinematic: Character 1 reads") == guessed
    fine = "Keeps the look of the reference pictures, warm indoor light.\n\n" + shots
    assert forge.keep_reference_look(fine, "Character 1 reads") == fine
    assert forge.keep_reference_look("[Shot 1] An animated wave.", "x") == "[Shot 1] An animated wave."


def test_runaway_is_refused_not_applied():
    words = " ".join(["absurd ridiculous nonsensical illogical unreasonable"] * 40)
    good = {"Summary": "A short summary.", "Detailed description": "[Shot 1] She reads. The camera holds."}
    assert forge.runaway(good) is None
    assert forge.runaway({**good, "Soundscape": "Room tone. " + words})[0] == "Soundscape"
    raw = "===SEGMENT: Summary===\nS.\n===SEGMENT: Detailed description===\n[Shot 1] " + words
    try:
        forge.parse_segments(raw, ["Summary", "Detailed description", "Soundscape", "Music"])
        assert False, "expected a runaway error"
    except forge.ForgeError as exc:
        assert exc.code == "runaway" and "Soundscape, Music" in exc.message


def test_music_warning_preserves_explicit_and_multilingual_requests():
    bundle = forge.load_bundle()
    for brief in ("She reads on the bed.", "She reads with a flute accompaniment.", "Eine Frau liest, mit Hintergrundmusik."):
        segments = {"Music": "A soft flute accompaniment."}
        assert forge.music_request_warning(bundle, brief, segments)
        assert segments["Music"] == "A soft flute accompaniment."
    assert not forge.music_request_warning(bundle, "Soft piano music plays.", {"Music": "Soft piano notes."})
    assert not forge.music_request_warning({"music_words": None}, "She reads.", {"Music": "Soft piano notes."})
    assert not forge.music_request_warning(bundle, "It takes place at night.", {"Music": "N/A"})


def test_runaway_allows_quoted_sentences_and_dialogue_tags():
    sentence = 'A placard reads "Please use the main entrance while repairs to this side doorway are in progress."'
    for body in ("[Shot 1] " + " ".join([sentence] * 6), "[Shot 1] " + " ".join(['<d>"Please use the main entrance while repairs to this side doorway are in progress."</d>'] * 6)):
        assert len(body) > 500
        assert forge.runaway({"Detailed description": body}) is None


def test_cut_repair_keeps_millisecond_order_near_clip_end():
    for stamps in (("09.800", "10.000", "10.000"), ("09.950", "10.000")):
        segments = {"Detailed description": "[Shot 1] A. " + " ".join(f"[Shot {n}] At 00:{stamp}, B." for n, stamp in enumerate(stamps, 2))}
        assert forge.repair_cut_times(10, segments)
        fields = forge.builder_fields(segments, "REF2VA")
        assert forge.check_prompt(fields, "REF2VA", 10, "", 7000) == []
        cuts = [int(m.group(2)) * 60 + float(m.group(3)) for m in forge._CUT.finditer(segments["Detailed description"])]
        assert all(a < b < 10 for a, b in zip([0, *cuts], cuts))
    segments = {"Detailed description": "[Shot 1] A. [Shot 2] At 00:09.999, B. [Shot 3] At 00:10.000, C. [Shot 4] At 00:10.000, D."}
    forge.repair_cut_times(10, segments)
    assert forge.check_prompt(forge.builder_fields(segments, "REF2VA"), "REF2VA", 10, "", 7000)


def test_typed_ref2va_validation_uses_the_typed_sections():
    from helper_minimax_h3_prompt_builder import validate_builder_state
    state = {"mode": "REF2VA", "prompt_mode": "simple", "ref": {}, "simple_prompt": "A woman waves."}
    assert validate_builder_state(state) == []
    state["simple_prompt"] = "subject_definitions:\n<Subject 1> is a woman.\nsummary:\nA short scene.\nretention_analysis:\nN/A\ndetailed_description:\n[Shot 1] She waves.\noverall_soundscape:\nRoom tone.\nnon_diegetic_music:\nN/A"
    assert validate_builder_state(state) == []
    state["simple_prompt"] = state["simple_prompt"].replace("A short scene.", "")
    assert validate_builder_state(state) == [{"level": "warn", "msg": "REF2VA summary is empty."}]
    state["prompt_mode"] = "structured"
    assert len(validate_builder_state(state)) == 2


def test_mixed_and_saved_references_keep_full_definitions(monkeypatch):
    class Backend:
        base = "https://fixture.invalid"
        def can_see(self, name):
            return False
        def chat(self, *args):
            segments = {
                "Subject definitions": "<Subject 1> from <Picture 1>. <Subject 2> from <Picture 2>. <Audio 1> supplies the voice. <Video 1> supplies motion.",
                "Summary": "A ten-second scene.",
                "Retention analysis": "<Audio 1>: fully_copy. <Video 1>: reference. <Subject 2>: fully_preserved.",
                "Detailed description": "[Shot 1] <Subject 1> waves, following <Video 1>, while <Audio 1> speaks.",
                "Soundscape": "Room tone.", "Music": "N/A",
            }
            return "\n".join(f"===SEGMENT: {label}===\n{body}" for label, body in segments.items()), {}
        def unload(self, name):
            return True
    monkeypatch.setattr(forge, "backends", lambda settings: {"openai": Backend()})
    for extra in ([{"kind": "audio"}, {"kind": "video"}], [{"kind": "image", "saved_reference": True}]):
        result = forge._generate({"mode": "REF2VA", "model": "openai:fixture", "brief": "She waves.", "duration": 10, "easy": True, "references": [pic("character-1"), *extra]}, None, None, None)
        assert result["easy"] is False
        assert "<Audio 1> supplies the voice" in result["fields"]["ref"]["subject_definitions"]
        assert "<Video 1>: reference" in result["fields"]["ref"]["retention_analysis"]
        assert "<Subject 2> from <Picture 2>" in result["fields"]["ref"]["subject_definitions"]
