from nodes import h3_prompting as hp


def test_export_preserves_director_canvas_and_shot_policy():
    bundle = hp.load_bundle()
    for spec in bundle["modes"].values():
        assert "Detail controls descriptive precision only, not shot count" in spec["system"]
        assert "If the canvas is unknown, omit aspect ratio" in spec["system"]
    for entry in bundle["detail_levels"].values():
        assert "not shot count" in entry["rule"] or "without adding cuts" in entry["rule"]


def pic(**kw):
    return {"kind": "image", "easy_role": "character-1", **kw}


def test_repaired_citations_determine_retention_before_analysis():
    cast = hp.easy_cast([pic()])
    segments = {"Detailed description": "[Shot 1] <Picture 1> runs. [Shot 2] <Picture 1> returns. [Shot 3] Empty street."}
    warnings = hp.easy_segments(cast, segments)
    assert "[Shot 1]-[Shot 2]" in segments["Retention analysis"]
    assert not any("never named" in w for w in warnings)


def test_dialogue_and_lyrics_are_not_rewritten():
    cast = hp.easy_cast([pic()])
    spoken = "<d>[English] The label reads <Picture 1>.</d>"
    segments = {"Detailed description": "[Shot 1] <Picture 1> says " + spoken}
    hp.easy_segments(cast, segments)
    assert spoken in segments["Detailed description"]
    assert "[Shot 1] <Subject 1> says" in segments["Detailed description"]


def test_five_and_thirty_two_people_survive_all_axes():
    for count in (5, 32):
        for axis in ("", "y", "z"):
            cast = hp.easy_cast([pic(picture_kinds=["character"], who=list(range(1, count + 1)), who_axis=axis)])
            assert len(cast["subjects"]) == count
            assert all(s["placements"] for s in cast["subjects"])
            assert "position 1 of" in hp._shown_in(cast["subjects"][0])


def test_empty_or_invalid_who_uses_legacy_label():
    for who in ([], [False, 0, 33, "2"]):
        cast = hp.easy_cast([pic(picture_kinds=["character"], who=who)])
        assert [s["number"] for s in cast["subjects"]] == [1]


def test_frame_only_people_have_visible_individual_description_targets():
    cast = hp.easy_cast([pic(easy_role="first-frame", picture_kinds=["first-frame"], who=[1, 2], who_axis="z")])
    targets = {tag: about for tag, _, about in hp.describe_targets(cast, ["<Picture 1>"])}
    assert set(targets) == {"<Subject 1>", "<Subject 2>", "<Picture 1>"}
    assert "nearest the camera" in targets["<Subject 1>"]
    assert "not the one behind" in targets["<Subject 1>"]
    assert hp.describe_targets(cast, []) == []
