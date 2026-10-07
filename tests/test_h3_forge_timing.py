"""Regression coverage for PromptForge's mechanical shot timing checks."""

import pytest

from nodes import h3_prompting as hp


@pytest.mark.parametrize("mode", (*hp.BASE_MODES, "REF2VA"))
@pytest.mark.parametrize("ratio", ("35:24", "16:09", "21:09"))
def test_aspect_ratio_is_not_a_shot_timestamp(mode, ratio):
    description = (
        f"2D-animated, cinematic, warm golden-hour lighting, {ratio}.\n"
        "[Shot 1] Wide view.\n[Shot 2] At 00:03.500, cut to a close-up."
    )
    fields = {"imd": description, "ref": {"detailed_description": description}}
    assert hp.check_prompt(fields, mode, 6, description, 7000) == []


@pytest.mark.parametrize("mode", ("T2VA", "REF2VA"))
@pytest.mark.parametrize("stamp, seconds", (("00:06", "6"), ("00:07.250", "7.25"), ("01:03", "63")))
def test_real_cut_at_or_past_clip_end_still_warns(mode, stamp, seconds):
    description = f"35:24 style. [Shot 1] Wide view. [Shot 2] At {stamp}, cut."
    fields = {"imd": description, "ref": {"detailed_description": description}}
    assert hp.check_prompt(fields, mode, 6, description, 7000) == [
        f"Shots run to {seconds}s but the clip is 6s. Regenerate, or fix the timestamps."
    ]


@pytest.mark.parametrize("stamp", ("00:02", "00:01", "00:00"))
def test_non_increasing_cut_times_still_warn(stamp):
    description = f"[Shot 1] Wide. [Shot 2] At 00:02, cut. [Shot 3] At {stamp}, cut."
    assert hp.check_prompt({"imd": description}, "T2VA", 6, description, 7000) == [
        "Cut timestamps must increase strictly. Edit the description before applying."
    ]


def test_non_timeline_numbers_in_single_shot_are_ignored():
    description = '35:24 style. [Shot 1] A clock reads 12:34; the sign says "16:09".'
    assert hp.check_prompt({"imd": description}, "T2VA", 6, description, 7000) == []


def test_character_limit_warning_is_preserved():
    description = "35:24 style. [Shot 1] Wide view."
    assert hp.check_prompt({"imd": description}, "T2VA", 6, description, 10) == [
        f"{len(description):,} characters; H3 takes 10. Lower Detail and regenerate."
    ]
