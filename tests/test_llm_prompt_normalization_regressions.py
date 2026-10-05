"""Prompt normalization preserves weights and the user-owned prefix."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nodes import llm_prompt_presets as presets
from nodes import nodes_llm_simple as simple


@pytest.mark.parametrize("weight", ["-0.5", ".5", "+1.2", "1.", "1e-2"])
def test_numeric_weights_remain_weights_in_comfyui(weight):
    from comfy.sd1_clip import token_weights
    original = f"(rain:{weight})"
    normalized = presets.escape_literal_parens(original)
    assert normalized == original
    assert token_weights(normalized, 1.0) == [("rain", float(weight))]


@pytest.mark.parametrize("text", [r"(rain \) storm:1.2)", r"(rain \( storm:1.2)",
                                 r"(rain \(storm\):1.2)"])
def test_weights_with_escaped_parentheses_preserve_comfyui_tokens(text):
    from comfy.sd1_clip import escape_important, token_weights
    normalized = presets.escape_literal_parens(text)
    assert token_weights(escape_important(normalized), 1.0) == token_weights(
        escape_important(text), 1.0)
    assert normalized == text
    assert presets.escape_literal_parens(normalized) == normalized


def test_literal_disambiguation_with_escaped_close_is_idempotent():
    text = r"watercolor (medium \) detail)"
    expected = r"watercolor \(medium \) detail\)"
    assert presets.escape_literal_parens(text) == expected
    assert presets.escape_literal_parens(expected) == expected


@pytest.mark.parametrize("prefix", ["  my_lora  ", "my_lora,", "my_lora,  ", "my_lora\n", "  "])
def test_prefix_is_preserved_byte_for_byte(prefix):
    result = simple.assemble("rain", {}, prefix)
    assert result.startswith(prefix)
    assert "rain" in result
    assert simple.assemble("", {}, prefix) == prefix


def test_prefix_separator_does_not_duplicate_existing_comma():
    assert simple.assemble("rain", {}, "my_lora,  ") == "my_lora,  rain"
    assert simple.assemble("rain", {}, "my_lora\n") == "my_lora\nrain"


def test_deduplication_ignores_generated_comma_spacing():
    assert simple.assemble("masterpiece,cat ears,best quality", {"tag_style": "space"},
                           "masterpiece, best_quality") == "masterpiece, best_quality, cat ears"


def test_literal_and_already_escaped_parentheses_are_unchanged_on_repeat():
    text = r"watercolor (medium), 2b \(nier:automata\), (rain:-.5)"
    expected = r"watercolor \(medium\), 2b \(nier:automata\), (rain:-.5)"
    assert presets.escape_literal_parens(text) == expected
    assert presets.escape_literal_parens(expected) == expected
