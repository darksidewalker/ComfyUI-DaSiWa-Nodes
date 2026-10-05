"""Prompt specifications and text composition (no inference ownership)."""

import re
from pathlib import Path

from . import h3_prompting

# The model presets are short hand-written guides, not copies of PromptForge's
# prompts: purpose-first and editable as plain markdown in data/llm_skills/.
# H3 is the exception and stays on the Director's data/h3_forge.json.
_SKILLS_DIR = Path(__file__).resolve().parents[1] / "data" / "llm_skills"
_SKILL_PRESETS = {
    "promptforge_wan22": "wan",
    "promptforge_ltx": "ltx",
    "promptforge_krea2": "krea2",
    "promptforge_anima": "anima",
    "promptforge_illustrious": "illustrious",
}


def load_skill(name):
    """A guide's frontmatter (`key: value` lines between `---`) plus its body as the system prompt."""
    text = (_SKILLS_DIR / f"{name}.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        raise ValueError(f"Skill {name}.md has no frontmatter")
    meta = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        if value.strip():
            meta[key.strip()] = value.strip()
    if not meta.get("output"):
        raise ValueError(f"Skill {name}.md does not name its output segment")
    spec = {"model": name, "output": meta["output"], "segments": [meta["output"]],
            "system": text[match.end():].strip()}
    if meta.get("tag_style"):
        spec["tag_style"] = meta["tag_style"]
    return spec


def preset_spec(preset):
    return load_skill(_SKILL_PRESETS[preset])


def exported_system(preset):
    return preset_spec(preset)["system"]


# Underscores between words, ported from PromptForge server/tagspacing.mjs.
# "Letter or digit" is [^\W_] because Python's re has no \p{L}.
_BETWEEN = re.compile(r"(?:(?<=[^\W_])|(?<=[)\\]))_(?=[^\W_]|[(\\])")
_EMOTICON = re.compile(r"^\(?[\W_]?[^\W_]?_[^\W_]?[\W_]?\)?$")
_SCORE = re.compile(r"^\(?score_\d", re.IGNORECASE)


def space_underscores(text):
    """Respell `best_quality` as `best quality`, tag by tag, for models that want spaces.

    The model writes underscores from habit even when its prompt spells every
    tag with spaces. Score tags (`score_7`) and emoticons (`o_o`, `^_^`) keep theirs.
    """
    pieces = []
    for piece in str(text or "").split(","):
        tag = piece.strip()
        if "_" in tag and not _SCORE.match(tag) and not _EMOTICON.match(tag):
            piece = _BETWEEN.sub(" ", piece)
        pieces.append(piece)
    return ",".join(pieces)


_WEIGHT = re.compile(r":\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*$")


def escape_literal_parens(text):
    """`watercolor (medium)` becomes `watercolor \\(medium\\)`; `(rain:1.2)` stays a weight.

    Ported from PromptForge server/parens.mjs. A tag model reads every bare
    pair as a weight, so a disambiguated tag left bare steers nothing. Models
    write them bare even when the guide shows them escaped.
    """
    src = str(text or "")
    out, i = [], 0
    while i < len(src):
        ch = src[i]
        if ch == "\\" and src[i + 1:i + 2] in ("(", ")"):
            out.append(src[i:i + 2])
            i += 2
            continue
        if ch != "(":
            out.append(ch)
            i += 1
            continue
        close, nested = i + 1, False
        while close < len(src):
            if src[close] == "\\" and src[close + 1:close + 2] in ("(", ")"):
                close += 2
                continue
            if src[close] == ")":
                break
            if src[close] == "(":
                nested = True
            close += 1
        inner = None if close >= len(src) else src[i + 1:close]
        if inner is None or nested:
            out.append(ch)
            i += 1
            continue
        out.append(src[i:close + 1] if _WEIGHT.search(inner) else f"\\({inner}\\)")
        i = close + 1
    return "".join(out)


def prompt_response(preset, raw):
    spec = preset_spec(preset)
    segments = h3_prompting.parse_segments(raw, spec["segments"])
    # Small models copy the code fence the guide's example sits in.
    result = re.sub(r"^```\w*\s*|\s*```$", "", segments[spec["output"]].strip()).strip()
    if not result:
        raise ValueError("The model returned an empty prompt segment")
    if spec.get("tag_style") == "space":
        result = escape_literal_parens(space_underscores(result))
    return result


def h3_request(brief, mode, duration, image_count):
    """Build H3 only after the caller has prepared and counted sampled images."""
    if mode not in ("T2VA", "I2VA", "FL2VA", "L2VA"):
        raise ValueError("Use the Director for H3 REF2VA and continuity prompts")
    required = {"T2VA": 0, "I2VA": 1, "FL2VA": 2, "L2VA": 1}[mode]
    if image_count != required:
        raise ValueError(f"H3 {mode} requires exactly {required} sampled reference images")
    bundle = h3_prompting.load_bundle()
    references = [{"kind": "image", "role": "keyframe"} for _ in range(required)]
    user = h3_prompting.build_user_message(
        bundle, brief, mode, duration, bundle["default_detail"],
        bundle["default_creativity"], references, bool(required),
        attached_labels=[f"<Picture {n}>" for n in range(1, required + 1)],
    )
    return bundle["modes"][mode]["system"], user


def h3_response(raw, mode, duration):
    """Project H3 segments through the same final prompt conversion as Director."""
    bundle = h3_prompting.load_bundle()
    segments = h3_prompting.parse_segments(raw, bundle["modes"][mode]["segments"])
    fields = h3_prompting.builder_fields(segments, mode)
    return h3_prompting.simple_prompt(fields, mode, duration)

_BASE_OUTPUT_RULES = (
    "Return only the requested prompt or caption. Do not add markdown, labels, quotes, "
    "explanations, safety notes, or alternative versions. Preserve concrete user intent "
    "and do not invent identities, logos, readable text, or copyrighted character names "
    "unless they are explicitly present in the input."
)


def _video_prompt_preset(model_name):
    if model_name == "LTX-2.3":
        return (
            f"{_BASE_OUTPUT_RULES}\n\n"
            "You enhance an input idea and optional reference image into a single LTX-2.3 video prompt. "
            "Write one flowing cinematic paragraph, 4-8 descriptive sentences. Use present tense. "
            "Include, when relevant: shot scale, subject identity, setting, lighting, color palette, "
            "surface textures, atmosphere, core action as a beginning-to-end motion, character physical cues, "
            "camera movement relative to the subject, pacing, depth, and audio such as ambience, music, or dialogue. "
            "If dialogue is requested, put the exact spoken words in quotation marks and specify language/accent when useful. "
            "Prefer visual cues over internal emotions. Avoid overloaded multi-action scenes, conflicting lighting, and text/logo requests. "
            "If an image is provided, preserve its visual identity and describe only motion/camera/style changes that fit it."
        )
    return (
        f"{_BASE_OUTPUT_RULES}\n\n"
        "You enhance an input idea and optional reference image into a single Wan2.2 video prompt. "
        "Write a detailed but compact cinematic paragraph in natural English. Include the main subject, visual style, "
        "scene/background, pose or action, motion progression, camera framing, lighting, atmosphere, and salient details. "
        "For image-to-video, preserve the reference image identity, composition, character/object appearance, and aspect logic while adding natural motion. "
        "Make the prompt concrete enough for Wan prompt extension or direct Wan generation. Avoid vague quality spam, contradictions, and excessive unrelated details."
    )


def _caption_preset(media, detail, style):
    length_rules = {
        "simple": "Keep it short: one sentence for natural language or 8-18 tags for tag output.",
        "detailed": "Include all important visible subjects, attributes, scene, pose/action, style, composition, lighting, and mood.",
        "very_detailed": "Be exhaustive but factual: include fine-grained visual attributes, spatial relationships, materials, expression/pose, camera/framing, lighting, color palette, and notable background elements.",
    }
    media_rules = {
        "image": "Caption a single image.",
        "video": "Caption sampled video frames as one coherent clip. Include temporal changes, camera movement, subject motion, scene continuity, and recurring visual details.",
    }
    style_rules = {
        "mixed": (
            "Output a mixed caption: start with concise booru-style comma-separated tags for concrete visual attributes, "
            "then add one natural-language sentence. Use underscores in tags. Avoid unsupported artist/character names."
        ),
        "tag": (
            "Output only comma-separated booru/Danbooru-style tags. Use lowercase, underscores, no sentences, no hashtags, no scores. "
            "Order tags from most important to least: subject count/type, character/object traits, pose/action, clothing, setting, composition, style, lighting, quality/meta tags. "
            "Use tag-like phrases compatible with WD14/Pony/Illustrious-style workflows."
        ),
        "natural": (
            "Output natural language only. Write clear descriptive English suitable for FLUX, Wan, LTX, SD3, and other language-prompted image/video models. "
            "Do not use booru syntax, tag lists, weight syntax, or comma-stuffed quality tags."
        ),
    }
    return (
        f"{_BASE_OUTPUT_RULES}\n\n"
        f"{media_rules[media]} {length_rules[detail]} {style_rules[style]} "
        "Describe only what is visible or strongly implied by the connected input. "
        "If no image/video is connected, caption the connected text prompt instead."
    )


_SYSTEM_PROMPT_PRESETS = {
    "custom": "",
    "enhance_video_ltx23": _video_prompt_preset("LTX-2.3"),
    "enhance_video_wan22": _video_prompt_preset("Wan2.2"),
}

for _media in ("image", "video"):
    for _detail in ("simple", "detailed", "very_detailed"):
        for _style in ("mixed", "tag", "natural"):
            _SYSTEM_PROMPT_PRESETS[f"caption_{_media}_{_detail}_{_style}"] = _caption_preset(_media, _detail, _style)

_EXPORTED_PRESET_IDS = tuple(_SKILL_PRESETS)
_SYSTEM_PROMPT_PRESET_LABELS = list(_SYSTEM_PROMPT_PRESETS.keys()) + list(_EXPORTED_PRESET_IDS) + ["promptforge_h3"]


def _resolve_system_prompt(system_prompt_preset, system_prompt):
    if system_prompt_preset == "custom":
        return str(system_prompt or "").strip()
    if system_prompt_preset in _EXPORTED_PRESET_IDS:
        return exported_system(system_prompt_preset)
    return _SYSTEM_PROMPT_PRESETS.get(system_prompt_preset, "").strip()


def _compose_user_text(system_prompt_preset, system_prompt, prompt, text_input, *, resolve_system=True):
    """Keep text ordering; H3 callers can defer system selection until image prep."""
    final_system = _resolve_system_prompt(system_prompt_preset, system_prompt) if resolve_system else ""
    final_prompt = str(prompt or "").strip()
    connected_text = str(text_input or "").strip()

    user_parts = []
    if final_prompt:
        user_parts.append(final_prompt)
    if connected_text:
        user_parts.append(connected_text)
    return final_system, "\n\n".join(user_parts).strip()


