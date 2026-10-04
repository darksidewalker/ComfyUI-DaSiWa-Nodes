"""Legacy prompt specifications and text composition (no inference ownership)."""

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

_SYSTEM_PROMPT_PRESET_LABELS = list(_SYSTEM_PROMPT_PRESETS.keys())


def _resolve_system_prompt(system_prompt_preset, system_prompt):
    if system_prompt_preset == "custom":
        return str(system_prompt or "").strip()
    return _SYSTEM_PROMPT_PRESETS.get(system_prompt_preset, "").strip()


def _compose_user_text(system_prompt_preset, system_prompt, prompt, text_input):
    final_system = _resolve_system_prompt(system_prompt_preset, system_prompt)
    final_prompt = str(prompt or "").strip()
    connected_text = str(text_input or "").strip()

    user_parts = []
    if final_prompt:
        user_parts.append(final_prompt)
    if connected_text:
        user_parts.append(connected_text)
    return final_system, "\n\n".join(user_parts).strip()


