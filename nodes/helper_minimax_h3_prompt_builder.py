# MiniMax H3 Prompt Builder Helper
"""
Pure contracts for MiniMax H3 prompt building. Adapted from ComfyUI-Fantastic-MiniMaxH3-PromptBuilder.
"""

# Mode definitions
MODES = ["T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"]

# Capacity per mode
MODE_CAPACITY = {
    "T2VA": {"Picture": 0, "Video": 0, "Audio": 0},
    "I2VA": {"Picture": 1, "Video": 0, "Audio": 0},
    "FL2VA": {"Picture": 2, "Video": 0, "Audio": 0},
    "L2VA": {"Picture": 1, "Video": 0, "Audio": 0},
    "REF2VA": {"Picture": 9, "Video": 3, "Audio": 3, "total": 12},
}

# Standard task and marker lists
TASK_TYPES = [
    "keyframe completion",
    "reference generation",
    "video editing",
    "video continuation",
    "audio reuse",
    "audio reference",
]

VISUAL_MARKERS = [
    "fully_preserved",
    "partially_preserved",
    "attribute_transfer",
    "weak_reference",
]
AUDIO_MARKERS = ["fully_copy", "partially_copy", "reference", "weak_reference"]

# Prompt-assembly styles. "structured" is the default and matches the official
# sectioned format; "simple" flattens the same fields into one plain block.
PROMPT_MODES = ("simple", "structured")


def align_frame_count(n: int) -> int:
    """Snap frame count to 17k+5 grid."""
    n = max(5, n)
    while n % 17 != 5:
        n += 1
    return n


def snapped_seconds(seconds: float) -> float:
    """Frame-aligned duration in seconds."""
    frames = align_frame_count(int(seconds * 24))
    return frames / 24.0


def fmt_ss(seconds: float) -> str:
    """Format snapped seconds (e.g. '5.10')."""
    return f"{round(snapped_seconds(seconds) * 100) / 100:.2f}"


def default_builder_state(mode: str = "T2VA") -> dict:
    """Return a fresh prompt-builder state tree for a given mode."""
    if mode == "REF2VA":
        return {
            "version": 2,
            "mode": mode,
            "duration": 5,
            "ref": {
                "subject_definitions": "",
                "summary": "",
                "retention_analysis": "",
                "detailed_description": "",
                "soundscape": "",
                "music": "",
            },
        }
    return {
        "version": 1,
        "mode": mode,
        "imd": "",
        "soundscape": "",
        "music": "",
        "duration": 5,
        "ref": {
            "subject_defs": [],
            "summary_types": ["reference generation"],
            "summary_text": "",
            "retention": [],
            "style_line": "",
            "detail": "",
            "soundscape": "",
            "music": "",
        },
    }


def _base_field_values(state: dict) -> dict:
    """Derive the base-mode (T2VA/I2VA/FL2VA/L2VA) prompt fields."""
    imd_raw = state.get("imd")
    imd = imd_raw.strip() if isinstance(imd_raw, str) else ""
    soundscape_raw = state.get("soundscape")
    soundscape = soundscape_raw.strip() if isinstance(soundscape_raw, str) else ""
    music_raw = state.get("music")
    music = (music_raw.strip() if isinstance(music_raw, str) else "") or "N/A"
    return {"imd": imd, "soundscape": soundscape, "music": music}


def build_base_prompt(state: dict) -> str:
    """Generate prompt for T2VA/I2VA/FL2VA/L2VA using official guide format."""
    mode = state.get("mode", "T2VA")
    duration_s = state.get("duration", 5)
    v = _base_field_values(state)

    S = fmt_ss(duration_s)
    head = ""

    if mode == "I2VA":
        head = ("For the target video, at 0.00 seconds into the target video, "
                "<Picture 1> (from [Shot 1]) is fully referenced.")
    elif mode == "FL2VA":
        head = ("How the reference pictures align with the target video — "
                f"Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
                f"Picture 2 (from Shot 2) aligns with the {S}-second mark of the target video.")
    elif mode == "L2VA":
        head = ("How the reference pictures align with the target video — "
                f"<Picture 1> (from [Shot 1]) aligns with the {S}-second mark of the target video.")

    body = (
        f"integrated_multimodal_description: {v['imd']}\n\n"
        f"overall_soundscape: {v['soundscape']}\n\n"
        f"non_diegetic_music: {v['music']}"
    )
    return f"{head}\n\n{body}" if head else body


def _ref_field_values(state: dict) -> dict:
    """Derive the six REF2VA prompt fields from v1 and/or v2 builder keys.

    This is the single source of truth for REF2VA field derivation so the
    structured and simple renderers (and prompt_payload) can never disagree.
    """
    ref = state.get("ref", {})

    subject_definitions = _ensure_str(ref.get("subject_definitions"))
    if not subject_definitions:
        defs_raw = ref.get("subject_defs") or []
        if isinstance(defs_raw, list):
            subject_definitions = "\n".join(_ensure_str(d["text"]) for d in defs_raw if isinstance(d, dict) and _ensure_str(d.get("text")))

    summary = _ensure_str(ref.get("summary"))
    if not summary:
        chosen = [t for t in TASK_TYPES if t in ref.get("summary_types", [])]
        types_str = " + ".join(chosen) or "reference generation"
        summary_text = _ensure_str(ref.get("summary_text"))
        if summary_text:
            summary = f"[{types_str}] {summary_text}"

    retention_analysis = _ensure_str(ref.get("retention_analysis"))
    if not retention_analysis:
        retention_rows = []
        for row in ref.get("retention", []):
            label = _ensure_str(row.get("label"))
            context = _ensure_str(row.get("context"))
            marker = _ensure_str(row.get("marker"))
            note = _ensure_str(row.get("note"))
            if not label or not marker:
                continue
            ctx_part = f" ({context})" if context else ""
            retention_rows.append(f"{label}{ctx_part}: {marker} - {note}")
        retention_analysis = "\n".join(retention_rows)

    detailed_description = _ensure_str(ref.get("detailed_description"))
    if not detailed_description:
        style_line = _ensure_str(ref.get("style_line"))
        detail = _ensure_str(ref.get("detail"))
        parts = [p for p in [style_line, detail] if p]
        detailed_description = "\n".join(parts)

    soundscape = _ensure_str(ref.get("soundscape"))
    music_raw = ref.get("music")
    music = (music_raw.strip() if isinstance(music_raw, str) else "") or "N/A"

    return {
        "subject_definitions": subject_definitions,
        "summary": summary,
        "retention_analysis": retention_analysis,
        "detailed_description": detailed_description,
        "soundscape": soundscape,
        "music": music,
    }


def build_ref_prompt(state: dict) -> str:
    """Generate the structured (sectioned) REF2VA prompt."""
    v = _ref_field_values(state)
    return (
        f"subject_definitions:\n{v['subject_definitions']}\n\n"
        f"summary:\n{v['summary']}\n\n"
        f"retention_analysis:\n{v['retention_analysis']}\n\n"
        f"detailed_description:\n{v['detailed_description']}\n\n"
        f"overall_soundscape:\n{v['soundscape']}\n\n"
        f"non_diegetic_music:\n{v['music']}"
    )


def _prompt_mode(state: dict) -> str:
    """Resolve the validated prompt-assembly style from the builder state."""
    mode = state.get("prompt_mode") or "structured"
    if not isinstance(mode, str):
        return "structured"
    mode = mode.strip().lower()
    return mode if mode in PROMPT_MODES else "structured"


def _simple_lines(state: dict) -> list[tuple[str, str]]:
    """Collect the (header, value) pairs for the flat "simple" rendering.

    Reuses the same field derivation as the structured renderers so both
    styles always carry identical content.
    """
    if state.get("mode", "T2VA") == "REF2VA":
        v = _ref_field_values(state)
        return [
            ("subject_definitions", v["subject_definitions"]),
            ("summary", v["summary"]),
            ("retention_analysis", v["retention_analysis"]),
            ("detailed_description", v["detailed_description"]),
            ("overall_soundscape", v["soundscape"]),
            ("non_diegetic_music", v["music"]),
        ]
    v = _base_field_values(state)
    return [
        ("integrated_multimodal_description", v["imd"]),
        ("overall_soundscape", v["soundscape"]),
        ("non_diegetic_music", v["music"]),
    ]


def _build_simple_prompt(state: dict) -> str:
    """Render the saved one-field simple prompt, with a legacy flat fallback.

    New simple-mode workflows store their editable text in ``simple_prompt``.
    Older simple-mode workflows have only the structured field values, so keep
    flattening those fields rather than discarding their prompt on load.
    """
    simple_prompt = state.get("simple_prompt")
    if isinstance(simple_prompt, str):
        return simple_prompt.strip()
    lines = [f"{header}: {value}" for header, value in _simple_lines(state) if value]
    return "\n".join(lines)


def has_builder_content(state: dict) -> bool:
    """Return whether a builder state contains authored prompt text."""
    if not isinstance(state, dict):
        return False
    if _ensure_str(state.get("simple_prompt")):
        return True
    if _ensure_str(state.get("imd")) or _ensure_str(state.get("soundscape")) or _ensure_str(state.get("music")) not in ("", "N/A"):
        return True
    ref = state.get("ref")
    return isinstance(ref, dict) and (
        any(_ensure_str(ref.get(key)) for key in (
            "subject_definitions", "summary", "retention_analysis",
            "detailed_description", "soundscape",
        )) or _ensure_str(ref.get("music")) not in ("", "N/A") or bool(ref.get("subject_defs")) or bool(ref.get("retention"))
    )


def migrate_legacy_prompt(builder: dict, timeline_state: dict, widget_prompt: object = "") -> bool:
    """Move a pre-builder Director prompt into lossless Simple-prompt state."""
    if not isinstance(builder, dict) or has_builder_content(builder):
        return False
    state = timeline_state if isinstance(timeline_state, dict) else {}
    raw_blocks = state.get("prompt_blocks")
    blocks = raw_blocks if isinstance(raw_blocks, list) else []
    ordered_blocks = sorted(enumerate(blocks), key=lambda pair: (
        float(pair[1].get("start", 0)) if isinstance(pair[1], dict) else 0,
        int(pair[1].get("order", pair[0])) if isinstance(pair[1], dict) else pair[0],
        pair[0],
    ))
    legacy_global = widget_prompt if _ensure_str(widget_prompt) else state.get("prompt")
    candidates = [
        state.get("resolved_prompt"),
        state.get("full_prompt"),
        "\n".join([_ensure_str(legacy_global), *[
            _ensure_str(block.get("text")) for _, block in ordered_blocks
            if isinstance(block, dict) and block.get("enabled", True)
        ]]),
    ]
    for candidate in candidates:
        text = _ensure_str(candidate)
        if text:
            builder["prompt_mode"] = "simple"
            builder["simple_prompt"] = text
            return True
    return False


def build_prompt(state: dict) -> str:
    """Mode-dispatched prompt assembly honoring the requested style.

    ``prompt_mode`` selects the assembly style: "structured" (default) uses the
    official sectioned layout; "simple" flattens the same fields into one
    block. Both render identical content, so the toggle never loses data.
    """
    if _prompt_mode(state) == "simple":
        return _build_simple_prompt(state)
    mode = state.get("mode", "T2VA")
    return build_ref_prompt(state) if mode == "REF2VA" else build_base_prompt(state)


def _ensure_str(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def normalize_ref_schema(ref: dict) -> None:
    """Ensure both v1 and v2 keys exist in ref for cross-mode compatibility.

    V1 keys (used by prompt_payload and legacy nodes):
        subject_defs, summary_types, summary_text, retention, style_line, detail

    V2 keys (used by REF2VA prompt builder UI):
        subject_definitions, summary, retention_analysis, detailed_description

    Mutates ref in-place; prefers existing values over derived ones.
    """
    # --- v2 ← v1 (for REF2VA builders expecting flat strings) ---

    if not _ensure_str(ref.get("subject_definitions")):
        defs_raw = ref.get("subject_defs") or []
        if isinstance(defs_raw, list):
            texts = [_ensure_str(d["text"]) for d in defs_raw if isinstance(d, dict) and _ensure_str(d.get("text"))]
            if texts:
                ref["subject_definitions"] = "\n".join(texts)

    if not _ensure_str(ref.get("summary")):
        summary_text = _ensure_str(ref.get("summary_text"))
        summary_types = ref.get("summary_types", ["reference generation"])
        if summary_text:
            chosen = [t for t in TASK_TYPES if t in summary_types]
            types_str = " + ".join(chosen) or "reference generation"
            ref["summary"] = f"[{types_str}] {summary_text}"

    if not _ensure_str(ref.get("retention_analysis")):
        rows = []
        for row in ref.get("retention", []):
            label = _ensure_str(row.get("label"))
            context = _ensure_str(row.get("context"))
            marker = _ensure_str(row.get("marker"))
            note = _ensure_str(row.get("note"))
            if not label or not marker:
                continue
            ctx = f" ({context})" if context else ""
            rows.append(f"{label}{ctx}: {marker} - {note}")
        if rows:
            ref["retention_analysis"] = "\n".join(rows)

    if not _ensure_str(ref.get("detailed_description")):
        style_line = _ensure_str(ref.get("style_line"))
        detail = _ensure_str(ref.get("detail"))
        parts = [p for p in [style_line, detail] if p]
        if parts:
            ref["detailed_description"] = "\n".join(parts)

    # --- v1 ← v2 (for prompt_payload expecting structured fields) ---

    if "subject_defs" not in ref:
        defs_text = _ensure_str(ref.get("subject_definitions"))
        ref["subject_defs"] = [{"text": line} for line in defs_text.split("\n") if line.strip()] if defs_text else []

    if "summary_text" not in ref:
        summary = _ensure_str(ref.get("summary"))
        # Strip leading "[task_type + task_type]" prefix if present.
        if summary and summary.startswith("[") and "]" in summary:
            _, rest = summary.split("]", 1)
            ref["summary_text"] = rest.lstrip()
        else:
            ref["summary_text"] = summary

    if "retention" not in ref:
        lines = (_ensure_str(ref.get("retention_analysis"))).split("\n")
        parsed = []
        for line in lines:
            line = line.strip()
            if not line or ": " not in line or " - " not in line:
                continue
            head, tail = line.rsplit(" - ", 1)
            marker, note = tail.rsplit(": ", 1) if ": " in tail else (tail, "")
            left, right = head.rsplit(": ", 1) if ": " in head else (head, "")
            label = left.strip()
            context = ""
            if "(" in label and ")" in label:
                label, ctx = label.rsplit("(", 1)
                context = ctx.rstrip(")").strip()
                label = label.strip()
            if label and marker.strip():
                parsed.append({"label": label, "context": context, "marker": marker.strip(), "note": note.strip()})
        ref["retention"] = parsed

    if "style_line" not in ref:
        dd = _ensure_str(ref.get("detailed_description"))
        first = dd.split("\n")[0].strip() if dd else ""
        ref["style_line"] = first

    if "detail" not in ref:
        dd = _ensure_str(ref.get("detailed_description"))
        rest = "\n".join(dd.split("\n")[1:]).strip() if dd else ""
        ref["detail"] = rest


def validate_builder_state(state: dict) -> list[dict]:
    """Return list of issues: {"level": "error"|"warn"|"info", "msg": str}."""
    issues = []
    mode = state.get("mode", "T2VA")

    if mode == "REF2VA":
        ref = state.get("ref", {})
        # Check both v2 and legacy v1 keys.
        has_summary = bool(_ensure_str(ref.get("summary")) or _ensure_str(ref.get("summary_text")))
        if not has_summary:
            issues.append({"level": "warn", "msg": "REF2VA summary is empty."})
        has_subjects = bool(_ensure_str(ref.get("subject_definitions"))) or bool(ref.get("subject_defs"))
        if not has_subjects:
            issues.append({"level": "warn", "msg": "REF2VA subject_definitions is empty."})

    return issues