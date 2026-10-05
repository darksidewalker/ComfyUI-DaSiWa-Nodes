"""Shared H3 bundle, references, prompt construction and segment conversion."""

import json
import math
import os
import re

from .llm_backends import ForgeError

BUNDLE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "h3_forge.json")
BASE_MODES = ("T2VA", "I2VA", "FL2VA", "L2VA")


_bundle_cache = {"mtime": None, "data": None}




def load_bundle(path=BUNDLE_PATH):
    mtime = os.path.getmtime(path)
    if _bundle_cache["mtime"] != mtime:
        with open(path, "r", encoding="utf-8") as fh:
            _bundle_cache["data"] = json.load(fh)
        _bundle_cache["mtime"] = mtime
    return _bundle_cache["data"]


def title_case(value):
    return re.sub(r"\b\w", lambda m: m.group(0).upper(), str(value).replace("_", " "))


# ── References: a port of PromptForge's server/references.mjs ─────────────

_ROLE_NOTE = {
    "pose": lambda l: f"pose — {l} supplies stance, limb position and gesture only; not identity, clothing, background or a literal target keyframe",
    "custom": lambda l: f"custom — use {l} only as specified by its reference instructions",
    "keyframe": lambda l: f"keyframe — {l} IS a frame of the video: give it its own line in subject_definitions and retention_analysis, naming the shot and moment it anchors",
    "motion": lambda l: f"motion — {l} gives structure only: pacing, cuts, camera",
    "subject": lambda l: f"subject — define a <Subject N> from it and cite {l} as its source; no standalone {l} line",
    "style": lambda l: f"style — rendering only, not its content: define a style <Subject N> citing {l}; no standalone {l} line",
}
_BASE_NOTE = {
    "keyframe": lambda l: f"keyframe — {l} is a frame of the video; describe it where it appears",
    "subject": lambda l: f"subject — who or what appears, taken from {l}",
    "style": lambda l: f"style — rendering only, taken from {l}",
}
_KIND_LABEL = {"image": "Picture", "video": "Video", "audio": "Audio"}
_STREAM_EMITS = {"video": ["Video"], "audio": ["Audio"], "both": ["Video", "Audio"]}


def _format_duration(seconds):
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return None
    if s <= 0:
        return None
    if s < 60:
        return f"{s:.1f}s"
    mins = int(s // 60)
    return f"{mins}m {round(s - mins * 60)}s"


# Group identity is an explicit Forge reference choice, never inferred from the brief.
_COUNT_WORD = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def _images(references):
    return [r for r in references if r.get("kind") == "image"]


def validate_references(references):
    """Bound untrusted request context without rewriting user intent."""
    if references is None:
        return []
    if not isinstance(references, list) or len(references) > 32:
        raise ForgeError("bad_references", "References must be a list of at most 32 items.")
    total = 0
    for ref in references:
        if not isinstance(ref, dict) or ref.get("kind") not in _KIND_LABEL:
            raise ForgeError("bad_references", "Each reference must name an image, video or audio kind.")
        for key, limit in (("instructions", 4000), ("keep", 2000), ("drop", 2000),
                           ("path", 4096), ("subject_group", 256), ("role", 64), ("stream", 16), ("easy_role", 32)):
            if key in ref:
                value = ref[key]
                if not isinstance(value, str) or len(value) > limit:
                    raise ForgeError("bad_references", f"Reference {key} must be text of at most {limit} characters.")
                total += len(value)
        if ref.get("role") == "custom" and not ref.get("instructions", "").strip():
            raise ForgeError("bad_references", "Custom references need nonblank reference instructions.")
    if total > 48000:
        raise ForgeError("bad_references", "Reference context exceeds 48,000 characters.")
    return references


def picture_groups(references):
    """Only groups of two or more subject pictures share a reference line."""
    by_id = {}
    for n, ref in enumerate(_images(references), 1):
        group = ref.get("subject_group")
        if ref.get("role") == "subject" and isinstance(group, str) and group:
            by_id.setdefault(group, []).append(n)
    return [{"pictures": nums} for nums in by_id.values() if len(nums) >= 2]


def _group_line(group, references):
    tags = [f"<Picture {n}>" for n in group["pictures"]]
    count = _COUNT_WORD[len(tags)] if len(tags) < len(_COUNT_WORD) else str(len(tags))
    every = "both" if len(tags) == 2 else f"all {count}"
    note = f"subject — ONE subject shown in {count} pictures: define a single <Subject N> citing {every}; no standalone picture lines"
    bits = [", ".join(tags), note]
    pictures = _images(references)
    for n in group["pictures"]:
        ref = pictures[n - 1]
        if ref.get("instructions"):
            bits.append(f"instructions (<Picture {n}>): {ref['instructions']}")
        if ref.get("keep"):
            bits.append(f"keep (<Picture {n}>): {ref['keep']}")
        if ref.get("drop"):
            bits.append(f"drop (<Picture {n}>): {ref['drop']}")
    return "- " + " · ".join(bits)


def format_references(references, mode):
    """Director label lines for each reference, and the labels of the pictures."""
    paired_total = sum(r.get("kind") == "video" and r.get("stream") == "both" and not r.get("saved_reference") for r in references)
    counters = {"Picture": 0, "Video": 0, "Audio": paired_total}
    paired_audio = 0
    lines, pictures = [], []
    base_mode = mode in BASE_MODES
    grouped = {}
    if mode == "REF2VA":
        for group in picture_groups(references):
            for n in group["pictures"]:
                grouped[n] = group
    for ref in references:
        kind = ref.get("kind")
        labels = _STREAM_EMITS.get(ref.get("stream"), ["Video"]) if kind == "video" else [_KIND_LABEL.get(kind, "Picture")]
        video_index = counters["Video"] + 1 if "Video" in labels else None
        for n, label in enumerate(labels):
            if label == "Audio" and kind == "video" and ref.get("stream") == "both" and not ref.get("saved_reference"):
                paired_audio += 1
                index = paired_audio
            else:
                counters[label] += 1
                index = counters[label]
            tag = f"<{label} {index}>"
            first = n == 0
            if first and kind == "image":
                pictures.append((ref, tag))
            bits = [tag]
            if label == "Audio" and kind == "video" and video_index:
                bits.append(f"synchronized audio track of <Video {video_index}>")
            elif label == "Audio" and kind == "video":
                bits.append("audio track only; the video picture is not referenced")
            elif label == "Audio":
                bits.append("voice — audio signal")
            elif label == "Picture" and counters["Picture"] in grouped:
                # One line for the group, where its first picture falls. The
                # other members are still attached and numbered.
                group = grouped[counters["Picture"]]
                if counters["Picture"] == group["pictures"][0]:
                    lines.append(_group_line(group, references))
                continue
            else:
                role = ref.get("role")
                note = (_BASE_NOTE.get(role) if base_mode else None) or _ROLE_NOTE.get(role)
                bits.append(note(tag) if note else f"role: {role or 'UNLABELLED'}")
                if base_mode and role in ("pose", "custom"):
                    bits[-1] = f"endpoint — {tag} remains the mode's actual target frame; reference intent cannot override first/last-frame alignment"
            length = _format_duration(ref.get("duration_seconds"))
            if length:
                bits.append(f"length: {length}")
            if first and ref.get("instructions"):
                bits.append(f"instructions: {ref['instructions']}")
            if first and ref.get("keep"):
                bits.append(f"keep: {ref['keep']}")
            if first and ref.get("drop"):
                bits.append(f"drop: {ref['drop']}")
            lines.append("- " + " · ".join(bits))
    return lines, pictures


# ── The user message: the h3 path of PromptForge's buildUserMessage ───────

# PromptForge's h3 detail ladder is written for a ~10 s clip (see its
# config/models.yaml). Forge scales the word range to the clip actually
# set on the node, so Detail 6 on a 5 s clip asks for half the words instead
# of cramming a 10 s clip's worth of shots into it. Only the first "N-M words"
# is the ladder's own number; "350-500 word range" later is a quote of
# MiniMax's guide and stays.
LADDER_SECONDS = 10
# ~5,000 characters of description, leaving room for the soundscape and music
# inside H3's 7,000-character prompt.
MAX_DESCRIPTION_WORDS = 800
_WORD_RANGE = re.compile(r"(\d+)-(\d+) words")


def scale_detail_rule(rule, duration):
    try:
        factor = float(duration) / LADDER_SECONDS
    except (TypeError, ValueError):
        return rule
    if factor <= 0 or abs(factor - 1) < 0.05:
        return rule

    def scaled(m):
        lo, hi = (max(10, int(round(int(n) * factor / 5.0) * 5)) for n in m.groups())
        hi = min(max(hi, lo + 10), MAX_DESCRIPTION_WORDS)
        return f"{min(lo, hi - 10)}-{hi} words"

    out = _WORD_RANGE.sub(scaled, rule, count=1)
    # Level 7 calls its range the guide's own, which stops being true once scaled.
    return out.replace(", the reference guide's own range", f" for this {float(duration):g}-second clip")


def output_canvas_context(canvas):
    """Only Director output dimensions establish the target aspect ratio."""
    if canvas is None:
        return ("Director output canvas is unknown (possibly externally overridden). "
                "Omit aspect ratio and resolution; do not infer them from references, examples or the brief.")
    if not isinstance(canvas, dict) or any(
        type(canvas.get(key)) is not int or not 1 <= canvas[key] <= 8192
        for key in ("width", "height")
    ):
        raise ForgeError("bad_canvas", "Output canvas requires integer width and height between 1 and 8192.")
    from math import gcd
    width, height = canvas["width"], canvas["height"]
    divisor = gcd(width, height)
    return (f"Director output canvas: {width}x{height} pixels; aspect ratio {width // divisor}:{height // divisor}. "
            "These dimensions are authoritative, including over conflicting format requests in the brief. "
            "Use this aspect ratio only; reference-image dimensions and example formats are not the output canvas.")


def build_user_message(bundle, brief, mode, duration, detail, creativity, references, carries_image, attached_labels=None, output_canvas=None, cast=None, shots=None):
    """`cast` is easy mode's (easy_cast): the brief's names become tags and the
    cast block replaces the labelled picture lines; video, audio and saved
    references keep theirs. `shots` is the Shots control: "Auto" or None sends
    nothing."""
    if cast is not None:
        brief = easy_brief(brief, cast)
    lines = [f'Brief: "{str(brief).strip()}"']
    settings = [f"Creativity: {title_case(creativity)}", f"Mode: {mode}"]
    if duration:
        settings.append(f"Duration: {duration} sec")
    lines.append(f"Settings (context for how to write, never text to include): {' · '.join(settings)}")
    lines.append(output_canvas_context(output_canvas))
    if shots_line(shots):
        lines.append(shots_line(shots))

    preset = bundle["creativity_presets"].get(creativity)
    if preset and preset.get("rule"):
        lines.append(f"Creativity - {title_case(creativity)}. {preset['rule']}")
        if cast is None and carries_image and any(r.get("kind") == "image" for r in references):
            lines.append(
                "A reference picture is attached. What it supplies, in the role it was given, stays exactly as "
                "the picture shows it at every Creativity setting. Creativity decides only what happens - the "
                "action, the camera, the cuts and the sound - never what is already in the picture."
            )

    table = bundle["detail_levels"]
    entry = table.get(str(detail)) or table.get(str(bundle["default_detail"]))
    if entry and entry.get("rule"):
        level = detail if str(detail) in table else bundle["default_detail"]
        lines.append(f"Detail level {level} of {len(table)} - {entry.get('label', level)}. {scale_detail_rule(entry['rule'], duration)}")

    if references and cast is not None:
        # Labelled pictures are in the cast block; everything else keeps its
        # reference line (videos, audio, saved references without a label).
        ref_lines, pictures = format_references(references, mode)
        labelled = [tag for ref, tag in pictures if ref.get("easy_role")]
        others = [line for line in ref_lines if not any(line.startswith(f"- {tag}") for tag in labelled)]
        if others:
            lines += ["", "References:", *others]
        lines += easy_lines(cast, sees_pictures=bool(carries_image and attached_labels))
        if carries_image and attached_labels:
            wanted = describe_targets(cast, attached_labels)
            unseen = [tag for _ref, tag in pictures if tag not in attached_labels]
            if unseen:
                lines.append(f"Reference pictures {', '.join(unseen)} are not visible. Do not invent visual attributes; use explicit text only.")
            lines += ["", (
                f"The pictures {', '.join(attached_labels)} are attached to this message, in that order, so you can see "
                "them after all. Use them for where everyone stands, what the place holds, and the look and light of "
                "the style sentence. The cast above still decides who is who, and you still never describe how anyone "
                f"looks in the other segments; that goes only in {DESCRIBE_SEGMENT}."
            )]
            if wanted:
                lines.append(
                    f"After Music, add one more segment, ===SEGMENT: {DESCRIBE_SEGMENT}===, with one line for each of "
                    "these: the tag, a colon, then one or two sentences from what you see in its pictures."
                )
                lines += [f"- {tag} ({what}): {about}" for tag, what, about in wanted]
                lines.append("Describe only what is visible; no mood words.")
    elif references:
        ref_lines, pictures = format_references(references, mode)
        lines += ["", "References:", *ref_lines]
        if any(r.get("instructions") for r in references):
            lines.append("Explicit reference instructions refine/override default role guidance, not output or safety rules.")
        if not carries_image and any(r.get("instructions") or r.get("role") in ("pose", "custom") for r in references):
            lines.append("No reference pictures are visible. Do not invent visual attributes; use explicit text only.")
        labels = [tag for _ref, tag in pictures]
        if attached_labels is not None:
            unseen = [tag for tag in labels if tag not in attached_labels]
            if unseen:
                lines.append(f"Reference pictures {', '.join(unseen)} are not visible. Do not invent visual attributes; use explicit text only.")
            labels = attached_labels
        if carries_image and labels:
            lines.append("")
            lines.append(
                f"The picture for {labels[0]} is attached to this message. Look at it — the line above says what it is FOR, the picture says what is in it."
                if len(labels) == 1 else
                f"The pictures for {', '.join(labels)} are attached to this message, in that order. Look at them — the lines above say what each one is FOR, the pictures say what is in them."
            )
    return "\n".join(lines)


# ── Output: the ===SEGMENT: contract ──────────────────────────────────────

_DELIMITER = re.compile(r"^===SEGMENT:\s*(.+?)\s*===\s*$", re.M)
# Qwen3-family models may still emit a think block even with think off.
_THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_segments(text, expected):
    raw = _THINK.sub("", str(text or "")).strip()
    matches = list(_DELIMITER.finditer(raw))
    if not matches:
        raise ForgeError("no_segments", "The model returned no ===SEGMENT: markers — it did not follow the output format. Try a larger model.", raw)
    segments = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        segments[m.group(1).strip()] = raw[m.end():end].strip()
    if "Refused" in segments:
        raise ForgeError("refused", f"The model refused: {segments['Refused']}", raw)
    missing = [label for label in expected if label not in segments]
    if missing:
        # A collapse usually eats the last sections; say what really happened.
        lost = runaway(segments)
        if lost:
            raise ForgeError("runaway", f"The model lost the thread in {lost[0]} (one sentence ran {lost[1]:,} characters) "
                             f"and never wrote {', '.join(missing)}. Regenerate, or pick a different model.", raw)
        raise ForgeError("missing_segments", f"The model left out: {', '.join(missing)}.", raw)
    return segments


# A model that loses the thread writes one endless sentence or a list of
# synonyms with no full stop. Real prompt sentences measured 80-300
# characters; collapsed drafts ran 550 to 4,600 (1 Oct 2026).
RUNAWAY_SENTENCE = 500


def runaway(segments):
    """The first segment holding a sentence too long to be prose, or None."""
    for label, body in segments.items():
        if label == "Subject definitions":
            continue  # written by code in easy mode, and one line per subject otherwise
        longest = max((len(s) for s in re.split(r"(?<=[.!?])(?:[\"'”’»]|</d>)*\s+", str(body or ""))), default=0)
        if longest > RUNAWAY_SENTENCE:
            return label, longest
    return None


def _bare(body, names):
    alts = "|".join(re.escape(n) for n in names if n)
    return re.sub(rf"^\s*(?:{alts})\s*:\s*", "", str(body or ""), flags=re.I).strip()


_REF_FIELDS = [
    ("Subject definitions", "subject_definitions"),
    ("Summary", "summary"),
    ("Retention analysis", "retention_analysis"),
    ("Detailed description", "detailed_description"),
    ("Soundscape", "soundscape"),
    ("Music", "music"),
]
_OFFICIAL = {"Soundscape": "overall_soundscape", "Music": "non_diegetic_music",
             "Detailed description": "integrated_multimodal_description"}


def builder_fields(segments, mode):
    """Segment bodies as the node's builder_state fields."""
    def value(label, *extra):
        return _bare(segments.get(label), [label, _OFFICIAL.get(label), *extra])
    if mode == "REF2VA":
        return {"ref": {key: value(label, key, "overall_soundscape" if key == "soundscape" else None,
                                   "detailed_description" if key == "detailed_description" else None)
                        for label, key in _REF_FIELDS}}
    return {
        "imd": value("Detailed description"),
        "soundscape": value("Soundscape"),
        "music": value("Music"),
    }


def group_warnings(subject_definitions, references):
    """Warn only when separate Subject entries explicitly cite split group members.

    Free-form prose without picture citations is inconclusive; don't pretend
    this mechanical check can judge character identity or visual similarity.
    """
    starts = list(re.finditer(r"(?m)^\s*(?:[-*]\s*)?<Subject\s+\d+>", str(subject_definitions or ""), re.I))
    blocks = [subject_definitions[m.start():starts[i + 1].start() if i + 1 < len(starts) else None]
              for i, m in enumerate(starts)]
    warnings = []
    for group in picture_groups(references):
        required = set(group["pictures"])
        cited = [{int(n) for n in re.findall(r"<Picture\s+(\d+)>", block, re.I)} & required for block in blocks]
        if len([part for part in cited if part]) >= 2 and not any(required <= part for part in cited):
            labels = ", ".join(str(n) for n in group["pictures"])
            warnings.append(f"Pictures {labels} were grouped as one subject, but the draft defines them under separate Subjects. Review subject_definitions before applying.")
    return warnings


def media_citation_warnings(text, references):
    """Check only numbered media existence; never infer visual correctness."""
    counters = {"Picture": 0, "Video": 0, "Audio": 0}
    available = set()
    for ref in references:
        labels = _STREAM_EMITS.get(ref.get("stream"), ["Video"]) if ref.get("kind") == "video" else [_KIND_LABEL[ref["kind"]]]
        for label in labels:
            counters[label] += 1
            available.add((label.lower(), counters[label]))
    pattern = r"<(Picture|Video|Audio)\s+(\d+)>"
    citations = {(kind.title(), int(n)) for kind, n in re.findall(pattern, str(text or ""), re.I)}
    return [f"<{kind} {n}> is undefined in the current references. Review or remap this citation; source-tail frames are not numbered references."
            for kind, n in sorted(citations) if (kind.lower(), n) not in available]
# ── Easy mode: a port of PromptForge's server/easy-mode.mjs ───────────────
#
# REF2VA where the person labels each picture - Character 1-4, Place, Style,
# First frame, Last frame, Pose, Custom - and code does the bookkeeping small models get
# wrong: numbering the subjects, writing Subject definitions from the labels,
# and Retention analysis from the shots the model wrote. The model writes one
# acting line per character, Summary, the shots, Soundscape and Music.
#
# No picture goes to the writer: H3 sees them in the Director, so the prompt
# only has to say who is in which picture. Measured in PromptForge, 1 Oct
# 2026, on two characters (one in two pictures) and a place: a 9B went 0/5
# without easy mode and 5/5 with it, and a 4B 5/5 in 4-5 s a run.

# "group-21" is a picture with Characters 2 and 1 in it, 2 on the left: the
# digits are the characters, left to right. The model never sees the picture,
# so position is what tells them apart; H3 sees it and is told who is where.
GROUP_ROLES = ("group-12", "group-21", "group-13", "group-31", "group-23", "group-32", "group-123")
EASY_ROLES = tuple(f"character-{n}" for n in range(1, 33)) + ("place", "style", "first-frame", "last-frame",
              "pose", "custom") + GROUP_ROLES
EASY_MODE = "REF2VA easy"
# Asked for only when the writer sees the pictures: a one or two sentence
# description per labelled picture, by what it was labelled as, folded into
# Retention analysis. Pose and custom pictures carry their own instructions.
DESCRIBE_SEGMENT = "Descriptions"
_DESCRIBE = {
    "character": ("a character", "what they look like - face, hair, build, outfit and anything they carry."),
    "place": ("the place", "the background - the setting, its landmarks, the light and the time of day."),
    "style": ("the style", "how it is drawn - the medium, linework, colour and shading."),
}
_DESCRIBE_FRAME = "a general description of the whole picture - who and what is in it, where, the framing and the light."
_DESCRIBE_POSE = "the pose only - stance, where the arms and legs are, the gesture and which way they face; not who it is or what they wear."


def describe_targets(cast, attached_labels=None):
    """Targets backed by attached pictures; None preserves the helper's old API."""
    visible = None if attached_labels is None else set(attached_labels)

    def attached(numbers):
        return visible is None or any(f"<Picture {n}>" in visible for n in numbers)

    out = [(s["tag"], *_DESCRIBE[s["kind"]]) for s in cast["subjects"]
           if s["kind"] in _DESCRIBE and attached(s["pictures"] + [p["picture"] for p in s.get("placements", [])])]
    out += [(f"<Picture {f['picture']}>", f"the {f['which']} frame", _DESCRIBE_FRAME)
            for f in cast["frames"] if attached([f["picture"]])]
    out += [(f"<Picture {u['picture']}>", "a pose", _DESCRIBE_POSE)
            for u in cast["uses"] if u["which"] == "pose" and attached([u["picture"]])]
    return out


def easy_vision_spec(spec, attached_labels):
    """A per-call contract: never mutate the exported bundle or blind fallback."""
    if not attached_labels:
        return spec
    system = spec["system"]
    # Replace the blind rule itself, rather than appending an instruction that
    # contradicts it. Fail clearly if a future export changes this contract.
    pattern = r"You have not seen the pictures,.*?Never name a medium yourself[^\n]*"
    system, count = re.subn(pattern, (
        "You can see only the attached pictures listed in the message. Use visible evidence "
        "for composition, spatial relationships, lighting and the overall style sentence, "
        "including the medium when visible. Do not infer visual attributes from unattached "
        "pictures. Keep character appearance out of the other segments: record it only in "
        "Descriptions. If no visible source establishes a look, use the brief or say the "
        "video keeps the look of the reference pictures."
    ), system, count=1, flags=re.S)
    if count != 1:
        raise ForgeError("bad_contract", "The labelled Forge system prompt's blind rule has changed.")
    system = system.replace(
        "Nothing about\nhow they look — the pictures carry that —",
        "Nothing about\nhow they look — visible appearance belongs only in Descriptions —"
    )
    system = system.replace(
        "Do not write Retention analysis. The app writes it from your shots.",
        "Do not write Retention analysis. The app writes it from your shots and Descriptions.\n\n"
        "**Descriptions.** After Music, write one line for each target requested in the message: "
        "its exact tag, a colon, and one or two sentences from attached pictures only. "
        "Follow each target's role; do not invent unseen details or describe unrequested targets. "
        "Describe only what is visible, with no mood words. The app folds these lines into "
        "Retention analysis (pose lines into Subject definitions), not an extra final field. "
        "If no targets are requested, write N/A."
    )
    labels = [*spec["segments"], DESCRIBE_SEGMENT]
    prefix, heading, _ = system.rpartition("## Segments to emit")
    if not heading:
        raise ForgeError("bad_contract", "The labelled Forge system prompt has no segment contract.")
    system = prefix + heading + "\n\n" + "\n".join(f"{i}. `{label}`" for i, label in enumerate(labels, 1))
    return {**spec, "system": system, "segments": labels}


_POSITIONS = {2: ("left", "right"), 3: ("left", "middle", "right")}


def _easy_role(ref):
    role = ref.get("easy_role")
    return role if role in EASY_ROLES else "character-1"


def easy_cast(references):
    """Subjects numbered characters first, then the place, then the style.

    Returns {"subjects": [{tag, kind, name, number, pictures, placements, refs}],
    "frames": [{picture, which, ref}], "uses": [{picture, which, ref}]};
    picture numbers count images only. A character's `pictures` are its own;
    `placements` are group pictures it shares, as [{picture, position}].
    `uses` are Pose and Custom pictures: no subject, only what they lend.
    A picture with no label at all (a saved reference) keeps its own
    reference line.
    """
    chars, place, style, frames, uses = {}, {"pictures": [], "refs": []}, {"pictures": [], "refs": []}, [], []

    def char(num):
        return chars.setdefault(num, {"pictures": [], "placements": [], "refs": []})

    for n, ref in enumerate(_images(references), 1):
        if not ref.get("easy_role"):
            continue
        role = _easy_role(ref)
        if role in ("pose", "custom"):
            uses.append({"picture": n, "which": role, "ref": ref})
            continue
        if role.startswith("group-"):
            nums = [int(d) for d in role.split("-", 1)[1]]
            for num, position in zip(nums, _POSITIONS[len(nums)]):
                char(num)["placements"].append({"picture": n, "position": position})
                char(num)["refs"].append(ref)
            continue
        if role.startswith("character-"):
            entry = char(int(role.rsplit("-", 1)[1]))
        elif role in ("place", "style"):
            entry = place if role == "place" else style
        else:
            frames.append({"picture": n, "which": "first" if role == "first-frame" else "last", "ref": ref})
            continue
        entry["pictures"].append(n)
        entry["refs"].append(ref)
    subjects = [{"kind": "character", "name": f"Character {num}", "number": num, **chars[num]} for num in sorted(chars)]
    if place["pictures"]:
        subjects.append({"kind": "place", "name": "the place", **place})
    if style["pictures"]:
        subjects.append({"kind": "style", "name": "the style", **style})
    for i, s in enumerate(subjects, 1):
        s["tag"] = f"<Subject {i}>"
    return {"subjects": subjects, "frames": frames, "uses": uses}


def _join_and(items):
    return " and ".join(items) if len(items) <= 2 else f"{', '.join(items[:-1])} and {items[-1]}"


def _picture_list(pictures):
    return _join_and([f"<Picture {p}>" for p in pictures])


def _shown_in(s):
    """Where a subject is shown: "in <Picture 1>, and on the left in <Picture 3>"."""
    own = f"in {_picture_list(s['pictures'])}" if s["pictures"] else ""
    shared = _join_and([f"{'in' if p['position'] == 'middle' else 'on'} the {p['position']} in <Picture {p['picture']}>"
                        for p in s.get("placements", [])])
    return f"{own}, and {shared}" if own and shared else own or shared


def easy_brief(brief, cast):
    """"Character 2" -> <Subject 2>, "the place" -> the place's tag, "picture 3"
    -> <Picture 3>. Only what the cast has; a bare "place" ("takes place") stays."""
    by_number = {s["number"]: s["tag"] for s in cast["subjects"] if s["kind"] == "character"}
    place = next((s["tag"] for s in cast["subjects"] if s["kind"] == "place"), None)
    last = max([p for s in cast["subjects"] for p in s["pictures"]]
               + [p["picture"] for s in cast["subjects"] for p in s.get("placements", [])]
               + [f["picture"] for f in cast["frames"] + cast["uses"]] + [0])
    text = re.sub(r"\b(?:character|char)\s*#?\s*(\d+)\b",
                  lambda m: by_number.get(int(m.group(1)), m.group(0)), str(brief), flags=re.I)
    if place:
        text = re.sub(r"\bthe (?:place|scenery|location)\b", place, text, flags=re.I)
    return re.sub(r"(?<!<)\b(?:picture|pic|image)\s*#?\s*(\d+)\b(?!>)",
                  lambda m: f"<Picture {int(m.group(1))}>" if 1 <= int(m.group(1)) <= last else m.group(0), text, flags=re.I)


def _notes(refs):
    def joined(key):
        seen = []
        for r in refs:
            value = str(r.get(key) or "").strip()
            if value and value not in seen:
                seen.append(value)
        return "; ".join(seen)
    return joined("instructions"), joined("keep"), joined("drop")


def _tail(refs):
    """The person's own notes on these pictures, for the end of a cast line."""
    instructions, keep, drop = _notes(refs)
    extra = " · ".join(x for x in (instructions and f"instructions: {instructions}", keep and f"keep: {keep}",
                                    drop and f"leave out: {drop}") if x)
    return f" · {extra}" if extra else ""


def easy_lines(cast, sees_pictures=False):
    """The cast block, in place of the per-picture reference lines."""
    lines = ["", "Cast (fixed). The person labelled every picture, and Subject definitions are already written from those labels. "
             "Use exactly these subjects: add none, merge none, split none. The video model sees the pictures itself, "
             "so never describe how anyone or anything looks."]
    if sees_pictures:
        lines[-1] = ("Cast (fixed). The person labelled every picture, and Subject definitions are already written from those labels. "
                     "Use exactly these subjects: add none, merge none, split none. "
                     "Describe visible reference appearance only in Descriptions; use attached sources only.")
    for s in cast["subjects"]:
        tail = _tail(s["refs"])
        pics = _picture_list(s["pictures"])
        if s["kind"] == "character":
            lines.append(f'- {s["tag"]} is "{s["name"]}" in the brief, a character, shown {_shown_in(s)}{tail}')
        elif s["kind"] == "place":
            lines.append(f"- {s['tag']} is the place, shown in {pics}. The shots happen here; anyone else in it is part of the place, not a subject{tail}")
        else:
            lines.append(f"- {s['tag']} is the rendering style, from {pics}: how the video looks, not what is in it{tail}")
    for f in cast["frames"]:
        lines.append((f"- <Picture {f['picture']}> is the first frame: [Shot 1] opens on it exactly" if f["which"] == "first"
                      else f"- <Picture {f['picture']}> is the last frame: the final shot ends on it exactly") + _tail([f["ref"]]))
    for u in cast["uses"]:
        lines.append((f"- <Picture {u['picture']}> gives a pose only: stance, limb position and gesture; not who anyone is, "
                      "their clothes or the background" if u["which"] == "pose"
                      else f"- <Picture {u['picture']}> is used only as its instructions say") + _tail([u["ref"]]))
    return lines


def _shots(description):
    parts = re.split(r"\[Shot (\d+)\]", str(description or ""))
    shots = {}
    for i in range(1, len(parts), 2):
        shots[int(parts[i])] = shots.get(int(parts[i]), "") + (parts[i + 1] if i + 1 < len(parts) else "")
    return shots


# A subject label opens a line, or follows a finished sentence with a colon or
# dash: models sometimes run every acting line into one paragraph. A mention
# inside a sentence ("smiles at <Subject 1>,") is not a label.
def _label_pattern(tags):
    return re.compile(rf"(?m)^\s*(?:[-*•]\s*)?(<(?:{tags}) \d+>)\s*(?:[:—–-]\s*)?|(?<=[.!?])[ \t]+(<(?:{tags}) \d+>)\s*[:—–]\s*")


_LABEL = _label_pattern("Subject")
# Descriptions also name first and last frames by picture.
_DESCRIBE_LABEL = _label_pattern("Subject|Picture")


def _acting(body, label=_LABEL):
    """The model's lines by tag: "<Subject 1>: ...", "- <Subject 1> — ..."."""
    text = str(body or "")
    marks = list(label.finditer(text))
    out = {}
    for i, m in enumerate(marks):
        tag = m.group(1) or m.group(2)
        line = " ".join(text[m.end():marks[i + 1].start() if i + 1 < len(marks) else len(text)].split())
        if line:
            out[tag] = f"{out[tag]} {line}" if tag in out else line
    return out


def _span(nums):
    if not nums:
        return None
    lo, hi = min(nums), max(nums)
    return f"[Shot {lo}]" if lo == hi else f"[Shot {lo}]-[Shot {hi}]"


def _sentence(text):
    t = str(text or "").strip()
    return t if not t or t[-1] in ".!?" else f"{t}."


# Words that name how a video is made. In easy mode the writer has not seen the
# pictures, so it cannot know any of these; a 4B wrote "Live-action, cinematic"
# for anime pictures despite being told not to.
_MEDIUM = re.compile(r"\b(live[- ]action|photo-?real\w*|realistic|anime|cartoon|animated|animation|2d|3d|cgi|pixel art|"
                     r"watercolou?r|oil painting|claymation|stop[- ]motion|cel[- ]shad\w*|film grain|cinematic)\b", re.I)


def keep_reference_look(description, brief):
    """The style line names no medium the idea did not: it says the pictures' look."""
    text = str(description or "")
    head, sep, rest = text.partition("[Shot")
    found = {m.lower() for m in _MEDIUM.findall(head)}
    asked = {m.lower() for m in _MEDIUM.findall(str(brief or ""))}
    if not sep or not head.strip() or not (found - asked):
        return text
    return "Keeps the look of the reference pictures.\n\n" + sep + rest


def easy_segments(cast, segments, description_targets=None):
    """Subject definitions and Retention analysis written in code, into the
    parsed segments. Returns the warnings (a character no shot names)."""
    acting = _acting(segments.get("Subject definitions"))
    # Written only when the writer saw the pictures: one or two sentences per
    # labelled picture, added to its retention line (a pose's to its definition).
    description_body = segments.pop(DESCRIBE_SEGMENT, "")
    descriptions = {tag: _sentence(line) for tag, line in _acting(description_body, _DESCRIBE_LABEL).items()
                    if not re.fullmatch(r"N/A[.!]?", line.strip(), re.I)}
    description_warnings = []
    if description_targets is not None:
        expected = set(description_targets)
        descriptions = {tag: text for tag, text in descriptions.items() if tag in expected}
        missing = [tag for tag in description_targets if tag not in descriptions]
        if missing:
            description_warnings.append(
                "Descriptions are missing or empty for: " + ", ".join(missing) +
                ". The base draft is available, but its visual descriptions are incomplete; regenerate or review before applying."
            )

    def described(tag):
        return f" As the pictures show: {descriptions[tag]}" if descriptions.get(tag) else ""
    shots = _shots(_bare(segments.get("Detailed description"), ["Detailed description", "integrated_multimodal_description"]))
    every = sorted(shots)
    last = every[-1] if every else None
    definitions, retention, warnings = [], [], description_warnings
    for s in cast["subjects"]:
        pics = _picture_list(s["pictures"])
        cited = [n for n in every if s["tag"] in shots[n]]
        if s["kind"] == "character":
            act = _sentence(acting.get(s["tag"]))
            definitions.append(f"{s['tag']} is the character {_shown_in(s)}; keep their appearance exactly as the pictures show."
                               + (f" In this scene: {act}" if act else ""))
            if every and not cited:
                warnings.append(f"{s['name']} ({s['tag']}) is never named in a shot. Check detailed_description, or say in the idea what {s['name']} does.")
            where = _span(cited) or (_span(every) if every else "every shot")
            retention.append(f"{s['tag']} (appears in {where}): fully_preserved — hold the same face, hair, build and outfit as its Subject definition in every shot."
                             + described(s["tag"]))
        elif s["kind"] == "place":
            definitions.append(f"{s['tag']} is the place in {pics}, where the video happens; keep it as the picture shows.")
            retention.append(f"{s['tag']} (appears in {_span(every) if every else 'every shot'}): fully_preserved — hold the same layout, landmarks, light and time of day."
                             + described(s["tag"]))
        else:
            definitions.append(f"{s['tag']} is the rendering style of {pics}, applied to the whole video and not its content.")
            retention.append(f"{s['tag']} (applies to every shot): fully_preserved — hold the same rendering throughout." + described(s["tag"]))
    for f in cast["frames"]:
        shot = "[Shot 1]" if f["which"] == "first" else (f"[Shot {last}]" if last else "the final shot")
        tag = f"<Picture {f['picture']}>"
        definitions.append(f"{tag} is the {f['which']} frame of {shot}.")
        retention.append(f"{tag} ({shot} {f['which']} frame): fully_preserved — the "
                         f"{'opening' if f['which'] == 'first' else 'closing'} composition, lighting and subject positions." + described(tag))
    for u in cast["uses"]:
        said = _sentence(str(u["ref"].get("instructions") or "").strip())
        said = said[:1].upper() + said[1:]
        definitions.append((f"<Picture {u['picture']}> gives the pose only: stance, limb position and gesture, "
                            "not identity, clothing or background." + (f" {said}" if said else "")
                            + described(f"<Picture {u['picture']}>"))
                           if u["which"] == "pose" else f"<Picture {u['picture']}>: {said}")
    segments["Subject definitions"] = "\n\n".join(definitions)
    segments["Retention analysis"] = "\n".join(retention)
    return warnings


def music_request_warning(bundle, brief, segments):
    """Flag possibly unsolicited music without deleting multilingual requests."""
    pattern = bundle.get("music_words")
    if not pattern or "Music" not in segments or re.search(pattern, str(brief or ""), re.I):
        return False
    if re.match(r"^\s*(?:non_diegetic_music:\s*)?N/A\s*$", segments["Music"] or "", re.I):
        return False
    return "The draft includes music; check that it matches your request, or set Music to N/A."


# ── Shots: a picked count, cut times that can play, and the count checked ──
# A port of PromptForge's server/shots.mjs. Measured there on eight models:
# "single shot" in the idea still came back as two or three shots on the small
# ones, and with a count picked every model wrote exactly that many. The small
# ones also put the last cut ON the final second ("At 00:10.000" in a 10 s
# clip), a shot that never plays; code moves those, since arithmetic about its
# own output is not something a small model does reliably.

def shot_count(shots):
    """The picked count as an int, or None for Auto / nothing picked."""
    try:
        n = int(shots)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def fold_shot_briefs(brief, shots, rows):
    """The idea with one "Shot N: ..." line per filled shot box. Port of
    foldShotBriefs in PromptForge's server/shots.mjs: empty boxes are skipped
    and keep their number, boxes past the picked count are dropped, and Auto
    folds nothing. Runs before anything reads the idea."""
    text = str(brief or "").strip()
    n = shot_count(shots)
    rows = rows[:n] if n and isinstance(rows, list) else []
    lines = [f"Shot {i}: {str(row).strip()}" for i, row in enumerate(rows, 1) if str(row or "").strip()]
    return "\n".join([text, *lines]).strip()


def shots_line(shots):
    """The instruction for the user message, or None when nothing was picked."""
    n = shot_count(shots)
    if not n:
        return None
    override = "This overrides any shot count in the detail level, the idea or the guide's examples."
    if n == 1:
        return f"Shots: 1 — one continuous shot. Write only [Shot 1]: no cuts and no later timestamps. {override}"
    return f"Shots: {n} — exactly {n} shots, [Shot 1] to [Shot {n}], no more and no fewer. {override}"


_CUT = re.compile(r"(\[Shot\s+\d+\]\s*At\s+)(\d+):(\d+(?:\.\d+)?)", re.I)


def _stamp(seconds):
    t = math.floor(seconds * 1000 + 0.5) / 1000
    return f"{int(t // 60):02d}:{t % 60:06.3f}"


def repair_cut_times(duration, segments):
    """Move cuts at or past the end of the clip, or not after the cut before
    them, evenly into the gap they belong in: [1.8, 4.2, 7.6, 10] in 10 s
    becomes [..., 7.6, 8.8]. Only the timestamp changes. Edits `segments` in
    place and returns the (from, to) pairs it moved."""
    try:
        end = float(duration)
    except (TypeError, ValueError):
        return []
    body = segments.get("Detailed description")
    if not body or end <= 0:
        return []
    cuts = [int(m.group(2)) * 60 + float(m.group(3)) for m in _CUT.finditer(body)]
    fixed = list(cuts)
    prev, i = 0.0, 0
    while i < len(cuts):
        if prev < cuts[i] < end:
            prev, i = cuts[i], i + 1
            continue
        # A run of bad cuts, up to the next one that is good where it stands.
        j = i
        while j < len(cuts) and not (prev < cuts[j] < end):
            j += 1
        upper = cuts[j] if j < len(cuts) else end
        for k in range(i, j):
            fixed[k] = prev + (upper - prev) * (k - i + 1) / (j - i + 1)
        prev, i = fixed[j - 1], j
    moved = [(_stamp(a), _stamp(b)) for a, b in zip(cuts, fixed) if _stamp(a) != _stamp(b)]
    if not moved:
        return []
    remaining = iter(fixed)

    def put(m):
        now = next(remaining)
        was = int(m.group(2)) * 60 + float(m.group(3))
        return m.group(0) if _stamp(was) == _stamp(now) else f"{m.group(1)}{_stamp(now)}"

    segments["Detailed description"] = _CUT.sub(put, body)
    return moved


def shot_count_warning(shots, segments):
    """A warning when the model wrote a different number of shots than was
    picked, or None."""
    n = shot_count(shots)
    body = segments.get("Detailed description")
    if not n or body is None:
        return None
    got = len(set(re.findall(r"\[Shot\s+(\d+)\]", body, re.I)))
    if got == n:
        return None
    return (f"You asked for {n} shot{'' if n == 1 else 's'} and the model wrote {got}. "
            "Regenerate, or edit the description before you apply it.")


# ── Simple prompt mode: a port of PromptForge's server/h3-simple.mjs ──────

_TIMESTAMP = re.compile(r"\b(\d{1,2}):(\d{2})(?:\.(\d+))?\b")


def check_prompt(fields, mode, duration, prompt_text, limit):
    """Mechanical checks on a finished prompt. Warnings, never repairs."""
    warnings = []
    description = fields["ref"]["detailed_description"] if mode == "REF2VA" else fields["imd"]
    try:
        clip = float(duration)
    except (TypeError, ValueError):
        clip = None
    if clip:
        stamps = [int(m) * 60 + int(s) + float(f"0.{frac}" if frac else 0)
                  for m, s, frac in _TIMESTAMP.findall(description)]
        if stamps and max(stamps) >= clip:
            warnings.append(f"Shots run to {max(stamps):g}s but the clip is {clip:g}s. Regenerate, or fix the timestamps.")
        cuts = [int(m.group(2)) * 60 + float(m.group(3)) for m in _CUT.finditer(description)]
        if any(b <= a for a, b in zip([0.0, *cuts], cuts)):
            warnings.append("Cut timestamps must increase strictly. Edit the description before applying.")
    if len(prompt_text) > limit:
        warnings.append(f"{len(prompt_text):,} characters; H3 takes {limit:,}. Lower Detail and regenerate.")
    return warnings


def _snapped_seconds_text(seconds):
    try:
        n = float(seconds)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    frames = max(5, int(n * 24))
    while frames % 17 != 5:
        frames += 1
    return f"{round(frames / 24, 2):.2f}"


def _last_shot(description):
    shots = [int(n) for n in re.findall(r"\[Shot\s+(\d+)\]", str(description or ""), re.I)]
    return max(shots) if shots else 1


def _alignment_line(mode, duration, shot):
    if mode == "I2VA":
        return "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
    if mode not in ("FL2VA", "L2VA"):
        return ""
    s = _snapped_seconds_text(duration)
    if s is None:
        return ""
    if mode == "FL2VA":
        return ("How the reference pictures align with the target video — "
                "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
                f"Picture 2 (from Shot {shot}) aligns with the {s}-second mark of the target video.")
    return ("How the reference pictures align with the target video — "
            f"<Picture 1> (from [Shot {shot}]) aligns with the {s}-second mark of the target video.")


def simple_prompt(fields, mode, duration):
    if mode == "REF2VA":
        ref = fields["ref"]
        names = [("subject_definitions", "subject_definitions"), ("summary", "summary"),
                 ("retention_analysis", "retention_analysis"), ("detailed_description", "detailed_description"),
                 ("soundscape", "overall_soundscape"), ("music", "non_diegetic_music")]
        return "\n\n".join(f"{name}:\n{ref[key] or ('N/A' if key == 'music' else '')}" for key, name in names)
    body = "\n\n".join([
        f"integrated_multimodal_description: {fields['imd']}",
        f"overall_soundscape: {fields['soundscape']}",
        f"non_diegetic_music: {fields['music'] or 'N/A'}",
    ])
    head = _alignment_line(mode, duration, _last_shot(fields["imd"]))
    return f"{head}\n\n{body}" if head else body


