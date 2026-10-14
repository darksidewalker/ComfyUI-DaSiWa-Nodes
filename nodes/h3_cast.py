"""The Cast list: one card per character, and one for the place, the style and
each pose, read from their pictures.

A port of PromptForge's Cast list (server/picture-labels.mjs, 13-14 Oct 2026).
The Forge panel keeps a card per character with a description a reader fills
from their pictures - from their people box where they have one - and the
person can edit ("green eyes, not blue"). That text IS the character's look
from then on: in labelled REF2VA it goes into Subject definitions as written
and Retention's kept list is made from it; in I2VA, FL2VA and L2VA it reaches
the writer beside where each one stands. A character with no card is drafted
as before.

The place, the style and each pose get a card the same way (the owner's call,
14 Oct 2026: one revamp of how the model sees and describes the pictures).
Their text goes into that subject's definition as written; a pose card keeps
only the stance (pose_only), since other characters take the pose.

The reader is one vision call per character, with the bundle's
picture_description system prompt and the message from describe_message.
"""

import re

from .h3_people import boxes_of, crop_for
from .h3_prompting import ForgeError, _join_and, describe_where, easy_cast, place_word

# How long a card's text may be. Not cut by sentences: the person wrote it.
CAST_TEXT_MAX = 600
MAX_CHARACTERS = 32
_NON_ADULT = re.compile(r"\b(?:girls?|boys?|maiden|children|child|kids?|teens?|teenage\w*|schoolgirls?|youthful|innocent|childlike|baby[- ]faced)\b", re.I)


def validate_card_age(text):
    if _NON_ADULT.search(str(text or "")):
        raise ForgeError("bad_cast", "Cast descriptions must follow the existing adult-only prompt policy. Describe an adult and remove minor/age-reducing terms.")
    return text
_ORDINAL = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth")


def _ordinal(i):
    return _ORDINAL[i] if i < len(_ORDINAL) else f"number {i + 1}"


def _cap(text):
    return text[:1].upper() + text[1:] if text else text


# Scene cards: "place", "style", and "pose@3" for the pose in <Picture 3>.
_SCENE_KEY = re.compile(r"^(?:place|style|pose@([1-9]\d?))$")


def given_cast(raw):
    """The cards the panel sent: {character number: text, "place": text,
    "style": text, "pose@N": text}. Empty cards are not cards."""
    out = {}
    if not isinstance(raw, dict):
        return out
    for k, v in list(raw.items())[:MAX_CHARACTERS * 2]:
        text = " ".join(str(v if isinstance(v, str) else "").split())[:CAST_TEXT_MAX]
        if not text:
            continue
        validate_card_age(text)
        key = str(k)
        if _SCENE_KEY.match(key):
            out[key] = text
            continue
        try:
            n = int(key)
        except ValueError:
            continue
        if 1 <= n <= MAX_CHARACTERS:
            out[n] = text
    return out


def character_cards(given):
    """Only the characters' cards: {number: text}."""
    return {k: v for k, v in given.items() if isinstance(k, int)}


# A pose card keeps only the stance: every clause about looks goes. PromptForge's
# poseOnly (12 Oct 2026): told plainly, r4 still wrote "the woman's long
# lavender braid" into a pose, and that line is what H3 applies to the person's
# own characters. Not "eyes": "eyes closed" is part of a pose; "face" is not on
# the list, since "facing forward" is a pose.
_LOOKS = re.compile(r"\b(?:hair\w*|braid\w*|ponytails?|pigtails?|twintails?|bangs|fringe|ahoge|eyelashes|eyebrows?|irises|pupils|skin|"
                    r"freckles?|makeup|tattoos?|clothes|clothing|outfits?|dress(?:es)?|skirts?|shirts?|blouses?|jackets?|coats?|suits?|"
                    r"bodysuits?|uniforms?|armou?r\w*|sweaters?|hoodies?|turtlenecks?|gloves?|boots?|shoes|hats?|glasses|ribbons?|red|blue|"
                    r"green|yellow|purple|violet|lavender|pink|orange|silver|golden|gold|blonde?|brunette|black|white|grey|gray|brown|teal|"
                    r"cyan|magenta)\b", re.I)
_CLAUSE = re.compile(r"(,\s*|;\s*|\s+while\s+|\s+with\s+|\s+and\s+(?=(?:her|his|their|its)\b))", re.I)


def pose_only(text):
    """A pose reading with every clause about looks taken out."""
    kept = []
    for sentence in re.findall(r"[^.!?]+[.!?]*", str(text or "")):
        parts = _CLAUSE.split(re.sub(r"[.!?]+$", "", sentence.strip()))
        out, dropped = "", False
        for i in range(0, len(parts), 2):
            attached = i > 0 and re.fullmatch(r"\s+with\s+", parts[i - 1], re.I)
            if not parts[i] or _LOOKS.search(parts[i]) or (dropped and attached):
                dropped = True
                continue
            dropped = False
            out += (parts[i - 1] if out and i > 0 else " " if out else "") + parts[i]
        out = re.sub(r"^(?:while|with|and)\s+", "", out.strip(), flags=re.I)
        if len(out.split()) >= 3:
            kept.append(f"{out[0].upper()}{out[1:]}.")
    return " ".join(kept)


# ── After the reader has written ──────────────────────────────────────────

def usable_reading(text):
    """The reader's answer without a leaked think block, code fences or a
    signed-off UNAVAILABLE, or None when nothing usable is left. A 27B signs
    off a real description with UNAVAILABLE on its own line now and then, and
    a Q6 once leaked its thinking into a reader call (PromptForge, 7 and 12 Oct
    2026)."""
    t = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.S | re.I)
    t = re.sub(r"^.*</think>", "", t, flags=re.S | re.I)
    t = "\n".join(line for line in t.split("\n") if not re.fullmatch(r"\s*```[a-z]*\s*", line, re.I)).strip()
    t = re.sub(r"\n\s*unavailable\.?$", "", t, flags=re.I).strip()
    if not t or re.fullmatch(r"unavailable\.?", t, re.I):
        return None
    return t


def two_sentences(text, most=2):
    """At most `most` sentences, whatever came back; a sentence said twice
    counts once, and a "Final Answer:" label goes."""
    t = re.sub(r"\bfinal answer\s*:\s*", "", " ".join(str(text or "").split()), flags=re.I).strip()
    if not t:
        return ""
    said, parts = set(), []
    for p in re.findall(r"[^.!?]+[.!?]+(?:[\"')\]]+)?", t) or [t]:
        p = p.strip()
        if p.lower() not in said:
            said.add(p.lower())
            parts.append(p)
    out = " ".join(parts[:most])
    return out if re.search(r"[.!?][\"')\]]*$", out) else f"{out}."


_KEPT = re.compile(r"^[ \t]*\**kept\**[ \t]*:\**[ \t]*(.+)$", re.I | re.M)


def split_description(text, sentences=2):
    """The look (for the card) without the reader's closing "Kept:" line."""
    return two_sentences(_KEPT.sub("", str(text or "")), sentences)


def group_lines(text, people):
    """A group reading split per person, "1: ..." to "N: ...", left to right;
    None unless all N are there."""
    found = {}
    for raw in str(text or "").split("\n"):
        m = re.match(r"^\s*\**\s*(?:character\s*|person\s*)?(\d{1,2})\s*\**\s*[:.)\-–—]\s*\**\s*(.+)$", raw, re.I)
        if m and int(m.group(1)) not in found:
            found[int(m.group(1))] = two_sentences(m.group(2))
    lines = [found.get(i + 1) for i in range(people)]
    return lines if all(lines) else None


def traits_of(line):
    """One sentence as a list of traits: "A young man with short dark hair
    wears a white shirt." -> "short dark hair, a white shirt"."""
    t = re.sub(r"[.!\s]+$", "", str(line or "").strip())
    t = re.sub(r"^(?:an?|the)\s+(?:[\w-]+\s+){0,3}?(?:man|woman|girl|boy|person|character|child|kid|teen\w*|lady|guy|figure)\b\s*(?:with\s+)?",
               "", t, flags=re.I)
    t = re.sub(r"\s*,?\s*\b(?:and\s+)?(?:is\s+)?(?:wears|wearing|dressed in|clad in)\b\s*", ", ", t, flags=re.I)
    t = re.sub(r"\s*,\s*,\s*", ", ", re.sub(r"^[,\s]+", "", t))
    return re.sub(r",\s*and\s+", " and ", t)


def kept_from_look(text):
    """A card as Retention's kept list: "A young woman with long green hair.
    She wears a red dress." -> "long green hair, a red dress"."""
    parts = []
    for s in re.findall(r"[^.!?]+[.!?]*", str(text or "")):
        p = re.sub(r"^(?:she|he|they|it)\b[\s,]*(?:(?:also\s+)?(?:has|have|is|are)\b\s*)?", "", traits_of(s), flags=re.I)
        p = re.sub(r"^[,\s]+|[.!\s]+$", "", p)
        if p:
            parts.append(p)
    return ", ".join(parts)


def written_look(text):
    """A card's text as it goes into the prompt: as written, with a full stop."""
    validate_card_age(text)
    look = " ".join(str(text or "").split())[:CAST_TEXT_MAX]
    return look if re.search(r"[.!?][\"')\]]*$", look) else f"{look}."


# ── What the reader reads for one character ───────────────────────────────

def frame_cast(references, mode):
    """I2VA, FL2VA and L2VA: the frames as a cast, so a character in them can
    be read and carded. FL2VA's second picture is the last frame."""
    refs = []
    for i, ref in enumerate([r for r in references if r.get("kind") == "image"]):
        which = "last" if mode == "L2VA" or (mode == "FL2VA" and i == 1) else "first"
        who = ref.get("who") if isinstance(ref.get("who"), list) else []
        refs.append({**ref, "easy_role": f"{which}-frame", "picture_kinds": [f"{which}-frame"], "who": who})
    return easy_cast(refs)


def cast_for(references, mode):
    """The cast a card is read from: labelled REF2VA pictures, or the frames."""
    if mode == "REF2VA":
        return easy_cast([r for r in references if r.get("kind") == "image" and (r.get("easy_role") or isinstance(r.get("picture_kinds"), list))])
    return frame_cast(references, mode)


def _ref_of(s, picture):
    for n, ref in zip(s.get("ref_pictures", []), s["refs"]):
        if n == picture:
            return ref
    return None


def _group_pictures(cast):
    """Pictures of several characters that nobody placed: no order picked and
    no boxes. [{picture, ref, members}], members in cast order."""
    groups = {}
    for s in cast["subjects"]:
        if s["kind"] != "character":
            continue
        for p in s.get("placements", []):
            ref = _ref_of(s, p["picture"])
            crop = crop_for(boxes_of(ref or {}), s["number"])
            g = groups.setdefault(p["picture"], {"picture": p["picture"], "ref": ref, "members": [], "placed": False})
            g["members"].append(s)
            if p.get("position") or crop:
                g["placed"] = True
    return [g for g in groups.values() if not g["placed"] and len(g["members"]) > 1]


def cast_unit(cast, number):
    """What the reader reads for Character `number`: their own pictures and
    the frames they stand in, each from their box where they have one, or a
    picture of several they share with nobody placed, read as a group. None
    when no picture shows them on their own."""
    s = next((x for x in cast["subjects"] if x["kind"] == "character" and x.get("number") == number), None)
    if s is None:
        return None
    framed = []
    for f in s.get("in_frames", []):
        ref = next((fr["ref"] for fr in cast["frames"] if fr["picture"] == f["picture"]), None)
        if ref and ref.get("path"):
            framed.append({"picture": f["picture"], "ref": ref, "crop": crop_for(boxes_of(ref), number),
                           "position": f.get("position")})
    placements = {p["picture"]: p for p in s.get("placements", [])}
    own = []
    for picture, ref in zip(s.get("ref_pictures", []), s["refs"]):
        if not ref.get("path") or any(o["picture"] == picture for o in own):
            continue
        crop = crop_for(boxes_of(ref), number) if picture in placements else None
        # A picture of several people with no order and no box cannot say
        # which one this is: it would describe the others too.
        if picture in placements and not placements[picture].get("position") and not crop:
            continue
        own.append({"picture": picture, "ref": ref, "crop": crop})
    if own or framed:
        others = lambda picture: [p.get("position") for o in cast["subjects"] if o is not s and o["kind"] == "character"
                                  for p in o.get("placements", []) if p["picture"] == picture and p.get("position")]
        unit = {"kind": "character", "name": s["name"], "tag": s["tag"], "pictures": framed + own, "frames": len(framed),
                "frame_position": next((f["position"] for f in framed if f.get("position") and not f["crop"]), None),
                "placements": [{"index": len(framed) + i + 1, "position": placements[o["picture"]]["position"],
                                "others": others(o["picture"])}
                               for i, o in enumerate(own) if o["picture"] in placements and not o["crop"]]}
        return unit
    g = next((g for g in _group_pictures(cast) if s in g["members"]), None)
    if g and g["ref"] and g["ref"].get("path"):
        people = len(g["members"])
        return {"kind": "group", "name": None, "tag": f"<Picture {g['picture']}> people",
                "pictures": [{"picture": g["picture"], "ref": g["ref"], "crop": None}], "frames": 0,
                "people": people, "index": g["members"].index(s), "max_tokens": min(640, 90 * people + 120)}
    return None


_DESCRIBE_KIND = {"place": "the place", "style": "a style", "pose": "a pose"}


def _clean(value):
    return str(value or "").strip()


def describe_message(unit, brief=""):
    """The reader's message for one character, PromptForge's describeMessage:
    which person this is, the person's notes on each picture, and the brief."""
    frames = unit.get("frames", 0)
    told = _clean(brief)[:1500]
    it = f'"{unit["name"]}"' if unit.get("name") else "this pose" if unit["kind"] == "pose" else "these characters"
    brief_lines = [
        f'The person\'s brief for the video{", where this is " + it if unit.get("name") else ""}:\n"""\n{told}\n"""',
        f"Use the brief to know what you are looking at, and call things what it calls them. Where it changes how {it} looks, "
        "such as hair, eyes or clothes, describe it the way the brief says, not as the picture shows, and leave the changed "
        "traits out of Kept. Anything the brief does not mention, describe as the picture shows.",
    ] if told else []
    own = [p["ref"] for p in unit["pictures"][frames:]]

    def notes(ref):
        # What this picture should contribute counts as what to keep.
        keep = "; ".join(x for x in dict.fromkeys([_clean(ref.get("instructions")), _clean(ref.get("keep"))]) if x)
        return keep, _clean(ref.get("drop"))
    keep = list(dict.fromkeys(k for k, _ in map(notes, own) if k))
    drop = list(dict.fromkeys(d for _, d in map(notes, own) if d))

    def the_one(position):
        return f"the one {describe_where(position)}"
    placed = []
    for p in unit.get("placements", []):
        nots = " or ".join(the_one(o) for o in p.get("others", []))
        placed.append(f"The {_ordinal(p['index'] - 1)} picture shows {len(p.get('others', [])) + 1} people. Describe ONLY "
                      f"{the_one(p['position'])}{', not ' + nots if nots else ''}. Every detail you give must belong to that one person.")
    cut = [i + 1 for i, p in enumerate(unit["pictures"]) if p["crop"] and i >= frames]
    if cut:
        placed.append(f"The {_join_and([_ordinal(i - 1) for i in cut])} {'picture is' if len(cut) == 1 else 'pictures are'} cut from a "
                      "picture of several people, around this one person. Describe only the person it is cut around, not anyone showing at the edges.")
    lines = []
    if frames:
        frame_cut = any(p["crop"] for p in unit["pictures"][:frames])
        who = f"the character {describe_where(unit['frame_position'])}" if unit.get("frame_position") and not frame_cut else "the character"
        if frame_cut:
            lines += [f"The first {'picture is a frame' if frames == 1 else f'{frames} pictures are frames'} of the video, cut around the one "
                      "person to describe: their face, build, hair and what they are wearing, colours included.",
                      "Describe only the person it is cut around, not anyone showing at the edges. Every detail you give must belong to them."]
        else:
            lines.append(f"The first picture is a frame of the video, and {who} in it is the one to describe: their face, build, hair and "
                         "what they are wearing, colours included." if frames == 1 else
                         f"The first {frames} pictures are frames of the video, and {who} in them is the one to describe: their face, "
                         "build, hair and what they are wearing, colours included.")
            if unit.get("frame_position"):
                lines.append("Describe ONLY that one person, not anyone else in the frame. Every detail you give must belong to them.")
        if own and keep:
            lines.append(f"The other {'picture is' if len(own) == 1 else 'pictures are'} a reference for: {'; '.join(keep)}. Describe that "
                         f"from {'it' if len(own) == 1 else 'them'}, in place of what the frame shows, and the rest from the frame.")
        elif own:
            lines.append(f"The other {'picture shows' if len(own) == 1 else 'pictures show'} the same character.")
        if drop:
            lines.append(f"The person wants to leave out: {'; '.join(drop)}.")
        lines += placed + brief_lines
        lines.append("Describe the whole character, not only the kept part. One or two sentences.")
        return "\n".join(lines)
    lines += placed
    if unit["kind"] == "group":
        lines.append(f"It shows {unit['people']} characters. Describe each of them in turn, left to right, one line each, starting with "
                     'their number and a colon: "1:" for the one furthest left, then "2:" and so on. Each line starts with who they are - a '
                     "adult man, an adult woman - then their hair (colour, length, style), eyes, build and what they are wearing, colours "
                     "included. Never what they are doing or where they are: the video puts them in a new scene.")
    elif unit["kind"] == "character":
        lines.append("This shows a character.")
        lines.append("Describe how they look, never what they are doing, holding or sitting on, or where they are: the video puts them in a new scene.")
    else:
        lines.append(f"This shows {_DESCRIBE_KIND[unit['kind']]}.")
    if unit["kind"] == "style":
        # The medium first, and never what is in the picture (PromptForge, 15
        # Oct 2026): told only "This shows a style.", r4 described the people
        # and objects of a style picture in 10 of 18 reads; with this, 12 of 12
        # named the medium right and 1 of 12 named what is in it.
        lines.append("Start with the medium in a few words: photorealistic, 3D render, semi-realistic, anime, cel-shaded, flat 2D "
                     "illustration, painterly illustration, oil painting, watercolour, pixel art. Then the linework, the shading and "
                     "the colour palette, and light only as how soft or harsh it is. Never what is in the picture, no people, animals, "
                     "objects, places or light sources: the video has its own content and takes only the look.")
    if unit["kind"] == "pose":
        # Stance only, said twice over (PromptForge, 12 Oct 2026); pose_only
        # makes sure in code.
        lines.append("Describe only the pose: the stance, where the arms and legs are, the gesture, the tilt of the head and which way they face.")
        if unit.get("takers", 0) > 1:
            lines.append(f"It shows {unit['takers']} people posing together. Give each one's pose, calling them only \"the one on the left\", "
                         "\"the one on the right\" and so on, and say how they stand to each other.")
        lines.append("Never their hair, eyes, face, skin, body, clothes, colours or the background, not even to tell them apart: "
                     "other characters take this pose. The Kept line lists the stance and gesture only.")
    if unit["kind"] == "place" and unit.get("also_people"):
        lines.append("Describe only the setting: the location, its layout, light and time of day. Leave out the people in it; they are described separately.")
    if len(unit["pictures"]) > 1:
        lines.append(f"These {len(unit['pictures'])} pictures show the same {'character' if unit['kind'] == 'character' else 'thing'}; describe it once.")
    # A note on one of several pictures is about that picture: "her face" on a
    # close-up beside a full-body shot (PromptForge, 14 Oct 2026).
    pairs = list(map(notes, own))
    if len(own) > 1 and len(set(pairs)) > 1:
        for i, (k, d) in enumerate(pairs):
            if k:
                lines.append(f"From the {_ordinal(i)} picture, take only: {k}.")
            if d:
                lines.append(f"From the {_ordinal(i)} picture, leave out: {d}.")
        lines.append("Take everything else from the other pictures.")
    else:
        if keep:
            lines.append(f"The person wants to keep: {'; '.join(keep)}.")
        if drop:
            lines.append(f"The person wants to leave out: {'; '.join(drop)}.")
    lines += brief_lines
    lines.append(f"{unit['people']} numbered lines, one per character, then the Kept line." if unit["kind"] == "group" else "One or two sentences.")
    return "\n".join(lines)


def cast_look(text, unit):
    """One card out of what the reader said: a character's line of a group
    reading, a pose's stance only, or the two sentences."""
    if unit["kind"] == "group":
        lines = group_lines(text, unit["people"])
        return unnamed(lines[unit["index"]]) if lines else ""
    if unit["kind"] == "pose":
        return pose_only(split_description(text, 4))
    look = split_description(text)
    return unnamed(look) if unit["kind"] == "character" else look


# "Character 2 is a young woman with...", "The character on the left has...":
# the brief names them, and r4 opens with the name (14 Oct 2026). The card goes
# into the prompt as written, and H3 cannot place "Character 2", so the opening
# is said the way the reader is asked to write it.
_NAMED = re.compile(r"^\s*(?:the\s+)?character(?:\s+\d+|\s+(?:on|in|at)\s+the\s+[\w -]+?)?\s+(is|has)\s+", re.I)


def unnamed(look):
    """A character card without the name or position it opens with."""
    m = _NAMED.match(str(look or ""))
    if not m:
        return look
    rest = look[m.end():]
    return _cap(rest) if m.group(1).lower() == "is" else f"A character with {rest}"


def scene_unit(cast, key):
    """What the reader reads for the place, the style, or the pose in a picture
    ("pose@3"), or None when the pictures have no such thing."""
    m = _SCENE_KEY.match(str(key or ""))
    if not m:
        return None
    if key in ("place", "style"):
        s = next((x for x in cast["subjects"] if x["kind"] == key), None)
        if s is None:
            return None
        pictures = [{"picture": n, "ref": ref, "crop": None} for n, ref in zip(s.get("ref_pictures", []), s["refs"]) if ref.get("path")]
        return {"kind": key, "name": s["name"], "tag": s["tag"], "pictures": pictures, "frames": 0,
                # A place in a picture of the characters: the setting only.
                "also_people": key == "place" and any("character" in (r.get("picture_kinds") or []) for r in s["refs"])} if pictures else None
    n = int(m.group(1))
    use = next((u for u in cast["uses"] if u["which"] == "pose" and u["picture"] == n), None)
    if use is None or not use["ref"].get("path"):
        return None
    return {"kind": "pose", "name": None, "tag": f"<Picture {n}>", "pictures": [{"picture": n, "ref": use["ref"], "crop": None}],
            "frames": 0, "takers": len(use.get("takers") or [])}


# ── Cards at Generate ─────────────────────────────────────────────────────

def carded_tags(cast, given):
    """{tag: card text} for what in this cast has a card: characters, the
    place and the style by their subject tag, a pose by its <Picture N>, its
    text already cut to the stance."""
    out = {s["tag"]: given[s["number"]] for s in cast["subjects"] if s["kind"] == "character" and s.get("number") in given}
    for s in cast["subjects"]:
        if s["kind"] in ("place", "style") and s["kind"] in given:
            out[s["tag"]] = given[s["kind"]]
    for u in cast["uses"]:
        text = given.get(f"pose@{u['picture']}") if u["which"] == "pose" else None
        stance = pose_only(text) if text else ""
        if stance:
            out[f"<Picture {u['picture']}>"] = stance
    return out


def framed_cast_lines(cast, given):
    """The cards, for the writer in I2VA, FL2VA and L2VA: beside where each one
    stands, never "Character 2" - the brief no longer says it, and given both,
    a writer put "(Character 2)" into the prompt, which H3 cannot place
    (PromptForge, 13 Oct 2026)."""
    lines = []
    for s in cast["subjects"]:
        if s["kind"] != "character" or s.get("number") not in given:
            continue
        f = (s.get("in_frames") or [None])[0]
        who = f"the character {place_word(f['position']) + ' ' if f.get('position') else ''}in the {f['which']} frame" if f else s["name"]
        lines.append(f"- {_cap(who)}: {written_look(given[s['number']])}")
    if not lines:
        return []
    return ["", "Cast, as the person describes them. Keep everyone looking like this in every shot, and never contradict it:", *lines]

