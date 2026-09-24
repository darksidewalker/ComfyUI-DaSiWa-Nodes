"""MiniMax H3 Forge: write a Director prompt with a local LLM before the run.

The Forge button on the H3 Director opens an overlay; this module is the
server half. It runs outside the ComfyUI queue on purpose: the prompt is
written and reviewed first, the LLM is unloaded, and only then does the person
press Run. The LLM and the video model are never resident together.

The prompting is PromptForge's h3 prompting, exported by its
scripts/export-h3-forge.mjs into data/h3_forge.json. Every system prompt and
every ladder rule comes from that bundle; this file only assembles the user
message, calls the model, and splits its ===SEGMENT: output into the node's
builder fields. Prompt-only: no tool calls.

Backend: Ollama on loopback, the same fixed URL nodes_llm.py uses.
"""

import asyncio
import base64
import io
import json
import os
import re
from urllib import error as urlerror
from urllib import request as urlrequest

try:
    from .helper_logging import log_dasiwa
except ImportError:  # pragma: no cover - direct test import
    from helper_logging import log_dasiwa

OLLAMA_URL = "http://127.0.0.1:11434"
BUNDLE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "h3_forge.json")
BASE_MODES = ("T2VA", "I2VA", "FL2VA", "L2VA")
IMAGE_MAX_EDGE = 1024
# Seven thousand characters is the H3 prompt ceiling; ~2.2 chars/token for
# this kind of prose with headroom for the segment markers.
NUM_PREDICT = 3500

# Models this module loaded into Ollama, so the Director can make sure they
# are gone before it runs even if a request was cut off mid-generation.
_FORGE_MODELS = set()

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


def format_references(references, mode):
    """Director label lines for each reference, and the labels of the pictures."""
    counters = {"Picture": 0, "Video": 0, "Audio": 0}
    lines, pictures = [], []
    base_mode = mode in BASE_MODES
    for ref in references:
        kind = ref.get("kind")
        labels = _STREAM_EMITS.get(ref.get("stream"), ["Video"]) if kind == "video" else [_KIND_LABEL.get(kind, "Picture")]
        video_index = counters["Video"] + 1 if "Video" in labels else None
        for n, label in enumerate(labels):
            counters[label] += 1
            tag = f"<{label} {counters[label]}>"
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
            else:
                role = ref.get("role")
                note = (_BASE_NOTE.get(role) if base_mode else None) or _ROLE_NOTE.get(role)
                bits.append(note(tag) if note else f"role: {role or 'UNLABELLED'}")
            length = _format_duration(ref.get("duration_seconds"))
            if length:
                bits.append(f"length: {length}")
            if first and ref.get("keep"):
                bits.append(f"keep: {ref['keep']}")
            if first and ref.get("drop"):
                bits.append(f"drop: {ref['drop']}")
            lines.append("- " + " · ".join(bits))
    return lines, pictures


# ── The user message: the h3 path of PromptForge's buildUserMessage ───────

def build_user_message(bundle, brief, mode, duration, detail, creativity, references, carries_image):
    lines = [f'Brief: "{str(brief).strip()}"']
    settings = [f"Creativity: {title_case(creativity)}", f"Mode: {mode}"]
    if duration:
        settings.append(f"Duration: {duration} sec")
    lines.append(f"Settings (context for how to write, never text to include): {' · '.join(settings)}")

    preset = bundle["creativity_presets"].get(creativity)
    if preset and preset.get("rule"):
        lines.append(f"Creativity - {title_case(creativity)}. {preset['rule']}")
        if any(r.get("kind") == "image" for r in references):
            lines.append(
                "A reference picture is attached. What it supplies, in the role it was given, stays exactly as "
                "the picture shows it at every Creativity setting. Creativity decides only what happens - the "
                "action, the camera, the cuts and the sound - never what is already in the picture."
            )

    table = bundle["detail_levels"]
    entry = table.get(str(detail)) or table.get(str(bundle["default_detail"]))
    if entry and entry.get("rule"):
        level = detail if str(detail) in table else bundle["default_detail"]
        lines.append(f"Detail level {level} of {len(table)} - {entry.get('label', level)}. {entry['rule']}")

    if references:
        ref_lines, pictures = format_references(references, mode)
        lines += ["", "References:", *ref_lines]
        labels = [tag for _ref, tag in pictures]
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


class ForgeError(Exception):
    def __init__(self, code, message, raw=None):
        super().__init__(message)
        self.code, self.message, self.raw = code, message, raw


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
        raise ForgeError("missing_segments", f"The model left out: {', '.join(missing)}.", raw)
    return segments


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


# ── Ollama ────────────────────────────────────────────────────────────────

def _ollama(path, payload=None, timeout=10):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urlrequest.Request(OLLAMA_URL + path, data=data, headers={"Content-Type": "application/json"})
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode() or "{}")


def list_models():
    models = []
    for m in _ollama("/api/tags").get("models", []):
        models.append({"name": m["name"], "size": m.get("size"), "parameters": (m.get("details") or {}).get("parameter_size")})
    return models


def can_see(model):
    try:
        caps = _ollama("/api/show", {"model": model}).get("capabilities") or []
    except Exception:
        return False
    return "vision" in caps


def unload(model):
    """Ask Ollama to drop a model now. Returns True when it is no longer loaded."""
    try:
        _ollama("/api/generate", {"model": model, "keep_alive": 0}, timeout=30)
    except Exception as exc:
        log_dasiwa("H3 Forge", f"unload of {model} failed: {exc}")
    return model not in loaded_models()


def loaded_models():
    try:
        return {m["name"] for m in _ollama("/api/ps").get("models", [])}
    except Exception:
        return set()


def unload_forge_models():
    """Backstop for the Director: make sure no Forge model is still resident."""
    if not _FORGE_MODELS:
        return
    for model in list(_FORGE_MODELS & loaded_models()):
        log_dasiwa("H3 Forge", f"{model} was still loaded; unloading before the Director runs")
        unload(model)
    _FORGE_MODELS.clear()


def _image_b64(path):
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((IMAGE_MAX_EDGE, IMAGE_MAX_EDGE))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def generate(body, input_directory=None, release_memory=None):
    """One Forge run. Blocking: call it off the event loop."""
    bundle = load_bundle()
    mode = body.get("mode")
    if mode not in bundle["modes"]:
        raise ForgeError("bad_mode", f"Forge does not write {mode or 'this mode'} prompts.")
    brief = str(body.get("brief") or "").strip()
    if not brief:
        raise ForgeError("no_brief", "Write what the clip should be first.")
    model = str(body.get("model") or "").strip()
    if not model:
        raise ForgeError("no_model", "Pick a model.")
    creativity = body.get("creativity") or bundle["default_creativity"]
    if creativity not in bundle["creativity_presets"]:
        creativity = bundle["default_creativity"]
    detail = body.get("detail") or bundle["default_detail"]
    duration = body.get("duration")
    references = [r for r in (body.get("references") or []) if isinstance(r, dict)]

    images = []
    sees = can_see(model)
    if sees and input_directory:
        from .helper_minimax_h3_director import resolve_input_path
        for ref in references:
            if ref.get("kind") == "image" and ref.get("path"):
                images.append(_image_b64(resolve_input_path(ref["path"], input_directory)))

    spec = bundle["modes"][mode]
    user = build_user_message(bundle, brief, mode, duration, detail, creativity, references, bool(images))
    sampling = bundle["creativity_presets"][creativity]
    message = {"role": "user", "content": user}
    if images:
        message["images"] = images

    if release_memory:
        release_memory()

    _FORGE_MODELS.add(model)
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": 0,
        "messages": [{"role": "system", "content": spec["system"]}, message],
        "options": {
            "num_ctx": int(body.get("num_ctx") or bundle["context_length"]),
            "num_predict": NUM_PREDICT,
            "temperature": sampling.get("temperature", 0.7),
            "top_p": sampling.get("top_p", 0.8),
        },
    }
    try:
        result = _ollama("/api/chat", payload, timeout=int(body.get("timeout") or 600))
    except urlerror.HTTPError as exc:
        detail_text = exc.read().decode(errors="replace")[:400]
        raise ForgeError("backend", f"Ollama returned {exc.code}: {detail_text}")
    except (urlerror.URLError, TimeoutError, OSError) as exc:
        raise ForgeError("backend", f"Could not reach Ollama at {OLLAMA_URL}: {exc}")
    finally:
        # keep_alive 0 already asked for this; confirm rather than assume.
        still = model in loaded_models() and not unload(model)
        if not still:
            _FORGE_MODELS.discard(model)

    raw = (result.get("message") or {}).get("content") or ""
    segments = parse_segments(raw, spec["segments"])
    fields = builder_fields(segments, mode)
    simple = simple_prompt(fields, mode, duration)
    return {
        "mode": mode,
        "fields": fields,
        "simple_prompt": simple,
        "warnings": check_prompt(fields, mode, duration, simple, bundle["max_output_chars"]),
        "model": model,
        "saw_images": len(images),
        "vision": sees,
        "unloaded": model not in loaded_models(),
        "stats": {
            "prompt_tokens": result.get("prompt_eval_count"),
            "output_tokens": result.get("eval_count"),
            "seconds": round((result.get("total_duration") or 0) / 1e9, 1),
        },
        "raw": raw,
    }


# ── Routes ────────────────────────────────────────────────────────────────

def register_routes():
    try:
        import folder_paths
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return
    server = getattr(PromptServer, "instance", None)
    if server is None:
        return

    def _release():
        from .nodes_llm import _release_all_model_memory
        _release_all_model_memory()

    @server.routes.get("/dasiwa/h3/forge/models")
    async def forge_models(_request):
        try:
            models = await asyncio.to_thread(list_models)
        except Exception as exc:
            return web.json_response({"error": "backend", "message": f"Ollama is not reachable at {OLLAMA_URL}: {exc}"}, status=502)
        bundle = load_bundle()
        return web.json_response({
            "models": models,
            "detail_levels": {k: v.get("label") for k, v in bundle["detail_levels"].items()},
            "creativity": list(bundle["creativity_presets"].keys()),
            "default_detail": bundle["default_detail"],
            "default_creativity": bundle["default_creativity"],
        })

    @server.routes.post("/dasiwa/h3/forge")
    async def forge(request):
        # Freeing memory while a workflow is sampling would pull its models
        # out from under it, so a busy queue is a refusal, not a wait.
        if server.prompt_queue.get_tasks_remaining() > 0:
            return web.json_response({"error": "busy", "message": "A workflow is running. Forge the prompt before you queue, or wait for it to finish."}, status=409)
        try:
            body = await request.json()
            result = await asyncio.to_thread(generate, body, folder_paths.get_input_directory(), _release)
        except ForgeError as exc:
            return web.json_response({"error": exc.code, "message": exc.message, "raw": exc.raw}, status=422)
        except Exception as exc:
            log_dasiwa("H3 Forge", f"failed: {exc}")
            return web.json_response({"error": "internal", "message": str(exc)}, status=500)
        log_dasiwa("H3 Forge", f"{result['model']} wrote a {result['mode']} prompt in {result['stats']['seconds']}s, unloaded={result['unloaded']}")
        return web.json_response(result)


register_routes()
