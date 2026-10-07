"""LoRA info backend for the Advanced LoRA Loader's "info" button.

Ports the rgthree Power Lora Loader info feature, scoped to this nodepack:
sha256 of the LoRA file, safetensors header metadata (trigger words), and a
Civitai lookup by sha256, cached next to the nodepack in lorainfo/<sha256>.json.
"""

import hashlib
import json
import os
from pathlib import Path, PureWindowsPath

import folder_paths
import aiohttp
from server import PromptServer

CHUNK_SIZE = 128 * 1024
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lorainfo")
CIVITAI_BY_HASH_URL = "https://civitai.com/api/v1/model-versions/by-hash/{}"
CIVITAI_BY_HASH_URL_RED = "https://civitai.red/api/v1/model-versions/by-hash/{}"
CIVITAI_BASE_COM = "https://civitai.com"
CIVITAI_BASE_RED = "https://civitai.red"


def civitai_base_url(domain: str) -> str:
    """Canonical base URL for a Civitai domain ("red" -> .red, anything else -> .com)."""
    return CIVITAI_BASE_RED if domain == "red" else CIVITAI_BASE_COM


def sha256_file(path: str) -> str:
    """Chunked sha256 of a (possibly large) LoRA file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(block)
    return h.hexdigest()


def header_metadata(path: str) -> dict:
    """Read safetensors __metadata__ from the file header without torch.

    Returns {} for non-safetensors / unreadable files. String values that are
    themselves JSON objects (the standard ss_* fields) are parsed in place.
    """
    try:
        with open(path, "rb") as f:
            size = int.from_bytes(f.read(8), "little", signed=False)
            if size <= 0:
                return {}
            header = json.loads(f.read(size))
    except Exception:
        return {}
    md = header.get("__metadata__") or {}
    if not isinstance(md, dict):
        return {}
    for key, value in list(md.items()):
        if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
            try:
                md[key] = json.loads(value)
            except Exception:
                pass
    return md


def trained_words_from_metadata(md: dict) -> list:
    """ss_tag_frequency ({bucket: {word: count}}) -> [{word, count}], aggregated.

    Buckets come from different training stages ("sks:045", "sks:090", ...);
    per-word counts are summed across buckets.
    """
    freq = md.get("ss_tag_frequency")
    if isinstance(freq, str):
        try:
            freq = json.loads(freq)
        except Exception:
            freq = None
    words = {}
    if isinstance(freq, dict):
        for bucket in freq.values():
            if not isinstance(bucket, dict):
                continue
            for tag, count in bucket.items():
                entry = words.setdefault(tag, {"word": tag, "count": 0})
                try:
                    entry["count"] += int(count)
                except (TypeError, ValueError):
                    pass
    return list(words.values())


def merge_civitai(info: dict, civ) -> bool:
    """Merge a Civitai model-versions API response into `info`. Returns True if changed.

    `civ` may be None (model not found on Civitai) — then nothing is touched.
    """
    if not civ:
        return False
    changed = False
    model = civ.get("model") if isinstance(civ.get("model"), dict) else {}
    if "name" not in info and civ.get("name"):
        model_name = model.get("name", "")
        version_name = civ["name"]
        info["name"] = f"{model_name} - {version_name}" if model_name else version_name
        changed = True
    for key in ("type", "baseModel"):
        value = civ.get(key) or model.get(key)
        if key not in info and value:
            info[key] = value
            changed = True
    word_map = {w["word"]: w for w in info.get("trainedWords", []) if isinstance(w, dict)}
    merged = False
    for word in list(civ.get("triggerWords", [])) + list(civ.get("trainedWords", [])):
        if not isinstance(word, str) or not word:
            continue
        entry = word_map.setdefault(word, {"word": word})
        entry["civitai"] = True
        merged = True
    if merged:
        for w in word_map.values():
            w.setdefault("count", 0)
        info["trainedWords"] = sorted(word_map.values(), key=lambda w: -w["count"])
        changed = True
    if civ.get("modelId") or civ.get("id"):
        # A model version exists on both mirrors with the same id/modelId
        # (confirmed byte-identical by-hash responses), so one lookup yields a
        # valid link for each domain. The panel shows both side by side.
        tail = f"?modelVersionId={civ['id']}" if civ.get("id") else ""
        link_tail = f"/models/{civ.get('modelId', '')}{tail}"
        info["links"] = info.get("links", []) + [
            f"{CIVITAI_BASE_COM}{link_tail}",
            f"{CIVITAI_BASE_RED}{link_tail}",
        ]
        changed = True
    if civ.get("images"):
        existing_urls = {im.get("url") for im in info.get("images", []) if isinstance(im, dict)}
        for img in civ["images"]:
            if not isinstance(img, dict):
                continue
            url = img.get("url")
            if not url or url in existing_urls:
                continue
            meta = img.get("meta") or {}
            info.setdefault("images", []).append({
                "url": url,
                "type": img.get("type"),
                "width": img.get("width"),
                "height": img.get("height"),
                "seed": meta.get("seed"),
                "positive": meta.get("prompt"),
                "negative": meta.get("negativePrompt"),
                "steps": meta.get("steps"),
                "sampler": meta.get("sampler"),
                "cfg": meta.get("cfgScale"),
                "model": meta.get("Model"),
            })
            changed = True
    return changed


def cache_read(sha: str):
    """Cached info for a file sha256, or None."""
    path = os.path.join(CACHE_DIR, f"{sha}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def cache_write(sha: str, data: dict):
    """Persist info next to the nodepack. Never raises (cache is convenience)."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, f"{sha}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def fetch_civitai(url: str):
    """Fetch the Civitai by-hash endpoint. Returns the response dict or None.

    Sync + monkeypatchable; the route runs it via asyncio.to_thread so a slow
    network never blocks the event loop.
    """
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-DaSiWa-Nodes/0.4"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _inside_lora_roots(path):
    """Resolve symlinks before checking the operator-configured LoRA roots."""
    try:
        target = Path(path).resolve()
        return any(target.is_relative_to(Path(root).resolve())
                   for root in folder_paths.get_folder_paths("loras"))
    except (OSError, ValueError, RuntimeError):
        return False


def _lora_path(lora_name):
    if not isinstance(lora_name, str) or not lora_name or "\x00" in lora_name:
        return None
    relative = PureWindowsPath(lora_name)
    if relative.drive or relative.root or ".." in relative.parts:
        return None
    path = folder_paths.get_full_path("loras", lora_name)
    return path if path and _inside_lora_roots(path) and os.path.isfile(path) else None


def _local_image_path(path):
    if not path or not _inside_lora_roots(path):
        return None
    for ext in ("jpg", "jpeg", "png", "webp"):
        candidate = f"{os.path.splitext(path)[0]}.{ext}"
        if _inside_lora_roots(candidate) and os.path.isfile(candidate):
            return os.path.realpath(candidate)
    return None


def _local_image_url(lora_name: str, path: str):
    """URL of a bounded sidecar image next to the LoRA, or None."""
    if _local_image_path(path):
        return f"/dasiwa/ltx2/loraimg?lora={lora_name}"
    return None


# ── Routes ────────────────────────────────────────────────────────────────────
@PromptServer.instance.routes.get("/dasiwa/ltx2/lorainfo")
async def lora_info(request):
    """LoRA info: sha256 + header metadata + cached/live Civitai data."""
    import asyncio
    lora_name = request.rel_url.query.get("lora", "")
    refresh = request.rel_url.query.get("refresh") in ("1", "true")
    path = _lora_path(lora_name)
    if path is None:
        return aiohttp.web.json_response({"status": 404, "error": "LoRA not found"}, status=404)

    sha = sha256_file(path)
    cached = None if refresh else cache_read(sha)
    cached = cached or {}
    info = {
        "file": lora_name,
        "sha256": sha,
        "imageLocal": _local_image_url(lora_name, path),
        "trainedWords": list(cached.get("trainedWords", [])),
        "images": list(cached.get("images", [])),
        "civitaiFound": bool(cached.get("civitaiFound")),
    }
    for key in ("name", "type", "baseModel"):
        if cached.get(key):
            info[key] = cached[key]
    if cached.get("links"):
        info["links"] = list(cached["links"])

    civitai = None if refresh else cached.get("civitai")
    civitai_error = None
    # Memoize *negative* results too: a "not found" is a valid outcome, so once
    # we've looked it up we must not re-hit the Civitai API on every (i) open.
    already = (not refresh) and bool(cached.get("civitaiLooked"))
    if civitai is None and not already:
        # .com first, then the .red mirror — a model published only to one of
        # them is still found (both expose the same by-hash API).
        for url in (CIVITAI_BY_HASH_URL.format(sha), CIVITAI_BY_HASH_URL_RED.format(sha)):
            civitai = await asyncio.to_thread(fetch_civitai, url)
            if civitai is not None:
                civitai["_sha256"] = sha
                civitai["_domain"] = "red" if "civitai.red" in url else "com"
                break
        civitai_error = "model not found on civitai (.com and .red)" if civitai is None else None
    elif civitai is None:
        # Memoized negative result: reuse the stored error message so re-opens
        # don't fall back to the generic "lookup unavailable".
        civitai_error = cached.get("civitaiError")

    merge_civitai(info, civitai)
    info["civitaiFound"] = bool(civitai)
    if civitai is None:
        info["civitaiError"] = civitai_error or "civitai lookup unavailable"
    metadata_words = trained_words_from_metadata(header_metadata(path))
    if metadata_words:
        known = {w.get("word") for w in info["trainedWords"]}
        for w in metadata_words:
            if w["word"] not in known:
                info["trainedWords"].append(w)
        info["trainedWords"] = sorted(info["trainedWords"], key=lambda w: -w.get("count", 0))

    # Rebuild the cache entry: fresh civitai response + merged display fields.
    # civitaiLooked flags that a by-hash lookup happened (found or not) so a
    # negative result isn't re-fetched on the next open.
    cache_write(sha, {**info, "civitai": civitai, "civitaiLooked": True, "raw": {}})
    return aiohttp.web.json_response({**info, "status": 200})


@PromptServer.instance.routes.get("/dasiwa/ltx2/loraimg")
async def lora_img(request):
    """Sidecar image next to the LoRA file (same basename, .jpg/.png/.jpeg/.webp)."""
    lora_name = request.rel_url.query.get("lora", "")
    path = _local_image_path(_lora_path(lora_name))
    if path:
        return aiohttp.web.FileResponse(path)
    return aiohttp.web.json_response({"status": 404, "error": "no image next to LoRA"}, status=404)
