"""Saved photo references and stable Director aliases."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

ALIAS = re.compile(r"<\s*refmod\s*_?\s*(\d+)(?:\s*:[^>]+)?\s*>", re.I)


def refmod_nodes():
    package = Path(__file__).resolve().parents[2] / "ComfyUI-MiniMaxH3Mod"
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve() == package / "nodes.py":
            return module
    if not (package / "__init__.py").is_file():
        raise RuntimeError("Install ComfyUI-MiniMaxH3Mod to use Director RefMods.")
    name = "dasiwa_refmod_dependency"
    spec = importlib.util.spec_from_file_location(name, package / "__init__.py", submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return sys.modules[name + ".nodes"]


def library():
    rn = refmod_nodes()
    result = []
    for name in rn._list_mod_names():
        meta = rn.read_refmod_meta(rn._find_mod_path(name)) or {}
        if meta.get("kind") not in ("image", "video"):
            continue
        result.append({"name": name, "kind": meta["kind"], "description": meta.get("description", ""),
                       "tokens": int(meta.get("latent_t", 1)) * (int(meta.get("latent_h", 0)) // 2) * (int(meta.get("latent_w", 0)) // 2)})
    return result


def load_selections(rows):
    if not isinstance(rows, list) or len(rows) > 8:
        raise ValueError("Director supports up to 8 RefMod slots.")
    rn = refmod_nodes() if rows else None
    mods, slots = [], set()
    for row in rows:
        slot = int(row["slot"])
        if slot < 1 or slot > 8 or slot in slots:
            raise ValueError("RefMod slots must be unique numbers from 1 to 8.")
        slots.add(slot)
        if not row.get("name") or not row.get("enabled", True):
            continue
        strength = float(row.get("strength", 1))
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError(f"RefMod {slot}: strength must be between 0 and 1.")
        if strength == 0:
            continue
        mod = copy.copy(rn._load_mod(row["name"]))
        if mod.kind not in ("image", "video"):
            raise ValueError(f"RefMod {slot}: select a standalone photo-based visual RefMod.")
        mod.director_slot = slot
        mod.custom_desc = str(row.get("description", mod.description or "")).strip()
        mods.append((mod, strength))
    return sorted(mods, key=lambda pair: pair[0].director_slot)


def active_mods(mods):
    result, slots = [], set()
    for index, (mod, strength) in enumerate(mods or []):
        strength = float(strength)
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("RefMod strength must be between 0 and 1.")
        if strength == 0:
            continue
        slot = int(getattr(mod, "director_slot", index + 1))
        if slot < 1 or slot in slots:
            raise ValueError("RefMod slot numbers must be positive and unique.")
        if mod.kind not in ("image", "video"):
            raise ValueError("Director RefMods currently supports standalone visual references.")
        slots.add(slot)
        result.append((mod, strength, slot))
    return result


def mappings(mods, images, videos):
    counts = {"image": len(images or {}), "video": len(videos or {})}
    tags, descriptions, names, lines = {}, {}, {}, []
    for mod, strength, slot in active_mods(mods):
        counts[mod.kind] += 1
        tag = f"<{'Picture' if mod.kind == 'image' else 'Video'} {counts[mod.kind]}>"
        tags[slot] = tag
        descriptions[slot] = getattr(mod, "custom_desc", mod.description or "")
        names[slot] = mod.name
        lines.append(f"<RefMod {slot}> -> {tag}: {mod.name}")
    return tags, descriptions, names, "\n".join(lines)


def translate(text, tags):
    def replace(match):
        slot = int(match[1])
        if slot not in tags:
            raise ValueError(f"<RefMod {slot}> has no active reference. Select it or remove the tag.")
        return tags[slot]
    return ALIAS.sub(replace, text) if isinstance(text, str) else text


def selection_stamp(timeline_data):
    rows = json.loads(timeline_data or "{}").get("refmods", [])
    rn = refmod_nodes() if rows else None
    return tuple((row.get("slot"), rn._file_stamp(rn._find_mod_path(row["name"])))
                 for row in rows if row.get("name") and row.get("enabled", True))


def register_routes():
    from aiohttp import web
    from server import PromptServer
    if not getattr(PromptServer, "instance", None):
        return

    @PromptServer.instance.routes.get("/dasiwa/director-refmods")
    async def list_refmods(request):
        try:
            return web.json_response({"mods": library()})
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
