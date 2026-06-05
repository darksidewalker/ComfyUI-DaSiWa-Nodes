"""DaSiWa LTX-2 LoRA Loader - LoRA Stacker for Video & Audio Branches

LTX-2.3 is unique because it generates video AND audio from a single model. 
The transformer has completely separate branches for each:
  - video keys (attn1, attn2, ff)
  - audio keys (audio_attn1, audio_attn2, audio_ff, cross-modal attention)

This node exploits that architecture with 10 independent LoRA slots, each with:
  - lora_str: master LoRA strength (applied to both branches)
  - vs: video branch multiplier (0.0-2.0, default 1.0)
  - as: audio branch multiplier (0.0-2.0, default 1.0)

Effective video strength = lora_str * vs
Effective audio strength = lora_str * as
"""

import os
import json
import folder_paths
import comfy.utils
import comfy.lora
try:
    from comfy.lora import load_lora_for_models as _load_lora
except (ImportError, AttributeError):
    from comfy.sd import load_lora_for_models as _load_lora
from aiohttp import web
from server import PromptServer

NUM_SLOTS = 10


def _is_audio_key(k):
    """Keys containing 'audio' = audio branch"""
    return "audio" in k.lower()


def _apply_slot(model, clip, lora_name, lora_str, vs, as_):
    """Apply a single LoRA slot to model and clip"""
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not lora_path or not os.path.isfile(lora_path):
        print(f"[DaSiWa LTX2] LoRA not found: {lora_name}")
        return model, clip

    weights = comfy.utils.load_torch_file(lora_path, safe_load=True)

    video_weights = {k: v for k, v in weights.items() if not _is_audio_key(k)}
    audio_weights = {k: v for k, v in weights.items() if _is_audio_key(k)}

    v_final = lora_str * vs
    a_final = lora_str * as_

    print(f"[DaSiWa LTX2] '{lora_name}' V:{len(video_weights)}@{v_final:.2f}  A:{len(audio_weights)}@{a_final:.2f}")

    if video_weights and v_final != 0.0:
        model, clip = _load_lora(model, clip, video_weights, v_final, v_final)
    if audio_weights and a_final != 0.0:
        model, clip = _load_lora(model, clip, audio_weights, a_final, a_final)

    return model, clip


# ── Key count endpoint ────────────────────────────────────────────────────────
@PromptServer.instance.routes.get("/dasiwa/ltx2/keycounts")
async def keycounts(request):
    """API endpoint to get video/audio key counts from a LoRA file"""
    lora_name = request.rel_url.query.get("lora", "")
    if not lora_name:
        return web.json_response({"v": 0, "a": 0})
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not lora_path or not os.path.isfile(lora_path):
        return web.json_response({"v": 0, "a": 0})
    try:
        import safetensors
        with safetensors.safe_open(lora_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
    except Exception:
        try:
            weights = comfy.utils.load_torch_file(lora_path, safe_load=True)
            keys = list(weights.keys())
        except Exception:
            return web.json_response({"v": -1, "a": -1})
    v = sum(1 for k in keys if not _is_audio_key(k))
    a = sum(1 for k in keys if _is_audio_key(k))
    return web.json_response({"v": v, "a": a})


# ── Node ──────────────────────────────────────────────────────────────────────
class DaSiWa_LTX2LoraLoader:
    """
    DaSiWa LTX-2 LoRA Loader
    
    10-slot LoRA stacker with independent video & audio branch control.
    Ideal for LTX-2.3 workflows where you need fine-grained control over
    how LoRAs affect video and audio generation independently.
    """
    DESCRIPTION = (
        "DaSiWa LTX-2 LoRA Loader: stacks multiple LoRAs for LTX video/audio models.\n"
        "Each slot has a master strength plus separate video and audio multipliers, "
        "so one LoRA can affect the visual branch, audio branch, or both."
    )

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL", {"description": "Base model that will receive the active LoRA stack."}),
                "clip": ("CLIP", {"description": "CLIP/text encoder paired with the model; LoRA weights are applied when compatible."}),
                "stack_data": ("STRING", {"default": "[]", "multiline": False, "description": "JSON-encoded LoRA slot data managed by the custom UI. Each slot stores on/off, LoRA file, master strength, video multiplier, and audio multiplier."}),
            },
            "hidden": {"available_loras": (lora_list, {"description": "Internal list of LoRA files used by the custom slot picker."})}
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply_stack"
    CATEGORY = "loaders/lora"

    def apply_stack(self, model, clip, stack_data, available_loras=None):
        """Parse and apply the LoRA stack"""
        m, c = model, clip
        try:
            data = json.loads(stack_data)
        except Exception as e:
            print(f"[DaSiWa LTX2] Failed to parse stack_data: {e}")
            return (m, c)
        
        for i, row in enumerate(data):
            if not row.get("on") or row.get("lora") in ("None", "", None):
                continue
            lora_str = float(row.get("str", 1.0))
            vs = float(row.get("vs", 1.0))
            as_ = float(row.get("as", 1.0))
            m, c = _apply_slot(m, c, row["lora"], lora_str, vs, as_)
        
        return (m, c)
