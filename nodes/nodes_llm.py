"""DaSiWa local LLM / VLM nodes.

These nodes intentionally keep the model object out of ComfyUI graph outputs.
That makes the unload-after-run mode much more reliable because the graph cache
does not hold a live reference to a very large model.
"""

import gc
import json
import os
import random
import re
from urllib import error as urlerror
from urllib import request as urlrequest
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

import folder_paths
try:
    from .helper_logging import log_dasiwa
except ImportError:
    from helper_logging import log_dasiwa


_LLM_CACHE = {}
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
_IMAGE_RESAMPLING = getattr(Image, "Resampling", None)
_PIL_LANCZOS = (
    getattr(_IMAGE_RESAMPLING, "LANCZOS")
    if _IMAGE_RESAMPLING is not None
    else getattr(Image, "LANCZOS", Image.BICUBIC)
)
_RESAMPLE_FILTERS = {
    "nearest": getattr(_IMAGE_RESAMPLING, "NEAREST") if _IMAGE_RESAMPLING is not None else Image.NEAREST,
    "box": getattr(_IMAGE_RESAMPLING, "BOX") if _IMAGE_RESAMPLING is not None else getattr(Image, "BOX", Image.BILINEAR),
    "bilinear": getattr(_IMAGE_RESAMPLING, "BILINEAR") if _IMAGE_RESAMPLING is not None else Image.BILINEAR,
    "hamming": getattr(_IMAGE_RESAMPLING, "HAMMING") if _IMAGE_RESAMPLING is not None else getattr(Image, "HAMMING", Image.BICUBIC),
    "bicubic": getattr(_IMAGE_RESAMPLING, "BICUBIC") if _IMAGE_RESAMPLING is not None else Image.BICUBIC,
    "lanczos": _PIL_LANCZOS,
}

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


def _ensure_llm_folder():
    llm_dir = os.path.join(folder_paths.models_dir, "llm")
    try:
        if hasattr(folder_paths, "add_model_folder_path"):
            folder_paths.add_model_folder_path("llm", llm_dir)
    except Exception as exc:
        log_dasiwa("LLM Nodes", f"Could not register models/llm folder: {exc}")
    return llm_dir


_LLM_DIR = _ensure_llm_folder()


@dataclass(frozen=True)
class _LoadedLLM:
    model: object
    tokenizer: object
    processor: object
    is_vision: bool


def _folder_paths_for_llm():
    try:
        return folder_paths.get_folder_paths("llm")
    except Exception:
        return [_LLM_DIR]


def _is_mmproj(name):
    """A llama.cpp vision projector: half of a vision GGUF, never a model on its own."""
    return "mmproj" in os.path.basename(name).lower()


def _find_mmproj(gguf_path):
    """The mmproj file beside a GGUF model, or None.

    Vision GGUFs ship as two files (model + mmproj). They are paired by folder:
    one mmproj next to the model is taken as its own; with several, the one
    whose name shares the longest start with the model's wins.
    """
    folder = os.path.dirname(gguf_path)
    try:
        found = [n for n in os.listdir(folder) if n.lower().endswith(".gguf") and _is_mmproj(n)]
    except OSError:
        return None
    if not found:
        return None
    def plain(n):
        return re.sub(r"[^a-z0-9]", "", os.path.splitext(n)[0].lower().replace("mmproj", ""))
    model = plain(os.path.basename(gguf_path))
    if len(found) > 1:
        found.sort(key=lambda n: -len(os.path.commonprefix([plain(n), model])))
        if not plain(found[0]) or not os.path.commonprefix([plain(found[0]), model]):
            return None
    elif len([n for n in os.listdir(folder) if n.lower().endswith(".gguf") and not _is_mmproj(n)]) > 1:
        # A shared folder with several models is ambiguous without a name match.
        if not plain(found[0]) or not os.path.commonprefix([plain(found[0]), model]):
            return None
    return os.path.join(folder, found[0])


def _list_llm_models():
    models = []
    seen = set()
    for base in _folder_paths_for_llm():
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            path = os.path.join(base, name)
            if name.startswith("."):
                continue
            if os.path.isdir(path):
                if os.path.isfile(os.path.join(path, "config.json")):
                    rel = name
                else:
                    rel = None
                    ggufs = []
                    for root, _, files in os.walk(path):
                        if "config.json" in files:
                            rel = os.path.relpath(root, base)
                            break
                        ggufs += [os.path.relpath(os.path.join(root, f), base) for f in sorted(files)
                                  if f.lower().endswith(".gguf") and not _is_mmproj(f)]
                    if not rel:
                        # A folder of GGUFs: a vision model keeps its mmproj beside it.
                        for gguf in ggufs:
                            if gguf not in seen:
                                models.append(gguf)
                                seen.add(gguf)
                if rel and rel not in seen:
                    models.append(rel)
                    seen.add(rel)
            elif _is_mmproj(name):
                continue
            elif name.lower().endswith((".safetensors", ".bin", ".gguf")) and name not in seen:
                models.append(name)
                seen.add(name)
    return ["None"] + models


def _resolve_model_path(model_name, custom_path, allow_gguf=False):
    path = (custom_path or "").strip()
    if path:
        path = os.path.expanduser(path)
        if not os.path.isabs(path):
            for base in _folder_paths_for_llm():
                candidate = os.path.join(base, path)
                if os.path.exists(candidate):
                    path = candidate
                    break
        if not os.path.exists(path):
            raise FileNotFoundError(f"LLM path does not exist: {path}")
        return _normalize_model_path(path, allow_gguf=allow_gguf)

    if not model_name or model_name == "None":
        raise ValueError("Choose an already installed LLM model or provide a custom_path.")

    for base in _folder_paths_for_llm():
        candidate = os.path.join(base, model_name)
        if os.path.exists(candidate):
            return _normalize_model_path(candidate, allow_gguf=allow_gguf)

    try:
        full_path = folder_paths.get_full_path("llm", model_name)
    except Exception:
        full_path = None
    if full_path and os.path.exists(full_path):
        return _normalize_model_path(full_path, allow_gguf=allow_gguf)

    raise FileNotFoundError(
        "LLM model is not already installed. Put the complete model folder in "
        "ComfyUI/models/llm or choose an existing local model. Runtime downloads "
        "are disabled for security."
    )


def _normalize_model_path(path, allow_gguf=False):
    if os.path.isfile(path):
        lower = path.lower()
        if lower.endswith(".gguf"):
            if allow_gguf:
                return path
            raise ValueError(
                "GGUF files are not supported by the transformers backend yet. "
                "Use a Hugging Face/transformers model folder, or add llama.cpp support later."
            )
        parent = os.path.dirname(path)
        if os.path.isfile(os.path.join(parent, "config.json")):
            return parent
        raise ValueError(
            "A single weight file is not enough for transformers chat inference. "
            "Place the full model folder in ComfyUI/models/llm, including config.json "
            "and tokenizer/processor files."
        )

    if not os.path.isfile(os.path.join(path, "config.json")):
        raise ValueError(
            f"Model folder is missing config.json: {path}. "
            "Use the complete Hugging Face model folder, not only the .safetensors file."
        )
    return path


def _torch_dtype(dtype):
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    return "auto"


def _device_for_inputs(model, requested_device):
    if requested_device == "cpu":
        return torch.device("cpu")
    try:
        return next(model.parameters()).device
    except Exception:
        if torch.cuda.is_available() and requested_device in ("auto", "cuda"):
            return torch.device("cuda")
        return torch.device("cpu")


def _to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _clear_llm_cache(model_path=None):
    keys = list(_LLM_CACHE.keys())
    removed = 0
    for key in keys:
        if model_path is None or key[0] == model_path:
            loaded = _LLM_CACHE.pop(key, None)
            if loaded is not None:
                try:
                    close = getattr(loaded.model, "close", None)
                    if callable(close):
                        close()
                    else:
                        loaded.model.to("cpu")
                except Exception:
                    pass
                del loaded
                removed += 1
    _cleanup_cuda()
    return removed


def _release_all_model_memory():
    """Release DaSiWa models and ask ComfyUI to unload its managed models too."""
    _clear_llm_cache()
    try:
        import comfy.model_management as model_management
        model_management.unload_all_models()
        model_management.soft_empty_cache()
    except ImportError:
        pass
    _cleanup_cuda()


def _generation_cache_kwargs(config, use_kv_cache):
    kwargs = {"use_cache": use_kv_cache}
    implementation = config.get("kv_cache_implementation", "default")
    if not use_kv_cache or implementation == "default":
        return kwargs
    kwargs["cache_implementation"] = implementation
    if implementation == "quantized":
        kwargs["cache_config"] = {
            "backend": config.get("kv_cache_quant_backend", "quanto"),
            "nbits": config.get("kv_cache_nbits", 4),
            "residual_length": config.get("kv_cache_residual_length", 128),
        }
    return kwargs


def _load_transformers_model(config, need_vision):
    try:
        from transformers import AutoProcessor, AutoTokenizer
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError(
            "DaSiWa LLM nodes require transformers and accelerate. "
            "Install them in the ComfyUI environment with: pip install transformers accelerate"
        ) from exc

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None
    try:
        from transformers import AutoModelForVision2Seq
    except ImportError:
        AutoModelForVision2Seq = None

    model_path = config["model_path"]
    task = config["task"]
    is_vision = need_vision or task == "vision"
    cache_key = (
        model_path,
        config["device"],
        config["dtype"],
        config["quantization"],
        task,
        config["attention_implementation"],
        is_vision,
    )

    if config["cache_mode"] == "cached" and cache_key in _LLM_CACHE:
        return _LLM_CACHE[cache_key]

    dtype = _torch_dtype(config["dtype"])
    common_kwargs = {
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
    }
    if dtype != "auto":
        common_kwargs["torch_dtype"] = dtype
    else:
        common_kwargs["torch_dtype"] = "auto"

    attn = config["attention_implementation"]
    if attn != "auto":
        common_kwargs["attn_implementation"] = attn

    device = config["device"]
    quantization = config["quantization"]
    if device == "auto" or quantization in ("8bit", "4bit"):
        common_kwargs["device_map"] = "auto"

    if quantization in ("8bit", "4bit"):
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "8-bit and 4-bit LLM loading requires bitsandbytes. "
                "Install it or choose quantization='none'."
            ) from exc
        common_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=quantization == "8bit",
            load_in_4bit=quantization == "4bit",
        )

    tokenizer = None
    processor = None
    if is_vision:
        try:
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=False,
            )
            tokenizer = getattr(processor, "tokenizer", None)
        except Exception as exc:
            log_dasiwa("LLM Nodes", f"AutoProcessor load failed, trying tokenizer only: {exc}")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=False,
        )

    model_classes = []
    if is_vision:
        model_classes.extend([AutoModelForImageTextToText, AutoModelForVision2Seq, AutoModelForCausalLM])
    else:
        model_classes.extend([AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForVision2Seq])
    model_classes = [cls for cls in model_classes if cls is not None]

    last_error = None
    model = None
    for model_cls in model_classes:
        try:
            model = model_cls.from_pretrained(model_path, **common_kwargs)
            break
        except Exception as exc:
            last_error = exc
    if model is None:
        raise RuntimeError(f"Could not load model with transformers: {last_error}") from last_error

    if "device_map" not in common_kwargs and device in ("cuda", "cpu"):
        target = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        model.to(target)

    model.eval()
    loaded = _LoadedLLM(model=model, tokenizer=tokenizer, processor=processor, is_vision=is_vision)
    if config["cache_mode"] == "cached":
        _LLM_CACHE[cache_key] = loaded
    return loaded


def _load_llama_cpp_model(config, need_vision):
    # Vision only when the caller names the model's mmproj (H3 Forge does);
    # the analyze node's llama.cpp path still sends text only.
    mmproj = config.get("llama_mmproj_path") or ""
    if need_vision and not mmproj:
        raise ValueError("The llama.cpp backend currently supports text-only GGUF models.")
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise ImportError(
            "GGUF loading requires llama-cpp-python. Install a CUDA-enabled build in the ComfyUI environment."
        ) from exc
    chat_handler = None
    if need_vision:
        try:
            from llama_cpp.llama_chat_format import MTMDChatHandler
        except ImportError as exc:
            raise ImportError(
                "GGUF vision needs llama-cpp-python 0.3.26 or newer. Update it in the ComfyUI environment."
            ) from exc
        chat_handler = MTMDChatHandler(clip_model_path=mmproj, verbose=False)
        try:
            # mtmd logs through its own hook, not llama.cpp's, and prints the
            # whole prompt on every image. Route it through llama-cpp-python's
            # filtered logger so verbose=False keeps the console quiet.
            import ctypes
            from llama_cpp import mtmd_cpp
            from llama_cpp._logger import llama_log_callback
            mtmd_cpp.mtmd_log_set(llama_log_callback, ctypes.c_void_p(0))
            mtmd_cpp.mtmd_helper_log_set(llama_log_callback, ctypes.c_void_p(0))
        except Exception:
            pass

    cache_key = (
        config["model_path"], config["llama_n_ctx"], config["llama_n_gpu_layers"],
        config["llama_n_threads"], config["llama_chat_format"], mmproj if need_vision else "",
    )
    if config["cache_mode"] == "cached" and cache_key in _LLM_CACHE:
        return _LLM_CACHE[cache_key]

    kwargs = {
        "model_path": config["model_path"],
        "n_ctx": config["llama_n_ctx"],
        "n_gpu_layers": config["llama_n_gpu_layers"],
        "verbose": False,
    }
    if config["llama_n_threads"] > 0:
        kwargs["n_threads"] = config["llama_n_threads"]
    if config["llama_chat_format"]:
        kwargs["chat_format"] = config["llama_chat_format"]
    if chat_handler is not None:
        kwargs["chat_handler"] = chat_handler
    loaded = _LoadedLLM(model=Llama(**kwargs), tokenizer=None, processor=None, is_vision=chat_handler is not None)
    if config["cache_mode"] == "cached":
        _LLM_CACHE[cache_key] = loaded
    return loaded


def _run_llama_cpp_generation(loaded, system_prompt, user_text, max_new_tokens, temperature, top_p,
                              repetition_penalty, seed):
    messages = _messages_for_prompt(system_prompt, user_text, 0)
    kwargs = {
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repetition_penalty,
    }
    # llama-cpp-python samples with a fixed seed unless given one, so -1 has to
    # pick a fresh seed here or every run returns the same text.
    kwargs["seed"] = seed if seed >= 0 else random.randrange(2**31)
    response = loaded.model.create_chat_completion(**kwargs)
    return response["choices"][0]["message"]["content"].strip(), 0


def _run_ollama_generation(config, system_prompt, user_text, max_new_tokens, temperature, top_p,
                           repetition_penalty, seed, pil_images, unload_after_request=False):
    if pil_images:
        raise ValueError("The Ollama backend currently supports text-only analysis.")
    messages = _messages_for_prompt(system_prompt, user_text, 0)
    options = {
        "num_predict": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repetition_penalty,
    }
    if seed >= 0:
        options["seed"] = seed
    payload_data = {
        "model": config["model_path"],
        "messages": messages,
        "stream": False,
        "options": options,
    }
    # Ollama owns its model memory in a separate server process; ComfyUI's CUDA
    # cleanup cannot release it. keep_alive=0 makes Ollama unload after this call.
    if unload_after_request:
        payload_data["keep_alive"] = 0
    payload = json.dumps(payload_data).encode()
    request = urlrequest.Request(OLLAMA_CHAT_URL, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlrequest.urlopen(request, timeout=config["ollama_timeout"]) as response:
            result = json.loads(response.read().decode())
    except (urlerror.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Could not reach Ollama at {OLLAMA_CHAT_URL}: {exc}") from exc
    try:
        return result["message"]["content"].strip(), 0
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Ollama returned an unexpected response: {result}") from exc


def _image_tensor_to_pil(image, resize_max_px, resize_algorithm):
    array = image.detach().cpu().clamp(0, 1).numpy()
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] > 3:
        array = array[..., :3]
    pil = Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")
    if resize_max_px and resize_max_px > 0:
        width, height = pil.size
        longest = max(width, height)
        if longest > resize_max_px:
            scale = resize_max_px / float(longest)
            resample = _RESAMPLE_FILTERS.get(resize_algorithm, _PIL_LANCZOS)
            pil = pil.resize((max(1, int(width * scale)), max(1, int(height * scale))), resample)
    return pil


def _select_frame_indices(frame_count, max_frames, frame_stride, strategy):
    if frame_count <= 0 or max_frames <= 0:
        return []

    max_frames = min(max_frames, frame_count)
    stride = max(1, frame_stride)

    if strategy == "first":
        return list(range(max_frames))
    if strategy == "last":
        return list(range(frame_count - max_frames, frame_count))
    if strategy == "middle":
        start = max(0, (frame_count - max_frames) // 2)
        return list(range(start, start + max_frames))
    if strategy == "every_nth":
        return list(range(0, frame_count, stride))[:max_frames]

    if max_frames == 1:
        return [frame_count // 2]
    return np.linspace(0, frame_count - 1, max_frames, dtype=int).tolist()


def _prepare_images(images, max_frames, frame_stride, frame_strategy, resize_max_px,
                    resize_algorithm):
    if images is None:
        return []
    if images.ndim == 3:
        images = images.unsqueeze(0)
    indices = _select_frame_indices(images.shape[0], max_frames, frame_stride, frame_strategy)
    return [
        _image_tensor_to_pil(
            images[index],
            resize_max_px,
            resize_algorithm,
        )
        for index in indices
    ]


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


def _messages_for_prompt(system_prompt, user_text, image_count):
    if image_count > 0:
        content = [{"type": "image"} for _ in range(image_count)]
        if image_count > 1:
            user_text = f"The attached images are sampled video/image-sequence frames in order.\n\n{user_text}"
        content.append({"type": "text", "text": user_text})
    else:
        content = user_text

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    return messages


def _apply_chat_template(tokenizer_or_processor, messages, fallback_text):
    template_fn = getattr(tokenizer_or_processor, "apply_chat_template", None)
    if template_fn is None:
        return fallback_text
    try:
        return template_fn(messages, tokenize=False, add_generation_prompt=True)
    except TypeError:
        return template_fn(messages, add_generation_prompt=True)
    except Exception as exc:
        log_dasiwa("LLM Nodes", f"Chat template failed, using plain prompt: {exc}")
        return fallback_text


def _run_generation(loaded, config, system_prompt, user_text, pil_images, max_new_tokens,
                    temperature, top_p, repetition_penalty, seed, max_input_tokens, use_kv_cache):
    model = loaded.model
    tokenizer = loaded.tokenizer
    processor = loaded.processor
    image_count = len(pil_images)

    if seed >= 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    messages = _messages_for_prompt(system_prompt, user_text, image_count)
    fallback = "\n\n".join(part for part in (system_prompt, user_text) if part)

    if image_count > 0:
        if processor is None:
            raise RuntimeError(
                "Images were connected, but this model has no AutoProcessor. "
                "Use a vision-language model folder with processor files."
            )
        prompt_text = _apply_chat_template(processor, messages, fallback)
        try:
            processor_kwargs = {
                "text": [prompt_text],
                "images": pil_images,
                "return_tensors": "pt",
                "padding": True,
            }
            if max_input_tokens > 0:
                processor_kwargs.update({"truncation": True, "max_length": max_input_tokens})
            inputs = processor(**processor_kwargs)
        except Exception as exc:
            if max_input_tokens > 0:
                processor_kwargs.pop("truncation", None)
                processor_kwargs.pop("max_length", None)
                try:
                    inputs = processor(**processor_kwargs)
                except Exception as fallback_exc:
                    raise RuntimeError(
                        "The selected model processor could not accept the connected images. "
                        "Try a Qwen-VL/LLaVA-style vision-language model or reduce frames/resolution."
                    ) from fallback_exc
            else:
                raise RuntimeError(
                    "The selected model processor could not accept the connected images. "
                    "Try a Qwen-VL/LLaVA-style vision-language model or reduce frames/resolution."
                ) from exc
        decoder = processor
    else:
        prompt_text = _apply_chat_template(tokenizer, messages, fallback)
        tokenizer_kwargs = {"return_tensors": "pt", "padding": True}
        if max_input_tokens > 0:
            tokenizer_kwargs.update({"truncation": True, "max_length": max_input_tokens})
        inputs = tokenizer([prompt_text], **tokenizer_kwargs)
        decoder = tokenizer

    device = _device_for_inputs(model, config["device"])
    inputs = _to_device(dict(inputs), device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
    }
    gen_kwargs.update(_generation_cache_kwargs(config, use_kv_cache))
    if temperature > 0:
        gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    else:
        gen_kwargs.update({"do_sample": False})

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    new_ids = output_ids[:, input_len:] if input_len else output_ids
    text = decoder.batch_decode(new_ids, skip_special_tokens=True)[0]
    return text.strip(), len(pil_images)


class DaSiWa_LLMModelSelector:
    DESCRIPTION = (
        "DaSiWa LLM Model Selector: choose a local transformers LLM/VLM folder "
        "from ComfyUI/models/llm and configure caching, dtype, device, and unload behavior."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_list_llm_models(), {"description": "Model folder under ComfyUI/models/llm. Use a full Hugging Face-style folder with config/tokenizer files."}),
                "custom_path": ("STRING", {"default": "", "description": "Optional absolute path, or relative path under ComfyUI/models/llm. Overrides model when set."}),
                "backend": (["transformers", "llama_cpp", "ollama"], {"default": "transformers", "description": "Transformers loads complete local model folders, llama.cpp loads local GGUF files, and Ollama calls its loopback API."}),
                "task": (["auto", "text", "vision"], {"default": "auto", "description": "Use vision when analyzing connected images/frame batches."}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto", "description": "Device placement for the model."}),
                "dtype": (["auto", "float16", "bfloat16", "float32"], {"default": "auto", "description": "Model dtype. Auto follows the model config when possible."}),
                "quantization": (["none", "8bit", "4bit"], {"default": "none", "description": "Optional bitsandbytes quantization. Requires bitsandbytes installed."}),
                "cache_mode": (["cached", "unload_after_run"], {"default": "unload_after_run", "description": "Cached is faster. Unload after run frees RAM/VRAM after every output."}),
                "attention_implementation": (["auto", "sdpa", "flash_attention_2", "eager"], {"default": "auto", "description": "Optional attention backend override."}),
                "kv_cache_implementation": (["default", "dynamic", "static", "quantized"], {"default": "default", "description": "Transformers KV-cache strategy. Quantized reduces generation memory but requires a compatible Transformers cache backend."}),
                "kv_cache_quant_backend": (["quanto", "hqq"], {"default": "quanto", "description": "Backend used only by the quantized Transformers KV cache."}),
                "kv_cache_nbits": (["2", "4", "8"], {"default": "4", "description": "Bit width used only by the quantized Transformers KV cache."}),
                "kv_cache_residual_length": ("INT", {"default": 128, "min": 0, "max": 16384, "step": 8, "description": "Recent KV tokens retained at full precision before older tokens are quantized."}),
                "llama_n_ctx": ("INT", {"default": 8192, "min": 512, "max": 1048576, "step": 512, "description": "llama.cpp context size for a local GGUF model."}),
                "llama_n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1, "description": "llama.cpp layers offloaded to GPU. -1 requests full offload; 0 is CPU-only."}),
                "llama_n_threads": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1, "description": "llama.cpp CPU threads. 0 lets llama.cpp choose."}),
                "llama_chat_format": ("STRING", {"default": "", "description": "Optional llama.cpp chat format, for example chatml. Leave empty to use GGUF metadata."}),
                "ollama_model": ("STRING", {"default": "", "description": "Ollama model name, for example qwen3:8b. Required when backend is ollama."}),
                "ollama_timeout": ("INT", {"default": 300, "min": 1, "max": 3600, "step": 1, "description": "Loopback Ollama request timeout in seconds."}),
            }
        }

    RETURN_TYPES = ("DASIWA_LLM_CONFIG",)
    RETURN_NAMES = ("llm_config",)
    FUNCTION = "select"
    CATEGORY = "DaSiWa/LLM"

    def select(self, model, custom_path, backend, task, device, dtype, quantization, cache_mode,
               attention_implementation, kv_cache_implementation, kv_cache_quant_backend,
               kv_cache_nbits, kv_cache_residual_length, llama_n_ctx, llama_n_gpu_layers,
               llama_n_threads, llama_chat_format, ollama_model, ollama_timeout):
        if backend == "ollama":
            model_path = ollama_model.strip()
            if not model_path:
                raise ValueError("Enter ollama_model when backend is ollama.")
        else:
            model_path = _resolve_model_path(
                model, custom_path, allow_gguf=backend == "llama_cpp",
            )
        return ({
            "model_path": model_path,
            "backend": backend,
            "task": task,
            "device": device,
            "dtype": dtype,
            "quantization": quantization,
            "cache_mode": cache_mode,
            "attention_implementation": attention_implementation,
            "kv_cache_implementation": kv_cache_implementation,
            "kv_cache_quant_backend": kv_cache_quant_backend,
            "kv_cache_nbits": int(kv_cache_nbits),
            "kv_cache_residual_length": kv_cache_residual_length,
            "llama_n_ctx": llama_n_ctx,
            "llama_n_gpu_layers": llama_n_gpu_layers,
            "llama_n_threads": llama_n_threads,
            "llama_chat_format": llama_chat_format.strip(),
            "ollama_timeout": ollama_timeout,
        },)


class DaSiWa_LLMAnalyze:
    DESCRIPTION = (
        "DaSiWa LLM Analyze: run a local text or vision-language model against "
        "connected text, image, or VHS/video frame batches and return a STRING response."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_config": ("DASIWA_LLM_CONFIG", {"description": "Config from DaSiWa LLM Model Selector."}),
                "system_prompt_preset": (_SYSTEM_PROMPT_PRESET_LABELS, {"default": "custom", "description": "Preset system instruction. Custom uses the system_prompt widget."}),
                "system_prompt": ("STRING", {"default": "You are a concise, helpful visual and text analysis assistant.", "multiline": True, "description": "Custom system prompt used only when system_prompt_preset is custom."}),
                "prompt": ("STRING", {"default": "Follow the selected system instruction for the connected text, image, or video input.", "multiline": True, "description": "User prompt/task instruction."}),
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 8192, "step": 1, "description": "Maximum generated response tokens."}),
                "max_input_tokens": ("INT", {"default": 0, "min": 0, "max": 131072, "step": 64, "description": "Optional prompt/context token limit. 0 lets the tokenizer/model decide."}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05, "description": "0 disables sampling for deterministic output."}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.01, "max": 1.0, "step": 0.01, "description": "Nucleus sampling value when temperature is above 0."}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05, "description": "Penalty for repeated text."}),
                "use_kv_cache": ("BOOLEAN", {"default": True, "description": "Keeps attention key/value cache during generation. Faster on, lower peak memory off for some models."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "description": "-1 leaves the current RNG state untouched."}),
                "max_frames": ("INT", {"default": 8, "min": 0, "max": 256, "step": 1, "description": "Maximum images/frames sent to the VLM. 0 ignores connected images."}),
                "frame_stride": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1, "description": "Stride used by the every_nth frame strategy."}),
                "frame_strategy": (["evenly_spaced", "first", "middle", "last", "every_nth"], {"default": "evenly_spaced", "description": "How to sample image batches from Load Image or VHS nodes."}),
                "resize_max_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 16, "description": "Downscale frames so the longest side is at most this many pixels. 0 disables resizing."}),
                "resize_algorithm": (["lanczos", "bicubic", "bilinear", "hamming", "box", "nearest"], {"default": "lanczos", "description": "Resampling filter used when resize_max_px downscales images."}),
                "memory_cleanup": (["off", "before_run", "after_run", "before_and_after"], {"default": "off", "description": "Optionally clear cached DaSiWa LLM models before and/or after this node runs."}),
            },
            "optional": {
                "images": ("IMAGE", {"description": "Native ComfyUI IMAGE input. Works with Load Image and VHS/image-sequence frame batches."}),
                "text_input": ("STRING", {"forceInput": True, "description": "Connected text to analyze."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "info")
    FUNCTION = "analyze"
    CATEGORY = "DaSiWa/LLM"

    def analyze(self, llm_config, system_prompt_preset, system_prompt, prompt, max_new_tokens,
                max_input_tokens, temperature, top_p, repetition_penalty, use_kv_cache,
                seed, max_frames, frame_stride, frame_strategy, resize_max_px,
                resize_algorithm, memory_cleanup, images=None, text_input=""):
        backend = llm_config.get("backend")
        if backend not in ("transformers", "llama_cpp", "ollama"):
            raise ValueError(f"Unsupported LLM backend: {backend}")

        final_system, user_text = _compose_user_text(
            system_prompt_preset,
            system_prompt,
            prompt,
            text_input,
        )
        if not user_text:
            user_text = "Analyze the provided input."

        pil_images = _prepare_images(
            images,
            max_frames,
            frame_stride,
            frame_strategy,
            resize_max_px,
            resize_algorithm,
        )
        loaded = None
        cleanup_before = memory_cleanup in ("before_run", "before_and_after")
        cleanup_after = memory_cleanup in ("after_run", "before_and_after")
        if cleanup_before:
            _release_all_model_memory()
        try:
            if backend == "transformers":
                loaded = _load_transformers_model(llm_config, need_vision=len(pil_images) > 0)
                response, image_count = _run_generation(
                    loaded, llm_config, final_system, user_text, pil_images, max_new_tokens,
                    temperature, top_p, repetition_penalty, seed, max_input_tokens, use_kv_cache,
                )
            elif backend == "llama_cpp":
                loaded = _load_llama_cpp_model(llm_config, need_vision=len(pil_images) > 0)
                response, image_count = _run_llama_cpp_generation(
                    loaded, final_system, user_text, max_new_tokens, temperature, top_p,
                    repetition_penalty, seed,
                )
            else:
                response, image_count = _run_ollama_generation(
                    llm_config, final_system, user_text, max_new_tokens, temperature, top_p,
                    repetition_penalty, seed, pil_images,
                    unload_after_request=(llm_config.get("cache_mode") == "unload_after_run" or cleanup_after),
                )
            info = (
                f"model={llm_config['model_path']}; "
                f"backend={backend}; "
                f"system_prompt_preset={system_prompt_preset}; "
                f"cache_mode={llm_config['cache_mode']}; "
                f"memory_cleanup={memory_cleanup}; "
                f"images_sent={image_count}; "
                f"resize_max_px={resize_max_px}; "
                f"resize_algorithm={resize_algorithm}; "
                f"max_input_tokens={max_input_tokens}; "
                f"use_kv_cache={use_kv_cache}; "
                f"kv_cache_implementation={llm_config.get('kv_cache_implementation', 'default')}"
            )
            return (response, info)
        finally:
            if llm_config.get("cache_mode") == "unload_after_run" or cleanup_after:
                _release_all_model_memory()
                if loaded is not None:
                    try:
                        close = getattr(loaded.model, "close", None)
                        if callable(close):
                            close()
                        else:
                            loaded.model.to("cpu")
                    except Exception:
                        pass
                    del loaded
                _cleanup_cuda()
