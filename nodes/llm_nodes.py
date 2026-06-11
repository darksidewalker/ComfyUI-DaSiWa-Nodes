"""DaSiWa local LLM / VLM nodes.

These nodes intentionally keep the model object out of ComfyUI graph outputs.
That makes the unload-after-run mode much more reliable because the graph cache
does not hold a live reference to a very large model.
"""

import gc
import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

import folder_paths


_LLM_CACHE = {}
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
        print(f"[DaSiWa LLM] Could not register models/llm folder: {exc}")
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
                    for root, _, files in os.walk(path):
                        if "config.json" in files:
                            rel = os.path.relpath(root, base)
                            break
                if rel and rel not in seen:
                    models.append(rel)
                    seen.add(rel)
            elif name.lower().endswith((".safetensors", ".bin", ".gguf")) and name not in seen:
                models.append(name)
                seen.add(name)
    return ["None"] + models


def _safe_repo_dir_name(repo_id, revision):
    repo_name = repo_id.strip().replace("/", "--")
    revision_name = (revision or "").strip()
    if revision_name and revision_name != "main":
        repo_name = f"{repo_name}--{revision_name.replace('/', '--')}"
    return repo_name


def _resolve_hf_repo_path(hf_repo_id, hf_revision, download_if_missing):
    repo_id = (hf_repo_id or "").strip()
    if not repo_id:
        return None

    local_name = _safe_repo_dir_name(repo_id, hf_revision)
    for base in _folder_paths_for_llm():
        candidate = os.path.join(base, local_name)
        if os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate

    nested_name = repo_id.replace("/", os.sep)
    for base in _folder_paths_for_llm():
        candidate = os.path.join(base, nested_name)
        if os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate

    if not download_if_missing:
        raise FileNotFoundError(
            f"HF repo '{repo_id}' is not present in models/llm. "
            "Enable download_if_missing to fetch it."
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Downloading Hugging Face repos requires huggingface_hub. "
            "Install it in the ComfyUI environment with: pip install huggingface_hub"
        ) from exc

    local_dir = os.path.join(_LLM_DIR, local_name)
    os.makedirs(local_dir, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"[DaSiWa LLM] Downloading HF repo '{repo_id}' to {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        revision=(hf_revision or "main").strip() or "main",
        local_dir=local_dir,
        token=token or None,
    )
    return _normalize_model_path(local_dir)


def _resolve_model_path(model_name, custom_path, hf_repo_id="", hf_revision="main", download_if_missing=False):
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
        return _normalize_model_path(path)

    hf_path = _resolve_hf_repo_path(hf_repo_id, hf_revision, download_if_missing)
    if hf_path:
        return _normalize_model_path(hf_path)

    if not model_name or model_name == "None":
        raise ValueError("Choose an LLM model or provide a custom_path.")

    for base in _folder_paths_for_llm():
        candidate = os.path.join(base, model_name)
        if os.path.exists(candidate):
            return _normalize_model_path(candidate)

    try:
        full_path = folder_paths.get_full_path("llm", model_name)
    except Exception:
        full_path = None
    if full_path and os.path.exists(full_path):
        return _normalize_model_path(full_path)

    raise FileNotFoundError(f"Could not find LLM model: {model_name}")


def _normalize_model_path(path):
    if os.path.isfile(path):
        lower = path.lower()
        if lower.endswith(".gguf"):
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
                    loaded.model.to("cpu")
                except Exception:
                    pass
                del loaded
                removed += 1
    _cleanup_cuda()
    return removed


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
        config["trust_remote_code"],
        task,
        config["attention_implementation"],
        is_vision,
    )

    if config["cache_mode"] == "cached" and cache_key in _LLM_CACHE:
        return _LLM_CACHE[cache_key]

    dtype = _torch_dtype(config["dtype"])
    common_kwargs = {
        "trust_remote_code": config["trust_remote_code"],
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
                trust_remote_code=config["trust_remote_code"],
            )
            tokenizer = getattr(processor, "tokenizer", None)
        except Exception as exc:
            print(f"[DaSiWa LLM] AutoProcessor load failed, trying tokenizer only: {exc}")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=config["trust_remote_code"],
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
        print(f"[DaSiWa LLM] Chat template failed, using plain prompt: {exc}")
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
        "use_cache": use_kv_cache,
    }
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
                "hf_repo_id": ("STRING", {"default": "", "description": "Optional Hugging Face repo id, for example Qwen/Qwen2.5-VL-7B-Instruct. Overrides model when set."}),
                "hf_revision": ("STRING", {"default": "main", "description": "HF branch, tag, or commit. Used when downloading or locating a repo copy."}),
                "download_if_missing": ("BOOLEAN", {"default": False, "description": "Download hf_repo_id into ComfyUI/models/llm if it is not already present."}),
                "backend": (["transformers"], {"default": "transformers", "description": "Inference backend. Transformers is supported in this version."}),
                "task": (["auto", "text", "vision"], {"default": "auto", "description": "Use vision when analyzing connected images/frame batches."}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto", "description": "Device placement for the model."}),
                "dtype": (["auto", "float16", "bfloat16", "float32"], {"default": "auto", "description": "Model dtype. Auto follows the model config when possible."}),
                "quantization": (["none", "8bit", "4bit"], {"default": "none", "description": "Optional bitsandbytes quantization. Requires bitsandbytes installed."}),
                "cache_mode": (["cached", "unload_after_run"], {"default": "unload_after_run", "description": "Cached is faster. Unload after run frees RAM/VRAM after every output."}),
                "trust_remote_code": ("BOOLEAN", {"default": False, "description": "Allow custom model code from the model folder."}),
                "attention_implementation": (["auto", "sdpa", "flash_attention_2", "eager"], {"default": "auto", "description": "Optional attention backend override."}),
            }
        }

    RETURN_TYPES = ("DASIWA_LLM_CONFIG",)
    RETURN_NAMES = ("llm_config",)
    FUNCTION = "select"
    CATEGORY = "DaSiWa/LLM"

    def select(self, model, custom_path, hf_repo_id, hf_revision, download_if_missing,
               backend, task, device, dtype, quantization, cache_mode, trust_remote_code,
               attention_implementation):
        model_path = _resolve_model_path(model, custom_path, hf_repo_id, hf_revision, download_if_missing)
        return ({
            "model_path": model_path,
            "hf_repo_id": hf_repo_id,
            "hf_revision": hf_revision,
            "backend": backend,
            "task": task,
            "device": device,
            "dtype": dtype,
            "quantization": quantization,
            "cache_mode": cache_mode,
            "trust_remote_code": bool(trust_remote_code),
            "attention_implementation": attention_implementation,
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
        if llm_config.get("backend") != "transformers":
            raise ValueError(f"Unsupported LLM backend: {llm_config.get('backend')}")

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
            _clear_llm_cache()
        try:
            loaded = _load_transformers_model(llm_config, need_vision=len(pil_images) > 0)
            response, image_count = _run_generation(
                loaded,
                llm_config,
                final_system,
                user_text,
                pil_images,
                max_new_tokens,
                temperature,
                top_p,
                repetition_penalty,
                seed,
                max_input_tokens,
                use_kv_cache,
            )
            info = (
                f"model={llm_config['model_path']}; "
                f"system_prompt_preset={system_prompt_preset}; "
                f"cache_mode={llm_config['cache_mode']}; "
                f"memory_cleanup={memory_cleanup}; "
                f"images_sent={image_count}; "
                f"resize_max_px={resize_max_px}; "
                f"resize_algorithm={resize_algorithm}; "
                f"max_input_tokens={max_input_tokens}; "
                f"use_kv_cache={use_kv_cache}"
            )
            return (response, info)
        finally:
            if llm_config.get("cache_mode") == "unload_after_run" or cleanup_after:
                _clear_llm_cache(llm_config.get("model_path"))
                if loaded is not None:
                    try:
                        loaded.model.to("cpu")
                    except Exception:
                        pass
                    del loaded
                _cleanup_cuda()
