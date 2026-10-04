"""Node-independent local model execution and memory lifecycle."""

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

# Backwards-compatible helper aliases; specifications have a single owner.
from .llm_prompt_presets import (
    _BASE_OUTPUT_RULES,
    _video_prompt_preset,
    _caption_preset,
    _SYSTEM_PROMPT_PRESETS,
    _SYSTEM_PROMPT_PRESET_LABELS,
    _resolve_system_prompt,
    _compose_user_text,
)


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
    # Callers pair vision GGUFs with the matching projector before loading.
    # Text-only requests keep the existing no-projector path.
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


def _messages_for_llama_cpp(system, user, images_b64=None, *, include_empty_system=False):
    """llama.cpp image_url messages, preserving text-only workflow contracts."""
    content = user
    if images_b64:
        content = [{"type": "text", "text": user}] + [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
            for b in images_b64]
    messages = []
    if system or include_empty_system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    return messages


def _run_llama_cpp_generation(loaded, system_prompt, user_text, max_new_tokens, temperature, top_p,
                              repetition_penalty, seed, images_b64=None):
    messages = _messages_for_llama_cpp(system_prompt, user_text, images_b64)
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
    return response["choices"][0]["message"]["content"].strip(), len(images_b64 or [])


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


