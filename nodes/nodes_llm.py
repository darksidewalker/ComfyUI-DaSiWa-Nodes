"""DaSiWa local LLM / VLM nodes.

These nodes intentionally keep the model object out of ComfyUI graph outputs.
That makes the unload-after-run mode much more reliable because the graph cache
does not hold a live reference to a very large model.
"""

from .llm_backends import run_workflow_server, pil_images_b64

from .llm_prompt_presets import (
    _BASE_OUTPUT_RULES,
    _video_prompt_preset,
    _caption_preset,
    _SYSTEM_PROMPT_PRESETS,
    _SYSTEM_PROMPT_PRESET_LABELS,
    _resolve_system_prompt,
    _compose_user_text,
    prompt_response,
    h3_request,
    h3_response,
)

# Compatibility aliases: implementations and their globals belong to runtime.
from .llm_runtime import (
    _LLM_CACHE,
    OLLAMA_CHAT_URL,
    _IMAGE_RESAMPLING,
    _PIL_LANCZOS,
    _RESAMPLE_FILTERS,
    _ensure_llm_folder,
    _LLM_DIR,
    _LoadedLLM,
    _folder_paths_for_llm,
    _is_mmproj,
    _find_mmproj,
    _list_llm_models,
    _resolve_model_path,
    _normalize_model_path,
    _torch_dtype,
    _device_for_inputs,
    _to_device,
    _cleanup_cuda,
    _clear_llm_cache,
    _release_all_model_memory,
    _generation_cache_kwargs,
    _load_transformers_model,
    _load_llama_cpp_model,
    _run_llama_cpp_generation,
    _run_ollama_generation,
    _image_tensor_to_pil,
    _select_frame_indices,
    _prepare_images,
    _messages_for_prompt,
    _apply_chat_template,
    _run_generation,
)


class DaSiWa_LLMModelSelector:
    DESCRIPTION = (
        "DaSiWa LLM Model Selector: choose a local model from ComfyUI/models/llm "
        "or an operator-configured server, and configure inference and caching."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_list_llm_models(), {"description": "Model folder under ComfyUI/models/llm. Use a full Hugging Face-style folder with config/tokenizer files."}),
                "custom_path": ("STRING", {"default": "", "description": "Optional absolute path, or relative path under ComfyUI/models/llm. Overrides model when set."}),
                "backend": (["transformers", "llama_cpp", "ollama", "openai", "ollama_server"], {"default": "transformers", "description": "Transformers loads local model folders; llama.cpp loads local GGUFs; ollama is legacy loopback. openai/ollama_server use operator-configured endpoints."}),
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
                "ollama_timeout": ("INT", {"default": 300, "min": 1, "max": 3600, "step": 1, "description": "Ollama/server request timeout in seconds."}),
            },
            "optional": {
                "server_model": ("STRING", {"default": "", "description": "Model ID on the operator-configured OpenAI-compatible or Ollama server."}),
            }
        }

    RETURN_TYPES = ("DASIWA_LLM_CONFIG",)
    RETURN_NAMES = ("llm_config",)
    FUNCTION = "select"
    CATEGORY = "DaSiWa/LLM"

    def select(self, model, custom_path, backend, task, device, dtype, quantization, cache_mode,
               attention_implementation, kv_cache_implementation, kv_cache_quant_backend,
               kv_cache_nbits, kv_cache_residual_length, llama_n_ctx, llama_n_gpu_layers,
               llama_n_threads, llama_chat_format, ollama_model, ollama_timeout, server_model=""):
        if backend in ("openai", "ollama_server"):
            model_path = str(server_model or "").strip()
            if not model_path:
                raise ValueError("Enter server_model for the selected server backend.")
        elif backend == "ollama":
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
        "DaSiWa LLM Analyze: run a local or external text/vision model against "
        "connected text, images or video frame batches, with model-aware prompt rewriting."
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
                "h3_mode": (["T2VA", "I2VA", "FL2VA", "L2VA"], {"default": "T2VA", "description": "Mode for the PromptForge H3 preset; REF2VA uses the Director."}),
                "h3_duration": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 120.0, "step": 0.1, "description": "Clip duration for the PromptForge H3 preset."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "info")
    FUNCTION = "analyze"
    CATEGORY = "DaSiWa/LLM"

    def analyze(self, llm_config, system_prompt_preset, system_prompt, prompt, max_new_tokens,
                max_input_tokens, temperature, top_p, repetition_penalty, use_kv_cache,
                seed, max_frames, frame_stride, frame_strategy, resize_max_px,
                resize_algorithm, memory_cleanup, images=None, text_input="",
                h3_mode="T2VA", h3_duration=10.0):
        backend = llm_config.get("backend")
        if backend not in ("transformers", "llama_cpp", "ollama", "openai", "ollama_server"):
            raise ValueError(f"Unsupported LLM backend: {backend}")

        final_system, user_text = _compose_user_text(
            system_prompt_preset,
            system_prompt,
            prompt,
            text_input,
            resolve_system=system_prompt_preset != "promptforge_h3",
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
        if system_prompt_preset == "promptforge_h3":
            final_system, user_text = h3_request(user_text, h3_mode, h3_duration, len(pil_images))
        loaded = None
        remote_status = None
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
                load_config = llm_config
                if pil_images:
                    projector = llm_config.get("llama_mmproj_path") or _find_mmproj(llm_config["model_path"])
                    if not projector:
                        raise ValueError("GGUF vision requires a matching mmproj file beside the model.")
                    load_config = dict(llm_config, llama_mmproj_path=projector)
                loaded = _load_llama_cpp_model(load_config, need_vision=len(pil_images) > 0)
                response, image_count = _run_llama_cpp_generation(
                    loaded, final_system, user_text, max_new_tokens, temperature, top_p,
                    repetition_penalty, seed, images_b64=pil_images_b64(pil_images),
                )
            elif backend in ("openai", "ollama_server"):
                response, image_count, remote_status = run_workflow_server(
                    llm_config, final_system, user_text, pil_images, max_new_tokens,
                    temperature, top_p, repetition_penalty, seed, cleanup_after,
                )
            else:
                response, image_count = _run_ollama_generation(
                    llm_config, final_system, user_text, max_new_tokens, temperature, top_p,
                    repetition_penalty, seed, pil_images,
                    unload_after_request=(llm_config.get("cache_mode") == "unload_after_run" or cleanup_after),
                )
            if system_prompt_preset == "promptforge_h3":
                response = h3_response(response, h3_mode, h3_duration)
            elif system_prompt_preset.startswith("promptforge_"):
                response = prompt_response(system_prompt_preset, response)
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
            if remote_status is not None:
                info += (f"; remote_unload_requested={remote_status['unload_requested']}"
                         f"; remote_unloaded={remote_status['unloaded']}")
                if backend == "openai" and repetition_penalty != 1.0:
                    info += "; repetition_penalty_supported=False"
            return (response, info)
        finally:
            if cleanup_after or (
                backend not in ("openai", "ollama_server")
                and llm_config.get("cache_mode") == "unload_after_run"
            ):
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
