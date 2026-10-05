"""Shared Forge model transports, independent of workflow nodes."""

import base64
import io
import json
import os
import re
import threading
import time
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from .helper_logging import log_dasiwa

IMAGE_MAX_EDGE = 1024
NUM_PREDICT = 3500


class ForgeError(Exception):
    def __init__(self, code, message, raw=None):
        super().__init__(message)
        self.code, self.message, self.raw = code, message, raw


DEFAULT_OLLAMA = "http://127.0.0.1:11434"


def _base_url(value, default=""):
    url = str(value or default).strip().rstrip("/")
    if not url:
        return ""
    if not re.match(r"^https?://[^\s/]+", url):
        raise ForgeError("bad_url", f"Server address must start with http:// or https://, got {url!r}.")
    return url


def _workflow_url(value, default=""):
    """Validate new workflow endpoints without echoing credential-bearing input."""
    value = str(value or default).strip().rstrip("/")
    if not value:
        return ""
    try:
        parsed = urlparse.urlsplit(value)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment
                or any(c.isspace() or ord(c) < 32 for c in value)):
            raise ValueError
        parsed.port  # Validate port syntax/range, including IPv6 endpoints.
    except ValueError:
        raise ForgeError("bad_url", "Workflow server address must be HTTP(S), without credentials, query or fragment.") from None
    return value


# User-editable ComfyUI Settings are not trusted global service configuration.
# Workflow fallback requires explicit operator opt-in and single-user mode.
SETTING_KEYS = {
    "ollama_url": "DaSiWa.H3Forge.OllamaURL",
    "openai_url": "DaSiWa.H3Forge.OpenAIURL",
    "openai_api_key": "DaSiWa.H3Forge.OpenAIKey",
}


def comfy_settings():
    """The LLM server values from ComfyUI's settings file, or {} when there is none."""
    try:
        import folder_paths
        path = os.path.join(folder_paths.get_user_directory(), "default", "comfy.settings.json")
        with open(path, encoding="utf-8") as handle:
            saved = json.load(handle)
    except Exception:
        return {}
    return {key: str(saved.get(setting_id) or "").strip() for key, setting_id in SETTING_KEYS.items()}


def workflow_server_settings():
    """Trusted global environment, or explicitly opted-in single-user settings.

    An endpoint and its key always come from the same configuration source."""
    saved = {}
    if os.environ.get("DASIWA_LLM_ALLOW_SETTINGS") == "1":
        from comfy.cli_args import args
        if not getattr(args, "multi_user", False):
            saved = comfy_settings()
    ollama_url = os.environ.get("DASIWA_LLM_OLLAMA_URL", "").strip()
    openai_url = os.environ.get("DASIWA_LLM_OPENAI_URL", "").strip()
    if openai_url:
        api_key = os.environ.get("DASIWA_LLM_OPENAI_API_KEY", "").strip()
    else:
        openai_url = saved.get("openai_url", "")
        api_key = saved.get("openai_api_key", "") if openai_url else ""
    return {
        "ollama_url": _workflow_url(ollama_url or saved.get("ollama_url", ""), DEFAULT_OLLAMA),
        "openai_url": _workflow_url(openai_url),
        "openai_api_key": api_key,
    }


# Server models as entries in a node's model list, beside the local files.
SERVER_CHOICE_PREFIXES = {"Ollama: ": "ollama_server", "Server: ": "openai"}
# Schema calls read a cache; only one discovery (automatic or explicit) runs.
LIST_TIMEOUT = 2
MODEL_CHOICES_TTL = 60
MODEL_CHOICES_RETRY = 10
_model_choices_cache = {}
_model_choices_condition = threading.Condition()
_model_choices_running = False
_model_choices_thread = None


def _server_choice_keys(settings):
    keys = [("ollama", settings["ollama_url"], "")]
    if settings["openai_url"]:
        keys.append(("openai", settings["openai_url"], settings["openai_api_key"]))
    return keys


def _cached_server_choices(keys):
    # Caller holds the condition. Keys (including credentials) are never logged.
    return [choice for key in keys for choice in _model_choices_cache.get(key, ([], 0))[0]]


def _discover_server_choices(keys, force=False):
    global _model_choices_running
    try:
        for key in keys:
            with _model_choices_condition:
                previous, due = _model_choices_cache.get(key, ([], 0))
                if not force and time.monotonic() < due:
                    continue
            kind, url, api_key = key
            try:
                server = Ollama(url) if kind == "ollama" else OpenAICompatible(url, api_key)
                prefix = "Ollama: " if kind == "ollama" else "Server: "
                choices = [prefix + m["id"][len(kind) + 1:]
                           for m in server.models(timeout=LIST_TIMEOUT)]
                delay = MODEL_CHOICES_TTL
            except Exception:
                # Retain last good results only for this exact backend/URL/key.
                choices, delay = previous, MODEL_CHOICES_RETRY
            with _model_choices_condition:
                _model_choices_cache[key] = (choices, time.monotonic() + delay)
                # Avoid retaining unbounded endpoints/credentials during churn.
                while len(_model_choices_cache) > 32:
                    del _model_choices_cache[next(iter(_model_choices_cache))]
    finally:
        with _model_choices_condition:
            _model_choices_running = False
            _model_choices_condition.notify_all()


def server_model_choices():
    """Return a cached snapshot with zero caller-thread network IO.

    Schedule one daemon discovery when due. Config changes while it is busy
    are isolated immediately and retried on a later schema call."""
    global _model_choices_running, _model_choices_thread
    try:
        keys = _server_choice_keys(workflow_server_settings())
    except ForgeError:
        return []
    with _model_choices_condition:
        out = _cached_server_choices(keys)
        if not _model_choices_running and any(
                time.monotonic() >= _model_choices_cache.get(key, ([], 0))[1] for key in keys):
            _model_choices_running = True
            try:
                _model_choices_thread = threading.Thread(
                    target=_discover_server_choices, args=(keys,),
                    name="dasiwa-server-models", daemon=True)
                _model_choices_thread.start()
            except Exception:
                _model_choices_thread = None
                _model_choices_running = False
                _model_choices_condition.notify_all()
        return out


def refresh_server_model_choices():
    """Blocking forced refresh for Forge's off-thread models route, not schemas.

    Bypass TTL/retry and return choices after discovery (LIST_TIMEOUT per
    backend). Serialize with other discovery without holding a lock during IO.
    If another discovery stays busy beyond the bounded wait, return the cache.
    Transient errors retain each backend's last good results for this config.
    """
    global _model_choices_running
    try:
        keys = _server_choice_keys(workflow_server_settings())
    except ForgeError:
        return []
    with _model_choices_condition:
        if not _model_choices_condition.wait_for(
                lambda: not _model_choices_running, timeout=2 * LIST_TIMEOUT + 1):
            return _cached_server_choices(keys)
        _model_choices_running = True
    _discover_server_choices(keys, force=True)
    with _model_choices_condition:
        return _cached_server_choices(keys)


def parse_server_choice(choice):
    """(backend, model id) for an 'Ollama: ' or 'Server: ' entry, else None."""
    for prefix, backend in SERVER_CHOICE_PREFIXES.items():
        if str(choice or "").startswith(prefix):
            return backend, choice[len(prefix):]
    return None


def _is_this_machine(url):
    host = re.sub(r"^https?://", "", url).split("/")[0].rsplit(":", 1)[0].strip("[]").lower()
    return host in ("127.0.0.1", "localhost", "::1", "0.0.0.0")


def _http(url, payload=None, timeout=10, headers=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urlrequest.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode() or "{}")


CANCELLED = "Cancelled. Nothing was applied to the node."


def _stream_lines(url, payload, timeout, cancel, headers=None):
    """POST and yield the response line by line, stopping when Cancel is pressed.

    Leaving the `with` closes the connection, which is what makes a server
    stop: Ollama and llama.cpp both abort a generation whose client is gone.
    The check runs between lines, so a cancel lands at the next token - or,
    during the prompt read before the first token, as soon as one arrives.
    """
    req = urlrequest.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **(headers or {})})
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            if cancel is not None and cancel.is_set():
                raise ForgeError("cancelled", CANCELLED)
            line = raw.decode("utf-8", "replace").strip()
            if line:
                yield line


def _image_b64(path):
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((IMAGE_MAX_EDGE, IMAGE_MAX_EDGE))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


class WorkflowInterrupt:
    """Bridge stream checks to ComfyUI's exception-based interruption contract."""

    def is_set(self):
        import comfy.model_management as management
        management.throw_exception_if_processing_interrupted()
        return False


def pil_images_b64(images):
    """Encode already-sampled frames without imposing Director's resize limit."""
    encoded = []
    for image in images:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        encoded.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return encoded


class Ollama:
    kind = "ollama"

    def __init__(self, base):
        self.base = base

    def models(self, timeout=10):
        out = []
        for m in _http(self.base + "/api/tags", timeout=timeout).get("models", []):
            details = m.get("details") or {}
            # Embedding models (nomic-embed-text and the like: BERT family)
            # cannot write, so they are not offered.
            families = " ".join([details.get("family") or "", *(details.get("families") or [])]).lower()
            if "bert" in families or "embed" in m["name"].lower():
                continue
            params = details.get("parameter_size")
            out.append({"id": f"ollama:{m['name']}", "label": f"{m['name']}{f' ({params})' if params else ''}"})
        return out

    def can_see(self, name):
        try:
            return "vision" in (_http(self.base + "/api/show", {"model": name}).get("capabilities") or [])
        except Exception:
            return False

    def loaded(self):
        try:
            return {m["name"] for m in _http(self.base + "/api/ps").get("models", [])}
        except Exception:
            return set()

    def unload(self, name):
        try:
            _http(self.base + "/api/generate", {"model": name, "keep_alive": 0}, timeout=30)
        except Exception:
            log_dasiwa("H3 Forge", "Ollama unload request failed.")
            return False
        # Ollama unloads in the background; checking at once reported a model
        # "still loaded" that was gone a second later.
        import time
        for _ in range(10):
            try:
                loaded = {m["name"] for m in _http(self.base + "/api/ps").get("models", [])}
            except Exception:
                return False
            if name not in loaded:
                return True
            time.sleep(0.5)
        return False

    def chat(self, name, system, user, images_b64, sampling, num_ctx, timeout, cancel=None,
             *, max_tokens=NUM_PREDICT, keep_alive=0, seed=-1, repetition_penalty=1.0):
        options = {"num_ctx": num_ctx, "num_predict": max_tokens,
                   "temperature": sampling.get("temperature", 0.7),
                   "top_p": sampling.get("top_p", 0.8), "presence_penalty": 0}
        if seed >= 0:
            options["seed"] = seed
        if repetition_penalty != 1.0:
            options["repeat_penalty"] = repetition_penalty
        message = {"role": "user", "content": user}
        if images_b64:
            message["images"] = images_b64
        parts, stats = [], {}
        for line in _stream_lines(self.base + "/api/chat", {
            "model": name, "stream": True, "think": False, "keep_alive": keep_alive,
            "messages": [{"role": "system", "content": system}, message],
            "options": options,
        }, timeout, cancel):
            try:
                chunk = json.loads(line)
                if not isinstance(chunk, dict):
                    raise ValueError
                if chunk.get("error"):
                    raise ForgeError("backend", f"Ollama: {chunk['error']}")
                content = (chunk.get("message") or {}).get("content") or ""
                if not isinstance(content, str):
                    raise ValueError
                parts.append(content)
                if chunk.get("done"):
                    stats = {"prompt_tokens": chunk.get("prompt_eval_count"), "output_tokens": chunk.get("eval_count")}
            except (ValueError, TypeError, AttributeError):
                raise ForgeError("backend", "Ollama returned an invalid stream chunk.") from None
        return "".join(parts), stats


class OpenAICompatible:
    kind = "openai"

    def __init__(self, base, api_key=""):
        self.base = base
        self.api = base if base.endswith("/v1") else base + "/v1"
        self.root = self.api[: -len("/v1")]
        # Sent only to this server: llama-server --api-key, llama-swap apiKeys,
        # LM Studio with authentication on, or a hosted OpenAI-compatible API.
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def models(self, timeout=10):
        return [{"id": f"openai:{m['id']}", "label": m["id"]}
                for m in _http(self.api + "/models", timeout=timeout, headers=self.headers).get("data", [])]

    def can_see(self, name):
        # No standard capability endpoint; the request itself is the test.
        return None

    def loaded(self):
        return set()

    def unload(self, name):
        # llama-swap has a per-model unload; a plain llama.cpp server or LM
        # Studio holds its model for the life of the process.
        # llama-swap answers "OK" as plain text, so the status is the answer:
        # parsing it as JSON failed and reported every unload as refused.
        try:
            req = urlrequest.Request(f"{self.root}/api/models/unload/{urlparse.quote(name, safe='')}", data=b"{}",
                                     headers={"Content-Type": "application/json", **self.headers})
            with urlrequest.urlopen(req, timeout=30) as resp:
                return 200 <= resp.status < 300
        except Exception:
            return False

    def chat(self, name, system, user, images_b64, sampling, num_ctx, timeout, cancel=None,
             *, max_tokens=NUM_PREDICT, keep_alive=0, seed=-1, repetition_penalty=1.0):
        content = user
        if images_b64:
            content = [{"type": "text", "text": user}] + [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in images_b64]
        parts, usage = [], {}
        payload = {
            "model": name, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": max_tokens,
            "temperature": sampling.get("temperature", 0.7), "top_p": sampling.get("top_p", 0.8),
            # Same reason as the Ollama call: a server-side default penalty
            # turns a long structured answer into word lists.
            "presence_penalty": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            # llama.cpp server honours this; others ignore unknown fields.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if seed >= 0:
            payload["seed"] = seed
        for line in _stream_lines(self.api + "/chat/completions", payload, timeout, cancel, self.headers):
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                if not isinstance(chunk, dict) or chunk.get("error"):
                    raise ValueError
                usage = chunk.get("usage") or usage
                if not isinstance(usage, dict):
                    raise ValueError
                for choice in chunk.get("choices") or []:
                    content = (choice.get("delta") or {}).get("content") or ""
                    if not isinstance(content, str):
                        raise ValueError
                    parts.append(content)
            except (ValueError, TypeError, AttributeError):
                raise ForgeError("backend", "OpenAI-compatible server returned an error or invalid stream chunk.") from None
        return "".join(parts), {"prompt_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")}


class _NoGqaWithoutFlash:
    """Keep transformers off PyTorch's math attention kernel while Forge generates.

    For grouped-query models (Qwen3, Qwen3-VL) with no mask, transformers passes
    enable_gqa=True to SDPA, trusting the flash kernel to take it. Where PyTorch
    has no flash kernel - the Windows builds - SDPA falls back to the math
    kernel, which materialises the whole attention matrix. Measured 24 Sep
    2026, torch 2.12+cu130 on a 5080, one layer at 8k tokens: 18.9 GiB and
    1.66 s, against 0.23 GiB and 0.016 s with the key/value heads repeated
    first. On a 9k-token REF2VA prompt that spilled a 4B model into shared
    system memory and took minutes.

    Scoped to Forge's own generate call, and a no-op wherever flash exists.
    """

    def __enter__(self):
        self._saved = None
        try:
            import torch
            from transformers.integrations import sdpa_attention
            if torch.cuda.is_available() and not torch.backends.cuda.is_flash_attention_available():
                self._saved = sdpa_attention.use_gqa_in_sdpa
                # transformers added `value` as a third argument in newer releases.
                sdpa_attention.use_gqa_in_sdpa = lambda attention_mask, key, value=None: False
        except Exception:
            self._saved = None
        return self

    def __exit__(self, *exc):
        if self._saved is not None:
            from transformers.integrations import sdpa_attention
            sdpa_attention.use_gqa_in_sdpa = self._saved
        return False


def _half_dtype():
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bfloat16"
    except Exception:
        pass
    return "float16"


class Local:
    """ComfyUI/models/llm, through nodes_llm.py's own loaders."""
    kind = "local"

    def _llm(self):
        from . import llm_runtime
        return llm_runtime

    def models(self):
        llm = self._llm()
        try:
            import llama_cpp  # noqa: F401
            has_llama_cpp = True
        except ImportError:
            has_llama_cpp = False
        out = []
        for name in llm._list_llm_models():
            if name == "None":
                continue
            gguf = name.lower().endswith(".gguf")
            if not gguf and os.path.splitext(name)[1].lower() in (".safetensors", ".bin"):
                continue  # a bare weight file cannot be loaded for chat
            if gguf and not has_llama_cpp:
                out.append({"id": f"local:{name}", "label": f"{name} (GGUF - needs llama-cpp-python installed)", "disabled": True})
            else:
                # "org--Model" is how Hugging Face downloads name folders; the
                # org prefix makes the model hard to find in the list.
                shown = name.split("--", 1)[-1]
                if gguf:
                    kind = "GGUF, sees pictures" if self._mmproj(llm, name) else "GGUF, text only - no mmproj file beside it"
                else:
                    kind = "transformers"
                if not gguf and self._compressed_tensors(llm, name):
                    kind += ", FP8 compressed-tensors: very slow in ComfyUI, get the normal version"
                out.append({"id": f"local:{name}", "label": f"{shown} ({kind})"})
        return out

    @staticmethod
    def _mmproj(llm, name):
        try:
            return llm._find_mmproj(llm._resolve_model_path(name, "", allow_gguf=True))
        except Exception:
            return None

    @staticmethod
    def _compressed_tensors(llm, name):
        """True for an llm-compressor checkpoint (quant_method compressed-tensors).

        Built for vLLM. Under transformers it re-quantizes activations in every
        layer on every token: measured 24 Sep 2026 on a Qwen3-VL-4B FP8 build,
        215 ms per token (~1,500 syncs and 8,400 launches per step) against
        39 ms for the same architecture in plain bf16.
        """
        try:
            path = llm._resolve_model_path(name, "")
            with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
        except Exception:
            return False
        quant = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config") or {}
        return quant.get("quant_method") == "compressed-tensors"

    def can_see(self, name):
        # A GGUF sees only with its mmproj projector beside it.
        return bool(self._mmproj(self._llm(), name)) if name.lower().endswith(".gguf") else True

    def loaded(self):
        return set()

    def unload(self, name):
        self._llm()._release_all_model_memory()
        return True

    def chat(self, name, system, user, images_b64, sampling, num_ctx, timeout, cancel=None):
        llm = self._llm()
        gguf = name.lower().endswith(".gguf")
        config = {
            "model_path": llm._resolve_model_path(name, "", allow_gguf=gguf),
            "backend": "llama_cpp" if gguf else "transformers",
            "task": "vision" if images_b64 else "text",
            "device": "auto", "dtype": _half_dtype(), "quantization": "none",
            "cache_mode": "unload_after_run", "attention_implementation": "auto",
            "kv_cache_implementation": "default", "kv_cache_quant_backend": "quanto",
            "kv_cache_nbits": 4, "kv_cache_residual_length": 128,
            "llama_n_ctx": num_ctx, "llama_n_gpu_layers": -1, "llama_n_threads": 0, "llama_chat_format": "",
        }
        temperature, top_p = sampling.get("temperature", 0.7), sampling.get("top_p", 0.8)
        loaded = None
        try:
            if gguf:
                if images_b64:
                    config["llama_mmproj_path"] = self._mmproj(llm, name)
                messages = llm._messages_for_llama_cpp(
                    system, user, images_b64, include_empty_system=True)
                loaded = llm._load_llama_cpp_model(config, need_vision=bool(images_b64))
                # Streamed here rather than through _run_llama_cpp_generation so
                # Cancel can stop it between tokens.
                parts = []
                for chunk in loaded.model.create_chat_completion(
                        messages=messages,
                        max_tokens=NUM_PREDICT, temperature=temperature, top_p=top_p, stream=True,
                        # llama-cpp-python samples with a fixed seed unless given
                        # one, so Regenerate would return the same draft every time.
                        seed=__import__("random").randrange(2**31)):
                    if cancel is not None and cancel.is_set():
                        raise ForgeError("cancelled", CANCELLED)
                    parts.append(((chunk.get("choices") or [{}])[0].get("delta") or {}).get("content") or "")
                text = "".join(parts)
            else:
                pil = []
                if images_b64:
                    from PIL import Image
                    pil = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in images_b64]
                loaded = llm._load_transformers_model(config, need_vision=bool(pil))
                if cancel is not None:
                    # _run_generation calls model.generate itself; hand it a stop
                    # check through that call. This model instance is Forge's own
                    # (unload_after_run), so nothing else sees the wrapper.
                    from transformers import StoppingCriteria, StoppingCriteriaList

                    class _Stop(StoppingCriteria):
                        def __call__(self, input_ids, scores, **kwargs):
                            return cancel.is_set()

                    plain = loaded.model.generate
                    loaded.model.generate = lambda *a, **k: plain(*a, stopping_criteria=StoppingCriteriaList([_Stop()]), **k)
                with _NoGqaWithoutFlash():
                    text, _ = llm._run_generation(loaded, config, system, user, pil, NUM_PREDICT,
                                                  temperature, top_p, 1.0, -1, 0, True)
                if cancel is not None and cancel.is_set():
                    raise ForgeError("cancelled", CANCELLED)
        finally:
            if loaded is not None:
                try:
                    close = getattr(loaded.model, "close", None)
                    close() if callable(close) else loaded.model.to("cpu")
                except Exception:
                    pass
                del loaded
            llm._release_all_model_memory()
        return text, {"prompt_tokens": None, "output_tokens": None}


def run_workflow_server(config, system, user, images, max_tokens,
                        temperature, top_p, repetition_penalty, seed,
                        cleanup_after=False):
    """Run a graph request against operator settings; return text/count/status."""
    model = config.get("model_path")
    if (not isinstance(model, str) or not model.strip() or "://" in model
            or model.startswith("//") or any(ord(c) < 32 for c in model)):
        raise ValueError("Enter a server model ID, not an endpoint URL")
    settings = workflow_server_settings()
    kind = config["backend"]
    if kind == "openai":
        if not settings["openai_url"]:
            raise ValueError("Set DASIWA_LLM_OPENAI_URL in the ComfyUI service environment. "
                             "Single-user operators may opt into ComfyUI Settings > DaSiWa > LLM servers "
                             "with DASIWA_LLM_ALLOW_SETTINGS=1; settings fallback is disabled in multi-user mode.")
        server = OpenAICompatible(settings["openai_url"], settings["openai_api_key"])
    elif kind == "ollama_server":
        server = Ollama(settings["ollama_url"])
    else:
        raise ValueError("Unsupported workflow server backend")
    unload = config.get("cache_mode") == "unload_after_run" or cleanup_after
    status = {"unload_requested": bool(unload), "unloaded": False}
    try:
        text, _ = server.chat(
            config["model_path"], system, user, pil_images_b64(images),
            {"temperature": temperature, "top_p": top_p},
            config.get("llama_n_ctx", 8192), config.get("ollama_timeout", 300),
            WorkflowInterrupt(), max_tokens=max_tokens,
            keep_alive=0 if unload else "5m", seed=seed,
            repetition_penalty=repetition_penalty,
        )
        return text, len(images), status
    except urlerror.HTTPError as exc:
        code = "auth" if exc.code in (401, 403) else "backend"
        raise ForgeError(code, f"Workflow server returned HTTP {exc.code}.") from None
    except urlerror.URLError:
        raise ForgeError("connection", "Could not reach the configured workflow server.") from None
    except ForgeError as exc:
        message = CANCELLED if exc.code == "cancelled" else "Workflow server rejected the request."
        raise ForgeError(exc.code, message) from None
    except ValueError:
        raise ForgeError("backend", "Workflow server returned an invalid response.") from None
    finally:
        if unload:
            try:
                status["unloaded"] = bool(server.unload(config["model_path"]))
            except Exception:
                # Cleanup is best effort and must never replace a stream error.
                status["unloaded"] = False


def backends(settings):
    """The sources this request may use, from the person's ComfyUI Settings."""
    settings = settings or {}
    out = {"local": Local(), "ollama": Ollama(_base_url(settings.get("ollama_url"), DEFAULT_OLLAMA))}
    openai_url = _base_url(settings.get("openai_url"))
    if openai_url:
        out["openai"] = OpenAICompatible(openai_url, str(settings.get("openai_api_key") or "").strip())
    return out


