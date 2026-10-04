"""Shared Forge model transports, independent of workflow nodes."""

import base64
import io
import json
import os
import re
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


class Ollama:
    kind = "ollama"

    def __init__(self, base):
        self.base = base

    def models(self):
        out = []
        for m in _http(self.base + "/api/tags").get("models", []):
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
        except Exception as exc:
            log_dasiwa("H3 Forge", f"unload of {name} failed: {exc}")
        # Ollama unloads in the background; checking at once reported a model
        # "still loaded" that was gone a second later.
        import time
        for _ in range(10):
            if name not in self.loaded():
                return True
            time.sleep(0.5)
        return False

    def chat(self, name, system, user, images_b64, sampling, num_ctx, timeout, cancel=None):
        message = {"role": "user", "content": user}
        if images_b64:
            message["images"] = images_b64
        parts, stats = [], {}
        for line in _stream_lines(self.base + "/api/chat", {
            "model": name, "stream": True, "think": False, "keep_alive": 0,
            "messages": [{"role": "system", "content": system}, message],
            "options": {"num_ctx": num_ctx, "num_predict": NUM_PREDICT,
                        "temperature": sampling.get("temperature", 0.7), "top_p": sampling.get("top_p", 0.8),
                        # Always sent, to override a Modelfile's chat default.
                        # Ollama's qwen3.5:9b ships presence_penalty 1.5, which
                        # on a prompt this long runs out of "allowed" words and
                        # writes synonym lists until the token cap: 1 of 4
                        # drafts usable at 1.5, 4 of 4 at 0 (1 Oct 2026).
                        "presence_penalty": 0},
        }, timeout, cancel):
            chunk = json.loads(line)
            if chunk.get("error"):
                raise ForgeError("backend", f"Ollama: {chunk['error']}")
            parts.append((chunk.get("message") or {}).get("content") or "")
            if chunk.get("done"):
                stats = {"prompt_tokens": chunk.get("prompt_eval_count"), "output_tokens": chunk.get("eval_count")}
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

    def models(self):
        return [{"id": f"openai:{m['id']}", "label": m["id"]}
                for m in _http(self.api + "/models", headers=self.headers).get("data", [])]

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

    def chat(self, name, system, user, images_b64, sampling, num_ctx, timeout, cancel=None):
        content = user
        if images_b64:
            content = [{"type": "text", "text": user}] + [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in images_b64]
        parts, usage = [], {}
        for line in _stream_lines(self.api + "/chat/completions", {
            "model": name, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": NUM_PREDICT,
            "temperature": sampling.get("temperature", 0.7), "top_p": sampling.get("top_p", 0.8),
            # Same reason as the Ollama call: a server-side default penalty
            # turns a long structured answer into word lists.
            "presence_penalty": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            # llama.cpp server honours this; others ignore unknown fields.
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout, cancel, self.headers):
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                parts.append((choice.get("delta") or {}).get("content") or "")
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
                content = user
                if images_b64:
                    config["llama_mmproj_path"] = self._mmproj(llm, name)
                    content = [{"type": "text", "text": user}] + [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in images_b64]
                loaded = llm._load_llama_cpp_model(config, need_vision=bool(images_b64))
                # Streamed here rather than through _run_llama_cpp_generation so
                # Cancel can stop it between tokens.
                parts = []
                for chunk in loaded.model.create_chat_completion(
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
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


def backends(settings):
    """The sources this request may use, from the person's ComfyUI Settings."""
    settings = settings or {}
    out = {"local": Local(), "ollama": Ollama(_base_url(settings.get("ollama_url"), DEFAULT_OLLAMA))}
    openai_url = _base_url(settings.get("openai_url"))
    if openai_url:
        out["openai"] = OpenAICompatible(openai_url, str(settings.get("openai_api_key") or "").strip())
    return out


