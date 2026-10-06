"""Independent MiniMax H3 latent upscale inference, B,24,T,H,W.

Network/normalization adapted from LBH-123-AI's
https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
and its MMH3-UltimateUpscale adaptation. No external node-pack imports.

MIT License

Copyright (c) 2026 LBH-123-AI
Copyright (c) 2026 bbaudio-2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import os
import re
import torch
from torch import nn
from torch.nn import functional as F

_FOLDER = "latent_upscale_models"
_EXTENSIONS = {".safetensors", ".sft"}


def _folders():
    import folder_paths
    if _FOLDER not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path(_FOLDER, os.path.join(folder_paths.models_dir, _FOLDER))
    folder_paths.folder_names_and_paths[_FOLDER][1].update(_EXTENSIONS)
    return folder_paths


def upscale_model_names():
    """Native combo candidates, not verified models; no checkpoint reads/sentinels."""
    return sorted(set(n for n in _folders().get_filename_list(_FOLDER)
                      if os.path.splitext(n)[1].lower() in _EXTENSIONS and "ltx" not in n.lower()))


def _resolve_model_path(name):
    if not isinstance(name, str) or os.path.splitext(name)[1].lower() not in _EXTENSIONS:
        raise ValueError("H3 accepts only .safetensors / .sft checkpoints")
    if os.path.isabs(name) or ".." in name.replace("\\", "/").split("/"):
        raise ValueError("Select a relative H3 checkpoint name from latent_upscale_models")
    path = _folders().get_full_path(_FOLDER, name)
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(f"H3 checkpoint not found: {name}")
    if any(os.path.splitext(p)[1].lower() not in _EXTENSIONS for p in (path, os.path.realpath(path))):
        raise ValueError("Resolved H3 checkpoint must be .safetensors / .sft")
    return path


LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523
]


def _make_norm_tensors(device, dtype):
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std


def _normalization(channels):
    return nn.GroupNorm(32, channels)


def _zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class _AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = _normalization(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, 1)
        self.k = nn.Conv3d(in_channels, in_channels, 1)
        self.v = nn.Conv3d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        b, c, t, hh, w = h.shape
        q = self.q(h).flatten(2).transpose(1, 2)
        k = self.k(h).flatten(2).transpose(1, 2)
        v = self.v(h).flatten(2).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).view(b, c, t, hh, w)
        return x + self.proj_out(h)


class _ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            _normalization(channels), nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = _normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(p=dropout),
            _zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip = (
            nn.Conv3d(channels, self.out_channels, 1)
            if self.out_channels != channels else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h


class _TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = _normalization(channels)
        self.dwconv = nn.Conv3d(channels, channels,
                                kernel_size=(kernel_size, 1, 1),
                                padding=(padding, 0, 0),
                                groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        identity = x
        h = self.norm(x)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return identity + h


class _LatentResizer3D(nn.Module):
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12,
                 channels=512, dropout=0.1, attn=False,
                 temporal_every=2, temporal_kernel=5, block_layouts=None):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks if block_layouts is None else 0):
            if (b == 1 or b == in_blocks - 1) and attn:
                self.in_blocks.append(_AttnBlock3D(channels))
            self.in_blocks.append(_ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(_TemporalConv(channels, temporal_kernel))

        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks if block_layouts is None else 0):
            if (b == 1 or b == out_blocks - 1) and attn:
                self.out_blocks.append(_AttnBlock3D(channels))
            self.out_blocks.append(_ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(_TemporalConv(channels, temporal_kernel))

        if block_layouts is not None:
            for stage, layout in block_layouts.items():
                for kind, kernel in layout:
                    block = (_ResBlockEmb3D(channels, embed_dim, dropout) if kind == "residual"
                             else _TemporalConv(channels, kernel) if kind == "temporal"
                             else _AttnBlock3D(channels))
                    getattr(self, stage).append(block)

        self.norm_out = _normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x, scale=None, target_size=None):
        if target_size is not None:
            size = target_size
        elif scale is not None:
            size = tuple(int(round(s * scale)) for s in x.shape[-3:])
        else:
            return x

        if size == x.shape[-3:]:
            return x

        scale_emb = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_emb)

        x = self.conv_in(x)
        for b in self.in_blocks:
            if isinstance(b, _ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)

        for b in self.out_blocks:
            if isinstance(b, _ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x

def _detect_arch(sd):
    """Infer exact module indices and per-block temporal kernels, not a cadence."""
    weight = sd.get("conv_in.weight")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 5:
        raise ValueError("Not an H3 3D upscaler: missing conv_in.weight")
    channels, inputs, *kernel = weight.shape
    if inputs != 24 or kernel != [3, 3, 3] or channels < 32 or channels % 32:
        raise ValueError("H3 requires 24 latent channels and 32-divisible hidden channels")
    layouts = {}
    for stage in ("in_blocks", "out_blocks"):
        indices = sorted({int(m.group(1)) for key in sd
                          if (m := re.match(rf"{stage}\.(\d+)\.", key))})
        if not indices or indices != list(range(len(indices))):
            raise ValueError(f"Invalid H3 {stage}: indices must be contiguous from zero")
        layout = []
        for index in indices:
            prefix = f"{stage}.{index}."
            markers = [prefix + marker in sd for marker in
                       ("in_layers.2.weight", "dwconv.weight", "q.weight")]
            if sum(markers) != 1:
                raise ValueError(f"Unsupported H3 block: {prefix}")
            if markers[0]:
                layout.append(("residual", None))
            elif markers[1]:
                shape = tuple(sd[prefix + "dwconv.weight"].shape)
                if (len(shape) != 5 or shape[:2] != (channels, 1)
                        or shape[3:] != (1, 1) or shape[2] < 1 or shape[2] % 2 == 0):
                    raise ValueError(f"Invalid H3 temporal kernel: {prefix} {shape}")
                layout.append(("temporal", shape[2]))
            else:
                layout.append(("attention", None))
        if not any(kind == "residual" for kind, _ in layout):
            raise ValueError(f"H3 {stage} needs residual blocks")
        layouts[stage] = layout
    return {"in_channels": 24, "channels": channels, "dropout": 0.0, "block_layouts": layouts}


def _validated_state_dict(raw):
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Empty or invalid H3 checkpoint")
    sd = ({k.removeprefix("upscaler."): v for k, v in raw.items()}
          if all(k.startswith("upscaler.") for k in raw) else raw)
    if any(not isinstance(v, torch.Tensor) or not v.is_floating_point() for v in sd.values()):
        raise ValueError("H3 checkpoint must contain floating-point weights only")
    config = _detect_arch(sd)
    with torch.device("meta"):
        expected = _LatentResizer3D(**config).state_dict()
    missing, extra = expected.keys() - sd.keys(), sd.keys() - expected.keys()
    mismatched = [k for k in sorted(expected.keys() & sd.keys()) if expected[k].shape != sd[k].shape]
    if missing or extra or mismatched:
        raise ValueError(f"Invalid H3 checkpoint: missing={sorted(missing)[:5]}, "
                         f"unexpected={sorted(extra)[:5]}, shape_mismatch={mismatched[:5]}")
    return sd, config


def _compute_dtype(device, mm, precision="auto"):
    explicit = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if precision in explicit:
        return explicit[precision]
    if precision != "auto":
        raise ValueError(f"Unknown H3 upscale precision: {precision}")
    if device.type == "cpu":
        return torch.float32
    # BF16 policy does not itself inspect --force-fp16 in this Core version.
    if getattr(getattr(mm, "args", None), "force_fp16", False):
        return torch.float16
    if mm.should_use_bf16(device=device):
        return torch.bfloat16
    if mm.should_use_fp16(device=device):
        return torch.float16
    return torch.float32


class H3LatentUpscaler:
    """Lazy per-execution owner; preserves input BCTHW, device and dtype.

    Native ModelPatcher/LoadedModel lifecycle, scoped to this model. The normal
    load_models_gpu calls global free_memory and could evict unrelated models;
    instead we register/load only this owner using LoadedModel.model_load.
    Full load is required for ordinary torch layers (no low-VRAM casting).
    These lifecycle internals were checked against ComfyUI d49e8885.
    Insufficient memory raises; no global eviction, fallback, or model cache.
    """

    def __init__(self, model_name, device, precision="auto"):
        if precision not in ("auto", "bf16", "fp16", "fp32"):
            raise ValueError(f"Unknown H3 upscale precision: {precision}")
        self.precision = precision
        self.model_name = model_name
        self.device = torch.device(device)
        self.model = self.patcher = self._loaded = self._mm = None
        self.dtype = None
        self._closed = False

    def _load(self):
        if self._closed:
            raise RuntimeError("H3LatentUpscaler is closed")
        if self.model is not None:
            return
        path = _resolve_model_path(self.model_name)
        from comfy import utils, model_management as mm
        from comfy.model_patcher import ModelPatcher
        raw = utils.load_torch_file(path, safe_load=True, device=torch.device("cpu"))
        sd, config = _validated_state_dict(raw)
        dtype = _compute_dtype(self.device, mm, self.precision)
        with torch.device("meta"):
            model = _LatentResizer3D(**config)
        model.load_state_dict(sd, strict=True, assign=True)
        model = model.to(dtype=dtype).eval().requires_grad_(False)
        del sd, raw
        self.model, self.dtype, self._mm = model, dtype, mm
        self.patcher = ModelPatcher(model, load_device=self.device, offload_device=torch.device("cpu"))
        self._loaded = mm.LoadedModel(self.patcher)
        try:
            self._loaded.model_load()
            mm.current_loaded_models.insert(0, self._loaded)
        except BaseException:
            self.close()
            raise

    @property
    def temporal_halo(self):
        """Sum of depthwise temporal radii across both stages (loads lazily).

        This is the explicit temporal-conv halo, NOT a guarantee of chunk/full
        equivalence: residual Conv3d, GroupNorm over THW and optional attention
        also couple frames. Bounded temporal chunks remain an approximation.
        """
        self._load()
        return sum((b.dwconv.kernel_size[0] - 1) // 2
                   for b in self.model.modules() if isinstance(b, _TemporalConv))

    def upscale(self, video, target_height_latent, target_width_latent):
        if self._closed:
            raise RuntimeError("H3LatentUpscaler is closed")
        if (not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[1] != 24
                or not video.is_floating_point() or any(d < 1 for d in video.shape)):
            raise ValueError("H3 video must be a floating-point B,24,T,H,W tensor")
        targets = (target_height_latent, target_width_latent)
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in targets):
            raise ValueError("H3 target latent height/width must be positive integers")
        h, w = video.shape[-2:]
        th, tw = targets
        if th < h or tw < w:
            raise ValueError("H3 supports spatial upscaling only, not downscaling")
        if (th, tw) == (h, w):
            return video
        self._load()
        if self._loaded.real_model is None:
            self._loaded.model_load()
            if not any(entry is self._loaded for entry in self._mm.current_loaded_models):
                self._mm.current_loaded_models.insert(0, self._loaded)
        self._mm.throw_exception_if_processing_interrupted()
        with torch.inference_mode():
            x = video.to(device=self.device, dtype=self.dtype).contiguous()
            mean, std = _make_norm_tensors(self.device, self.dtype)
            x = (x - mean) / std
            result = self.model(x, scale=(th / h + tw / w) / 2,
                                target_size=(video.shape[2], th, tw))
            result = result * std + mean
            return result.to(device=video.device, dtype=video.dtype)

    def close(self):
        if self._closed:
            return
        try:
            if self._loaded is not None:
                registry = self._mm.current_loaded_models
                registry[:] = [entry for entry in registry if entry is not self._loaded]
                if self._loaded.model_finalizer is not None:
                    self._loaded.model_unload()
                else:
                    self.patcher.detach()
                self.patcher.cleanup()
        finally:
            self._loaded = self.patcher = self.model = self._mm = None
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("H3LatentUpscaler is closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
