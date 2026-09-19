"""
MiniMax H3 Latent Upscaler - Consolidated single node for DaSiWa Nodes.

Merges the 2D (LatentResizer + Temporal Conv) and 3D (pure 3D conv with temporal
chunking) backbones into one node with a backbone selector. Supports three resize
modes: scale by multiplier, target dimensions, and megapixels.

Features:
- Backbone selection: 2d (fast, lightweight) or 3d (temporally coherent)
- Three resize modes in one node
- Auto device detection (cuda > mps > rocm > cpu)
- Batch size > 1 support
- NestedTensor (AV latent) handling - upscales video, passes audio through
- Per-channel mean/std normalization for H3 latents (24 channels)
- Auto architecture detection from checkpoint
- Temporal chunking for long videos (3D backbone)
- Optional model offload after inference (force_unload)
- ROCm (AMD GPU) and MPS (Apple Silicon) support

Model placement: ComfyUI/models/latent_upscale_models/
"""

import os
import gc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import folder_paths
import re

from .helper_logging import log_dasiwa

try:
    from einops import rearrange
except ImportError:
    def rearrange(x, pattern):
        raise ImportError("einops is required for MiniMax H3 Latent Upscaler. Install with: pip install einops")

try:
    import comfy.model_management as mm
    HAS_COMFY_MM = True
except ImportError:
    HAS_COMFY_MM = False

try:
    import comfy.sample
    import comfy.samplers
    import comfy.nested_tensor
    import latent_preview
    HAS_SAMPLER = True
except ImportError:
    HAS_SAMPLER = False


# ==========================================
# Register model folder
# ==========================================
_LATENT_UPSCALE_FOLDER = "latent_upscale_models"
if _LATENT_UPSCALE_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        _LATENT_UPSCALE_FOLDER,
        os.path.join(folder_paths.models_dir, _LATENT_UPSCALE_FOLDER)
    )

VAE_DOWNSAMPLE = 16

# ==========================================
# MiniMax H3 latent normalization stats (24 channels)
# ==========================================
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


# ==========================================
# Device helper functions (ROCm/MPS/CUDA auto-detect)
# ==========================================
def _is_rocm_build():
    return getattr(torch.version, "hip", None) is not None


def _resolve_device(backend):
    """Resolve device string to torch.device with auto-detection."""
    if backend == "auto":
        if torch.cuda.is_available() and not _is_rocm_build():
            return torch.device("cuda")
        if torch.cuda.is_available() and _is_rocm_build():
            return torch.device("cuda")  # ROCm uses cuda device type
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if backend == "cpu":
        return torch.device("cpu")

    if backend == "cuda":
        if not torch.cuda.is_available():
            log_dasiwa("MiniMaxH3 Upscaler", "CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("cuda")

    if backend == "rocm":
        if not _is_rocm_build():
            raise RuntimeError("ROCm was selected, but this PyTorch build has no HIP/ROCm support.")
        if not torch.cuda.is_available():
            raise RuntimeError("ROCm was selected, but PyTorch cannot access an AMD GPU.")
        return torch.device("cuda")

    if backend == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            log_dasiwa("MiniMaxH3 Upscaler", "MPS requested but not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("mps")

    raise ValueError(f"Unsupported device backend: {backend}")


def _backend_label(device):
    if device.type == "cuda" and _is_rocm_build():
        return f"ROCm/HIP {torch.version.hip}"
    if device.type == "cuda":
        return f"CUDA {getattr(torch.version, 'cuda', None) or 'unknown'}"
    if device.type == "mps":
        return "MPS (Apple Silicon)"
    return "CPU"


# ==========================================
# 2D backbone components (LatentResizer + Temporal Conv)
# ==========================================
def normalization(channels):
    return nn.GroupNorm(32, channels)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = normalization(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q = rearrange(self.q(h), "b c h w -> b 1 (h w) c")
        k = rearrange(self.k(h), "b c h w -> b 1 (h w) c")
        v = rearrange(self.v(h), "b c h w -> b 1 (h w) c")
        h = F.scaled_dot_product_attention(q, k, v)
        h = rearrange(h, "b 1 (h w) c -> b c h w", h=x.shape[-2], w=x.shape[-1])
        return x + self.proj_out(h)


class ResBlockEmb(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip = (
            nn.Conv2d(channels, self.out_channels, 1)
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


class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = normalization(channels)
        self.dwconv = nn.Conv3d(channels, channels,
                                kernel_size=(kernel_size, 1, 1),
                                padding=(padding, 0, 0),
                                groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        identity = x
        B, C, T, H, W = x.shape
        h = rearrange(x, "b c t h w -> (b t) c h w")
        h = self.norm(h)
        h = rearrange(h, "(b t) c h w -> b c t h w", b=B, t=T)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return identity + h


class LatentResizer(nn.Module):
    """2D backbone (matches training code)."""
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12,
                 channels=640, dropout=0.1, attn=False):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            if (b == 1 or b == in_blocks - 1) and attn:
                self.in_blocks.append(AttnBlock(channels))
            self.in_blocks.append(ResBlockEmb(channels, embed_dim, dropout))
        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            if (b == 1 or b == out_blocks - 1) and attn:
                self.out_blocks.append(AttnBlock(channels))
            self.out_blocks.append(ResBlockEmb(channels, embed_dim, dropout))
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels, in_channels, 3, padding=1)

    def forward(self, x, scale=None, target_hw=None):
        if target_hw is not None:
            size = target_hw
        elif scale is not None:
            size = tuple(int(round(s * scale)) for s in x.shape[-2:])
        else:
            return x
        if size == x.shape[-2:]:
            return x

        scale_emb = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_emb)

        x = self.conv_in(x)
        for b in self.in_blocks:
            if isinstance(b, ResBlockEmb):
                x = b(x, emb)
            else:
                x = b(x)
        x = F.interpolate(x, size=size, mode="bilinear")
        for b in self.out_blocks:
            if isinstance(b, ResBlockEmb):
                x = b(x, emb)
            else:
                x = b(x)
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


class VideoLatentResizer(nn.Module):
    """5D wrapper with Temporal blocks (matches training code)."""
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12,
                 channels=640, dropout=0.1, attn=False,
                 temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.resizer = LatentResizer(
            in_channels=in_channels,
            in_blocks=in_blocks,
            out_blocks=out_blocks,
            channels=channels,
            dropout=dropout,
            attn=attn,
        )
        self.temporal_blocks = nn.ModuleList()
        if temporal_every > 0:
            self.temporal_blocks.append(TemporalConv(channels, temporal_kernel))
            self.temporal_blocks.append(TemporalConv(channels, temporal_kernel))
        self.temporal_every = temporal_every
        self.temporal_kernel = temporal_kernel

    def forward(self, x, scale=None, target_hw=None):
        B, C, T, H, W = x.shape
        if target_hw is not None:
            size = target_hw
        elif scale is not None:
            size = (int(round(H * scale)), int(round(W * scale)))
        else:
            size = (H, W)

        if len(self.temporal_blocks) == 0:
            x_flat = rearrange(x, "b c t h w -> (b t) c h w")
            out = self.resizer(x_flat, scale=scale, target_hw=size)
            return rearrange(out, "(b t) c h w -> b c t h w", b=B, t=T)

        x_flat = rearrange(x, "b c t h w -> (b t) c h w")
        emb = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.resizer.embed(emb)

        out = self.resizer.conv_in(x_flat)
        for i, block in enumerate(self.resizer.in_blocks):
            if isinstance(block, ResBlockEmb):
                emb_t = emb.expand(B * T, -1)
                out = block(out, emb_t)
            else:
                out = block(out)
            if i % self.temporal_every == 0:
                out_3d = rearrange(out, "(b t) c h w -> b c t h w", b=B, t=T)
                out_3d = self.temporal_blocks[0](out_3d)
                out = rearrange(out_3d, "b c t h w -> (b t) c h w")

        out = F.interpolate(out, size=size, mode="bilinear")

        for i, block in enumerate(self.resizer.out_blocks):
            if isinstance(block, ResBlockEmb):
                emb_t = emb.expand(B * T, -1)
                out = block(out, emb_t)
            else:
                out = block(out)
            if i % self.temporal_every == 0:
                out_3d = rearrange(out, "(b t) c h w -> b c t h w", b=B, t=T)
                out_3d = self.temporal_blocks[1](out_3d)
                out = rearrange(out_3d, "b c t h w -> (b t) c h w")

        out = self.resizer.norm_out(out)
        out = F.silu(out)
        out = self.resizer.conv_out(out)
        out = rearrange(out, "(b t) c h w -> b c t h w", b=B, t=T)
        return out


# ==========================================
# 3D backbone components (pure 3D conv with temporal chunking)
# ==========================================
class AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = normalization(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, 1)
        self.k = nn.Conv3d(in_channels, in_channels, 1)
        self.v = nn.Conv3d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q = rearrange(self.q(h), "b c t h w -> b 1 (t h w) c")
        k = rearrange(self.k(h), "b c t h w -> b 1 (t h w) c")
        v = rearrange(self.v(h), "b c t h w -> b 1 (t h w) c")
        h = F.scaled_dot_product_attention(q, k, v)
        h = rearrange(h, "b 1 (t h w) c -> b c t h w", t=x.shape[2], h=x.shape[3], w=x.shape[4])
        return x + self.proj_out(h)


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(p=dropout),
            zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)),
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


class LatentResizer3D(nn.Module):
    """Pure-3D backbone with temporal chunking for long videos."""
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12,
                 channels=512, dropout=0.1, attn=False,
                 temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            if (b == 1 or b == in_blocks - 1) and attn:
                self.in_blocks.append(AttnBlock3D(channels))
            self.in_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(TemporalConv(channels, temporal_kernel))

        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            if (b == 1 or b == out_blocks - 1) and attn:
                self.out_blocks.append(AttnBlock3D(channels))
            self.out_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(TemporalConv(channels, temporal_kernel))

        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x, scale=None, target_size=None, enable_chunking=True):
        if target_size is not None:
            size = target_size
        elif scale is not None:
            size = tuple(int(round(s * scale)) for s in x.shape[-3:])
        else:
            return x

        if size == x.shape[-3:]:
            return x

        B, C, T, H, W = x.shape

        tk = 0
        for b in self.in_blocks:
            if isinstance(b, TemporalConv):
                tk = b.dwconv.weight.shape[2]
                break

        overlap = tk
        chunk = 32

        if not enable_chunking or T <= chunk:
            return self._forward_seg(x, scale, size)

        log_dasiwa("MiniMaxH3 Upscaler", f"Temporal chunking: T={T}, chunks={(T + chunk - 1) // chunk}, overlap={overlap}")

        x_padded = F.pad(x, (0, 0, 0, 0, overlap, overlap), mode='replicate')

        out_full = torch.zeros(B, C, T, size[-2], size[-1], device=x.device, dtype=x.dtype)
        weight_full = torch.zeros(1, 1, T, 1, 1, device=x.device, dtype=x.dtype)

        start = 0
        while start < T:
            seg_start = start
            seg_end = min(T, start + chunk)
            out_start = max(0, seg_start - overlap)
            out_end = min(T, seg_end + overlap)
            lo = max(0, out_start - overlap)
            hi = min(T + 2 * overlap, out_end + overlap)

            seg = x_padded[:, :, lo:hi].contiguous()
            seg_size = (hi - lo, size[-2], size[-1])
            seg_out = self._forward_seg(seg, scale, seg_size)

            s0 = (out_start + overlap) - lo
            s1 = s0 + (out_end - out_start)
            valid_out = seg_out[:, :, s0:s1]
            n_valid = out_end - out_start

            weight = torch.ones(n_valid, device=x.device, dtype=x.dtype)
            if seg_start > out_start:
                blend_len = seg_start - out_start
                weight[:blend_len] = torch.arange(1, blend_len + 1, device=x.device, dtype=x.dtype) / (blend_len + 1)
            if out_end > seg_end:
                blend_len = out_end - seg_end
                weight[-blend_len:] = torch.arange(blend_len, 0, -1, device=x.device, dtype=x.dtype) / (blend_len + 1)

            out_full[:, :, out_start:out_end] += valid_out * weight.view(1, 1, n_valid, 1, 1)
            weight_full[:, :, out_start:out_end] += weight.view(1, 1, n_valid, 1, 1)

            start += chunk
            del seg, seg_out, valid_out
            if start % (chunk * 4) == 0:
                gc.collect()

        out_full = out_full / weight_full.clamp(min=1e-8)
        return out_full

    def _forward_seg(self, x, scale, size):
        scale_emb = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_emb)

        x = self.conv_in(x)
        for b in self.in_blocks:
            if isinstance(b, ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)

        for b in self.out_blocks:
            if isinstance(b, ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


# ==========================================
# Model loading with architecture auto-detection
# ==========================================
MODEL_CACHE = {}


def get_models_dir():
    return folder_paths.get_folder_paths(_LATENT_UPSCALE_FOLDER)[0]


def scan_models():
    """Scan all registered latent_upscale_models directories for .safetensors files.

    Only .safetensors are listed — pickle formats (.pth/.pt/.ckpt) are excluded
    due to CVE-2025-32434 (arbitrary code execution via torch.load).
    """
    all_names = []
    dirs = folder_paths.get_folder_paths(_LATENT_UPSCALE_FOLDER)
    for d in dirs:
        try:
            for f in os.listdir(d):
                if f.lower().endswith(('.safetensors', '.sft')) and f not in all_names:
                    all_names.append(f)
        except OSError:
            pass
    if not all_names:
        return [f"(place .safetensors models in: {dirs[0] if dirs else 'latent_upscale_models'})"]
    return sorted(all_names)


def _load_raw_sd(path):
    """Load model state dict. Only .safetensors supported (security)."""
    if not path.lower().endswith(('.safetensors', '.sft')):
        raise ValueError(
            f"Only .safetensors models are supported for security reasons "
            f"(CVE-2025-32434). Got: {os.path.basename(path)}")

    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            sd = {k: f.get_tensor(k) for k in f.keys()}
    except ImportError:
        from safetensors.torch import load_file
        sd = load_file(path, device='cpu')

    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    return sd


def _extract_upscaler_sd(sd):
    """Handle merged checkpoints with 'upscaler.' prefix."""
    if any(k.startswith("upscaler.") for k in sd):
        return {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    return sd


def _detect_arch_2d(sd):
    """Detect 2D architecture from state dict keys."""
    cfg = {
        "in_channels": 24, "in_blocks": 12, "out_blocks": 12,
        "channels": 640, "dropout": 0.1, "attn": False,
        "temporal_every": 2, "temporal_kernel": 5,
    }

    if 'resizer.conv_in.weight' in sd:
        w = sd['resizer.conv_in.weight']
        cfg["in_channels"] = w.shape[1]
        cfg["channels"] = w.shape[0]

    in_ids = set()
    out_ids = set()
    for k in sd.keys():
        m = re.match(r'resizer\.in_blocks\.(\d+)\.in_layers\.', k)
        if m:
            in_ids.add(int(m.group(1)))
        m = re.match(r'resizer\.out_blocks\.(\d+)\.in_layers\.', k)
        if m:
            out_ids.add(int(m.group(1)))

    if in_ids:
        cfg["in_blocks"] = len(in_ids)
    if out_ids:
        cfg["out_blocks"] = len(out_ids)

    has_temporal = any('temporal_blocks' in k for k in sd)
    if has_temporal:
        for k in sd.keys():
            if 'temporal_blocks.0.dwconv.weight' in k:
                cfg["temporal_kernel"] = sd[k].shape[2]
                break
        cfg["temporal_every"] = 2
    else:
        cfg["temporal_every"] = 0

    cfg["attn"] = False
    return cfg


def _detect_arch_3d(sd):
    """Detect 3D architecture from state dict keys."""
    cfg = {
        "in_channels": 24, "in_blocks": 12, "out_blocks": 12,
        "channels": 512, "dropout": 0.1, "attn": False,
        "temporal_every": 2, "temporal_kernel": 5,
    }

    conv_key = 'conv_in.weight'
    if conv_key in sd:
        cfg["in_channels"] = sd[conv_key].shape[1]
        cfg["channels"] = sd[conv_key].shape[0]

    in_ids, out_ids = set(), set()
    for k in sd.keys():
        m = re.match(r'in_blocks\.(\d+)\.in_layers\.', k)
        if m:
            in_ids.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.in_layers\.', k)
        if m:
            out_ids.add(int(m.group(1)))

    if in_ids:
        cfg["in_blocks"] = len(in_ids)
    if out_ids:
        cfg["out_blocks"] = len(out_ids)

    has_temporal = any('dwconv.weight' in k for k in sd)
    if has_temporal:
        cfg["temporal_every"] = 2
        for k in sd.keys():
            if 'dwconv.weight' in k and k.endswith('dwconv.weight'):
                cfg["temporal_kernel"] = sd[k].shape[2]
                break
    else:
        cfg["temporal_every"] = 0

    cfg["attn"] = False
    return cfg


def load_model(name, device, precision, backbone):
    backend_lbl = _backend_label(device)
    cache_key = f"{name}::{backend_lbl}::{precision}::{backbone}"
    if cache_key in MODEL_CACHE:
        model = MODEL_CACHE[cache_key]
        return model.to(device, non_blocking=True)

    try:
        path = folder_paths.get_full_path_or_raise(_LATENT_UPSCALE_FOLDER, name)
    except Exception as e:
        raise FileNotFoundError(f"Model file not found: {name}") from e

    raw_sd = _load_raw_sd(path)
    up_sd = _extract_upscaler_sd(raw_sd)

    if backbone == "2d":
        cfg = _detect_arch_2d(up_sd)
        model = VideoLatentResizer(
            in_channels=cfg["in_channels"],
            in_blocks=cfg["in_blocks"],
            out_blocks=cfg["out_blocks"],
            channels=cfg["channels"],
            dropout=cfg["dropout"],
            attn=cfg["attn"],
            temporal_every=cfg["temporal_every"],
            temporal_kernel=cfg["temporal_kernel"],
        )
    else:  # 3d
        cfg = _detect_arch_3d(up_sd)
        model = LatentResizer3D(
            in_channels=cfg["in_channels"],
            in_blocks=cfg["in_blocks"],
            out_blocks=cfg["out_blocks"],
            channels=cfg["channels"],
            dropout=cfg["dropout"],
            attn=cfg["attn"],
            temporal_every=cfg["temporal_every"],
            temporal_kernel=cfg["temporal_kernel"],
        )

    missing, unexpected = model.load_state_dict(up_sd, strict=False)
    if missing:
        log_dasiwa("MiniMaxH3 Upscaler", f"Missing keys: {missing[:5]}...")
    if unexpected:
        log_dasiwa("MiniMaxH3 Upscaler", f"Unexpected keys: {unexpected[:5]}...")

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(precision, torch.float32)
    model = model.to(device).eval().requires_grad_(False)
    if dtype != torch.float32:
        model = model.to(dtype)

    MODEL_CACHE[cache_key] = model
    log_dasiwa("MiniMaxH3 Upscaler", f"Loaded upscale model: {name}")
    log_dasiwa("MiniMaxH3 Upscaler",
              f"  Params: {sum(p.numel() for p in model.parameters()):,} | "
              f"Backbone: {backbone} | Temporal: {'on' if cfg['temporal_every'] > 0 else 'off'} "
              f"(every={cfg['temporal_every']}, kernel={cfg['temporal_kernel']}) | "
              f"Backend: {backend_lbl} | Precision: {precision}")
    return model


# ==========================================
# NestedTensor (AV latent) handling helpers
# ==========================================
def _is_nested_tensor(samples):
    """Detect if samples is a MiniMax H3 AV latent (NestedTensor with video+audio)."""
    if isinstance(samples, dict):
        return "video" in samples or "audio" in samples
    # ComfyUI NestedTensor has .tensors list with [video, audio]
    if hasattr(samples, 'tensors') and len(getattr(samples, 'tensors', [])) >= 1:
        return True
    return False


def _extract_video_component(samples):
    """Extract video latent from NestedTensor or dict."""
    if isinstance(samples, dict):
        return samples.get("video", samples)
    # ComfyUI NestedTensor: .tensors[0] is video, .tensors[1] is audio
    if hasattr(samples, 'tensors'):
        tensors = samples.tensors
        if len(tensors) >= 1:
            return tensors[0]
    if hasattr(samples, 'video'):
        return samples.video
    return samples


def _extract_audio_component(samples):
    """Extract audio latent from NestedTensor or dict."""
    if isinstance(samples, dict):
        return samples.get("audio")
    if hasattr(samples, 'tensors'):
        tensors = samples.tensors
        if len(tensors) >= 2:
            return tensors[1]
    if hasattr(samples, 'audio'):
        return samples.audio
    return None


def _reconstruct_nested_tensor(original_samples, upscaled_video):
    """Reconstruct NestedTensor with upscaled video and original audio."""
    if isinstance(original_samples, dict):
        result = dict(original_samples)
        result["video"] = upscaled_video
        return result
    # ComfyUI NestedTensor: rebuild from components
    if hasattr(original_samples, 'tensors'):
        try:
            import comfy.nested_tensor
            audio = original_samples.tensors[1] if len(original_samples.tensors) >= 2 else None
            if audio is not None:
                return comfy.nested_tensor.NestedTensor((upscaled_video, audio))
        except Exception:
            pass
    # Fallback: just return the video tensor
    return upscaled_video


# ==========================================
# Tiled resample mode helpers
# ==========================================
try:
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
except Exception:
    FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    FRAME_RESCALE = 5.0 / 3.0

ALIGN_TILE = 2


def _frames_for_tokens(n):
    return sum(FRAME_PER_TOKEN[i % 5] for i in range(n))


def _tokens_for_frames(f):
    n, acc = 0, 0
    while acc < f:
        acc += FRAME_PER_TOKEN[n % 5]
        n += 1
    return n


def _clip_tokens(n):
    return (n - 5) // 17 * 5 + 2 if n >= 5 else 1


def _snap_clip_frames(v):
    return 5 + 17 * max(1, round((v - 5) / 17)) if v >= 5 else max(1, int(v))


def _snap_overlap_frames(v):
    return 0 if v <= 0 else 5 + 17 * max(0, round((v - 5) / 17))


def _compute_h3_segments_adaptive(tv, chunk_frames, overlap_frames):
    tc = _clip_tokens(chunk_frames)
    to = _clip_tokens(overlap_frames) if overlap_frames > 0 else 0
    if to >= tc:
        to = max(0, tc - 1)
    if tc >= tv:
        return [(0, 0, tv, _frames_for_tokens(tv))], _frames_for_tokens(tv)
    hop = max(1, tc - to)
    bounds, prev_k0, i = [], -1, 0
    while True:
        k0 = i * hop
        if k0 + tc >= tv:
            k1 = tv
            k0 = max(k0, tv - tc)
            if prev_k0 >= 0 and k0 <= prev_k0:
                k0 = prev_k0 + 1
            if k0 >= tv:
                break
            bounds.append((k0, _frames_for_tokens(k0), k1, _frames_for_tokens(k1)))
            break
        bounds.append((k0, _frames_for_tokens(k0), k0 + tc, _frames_for_tokens(k0 + tc)))
        prev_k0 = k0
        i += 1
    return bounds, _frames_for_tokens(tv)


def _px_to_lat(px):
    return max(ALIGN_TILE, (round(px / VAE_DOWNSAMPLE) // ALIGN_TILE) * ALIGN_TILE)


def _snap_align(v):
    return max(0, int(round(v / ALIGN_TILE)) * ALIGN_TILE)


def _compute_spatial_grid(h, w, th, tw, ol_h, ol_w, min_th=0, min_tw=0):
    def _grid_1d(size, tile, ol, min_tile):
        if size <= tile:
            return [0], [size], [0]
        sh = tile - ol
        n = math.ceil((size - ol) / sh)
        if (n - 1) * sh + tile < size:
            n += 1
        rows = [i * sh for i in range(n)]
        trows = [min(tile, size - r) for r in rows]
        if min_tile > 0 and n >= 2:
            edge = size - rows[-1]
            if edge < min_tile:
                new_last = size - min_tile
                if rows[-2] < new_last < rows[-2] + trows[-2]:
                    rows[-1] = new_last
                    trows[-1] = size - new_last
        ovl = [0] * n
        for i in range(1, n):
            ovl[i] = max(0, rows[i - 1] + trows[i - 1] - rows[i])
        return rows, trows, ovl

    rows, trows, row_ovl = _grid_1d(h, th, ol_h, min_th)
    cols, tcols, col_ovl = _grid_1d(w, tw, ol_w, min_tw)
    return rows, cols, trows, tcols, row_ovl, col_ovl


def _spatial_fade_mask(tile_h, tile_w, ovh, ovw, done_top, done_left, fade_h=0, fade_w=0):
    mask = torch.ones(tile_h, tile_w, dtype=torch.float32)

    def profile(n, ov, fade):
        p = torch.ones(n, dtype=torch.float32)
        f = min(fade, ov)
        frozen = ov - f
        p[:frozen] = 0.0
        if f > 0:
            p[frozen:ov] = torch.linspace(0.0, 1.0, f)
        return p

    if done_left and ovw > 0:
        mask = torch.minimum(mask, profile(tile_w, ovw, fade_w)[None, :])
    if done_top and ovh > 0:
        mask = torch.minimum(mask, profile(tile_h, ovh, fade_h)[:, None])
    return mask


def _crossfade(a, b, dim):
    n = a.shape[dim]
    w = torch.linspace(0.0, 1.0, n, device=a.device, dtype=a.dtype)
    shape = [1] * a.ndim
    shape[dim] = n
    return a + (b - a) * w.view(shape)


def _temporal_append(acc_v, acc_a, chunk_v, chunk_a, index, k0, f0):
    if acc_v is None:
        return chunk_v, chunk_a
    gi, agi = k0, round(f0 * FRAME_RESCALE)
    total_v = max(acc_v.shape[2], gi + chunk_v.shape[2])
    total_a = max(acc_a.shape[-1], agi + chunk_a.shape[-1])
    rv = torch.zeros((1, acc_v.shape[1], total_v, acc_v.shape[3], acc_v.shape[4]),
                     device=acc_v.device, dtype=acc_v.dtype)
    ra = torch.zeros((1, 32, 2, total_a), device=acc_a.device, dtype=acc_a.dtype)
    rv[:, :, :acc_v.shape[2]] = acc_v
    ra[:, :, :, :acc_a.shape[-1]] = acc_a
    v, a = chunk_v, chunk_a
    if index > 0:
        ov = min(acc_v.shape[2] - gi, v.shape[2])
        if ov > 0:
            rv[:, :, gi:gi + ov] = _crossfade(rv[:, :, gi:gi + ov], v[:, :, :ov], dim=2)
            v = v[:, :, ov:]
        gi += ov
        ova = min(acc_a.shape[-1] - agi, a.shape[-1])
        if ova > 0:
            ra[:, :, :, agi:agi + ova] = _crossfade(ra[:, :, :, agi:agi + ova], a[:, :, :, :ova], dim=3)
            a = a[:, :, :, ova:]
        agi += ova
    if v.shape[2] > 0:
        rv[:, :, gi:gi + v.shape[2]] = v
    if a.shape[-1] > 0:
        ra[:, :, :, agi:agi + a.shape[-1]] = a
    return rv, ra


def _audio_range(f0, f1):
    return round(f0 * FRAME_RESCALE), round(f1 * FRAME_RESCALE)


def _is_h3_av_latent(samples):
    if isinstance(samples, dict):
        video = samples.get("video")
        audio = samples.get("audio")
        if video is not None and audio is not None:
            return (video.ndim == 5 and video.shape[1] == 24 and
                    audio.ndim == 4 and audio.shape[1] == 32)
    if hasattr(samples, 'tensors'):
        tensors = samples.tensors
        if len(tensors) == 2:
            return (tensors[0].ndim == 5 and tensors[0].shape[1] == 24 and
                    tensors[1].ndim == 4 and tensors[1].shape[1] == 32)
    return False


def _get_h3_video_audio(samples):
    if isinstance(samples, dict):
        return samples["video"], samples.get("audio")
    if hasattr(samples, 'tensors'):
        return samples.tensors[0], samples.tensors[1]
    raise ValueError("Cannot extract video/audio from latent")


def _make_h3_nested(video, audio):
    try:
        return comfy.nested_tensor.NestedTensor((video, audio))
    except Exception:
        return {"video": video, "audio": audio}


def _auto_tile_chunk_sizes(h_lat, w_lat, t_tokens):
    """Auto-calculate optimal tile and chunk sizes based on VRAM."""
    if HAS_COMFY_MM and hasattr(mm, 'get_free_memory'):
        free_vram = mm.get_free_memory() / (1024 ** 3)  # GB
    else:
        free_vram = 8.0

    # Tile size based on VRAM (latent space)
    if free_vram >= 24:
        tile_lat = 768
        chunk_frames = 170
    elif free_vram >= 16:
        tile_lat = 576
        chunk_frames = 136
    elif free_vram >= 12:
        tile_lat = 512
        chunk_frames = 102
    elif free_vram >= 8:
        tile_lat = 384
        chunk_frames = 68
    else:
        tile_lat = 320
        chunk_frames = 34

    # Snap to grid and clamp to actual size
    tw = min(tile_lat, w_lat)
    th = min(tile_lat, h_lat)
    tw = _px_to_lat(tw * VAE_DOWNSAMPLE) // VAE_DOWNSAMPLE * ALIGN_TILE
    th = _px_to_lat(th * VAE_DOWNSAMPLE) // VAE_DOWNSAMPLE * ALIGN_TILE

    # Overlap: 25% of tile, at least 32px in latent space
    ol_w = max(ALIGN_TILE, min(tw - ALIGN_TILE, _snap_align(tw * 0.25)))
    ol_h = max(ALIGN_TILE, min(th - ALIGN_TILE, _snap_align(th * 0.25)))

    # Fade: 50% of overlap
    fw = min(ol_w, int(round(ol_w * 0.5)))
    fh = min(ol_h, int(round(ol_h * 0.5)))

    # Temporal chunk in frames (snap to H3 grid)
    chunk_frames = _snap_clip_frames(chunk_frames)
    temporal_overlap = _snap_overlap_frames(chunk_frames // 4)

    return {
        "tile_w": tw, "tile_h": th,
        "overlap_w": ol_w, "overlap_h": ol_h,
        "fade_w": fw, "fade_h": fh,
        "chunk_frames": chunk_frames,
        "temporal_overlap": temporal_overlap,
    }


def _trim_keyframe(kf, f0, f1):
    idx = kf["resolved_frame_index"]
    latent, audio_latent = kf.get("latent"), kf.get("audio_latent")
    if latent is None and audio_latent is None:
        return None if (idx < f0 or idx >= f1) else {"resolved_frame_index": idx - f0}
    out = {}
    if latent is not None:
        t_start = t_end = None
        pos = idx
        for k in range(latent.shape[2]):
            span = FRAME_PER_TOKEN[k % 5]
            if f0 <= pos and pos + span <= f1:
                if t_start is None:
                    t_start = k
                t_end = k + 1
            pos += span
        if t_start is None:
            return None
        out["latent"] = latent[:, :, t_start:t_end].contiguous()
        out["resolved_frame_index"] = idx + _frames_for_tokens(t_start) - f0
    if audio_latent is not None:
        rt = audio_latent.shape[-1]
        a_start = max(0, math.ceil((f0 - idx) * FRAME_RESCALE))
        a_end = min(rt, math.floor((f1 - idx) / FRAME_RESCALE))
        if a_end > a_start:
            out["audio_latent"] = audio_latent[..., a_start:a_end].contiguous()
            if "resolved_frame_index" not in out:
                out["resolved_frame_index"] = max(0, idx - f0)
    return out if ("latent" in out or "audio_latent" in out) else None


def _reanchor_conditioning(cond, f0, f1, spatial):
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            trimmed = [kf for kf in (_trim_keyframe(kf, f0, f1) for kf in kfs) if kf is not None]
            if trimmed:
                if spatial is not None:
                    for kf in trimmed:
                        lt = kf.get("latent")
                        if lt is not None and (lt.shape[3] != spatial[0] or lt.shape[4] != spatial[1]):
                            B, C, T, H, W = lt.shape
                            kf["latent"] = F.interpolate(
                                lt.view(B * T, C, H, W), size=spatial, mode="bilinear",
                                align_corners=False).view(B, C, T, spatial[0], spatial[1])
                nd["minimax_keyframes"] = trimmed
            else:
                nd.pop("minimax_keyframes", None)
        out.append([tensor, nd])
    return out


def _anchor_conditioning(cond, prev_video, f0, strength):
    t = _tokens_for_frames(f0)
    if t >= prev_video.shape[2]:
        raise ValueError("previous result does not reach the current segment start")
    anchor_kf = {"resolved_frame_index": 0, "latent": prev_video[:, :, t:t + 1].contiguous()}
    aug = max(0.0, min(1.0, float(strength)))
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            kept = [kf for kf in kfs if kf.get("resolved_frame_index") != 0 or "latent" not in kf]
            nd["minimax_keyframes"] = [anchor_kf] + kept
        else:
            nd["minimax_keyframes"] = [anchor_kf]
        nd["minimax_visual_cond_noise_aug"] = aug
        out.append([tensor, nd])
    return out


def _crop_keyframes_to_tile(cond, src_h, src_w, r0, c0, tr, tc):
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            cropped = []
            for kf in kfs:
                nkf = dict(kf)
                lt = kf.get("latent")
                if lt is not None:
                    if lt.shape[3] == src_h and lt.shape[4] == src_w:
                        nkf["latent"] = lt[:, :, :, r0:r0 + tr, c0:c0 + tc].contiguous()
                    else:
                        lt_r = F.interpolate(
                            lt.to(torch.float32), size=(src_h, src_w),
                            mode="bilinear", align_corners=False)
                        nkf["latent"] = lt_r[:, :, :, r0:r0 + tr, c0:c0 + tc].contiguous()
                cropped.append(nkf)
            nd["minimax_keyframes"] = cropped
        out.append([tensor, nd])
    return out


def _build_guider(model, cond, negative, cfg):
    guider = comfy.samplers.CFGGuider(model)
    if negative is not None:
        guider.set_conds(cond, negative)
        guider.set_cfg(cfg)
    else:
        guider.inner_set_conds({"positive": cond})
    return guider


def _sample_piece(piece, guider, noise_tensor, seed, sampler, sigmas):
    latent = dict(piece)
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model=guider.model_patcher.model, latent_image=latent_image)
    latent["samples"] = latent_image
    samples = guider.sample(noise_tensor, latent_image, sampler, sigmas,
                            denoise_mask=latent.get("noise_mask"),
                            disable_pbar=True, seed=seed)
    return samples.to(comfy.model_management.intermediate_device())


def _run_tiled_resample(latent, conditioning, model, noise, sampler, sigmas,
                        negative=None, cfg=1.0):
    """Tiled spatial+temporal resample upscale with auto-calculated tile sizes."""
    if not HAS_SAMPLER:
        raise RuntimeError("Sampler modules not available for tiled mode")

    samples = latent["samples"]
    video, audio = _get_h3_video_audio(samples)
    B, C, T, H, W = video.shape

    if B != 1:
        raise ValueError("Tiled mode only supports batch size 1")

    # Auto-calculate tile/chunk sizes
    sizes = _auto_tile_chunk_sizes(H, W, T)
    log_dasiwa("MiniMaxH3 Upscaler", f"Tiled mode auto-sizes: tile={sizes['tile_w']}x{sizes['tile_h']}, "
              f"chunk={sizes['chunk_frames']}f, overlap={sizes['overlap_w']}x{sizes['overlap_h']}")

    # Temporal segmentation
    bounds, _ = _compute_h3_segments_adaptive(T, sizes["chunk_frames"], sizes["temporal_overlap"])

    # Spatial grid
    rows, cols, trows, tcols, row_ovl, col_ovl = _compute_spatial_grid(
        H, W, sizes["tile_h"], sizes["tile_w"],
        sizes["overlap_h"], sizes["overlap_w"])
    nrows, ncols = len(rows), len(cols)

    # Noise
    noise_v = noise.generate_noise({"samples": torch.zeros_like(video, dtype=torch.float32)})
    noise_a = noise.generate_noise({"samples": torch.zeros_like(audio, dtype=torch.float32)})

    acc_v = acc_a = None

    for i, (k0, f0, k1, f1) in enumerate(bounds):
        log_dasiwa("MiniMaxH3 Upscaler", f"Tiled chunk {i}/{len(bounds)}: frames {f0}-{f1}")

        chunk_v = video[:, :, k0:k1].contiguous()
        a0, a1 = _audio_range(f0, f1)
        a1 = min(a1, audio.shape[-1])
        chunk_a = audio[:, :, :, a0:a1].contiguous()

        # Reanchor conditioning for this time segment
        cond_i = _reanchor_conditioning(conditioning, f0, f1, (H, W))

        # Frame-0 anchor from previous chunk
        if i > 0 and acc_v is not None:
            cond_i = _anchor_conditioning(cond_i, acc_v, f0, 0.999)

        chunk_out = chunk_v.clone()
        noise_vc = noise_v[:, :, k0:k1]
        noise_ac = noise_a[:, :, :, a0:a1]

        # Spatial tiling inner loop
        for ri in range(nrows):
            for cj in range(ncols):
                comfy.model_management.throw_exception_if_processing_interrupted()
                r0, c0 = rows[ri], cols[cj]
                tr, tc = trows[ri], tcols[cj]
                ovh, ovw = row_ovl[ri], col_ovl[cj]

                tile = chunk_out[:, :, :, r0:r0 + tr, c0:c0 + tc].clone()

                # Build noise mask (freeze overlap bands)
                m = _spatial_fade_mask(tr, tc, ovh, ovw,
                                       done_top=(ri > 0), done_left=(cj > 0),
                                       fade_h=sizes["fade_h"], fade_w=sizes["fade_w"])
                mv = m[None, None, None].to(chunk_out.device)
                ma = torch.zeros((1, 32, 2, chunk_a.shape[-1]),
                                 device=chunk_a.device, dtype=torch.float32)

                piece = {
                    "samples": _make_h3_nested(tile, chunk_a),
                    "noise_mask": _make_h3_nested(mv, ma)
                }
                tile_noise = _make_h3_nested(
                    noise_vc[:, :, :, r0:r0 + tr, c0:c0 + tc].contiguous(),
                    noise_ac.contiguous())

                # Crop conditioning to tile
                cond_tile = _crop_keyframes_to_tile(cond_i, H, W, r0, c0, tr, tc)
                guider = _build_guider(model, cond_tile, negative, cfg)

                out = _sample_piece(piece, guider, tile_noise, noise.seed, sampler, sigmas)
                tile_v = (out.tensors[0] if hasattr(out, 'tensors') else out).to(chunk_out.device)

                # Blend into accumulated result
                region = chunk_out[:, :, :, r0:r0 + tr, c0:c0 + tc].clone()
                if cj > 0 and ovw > 0:
                    wts = torch.linspace(0.0, 1.0, ovw, device=region.device,
                                         dtype=region.dtype).view(1, 1, 1, 1, ovw)
                    region[:, :, :, :, :ovw] = (region[:, :, :, :, :ovw] * (1.0 - wts) +
                                                tile_v[:, :, :, :, :ovw] * wts)
                if ri > 0 and ovh > 0:
                    wts = torch.linspace(0.0, 1.0, ovh, device=region.device,
                                         dtype=region.dtype).view(1, 1, 1, ovh, 1)
                    region[:, :, :, :ovh, :] = (region[:, :, :, :ovh, :] * (1.0 - wts) +
                                                tile_v[:, :, :, :ovh, :] * wts)
                # Preserve frozen bands
                band = torch.zeros((1, 1, 1, tr, tc), dtype=torch.bool, device=region.device)
                if cj > 0 and ovw > 0:
                    band[:, :, :, :, :ovw] = True
                if ri > 0 and ovh > 0:
                    band[:, :, :, :ovh, :] = True
                region = torch.where(band, region, tile_v)
                chunk_out[:, :, :, r0:r0 + tr, c0:c0 + tc] = region

                comfy.model_management.soft_empty_cache()

        # Temporal stitch
        acc_v, acc_a = _temporal_append(acc_v, acc_a, chunk_out, chunk_a, i, k0, f0)

    return {"samples": _make_h3_nested(acc_v, acc_a)}


# ==========================================
# ComfyUI node (consolidated single node)
# ==========================================
class DaSiWa_MiniMaxH3LatentUpscaler:
    """MiniMax H3 Latent Upscaler - consolidated 2D/3D backbone with three resize modes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["neural", "tiled_resample"], {"default": "neural"}),
                "latent": ("LATENT",),
                "model_name": (scan_models(),),
                "backbone": (["2d", "3d"], {"default": "3d"}),
                "resize_mode": (["scale", "target_dimensions", "megapixels"], {"default": "scale"}),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05}),
                "target_width": ("INT", {"default": 1280, "min": 64, "max": 8192, "step": 8}),
                "target_height": ("INT", {"default": 704, "min": 64, "max": 8192, "step": 8}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1}),
                "align_grid": ("INT", {"default": 32, "min": 8, "max": 64, "step": 8}),
                "device": (["auto", "cuda", "rocm", "mps", "cpu"], {"default": "auto"}),
                "precision": (["fp32", "fp16", "bf16"], {"default": "fp16"}),
            },
            "optional": {
                "force_unload": ("BOOLEAN", {"default": True}),
                "enable_temporal_chunking": ("BOOLEAN", {"default": True}),
                # Tiled resample mode inputs (ignored in neural mode)
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "noise": ("NOISE",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "video/MinimaxH3"

    def run(self, mode, latent, model_name, backbone, resize_mode, scale, target_width,
            target_height, megapixels, align_grid, device, precision,
            force_unload=True, enable_temporal_chunking=True,
            model=None, conditioning=None, negative=None, noise=None,
            sampler=None, sigmas=None, cfg=1.0):
        if mode == "tiled_resample":
            if model is None or conditioning is None or noise is None or sampler is None or sigmas is None:
                raise ValueError("Tiled resample mode requires MODEL, CONDITIONING, NOISE, SAMPLER, and SIGMAS inputs")
            return (_run_tiled_resample(latent, conditioning, model, noise, sampler, sigmas,
                                        negative=negative, cfg=cfg),)

        # Neural upscale mode (original path)
        if model_name.startswith('('):
            raise ValueError("Please place model files into the latent_upscale_models directory")

        # Handle NestedTensor (AV latent) - extract video component
        src = latent["samples"]
        log_dasiwa("MiniMaxH3 Upscaler", f"Input samples type: {type(src).__name__}")
        is_nested = _is_nested_tensor(src)
        video_samples = _extract_video_component(src)
        log_dasiwa("MiniMaxH3 Upscaler", f"Extracted video type: {type(video_samples).__name__}, shape: {video_samples.shape if hasattr(video_samples, 'shape') else 'N/A'}")

        orig_dtype = video_samples.dtype
        was_4d = (video_samples.dim() == 4)

        dev = _resolve_device(device)
        compute_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]

        # Handle batch > 1: process each batch item separately
        b_size = video_samples.shape[0]
        if b_size > 1:
            log_dasiwa("MiniMaxH3 Upscaler", f"Batch size {b_size}: processing items individually")

        results = []
        for b_idx in range(b_size):
            s = video_samples[b_idx:b_idx+1].to(device=dev, dtype=compute_dtype, copy=True)
            if was_4d:
                s = s.unsqueeze(2)  # (1, C, 1, H, W)

            b, c, t, h_in, w_in = s.shape

            # Calculate target size based on resize mode
            if resize_mode == "scale":
                scale_val = scale
                w_pixel_target = w_in * VAE_DOWNSAMPLE * scale_val
                h_pixel_target = h_in * VAE_DOWNSAMPLE * scale_val
                effective_scale = scale_val
            elif resize_mode == "target_dimensions":
                w_pixel_target = float(target_width)
                h_pixel_target = float(target_height)
                effective_scale = (w_pixel_target / (w_in * VAE_DOWNSAMPLE) +
                                   h_pixel_target / (h_in * VAE_DOWNSAMPLE)) / 2.0
            elif resize_mode == "megapixels":
                target_pixels = megapixels * 1024 * 1024
                aspect_ratio = w_in / h_in
                h_pixel_target = (target_pixels / aspect_ratio) ** 0.5
                w_pixel_target = h_pixel_target * aspect_ratio
                effective_scale = (w_pixel_target / (w_in * VAE_DOWNSAMPLE) +
                                   h_pixel_target / (h_in * VAE_DOWNSAMPLE)) / 2.0
            else:
                raise ValueError(f"Unsupported resize mode: {resize_mode}")

            # Pixel-space alignment
            w_pixel_aligned = round(w_pixel_target / align_grid) * align_grid
            h_pixel_aligned = round(h_pixel_target / align_grid) * align_grid
            w_pixel_final = round(w_pixel_aligned / VAE_DOWNSAMPLE) * VAE_DOWNSAMPLE
            h_pixel_final = round(h_pixel_aligned / VAE_DOWNSAMPLE) * VAE_DOWNSAMPLE

            w_out = max(1, int(w_pixel_final // VAE_DOWNSAMPLE))
            h_out = max(1, int(h_pixel_final // VAE_DOWNSAMPLE))

            if effective_scale < 1.0 and (w_out < w_in or h_out < h_in):
                raise ValueError("This model only supports upscaling (effective scale >= 1.0).")

            if w_out == w_in and h_out == h_in:
                log_dasiwa("MiniMaxH3 Upscaler", f"Batch {b_idx}: no size change, returning input")
                out = s
            else:
                log_dasiwa("MiniMaxH3 Upscaler",
                           f"Batch {b_idx}: Latent {w_in}x{h_in} -> {w_out}x{h_out} | "
                           f"Pixels {w_out * VAE_DOWNSAMPLE}x{h_out * VAE_DOWNSAMPLE} | scale={effective_scale:.3f}")

                # Load model (cached)
                model = load_model(model_name, dev, precision, backbone)

                # Normalize
                norm_mean, norm_std = _make_norm_tensors(dev, compute_dtype)
                s_norm = (s - norm_mean) / norm_std
                log_dasiwa("MiniMaxH3 Upscaler", f"  Input range: [{s.min().item():.4f}, {s.max().item():.4f}]")
                log_dasiwa("MiniMaxH3 Upscaler", f"  Normalized range: [{s_norm.min().item():.4f}, {s_norm.max().item():.4f}]")
                del s

                with torch.inference_mode():
                    if backbone == "2d":
                        target_hw = (h_out, w_out)
                        out = model(s_norm, scale=effective_scale, target_hw=target_hw)
                    else:  # 3d
                        target_size = (t, h_out, w_out)
                        out = model(s_norm, scale=effective_scale, target_size=target_size,
                                    enable_chunking=enable_temporal_chunking)

                del s_norm
                log_dasiwa("MiniMaxH3 Upscaler", f"  Model output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
                out = out * norm_std + norm_mean
                log_dasiwa("MiniMaxH3 Upscaler", f"  Denormalized range: [{out.min().item():.4f}, {out.max().item():.4f}]")

            # Restore dimensions
            if was_4d:
                out = out.squeeze(2)

            results.append(out.to(device="cpu", dtype=orig_dtype, non_blocking=True))

        # Stack batch results back together
        final_video = torch.cat(results, dim=0)

        # Reconstruct NestedTensor if input was nested
        if is_nested:
            final_samples = _reconstruct_nested_tensor(src, final_video)
        else:
            final_samples = final_video

        # VRAM cleanup
        if dev.type == "cuda":
            if force_unload and model_name in MODEL_CACHE:
                cache_key = f"{model_name}::{_backend_label(dev)}::{precision}::{backbone}"
                if cache_key in MODEL_CACHE:
                    MODEL_CACHE[cache_key].to("cpu", non_blocking=True)
                    log_dasiwa("MiniMaxH3 Upscaler", "Model offloaded to CPU. VRAM released.")
            if HAS_COMFY_MM:
                mm.soft_empty_cache()
            else:
                torch.cuda.empty_cache()
            gc.collect()

        return ({"samples": final_samples},)


# ==========================================
# Node registration
# ==========================================
NODE_CLASS_MAPPINGS = {
    "DaSiWa_MiniMaxH3LatentUpscaler": DaSiWa_MiniMaxH3LatentUpscaler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3LatentUpscaler": "Minimax H3 Latent Upscaler",
}
