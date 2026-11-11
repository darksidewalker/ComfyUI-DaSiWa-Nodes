"""MiniMax H3 de-rope: re-render fast-motion spans as a slowed copy of the clip, then drop the holds.

One latent token covers four frames, so whip-fast motion smears. The baseline latent's own jerk
profile picks the hot spans, those frames are held (slowed), the slowed clip is re-generated
video-to-video at partial denoise and the original clock is recovered by exact frame selection.

Ported and condensed from MATLOWAI's ComfyUI-MAINodes Motion Lab (motion.py at db1f62d:
H3JerkOracle, H3TimeSmear, H3AudioSmear, H3V2VInit, H3ExactRecover; GPL-3.0-or-later,
Copyright (C) 2026 MATLOWAI). Prepare folds oracle + smear + both encodes + the AV init into one
node; the truncated schedule is core BasicScheduler -> SplitSigmasDenoise, and the final audio is
the baseline track (MAINodes' reference_mix 1.0 default), so neither needs a node here.
"""
import json
import math

import numpy as np
import torch

LEGAL_STEP = 17  # H3 pixel lengths are 17k+5
MAX_END_TAIL = LEGAL_STEP
BRIDGE_TOKENS = 8
AUDIO_FOLLOW = 0.5  # "follow the original performance": pass 2 keeps the seeded timing, re-renders detail

PRESETS = {
    "balanced": {"q": 0.75, "d_max": 4},
    "max quality": {"q": 0.70, "d_max": 4},
    "economy": {"q": 0.85, "d_max": 3},
}


def _tok_start_frame(t):
    c, i = divmod(t, 5)
    return c * 17 + (0 if i == 0 else 4 * (i - 1) + 1)


def _frame_token(f, t_lat):
    for t in range(t_lat - 1, -1, -1):
        if _tok_start_frame(t) <= f:
            return t
    return 0


def _legal_ceil(n):
    return LEGAL_STEP * max(2, -(-(n - 5) // LEGAL_STEP)) + 5


def _token_count(frames):
    return (frames - 5) // LEGAL_STEP * 5 + 2


def jerk_profile(video):
    """Per-token |d3| of the latent values, averaged over c/h/w, with the (1,4,4,4,4) grid bias divided out."""
    v = video.detach().float().cpu().numpy()
    j = np.abs(np.diff(v, n=3, axis=2)).mean(axis=(0, 1, 3, 4))
    prof = np.pad(j, (1, v.shape[2] - len(j) - 1), mode="edge")
    for ph in range(5):
        m = prof[ph::5].mean()
        if m > 0:
            prof[ph::5] /= m
    return prof


def plan_holds(prof, length, q, d_max, bridge=BRIDGE_TOKENS):
    """Profile -> per-frame integer holds: quantile threshold, bridged valleys, C1 ramp shoulders."""
    prof = np.asarray(prof, dtype=np.float64)
    t_lat = len(prof)
    tok_d = np.where(prof >= np.quantile(prof, q), d_max, 1).astype(int)
    hot = np.where(tok_d == d_max)[0]
    for a, b in zip(hot[:-1], hot[1:]):
        if 1 < b - a <= bridge:
            tok_d[a:b + 1] = d_max
    for _ in range(d_max - 1):  # relax until |delta d| <= 1
        left = np.concatenate([[1], tok_d[:-1]])
        right = np.concatenate([tok_d[1:], [1]])
        tok_d = np.maximum(tok_d, np.maximum(left, right) - 1)
    return [int(tok_d[_frame_token(f, t_lat)]) for f in range(length)]


def expand_to_end(holds):
    """Lift a short rate-1 tail behind an expansion span to that span's rate (no jump back to real time at the end)."""
    holds = [int(h) for h in holds]
    n = len(holds)
    tail = 0
    while tail < n and holds[n - 1 - tail] == 1:
        tail += 1
    if tail == 0 or tail == n or tail > MAX_END_TAIL:
        return holds
    start = n - tail - 1
    r = holds[start]
    while start > 0 and holds[start - 1] == r:
        start -= 1
    out = holds[:start] + [r] * (n - start)
    q, rem = divmod(_legal_ceil(sum(out)) - sum(out), n - start)
    for i in range(start, n):
        out[i] += q
    for i in range(n - rem, n):
        out[i] += 1
    return out


def final_holds(prof, length, preset):
    p = PRESETS[preset]
    holds = expand_to_end(plan_holds(prof, length, p["q"], p["d_max"]))
    holds[-1] += _legal_ceil(sum(holds)) - sum(holds)  # tail pad lives in the last hold
    return holds


def _runs(holds):
    runs = []
    for h in holds:
        if runs and runs[-1][0] == h:
            runs[-1][1] += 1
        else:
            runs.append([h, 1])
    return runs


def smear_audio(audio, holds, fps):
    """Stretch the world-clock track onto the dilated clock, pitch preserved (phase vocoder per hold run)."""
    import torchaudio

    wav = audio["waveform"].detach().float().cpu()
    sr = audio["sample_rate"]
    b, c, n = wav.shape
    x = wav.reshape(b * c, n)
    n_fft, hop = 2048, 512
    window = torch.hann_window(n_fft)
    phase_adv = torch.linspace(0, math.pi * hop, n_fft // 2 + 1)[..., None]
    spf = sr / float(fps)
    xfade = max(1, int(round(0.005 * sr)))
    segs, joins, cursor, prev_tgt = [], [], 0.0, 0
    for h, count in _runs(holds):
        src = count * spf
        tgt = int(round(h * count * spf))
        s0, s1 = int(round(cursor)), int(round(cursor + src))
        cursor += src
        f = 0 if not segs else min(xfade, tgt, prev_tgt, s0 * max(h, 1))
        prev_tgt = tgt
        seg = x[:, s0 - max(1, f // max(h, 1)):min(s1, n)] if f else x[:, s0:min(s1, n)]
        if seg.shape[1] == 0:
            segs.append(torch.zeros(x.shape[0], tgt))
            joins.append(0)
            continue
        if h > 1:
            spec = torch.stft(seg, n_fft, hop, window=window, return_complex=True)
            # istft can cover at most n_frames*hop samples; clamp so a clipped last run never overruns it
            rate = min(1.0 / h, spec.shape[-1] * hop / float(f + tgt))
            spec = torchaudio.functional.phase_vocoder(spec, rate, phase_adv)
            seg = torch.istft(spec, n_fft, hop, window=window, length=f + tgt)
        if seg.shape[1] < f + tgt:
            seg = torch.nn.functional.pad(seg, (0, f + tgt - seg.shape[1]))
        segs.append(seg[:, :f + tgt])
        joins.append(f)
    parts = []
    for seg, f in zip(segs, joins):
        if f:
            t = torch.arange(1, f + 1, dtype=seg.dtype) / f
            prev = parts[-1].clone()
            prev[:, -f:] = prev[:, -f:] * torch.cos(t * (math.pi / 2)) + seg[:, :f] * torch.sin(t * (math.pi / 2))
            parts[-1] = prev
        parts.append(seg[:, f:])
    return {"waveform": torch.cat(parts, dim=1).reshape(b, c, -1).contiguous(), "sample_rate": sr}


def remap_conditioning(conditioning, holds, keep=True):
    """Move H3 keyframes (`minimax_keyframes`, absolute frame indices) onto the slowed clock.

    A single-frame keyframe at world frame f moves to the first copy of f, the frame Recover keeps,
    so an FL2VA last frame still pins the recovered last frame. Multi-frame or audio keyframes would
    need stretching too; they are dropped (the init already carries their content) and counted.
    References (`minimax_refs`) and text carry no frame index and stay untouched. keep=False drops
    every keyframe: a continuation's keyframes index the sampled clip, not the decoded timeline.
    """
    starts = hold_starts(holds, sum(holds))
    out, dropped = [], 0
    for tensor, metadata in conditioning:
        md = dict(metadata)
        if "minimax_keyframes" in md:
            kept = []
            for kf in md["minimax_keyframes"]:
                f, video = kf.get("resolved_frame_index"), kf.get("latent")
                single = (video is None or video.shape[2] == 1) and kf.get("audio_latent") is None
                if keep and single and isinstance(f, int) and 0 <= f < len(starts):
                    kept.append(dict(kf, resolved_frame_index=starts[f]))
                else:
                    dropped += 1
            md["minimax_keyframes"] = kept
        out.append([tensor, md])
    return out, dropped


def hold_starts(holds, frames):
    starts, cur = [], 0
    for h in holds:
        starts.append(cur)
        cur += h
    if cur != frames:
        raise ValueError(f"De-rope recover: hold map covers {cur} frames, the batch has {frames}")
    return starts


class DaSiWaH3DeropePrepare:
    DESCRIPTION = (
        "Reads the jerk profile of the baseline H3 latent, holds the frames of hot motion spans, encodes the "
        "slowed clip (and the baseline audio stretched onto the same clock) and returns the AV latent for a "
        "partial-denoise second pass. Wire hold_map into De-rope Recover after that pass."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "decoded baseline frames"}),
            "samples": ("LATENT", {"tooltip": "baseline H3 latent the frames were decoded from"}),
            "vae": ("VAE",),
            "preset": (list(PRESETS), {"default": "balanced",
                       "tooltip": "balanced: q 0.75 x4; max quality: q 0.70 x4 (wider spans); economy: q 0.85 x3 (cheaper)"}),
            "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
        }, "optional": {
            "audio": ("AUDIO", {"tooltip": "baseline audio; seeds pass 2 so dialogue keeps its timing"}),
            "audio_vae": ("VAE",),
            "conditioning": ("CONDITIONING", {"tooltip": "generation conditioning; its keyframes are moved onto the slowed clock"}),
            "keep_keyframes": ("BOOLEAN", {"default": True, "tooltip": "off for continuations: their keyframes index the sampled clip, not these frames, so pass 2 runs without them (the init carries their content)"}),
        }}

    RETURN_TYPES = ("LATENT", "STRING", "STRING", "CONDITIONING")
    RETURN_NAMES = ("latent", "hold_map", "report", "conditioning")
    FUNCTION = "prepare"
    CATEGORY = "DaSiWa/MiniMax H3"

    def prepare(self, images, samples, vae, preset, frame_rate, audio=None, audio_vae=None, conditioning=None,
                keep_keyframes=True):
        import comfy.nested_tensor
        from comfy_extras.nodes_minimax_h3 import temporal_shape

        z = samples["samples"]
        video = z.tensors[0] if getattr(z, "is_nested", False) else z
        n = images.shape[0]
        if (n - 5) % LEGAL_STEP or video.shape[2] != _token_count(n):
            raise ValueError(f"De-rope needs the baseline on H3's 17k+5 grid with its own latent: "
                             f"{n} frames, {video.shape[2]} latent tokens")
        holds = final_holds(jerk_profile(video), n, preset)
        length = sum(holds)
        idx = torch.tensor([i for i, h in enumerate(holds) for _ in range(h)])
        encoded = vae.encode(images.detach().cpu()[idx][:, :, :, :3])
        _, t_lat, audio_t = temporal_shape(length)
        if encoded.shape[2] != t_lat:
            raise ValueError(f"De-rope: {length} smeared frames encoded to {encoded.shape[2]} tokens, H3 needs {t_lat}")
        out_audio = torch.zeros(encoded.shape[0], 32, 2, audio_t, device=encoded.device, dtype=encoded.dtype)
        seeded = audio is not None and audio_vae is not None
        if seeded:
            stretched = smear_audio(audio, holds, frame_rate)
            wave, sr = stretched["waveform"], stretched["sample_rate"]
            vae_sr = getattr(audio_vae, "audio_sample_rate", 44100)
            if vae_sr != sr:
                import comfy.audio
                wave = comfy.audio.resample(wave, sr, vae_sr)
            a = audio_vae.encode(wave.movedim(1, -1)).to(device=encoded.device, dtype=encoded.dtype)
            if a.dim() == 3:
                a = a[None]
            # the encode is sized by the wav length and can land a tick off the video grid
            a = a[..., :audio_t] if a.shape[-1] >= audio_t else torch.nn.functional.pad(a, (0, audio_t - a.shape[-1]))
            out_audio = a[:1].expand(encoded.shape[0], -1, -1, -1).contiguous()
        latent = {"samples": comfy.nested_tensor.NestedTensor((encoded, out_audio))}
        if seeded:
            latent["noise_mask"] = comfy.nested_tensor.NestedTensor((
                torch.ones(1, 1, t_lat, encoded.shape[3], encoded.shape[4]),
                torch.full((1, 32, 2, audio_t), AUDIO_FOLLOW)))
        held = sum(1 for h in holds[:n] if h > 1)
        report = f"de-rope {preset}: {held} of {n} frames held, {n} -> {length} frames, peak x{max(holds)}"
        if conditioning is not None:
            conditioning, dropped = remap_conditioning(conditioning, holds, keep_keyframes)
            if dropped:
                report += f"; {dropped} keyframe(s) not carried into pass 2"
        print(f"[DaSiWa] H3 {report}")
        return (latent, json.dumps({"holds": holds, "world_len": n}), report, conditioning)


class DaSiWaH3DeropeRecover:
    DESCRIPTION = "Drops the held frames of a de-roped second pass: exact frame selection back to the original clock."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "hold_map": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "recover"
    CATEGORY = "DaSiWa/MiniMax H3"

    def recover(self, images, hold_map):
        holds = json.loads(hold_map)["holds"]
        return (images[torch.tensor(hold_starts(holds, images.shape[0]))].cpu(),)
