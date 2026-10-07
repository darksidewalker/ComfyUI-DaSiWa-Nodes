"""Internal H3 planning and per-step tiled diffusion (no node registration).

Global window-layout technique adapted from bbaudio's
Comfyui-MMH3-UltimateUpscale/nodes/tiled_diffusion.py, itself based on
shiimizu/ComfyUI-TiledDiffusion (MultiDiffusion / Mixture of Diffusers).
Distributed under this project's GPL-3.0 license. All tensor work is forward-local.
Memory planning is a conservative heuristic, NOT an optimality or VRAM guarantee.
"""
import math

import torch
import torch.nn.functional as F
import comfy.utils
import comfy.model_management
from comfy.ldm.common_dit import pad_to_patch_size
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE, PackedLayout


def _frames(tokens):
    q, r = divmod(tokens, len(FRAME_PER_TOKEN))
    return q * sum(FRAME_PER_TOKEN) + sum(FRAME_PER_TOKEN[:r])


def temporal_ranges(total_tokens, chunk_tokens, overlap_tokens):
    """Use period-aligned starts so each local native grid has the same phase.

    Intermediate ends are period-aligned; only the last end may be partial.
    A requested chunk smaller than two periods cannot split with overlap.
    """
    total, chunk, overlap = map(int, (total_tokens, chunk_tokens, overlap_tokens))
    if total < 0 or chunk <= 0 or overlap < 0 or overlap >= chunk:
        raise ValueError("Invalid temporal total/chunk/overlap")
    if not total:
        return []
    if total <= chunk:
        return [(0, total)]
    period = len(FRAME_PER_TOKEN)
    chunk = chunk // period * period
    overlap = math.ceil(overlap / period) * period
    if chunk <= overlap or chunk < period:
        raise ValueError("Temporal chunk must permit a native-grid-period stride")
    result = []
    start = 0
    while start < total:
        end = min(total, start + chunk)
        result.append((start, end))
        if end == total:
            break
        start += chunk - overlap
    return result


def plan_tiles(video_shape, target_width, target_height, memory_bytes,
               model_bytes=0, text_tokens=0, ref_tokens=0, *, streaming_weights=False,
               reserved_bytes=0, audio_tokens=0, latent_element_size=2, enforce_budget=True,
               spatial_tiling=True, temporal_chunking=True, keyframe_tokens=0):
    """Budget managed weight residency, full-window buffers and attention rows.

    Streaming reserves 40% of the pool for weights/casts rather than requiring
    the complete checkpoint in VRAM. This is our throughput heuristic, not a
    fixed residency requirement of ComfyUI's on-demand allocator.
    """
    if len(video_shape) != 5 or min(video_shape) <= 0:
        raise ValueError("Expected positive B,C,T,H,W video_shape")
    width, height = int(target_width), int(target_height)
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("Target dimensions must be positive multiples of 32")
    if min(memory_bytes, model_bytes, text_tokens, ref_tokens, reserved_bytes, audio_tokens, keyframe_tokens) < 0:
        raise ValueError("Memory and conditioning estimates must be nonnegative")
    pool = max(0, int(memory_bytes) - int(reserved_bytes))
    weight_bytes = min(int(model_bytes), int(pool * 0.4)) if streaming_weights else int(model_bytes)
    budget = max(0, pool - weight_bytes) * 0.65
    total = int(video_shape[2])
    period = len(FRAME_PER_TOKEN)
    chunks = list(range(2 * period, min(total, 30) + 1, period)) if temporal_chunking else []
    if not temporal_chunking or total <= 30 and total not in chunks:
        chunks.append(total)
    min_w, min_h = (min(width, 512), min(height, 512)) if spatial_tiling else (width, height)

    def target_rows(tokens, keyframes):
        # Full spatial window remains packed outside the tile wrapper. Account
        # for sampler input/noise/masks/x0 and fp32 blending/division buffers.
        elements = video_shape[0] * video_shape[1] * tokens * (height // 16) * (width // 16)
        buffers = elements * (6 * latent_element_size + 8)
        # Keyframes also exist on the full target grid outside the wrapper.
        buffers += video_shape[0] * video_shape[1] * keyframes * (height // 16) * (width // 16) * (latent_element_size + 4)
        return int((budget - buffers) / (96 * 1024)) - int(text_tokens) - int(ref_tokens) - int(audio_tokens)

    best = None
    for chunk in chunks:
        # prepare_conditioning can add one previous-window anchor.
        keyframes = int(keyframe_tokens) + int(chunk < total)
        rows = target_rows(chunk, keyframes)
        ranges = temporal_ranges(total, chunk, period if chunk < total else 0)
        sampled_time = sum(end - start for start, end in ranges)
        widths = range(min_w, width + 1, 32) if spatial_tiling else (width,)
        for tw in widths:
            max_h = min(height, rows // ((tw // 32) * (chunk + keyframes)) * 32) if enforce_budget else height
            heights = range(min_h, max_h + 1, 32) if spatial_tiling else (height,)
            for th in heights:
                area = (tw // 32) * (th // 32)
                if enforce_budget and area * (chunk + keyframes) > rows:
                    continue
                overlap = min(max(32, math.ceil(min(tw, th) / 128) * 32), min(tw, th) - 32) if spatial_tiling else 0
                # Exactly the count from _starts, including its pulled-back last
                # tile. Do not allocate tile/window lists per candidate.
                nx = (width - overlap - 1) // (tw - overlap) + 1
                ny = (height - overlap - 1) // (th - overlap) + 1
                forwards = nx * ny * len(ranges)
                processed_rows = nx * ny * area * sampled_time
                # Geometric/offload proxy, not a measured attention runtime:
                # fewest full-model calls, least repeated work, longest context.
                score = (forwards, processed_rows, -chunk, abs(tw * height - th * width))
                if best is None or score < best[0]:
                    best = (score, tw, th, overlap, chunk, rows)
    if best is None:
        raise MemoryError("H3 refinement budget cannot fit the minimum "
                          f"{min_w}x{min_h}px window "
                          f"(spatial_tiling={spatial_tiling}, temporal_chunking={temporal_chunking}). "
                          "Enable tiling/chunking if disabled. "
                          "Increase memory_budget_mb if capped, reduce references/target size, "
                          "use a smaller diffusion model, or set denoise=0.")
    score, tw, th, overlap, chunk, rows = best
    mode = 'streamed' if streaming_weights else 'resident'
    return dict(tile_width=tw, tile_height=th, overlap=overlap, target_rows=rows,
                chunk_tokens=chunk, temporal_overlap_tokens=period if chunk < total else 0,
                forwards_per_step=score[0], processed_video_rows=score[1],
                keyframe_rows_per_forward=(tw // 32) * (th // 32) * (int(keyframe_tokens) + int(chunk < total)),
                explanation=(f"Heuristic: target={width}x{height}px, pool={pool / 1024**3:.2f}GiB, {mode} weight reserve="
                             f"{weight_bytes / 1024**3:.2f}GiB; {rows} target rows after full-window buffers, "
                             f"text/references/audio and 35% headroom; {tw}x{th}px tiles "
                             f"(minimum {min_w}x{min_h}px), {chunk} temporal tokens; "
                             f"{score[0]} forwards/step, {score[1]} processed video rows. Native period={period}. "
                             "Grid search minimizes calls then repeated rows, not measured runtime. "
                             "Not a VRAM guarantee; attention workspace still costs memory."))


def _resize_video(video, hw):
    if tuple(video.shape[-2:]) == tuple(hw):
        return video
    b, c, t, h, w = video.shape
    # B,C,T is not contiguous B,T,C: explicitly permute before flattening.
    images = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    images = F.interpolate(images.float(), size=hw, mode="bilinear", align_corners=False)
    return images.reshape(b, t, c, *hw).permute(0, 2, 1, 3, 4).to(video.dtype)


def _cut_keyframes(keyframes, f0, f1, hw):
    out = []
    for kf in keyframes or ():
        origin = kf["resolved_frame_index"]
        video, audio = kf.get("latent"), kf.get("audio_latent")
        if video is not None:
            video = _resize_video(video, hw)
            # Single-token anchors retain exact native global positions even
            # when clipping begins at a nonzero keyframe-local grid phase.
            for i in range(video.shape[2]):
                pos = origin + _frames(i)
                if f0 <= pos and pos + FRAME_PER_TOKEN[i % len(FRAME_PER_TOKEN)] <= f1:
                    item = dict(kf)
                    item.pop("audio_latent", None)
                    item.update(resolved_frame_index=pos - f0, latent=video[:, :, i:i + 1].clone())
                    out.append(item)
        if audio is not None:
            a = max(0, math.ceil((f0 - origin) * FRAME_RESCALE))
            b = min(audio.shape[-1], math.floor((f1 - origin) * FRAME_RESCALE))
            if b > a:
                item = dict(kf)
                item.pop("latent", None)
                item.update(resolved_frame_index=origin + a / FRAME_RESCALE - f0,
                            audio_latent=audio[..., a:b].clone())
                out.append(item)
        if video is None and audio is None and f0 <= origin < f1:
            out.append(dict(kf, resolved_frame_index=origin - f0))
    return out


def prepare_conditioning(conditioning, start_token, end_token, spatial_latent_hw, previous_video=None):
    """Copy Comfy CONDITIONING, reanchor keyframes, leave refs/audio/layout metadata.

    previous_video is an accumulated full-timeline B,C,T,H,W tensor; its token
    at start_token anchors the new chunk (not the previous chunk's last token).
    Accepts public CONDITIONING metadata (minimax_keyframes), not the internal
    apply_model payload. User 'layout' metadata is preserved; native extra_conds
    builds a fresh packed layout. Parent may replace the anchor using VAE.
    """
    if start_token < 0 or end_token <= start_token or min(spatial_latent_hw) <= 0:
        raise ValueError("Invalid conditioning window/grid")
    f0, f1 = _frames(start_token), _frames(end_token)
    out = []
    for tensor, metadata in conditioning:
        md = dict(metadata)
        kfs = _cut_keyframes(md.get("minimax_keyframes"), f0, f1, spatial_latent_hw)
        if previous_video is not None:
            if start_token >= previous_video.shape[2]:
                raise ValueError("Previous video does not cover the chunk start token")
            anchor = _resize_video(previous_video[:, :, start_token:start_token + 1], spatial_latent_hw).clone()
            kfs = [k for k in kfs if not (k["resolved_frame_index"] == 0 and k.get("latent") is not None)]
            kfs.insert(0, dict(resolved_frame_index=0, latent=anchor))
        if "minimax_keyframes" in md or kfs:
            md["minimax_keyframes"] = kfs
        out.append([tensor, md])
    return out


def append_video(accum, chunk, start_token):
    """Append/crossfade video tokens without touching inputs or any audio."""
    start = int(start_token)
    if start < 0 or chunk.ndim != 5:
        raise ValueError("Invalid video chunk/start")
    if accum is None:
        if start:
            raise ValueError("First video chunk must start at zero")
        return chunk.clone()
    if start > accum.shape[2] or accum.shape[:2] + accum.shape[3:] != chunk.shape[:2] + chunk.shape[3:]:
        raise ValueError("Video chunks have a gap or incompatible grids")
    chunk = chunk.to(device=accum.device, dtype=accum.dtype)
    end = start + chunk.shape[2]
    result = accum.new_empty((*accum.shape[:2], max(end, accum.shape[2]), *accum.shape[3:]))
    result[:, :, :accum.shape[2]] = accum
    overlap = min(accum.shape[2] - start, chunk.shape[2])
    if overlap:
        weight = torch.arange(1, overlap + 1, device=accum.device, dtype=torch.float32) / (overlap + 1)
        weight = weight.view(1, 1, -1, 1, 1)
        old = accum[:, :, start:start + overlap].float()
        result[:, :, start:start + overlap] = torch.lerp(old, chunk[:, :, :overlap].float(), weight).to(accum.dtype)
    result[:, :, start + overlap:end] = chunk[:, :, overlap:]
    return result


def _ceil2(n):
    return (n + 1) // 2 * 2


def _starts(total, tile, overlap):
    # Align against the padded GLOBAL grid, never total-tile for an odd total.
    padded = _ceil2(total)
    if tile >= padded:
        return [0]
    starts = list(range(0, padded - tile + 1, tile - overlap))
    if starts[-1] + tile < padded:
        starts.append(padded - tile)
    return starts


class _WindowLayout:
    def __init__(self, full, keyframes, t, hp, wp, y, x, h, w):
        rows = (torch.arange(y // 2, (y + _ceil2(h)) // 2)[:, None] * (wp // 2)
                + torch.arange(x // 2, (x + _ceil2(w)) // 2)[None, :]).reshape(-1)
        frame_rows = (hp // 2) * (wp // 2)
        kf_t = iter(k["latent"].shape[2] for k in keyframes or () if k.get("latent") is not None)
        pos, ip, iu, ap, au, segments = [], [], [], [], [], []
        cursor = 0
        for a, b, kind in full.segments:
            p = full.position_ids[a:b]
            if kind in ("video", "cond"):
                vt = t if kind == "video" else next(kf_t)
                p = p.reshape(vt, frame_rows, 3)[:, rows].reshape(-1, 3)
            n = len(p)
            pos.append(p)
            segments.append((cursor, cursor + n, kind))
            indices = torch.arange(cursor, cursor + n)
            if kind in ("video", "cond", "ref_img"):
                ip.append(indices)
                iu.append(torch.full((n,), kind == "video", dtype=torch.bool))
            elif kind in ("audio", "cond_audio", "ref_audio"):
                ap.append(indices)
                au.append(torch.full((n,), kind == "audio", dtype=torch.bool))
            cursor += n
        self.position_ids = torch.cat(pos)
        self.img_pos, self.img_update = torch.cat(ip), torch.cat(iu)
        self.audio_pos, self.audio_update = torch.cat(ap), torch.cat(au)
        self.segments, self.seq_len = segments, cursor
        self.signature = (full.signature[0], t, _ceil2(h), _ceil2(w), full.signature[-1])


class H3TiledDiffusion:
    """Packed-AV model_function_wrapper; preserves native mask/x0 semantics.

    model_function is Comfy's apply_model (returns denoised x0), not raw DiT
    velocity. Masks are forwarded, not multiplied onto x0 a second time.
    No tensors are retained on this wrapper between forwards.
    """
    def __init__(self, tile_width_px, tile_height_px, overlap_px):
        if min(tile_width_px, tile_height_px) <= 0 or overlap_px < 0 or any(
                v % 32 for v in (tile_width_px, tile_height_px, overlap_px)):
            raise ValueError("Tile sizes and overlap must be multiples of 32 pixels")
        if overlap_px >= min(tile_width_px, tile_height_px):
            raise ValueError("Overlap must be smaller than both tiles")
        self.tile_w, self.tile_h, self.overlap = tile_width_px // 16, tile_height_px // 16, overlap_px // 16

    def __call__(self, model_function, args):
        c = args["c"]
        if c.get("control") is not None:
            raise ValueError("H3 tiled diffusion does not support spatial ControlNet")
        if "minimax_payload" not in c or len(c.get("latent_shapes", ())) != 2:
            raise ValueError("H3 tiled diffusion requires packed video/audio and minimax_payload")
        video, audio = comfy.utils.unpack_latents(args["input"], c["latent_shapes"])
        b, _, t, h, w = video.shape
        if b != 1:
            raise ValueError("MiniMax H3 supports batch size 1")
        payload = c["minimax_payload"]
        kfs = [dict(k) for k in payload.get("keyframes", ())]
        for k in kfs:
            if k.get("latent") is not None:
                k["latent"] = pad_to_patch_size(_resize_video(k["latent"], (h, w)), (1, 2, 2))
        hp, wp = _ceil2(h), _ceil2(w)
        text = c.get("c_crossattn")
        text_len = text.shape[1] if text is not None else 0
        full = payload.get("layout")
        signature = (text_len, t, hp, wp, audio.shape[-1])
        if full is None or full.signature != signature:
            full = PackedLayout(*signature, keyframes=kfs, refs=payload.get("refs"))
        vbuf = torch.zeros(video.shape, dtype=torch.float32, device=video.device)
        weights = torch.zeros((1, 1, 1, h, w), dtype=torch.float32, device=video.device)
        abuf = torch.zeros(audio.shape, dtype=torch.float32, device=audio.device)
        count = 0
        first_audio_prediction = None
        mask = c.get("denoise_mask")
        audio_mask = c.get("audio_denoise_mask")
        if mask is not None and mask.ndim == 3 and mask.shape == args["input"].shape:
            mask, audio_mask = comfy.utils.unpack_latents(mask, c["latent_shapes"])
        if mask is not None:
            mask = torch.broadcast_to(mask, (b, mask.shape[1], t, h, w))
        for y in _starts(h, self.tile_h, self.overlap):
            for x in _starts(w, self.tile_w, self.overlap):
                comfy.model_management.throw_exception_if_processing_interrupted()
                y1, x1 = min(h, y + self.tile_h), min(w, x + self.tile_w)
                th, tw = y1 - y, x1 - x
                p = dict(payload)
                tile_kfs = []
                for k in kfs:
                    nk = dict(k)
                    if nk.get("latent") is not None:
                        nk["latent"] = k["latent"][..., y:y + _ceil2(th), x:x + _ceil2(tw)].clone()
                    tile_kfs.append(nk)
                p["keyframes"] = tile_kfs
                refs = payload.get("refs", ())
                p["cond_video_latents"] = [k["latent"] for k in tile_kfs if k.get("latent") is not None] + [r["latent"] for r in refs if r.get("latent") is not None]
                p["layout"] = _WindowLayout(full, kfs, t, hp, wp, y, x, th, tw)
                ct = dict(c)
                ct["transformer_options"] = dict(c.get("transformer_options", {}))
                ct["minimax_payload"] = p
                if mask is not None:
                    ct["denoise_mask"] = mask[..., y:y1, x:x1].clone()
                if audio_mask is not None:
                    ct["audio_denoise_mask"] = audio_mask.clone()
                packed, shapes = comfy.utils.pack_latents([video[..., y:y1, x:x1], audio])
                ct["latent_shapes"] = shapes
                vo, ao = comfy.utils.unpack_latents(model_function(packed, args["timestep"], **ct), shapes)
                # Positive fp32 tent weights avoid fp16 gaussian underflow.
                yy = 1 - (torch.arange(th, device=video.device).float() - (th - 1) / 2).abs() / ((th + 1) / 2)
                xx = 1 - (torch.arange(tw, device=video.device).float() - (tw - 1) / 2).abs() / ((tw + 1) / 2)
                weight = (yy[:, None] * xx[None, :])[None, None, None]
                vbuf[..., y:y1, x:x1] += vo.float() * weight
                weights[..., y:y1, x:x1] += weight
                if count == 0 and audio_mask is not None:
                    first_audio_prediction = ao.clone()
                abuf += ao.float()
                count += 1
        out_audio = (abuf / count).to(audio.dtype)
        if first_audio_prediction is not None:
            # Zero-mask x0 is independent of the tile's predicted velocity.
            # Keep native x0 (including AV schedule scaling), not input audio,
            # exactly, rather than introduce summation drift on frozen samples.
            out_audio = torch.where(audio_mask.to(audio.device) == 0,
                                    first_audio_prediction.to(audio.dtype), out_audio)
        result, _ = comfy.utils.pack_latents([(vbuf / weights).to(video.dtype), out_audio])
        return result
