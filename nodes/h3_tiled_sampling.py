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


# Preferred equal-split geometry adapted from Talon Upscale V2 (GPL-3.0).
def _even_split(total, count, overlap):
    if count <= 1:
        return total
    return min(total, math.ceil((total + (count - 1) * overlap) / count / 32) * 32)


def auto_local_tiles(source_width, source_height, target_width, target_height, max_tile_megapixels=2.1):
    """Prefer few source-shaped equal tiles, not a measured runtime optimum."""
    dimensions = (source_width, source_height, target_width, target_height)
    if any(not math.isfinite(float(v)) or v <= 0 for v in dimensions):
        raise ValueError('Source and target dimensions must be finite and positive')
    sw, sh, tw, th = map(int, dimensions)
    if min(sw, sh, tw, th) <= 0 or tw % 32 or th % 32:
        raise ValueError('Target dimensions must be positive multiples of 32')
    cap = float(max_tile_megapixels) * 1_000_000
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError('max_tile_megapixels must be finite and positive')
    if tw * th <= cap:
        return tw, th, 0
    aspect = sw / sh
    min_w, min_h = min(tw, 512), min(th, 512)

    def search(overlap):
        shaped, any_shape = None, None
        # Enough splits to reach the spatial floor, without an arbitrary grid cap.
        max_nx = max(1, math.ceil((tw - overlap) / (min_w - overlap)))
        max_ny = max(1, math.ceil((th - overlap) / (min_h - overlap)))
        for nx in range(1, max_nx + 1):
            for ny in range(1, max_ny + 1):
                w, h = _even_split(tw, nx, overlap), _even_split(th, ny, overlap)
                if w < min_w or h < min_h or w * h > cap or overlap > min(w, h) - 32:
                    continue
                drift = abs(math.log((w / h) / aspect))
                key = (nx * ny, drift, nx)
                if any_shape is None or key < any_shape[0]:
                    any_shape = (key, w, h)
                if drift <= math.log(1.6) and (shaped is None or key < shaped[0]):
                    shaped = (key, w, h)
        return shaped or any_shape

    overlap = min(128, min_w - 32, min_h - 32)
    best = search(overlap)
    if best is None:
        overlap = min(64, min_w - 32, min_h - 32)
        best = search(overlap)
    if best is None:
        raise ValueError('max_tile_megapixels is too small for this output size')
    return best[1], best[2], overlap


def plan_tiles(video_shape, target_width, target_height, memory_bytes,
               model_bytes=0, text_tokens=0, ref_tokens=0, *, streaming_weights=False,
               reserved_bytes=0, audio_tokens=0, latent_element_size=2, enforce_budget=True,
               spatial_tiling=True, temporal_chunking=True, keyframe_tokens=0, workspace_reserve_pct=25.0,
               anchor_keyframe=True, croppable_ref_tokens=0):
    """Budget managed weight residency, full-window buffers and attention rows.

    anchor_keyframe: temporal windows after the first carry one previous-window
    anchor keyframe token (crossfade joins). Prefix joins add no tokens.
    croppable_ref_tokens: reference rows that are cropped to each tile, so a
    forward only sees their tile-area share (ref_tokens counts the rest).

    Streaming reserves 40% of the pool for weights/casts rather than requiring
    the complete checkpoint in VRAM. This is our throughput heuristic, not a
    fixed residency requirement of ComfyUI's on-demand allocator.
    """
    if len(video_shape) != 5 or min(video_shape) <= 0:
        raise ValueError("Expected positive B,C,T,H,W video_shape")
    width, height = int(target_width), int(target_height)
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("Target dimensions must be positive multiples of 32")
    if min(memory_bytes, model_bytes, text_tokens, ref_tokens, reserved_bytes, audio_tokens, keyframe_tokens,
           croppable_ref_tokens) < 0:
        raise ValueError("Memory and conditioning estimates must be nonnegative")
    pool = max(0, int(memory_bytes) - int(reserved_bytes))
    weight_bytes = min(int(model_bytes), int(pool * 0.4)) if streaming_weights else int(model_bytes)
    reserve = float(workspace_reserve_pct)
    if not math.isfinite(reserve) or not 0 <= reserve <= 100:
        raise ValueError("Workspace reserve must be finite and between 0 and 100 percent")
    # Reserve applies after device reservations and managed weight residency.
    budget = max(0, pool - weight_bytes) * (1 - reserve / 100)
    if enforce_budget and reserve == 100:
        raise MemoryError("H3 Memory Planning: 100% workspace reserve leaves no refinement budget. "
                          "Lower memory_reserve_pct, disable Memory Planning at your own OOM risk, "
                          "or set denoise=0.")
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

    preferred = (auto_local_tiles(video_shape[4] * 16, video_shape[3] * 16, width, height)
                 if spatial_tiling else (width, height, 0))
    pw, ph, po = preferred

    def search(fixed):
        best = None
        for chunk in chunks:
            # prepare_conditioning can add one previous-window anchor (crossfade joins only).
            keyframes = int(keyframe_tokens) + int(anchor_keyframe and chunk < total)
            rows = target_rows(chunk, keyframes)
            ranges = temporal_ranges(total, chunk, period if chunk < total else 0)
            sampled_time = sum(end - start for start, end in ranges)
            widths = (pw,) if fixed else range(min_w, pw + 1, 32)
            for tw in widths:
                max_h = min(ph, rows // ((tw // 32) * (chunk + keyframes)) * 32) if enforce_budget else ph
                heights = (ph,) if fixed else range(min_h, max_h + 1, 32)
                for th in heights:
                    area = (tw // 32) * (th // 32)
                    cropped = math.ceil(int(croppable_ref_tokens) * tw * th / (width * height))
                    if enforce_budget and area * (chunk + keyframes) + cropped > rows:
                        continue
                    overlap = po if fixed else min(max(32, math.ceil(min(tw, th) / 128) * 32), min(tw, th) - 32)
                    # Same count as _starts on the even target grid.
                    nx = (width - overlap - 1) // (tw - overlap) + 1
                    ny = (height - overlap - 1) // (th - overlap) + 1
                    forwards = nx * ny * len(ranges)
                    processed_rows = nx * ny * area * sampled_time
                    score = (forwards, processed_rows, -chunk, abs(tw * height - th * width))
                    if best is None or score < best[0]:
                        best = (score, tw, th, overlap, chunk, rows)
        return best

    best = search(fixed=True)
    policy = 'preferred' if spatial_tiling else 'full_canvas'
    if best is None and spatial_tiling:
        best = search(fixed=False)
        policy = 'memory_fallback'

    if best is None:
        raise MemoryError("H3 refinement budget cannot fit the minimum "
                          f"{min_w}x{min_h}px window "
                          f"(spatial_tiling={spatial_tiling}, temporal_chunking={temporal_chunking}). "
                          "Enable tiling/chunking if disabled. "
                          "Lower memory_reserve_pct or disable Memory Planning at your own OOM risk. "
                          "Increase memory_budget_mb if capped, reduce references/target size, "
                          "use a smaller diffusion model, or set denoise=0.")
    score, tw, th, overlap, chunk, rows = best
    mode = 'streamed' if streaming_weights else 'resident'
    coordinates = 'local' if tw < width or th < height else 'full_canvas'
    return dict(layout_policy=policy, tile_coordinates=coordinates,
                tile_width=tw, tile_height=th, overlap=overlap, target_rows=rows,
                chunk_tokens=chunk, temporal_overlap_tokens=period if chunk < total else 0,
                forwards_per_step=score[0], processed_video_rows=score[1],
                keyframe_rows_per_forward=(tw // 32) * (th // 32) * (int(keyframe_tokens) + int(anchor_keyframe and chunk < total)),
                explanation=(f"Heuristic: target={width}x{height}px, pool={pool / 1024**3:.2f}GiB, {mode} weight reserve="
                             f"{weight_bytes / 1024**3:.2f}GiB; layout={policy}; coordinates={coordinates}; "
                             f"budget enforcement={'on' if enforce_budget else 'off'}; "
                             f"{'preferred layout exceeds budget; smaller spatial search; ' if policy == 'memory_fallback' else ''}"
                             f"{rows} target rows after full-window buffers, "
                             f"text/references/audio and {reserve:g}% headroom; {tw}x{th}px tiles "
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


REFERENCE_MODES = ('full', 'crop_video', 'crop_all')
VIDEO_REFERENCE_KINDS = ('video', 'video_audio')


def _crop_span(a, b, n, full):
    """[a, b) of a `full`-long target axis mapped onto an `n`-long reference axis.

    Snapped outward to the DiT's 2x2 patch grid (the native patchify needs whole patches; references
    are encoded on the 32 px grid, so n is even) and never smaller than one patch.
    """
    s = (a * n // full) // 2 * 2
    e = min(n, -((-b * n) // full) + 1) // 2 * 2 if n % 2 == 0 else min(n, -((-b * n) // full))
    if e - s < min(2, n):
        e = min(n, s + 2)
        s = max(0, e - 2)
    return s, e


def crop_reference(ref, y0, y1, x0, x1, h, w):
    """The part of one reference block under the tile [y0:y1, x0:x1] of the h x w target grid.

    Proportional box, so a reference at another resolution (Reference to Video encodes at its own
    size) still maps onto the same picture area. Audio-only blocks are returned unchanged.
    """
    latent = ref.get("latent")
    if latent is None or not ref.get("latent_h") or not ref.get("latent_w"):
        return ref
    rh, rw = latent.shape[-2], latent.shape[-1]
    ry0, ry1 = _crop_span(y0, y1, rh, h)
    rx0, rx1 = _crop_span(x0, x1, rw, w)
    out = dict(ref)
    out.update(latent=latent[..., ry0:ry1, rx0:rx1].clone(), latent_h=ry1 - ry0, latent_w=rx1 - rx0)
    return out


def crops_reference(ref, mode):
    if mode == 'crop_all':
        return ref.get("latent") is not None
    return mode == 'crop_video' and ref.get("kind") in VIDEO_REFERENCE_KINDS


class H3TiledDiffusion:
    """MiniMax H3 model_function_wrapper; preserves native mask/x0 semantics.

    Handles the packed video+audio latent of the H3 video models and the single 5-D stream of
    MiniMax-H3-Image. model_function is Comfy's apply_model (returns denoised x0), not raw DiT
    velocity. Masks are forwarded, not multiplied onto x0 a second time.
    No tensors are retained on this wrapper between forwards.

    references: 'full' gives every tile every reference block (character/style references);
    'crop_video' crops reference videos to the tile (the plate of a video edit, so each tile
    follows its own part of it and attends to a tile-sized share of its tokens); 'crop_all' also
    crops reference images (an image that is a frame of this shot). Cropping needs local
    coordinates: the native layout is rebuilt for the cropped blocks.
    """
    def __init__(self, tile_width_px, tile_height_px, overlap_px, coordinates='global', references='full'):
        if coordinates not in ('global', 'local'):
            raise ValueError('Tile coordinates must be global or local')
        if references not in REFERENCE_MODES:
            raise ValueError(f"Tile references must be one of {', '.join(REFERENCE_MODES)}")
        if references != 'full' and coordinates != 'local':
            raise ValueError('Cropped tile references need local tile coordinates')
        self.coordinates, self.references = coordinates, references
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
        if "minimax_payload" not in c:
            raise ValueError("H3 tiled diffusion requires a native MiniMax H3 model (minimax_payload)")
        shapes = c.get("latent_shapes")
        packed = shapes is not None and len(shapes) == 2
        if packed:
            video, audio = comfy.utils.unpack_latents(args["input"], shapes)
        elif shapes is not None and len(shapes) > 1:
            raise ValueError("H3 tiled diffusion supports packed video+audio or one image stream")
        else:
            video, audio = args["input"], None
        if video.ndim != 5:
            raise ValueError("H3 tiled diffusion expects 5-D H3 latents [B,C,T,H,W]")
        b, _, t, h, w = video.shape
        if b != 1:
            raise ValueError("MiniMax H3 supports batch size 1")
        # The image model has no packed AV layout to window: its tiles always use local coordinates.
        global_coordinates = self.coordinates == 'global' and packed
        payload = c["minimax_payload"]
        kfs = [dict(k) for k in payload.get("keyframes", ())]
        for k in kfs:
            if k.get("latent") is not None:
                k["latent"] = pad_to_patch_size(_resize_video(k["latent"], (h, w)), (1, 2, 2))
        hp, wp = _ceil2(h), _ceil2(w)
        text = c.get("c_crossattn")
        text_len = text.shape[1] if text is not None else 0
        full = None
        if global_coordinates:
            full = payload.get("layout")
            signature = (text_len, t, hp, wp, audio.shape[-1])
            if full is None or full.signature != signature:
                full = PackedLayout(*signature, keyframes=kfs, refs=payload.get("refs"))
        vbuf = torch.zeros(video.shape, dtype=torch.float32, device=video.device)
        weights = torch.zeros((1, 1, 1, h, w), dtype=torch.float32, device=video.device)
        abuf = torch.zeros(audio.shape, dtype=torch.float32, device=audio.device) if packed else None
        count = 0
        first_audio_prediction = None
        mask = c.get("denoise_mask")
        audio_mask = c.get("audio_denoise_mask")
        if packed and mask is not None and mask.ndim == 3 and mask.shape == args["input"].shape:
            mask, audio_mask = comfy.utils.unpack_latents(mask, c["latent_shapes"])
        # A mask without this latent's spatial grid (image stream) is forwarded untouched.
        spatial_mask = mask is not None and (packed or (mask.ndim == 5 and tuple(mask.shape[-2:]) == (h, w)))
        if spatial_mask:
            mask = torch.broadcast_to(mask, (b, mask.shape[1], t, h, w))
        refs = payload.get("refs") or ()
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
                tile_refs = refs
                if self.references != 'full' and refs:
                    tile_refs = [crop_reference(r, y, y1, x, x1, h, w) if crops_reference(r, self.references) else r
                                 for r in refs]
                    p["refs"] = tile_refs
                p["cond_video_latents"] = ([k["latent"] for k in tile_kfs if k.get("latent") is not None]
                                           + [r["latent"] for r in tile_refs if r.get("latent") is not None])
                if global_coordinates:
                    p["layout"] = _WindowLayout(full, kfs, t, hp, wp, y, x, th, tw)
                else:
                    # Native PackedLayout normalizes this tile's spatial grid;
                    # retain resolved temporal positions and (unless cropped) AV refs.
                    p.pop("layout", None)
                ct = dict(c)
                ct["transformer_options"] = dict(c.get("transformer_options", {}))
                ct["minimax_payload"] = p
                if spatial_mask:
                    ct["denoise_mask"] = mask[..., y:y1, x:x1].clone()
                if audio_mask is not None:
                    ct["audio_denoise_mask"] = audio_mask.clone()
                if packed:
                    packed_tile, tile_shapes = comfy.utils.pack_latents([video[..., y:y1, x:x1], audio])
                    ct["latent_shapes"] = tile_shapes
                    vo, ao = comfy.utils.unpack_latents(model_function(packed_tile, args["timestep"], **ct), tile_shapes)
                else:
                    vo, ao = model_function(video[..., y:y1, x:x1], args["timestep"], **ct), None
                # Positive fp32 tent weights avoid fp16 gaussian underflow.
                yy = 1 - (torch.arange(th, device=video.device).float() - (th - 1) / 2).abs() / ((th + 1) / 2)
                xx = 1 - (torch.arange(tw, device=video.device).float() - (tw - 1) / 2).abs() / ((tw + 1) / 2)
                weight = (yy[:, None] * xx[None, :])[None, None, None]
                vbuf[..., y:y1, x:x1] += vo.float() * weight
                weights[..., y:y1, x:x1] += weight
                if packed:
                    if count == 0 and audio_mask is not None:
                        first_audio_prediction = ao.clone()
                    abuf += ao.float()
                count += 1
        if not packed:
            return (vbuf / weights).to(video.dtype)
        out_audio = (abuf / count).to(audio.dtype)
        if first_audio_prediction is not None:
            # Zero-mask x0 is independent of the tile's predicted velocity.
            # Keep native x0 (including AV schedule scaling), not input audio,
            # exactly, rather than introduce summation drift on frozen samples.
            out_audio = torch.where(audio_mask.to(audio.device) == 0,
                                    first_audio_prediction.to(audio.dtype), out_audio)
        result, _ = comfy.utils.pack_latents([(vbuf / weights).to(video.dtype), out_audio])
        return result


TILE_MEMORY_KEY = 'dasiwa_tile_memory_estimate'
RESIDENCY_MODES = ('auto', 'keep', 'stream')


class TileMemoryEstimate:
    """Size ComfyUI's pre-sampling memory reservation for one tile and choose how much of the model stays loaded.

    prepare_sampling estimates the model's working memory from the complete noise shape and
    evicts/offloads weights to make room for it, although a tiled model call only ever runs one
    tile: a 4K H3 window is estimated at ~100 GB, which forces maximum offload. Core's context
    windows clamp the estimate the same way (comfy/context_windows.py _prepare_sampling_wrapper)
    but skip packed AV latents, whose stream shapes are not known at that point; the OUTER_SAMPLE
    half of this pair records them, because outer_sample receives latent_shapes before it calls
    prepare_sampling. The full-frame sampler buffers stay resident beside the tile forward, so
    1/64 of the full latent is added back.

    Weight residency is a separate choice. ComfyUI loads into whatever VRAM its estimate leaves
    free, so the reported size also decides how much of the model is resident:
      keep   - the tile estimate: as much of the model as fits stays loaded;
      stream - as little as possible stays loaded; the weights stream from RAM every step;
      auto   - keep when the whole model fits beside one tile's working memory, else stream
               (a model that cannot fit gained nothing from being partly loaded on a
               compute-bound card, and streaming uses less VRAM).
    max_resident_gb (0 = no cap) limits the resident weights in auto/keep. Dynamic (aimdo)
    models load weights on demand and only get the tile estimate. None of this changes the samples.
    """
    BUFFER_SHARE = 64

    def __init__(self, tile_h, tile_w, residency='auto', max_resident_gb=0.0):
        if tile_h <= 0 or tile_w <= 0:
            raise ValueError('Tile memory estimate needs positive latent tile sizes')
        if residency not in RESIDENCY_MODES:
            raise ValueError(f"weight_residency must be one of {', '.join(RESIDENCY_MODES)}")
        if isinstance(max_resident_gb, bool) or not isinstance(max_resident_gb, (int, float)) \
                or not math.isfinite(max_resident_gb) or max_resident_gb < 0:
            raise ValueError('max_resident_gb must be a finite number of GB, 0 for no cap')
        self.tile_h, self.tile_w = int(tile_h), int(tile_w)
        self.residency, self.max_resident_gb = residency, float(max_resident_gb)
        self.latent_shapes = None
        self.decisions = []  # one plain record per sampling run (numbers/strings only)

    def outer_sample(self, executor, *args, **kwargs):
        self.latent_shapes = kwargs.get('latent_shapes')
        try:
            return executor(*args, **kwargs)
        finally:
            self.latent_shapes = None

    def prepare_sampling(self, executor, model, noise_shape, conds, *args, **kwargs):
        shape = self.residency_shape(model, self.estimate_shape(noise_shape), conds)
        return executor(model, shape, conds, *args, **kwargs)

    def estimate_shape(self, noise_shape):
        shape = list(noise_shape)
        streams = self.latent_shapes
        if len(shape) == 3 and shape[1] == 1:
            if streams is None or len(streams) != 2 or len(streams[0]) != 5:
                return shape  # unknown packing: keep Core's full estimate
            video = list(streams[0])
            full = math.prod(video[1:])
            video[-2], video[-1] = min(video[-2], self.tile_h), min(video[-1], self.tile_w)
            tile = math.prod(video[1:]) + math.prod(streams[1][1:])
            shape[-1] = min(shape[-1], tile + math.ceil(full / self.BUFFER_SHARE))
            return shape
        if len(shape) in (4, 5):
            h, w = shape[-2], shape[-1]
            tw = min(w, self.tile_w)
            lead = math.prod(shape[2:-2])  # memory_required's area skips the channel axis
            target = lead * min(h, self.tile_h) * tw + math.ceil(lead * h * w / self.BUFFER_SHARE)
            shape[-2], shape[-1] = min(h, -(-target // (lead * tw))), tw
        return shape

    def wanted_resident(self, model_bytes, tile_minimum, reserved, room):
        """Bytes of weights to keep loaded; room = VRAM this model can have (free + its own + other loaded models)."""
        if self.residency == 'stream':
            want = 0
        elif self.residency == 'auto' and model_bytes + tile_minimum + reserved > room:
            want = 0
        else:
            want = model_bytes
        if self.max_resident_gb > 0:
            want = min(want, int(self.max_resident_gb * 2**30))
        return want

    def residency_shape(self, model, shape, conds):
        """An estimate shape whose minimum makes Core load `wanted_resident` bytes (never below the tile estimate).

        Core loads max(0, current_free - minimum_required) bytes of a non-dynamic model, after
        freeing room for it; minimum_required is estimate_memory's minimum plus the reserved VRAM.
        """
        import comfy.model_management as mm
        import comfy.sampler_helpers as sh
        if self.residency == 'keep' and self.max_resident_gb <= 0:
            return shape
        device = getattr(model, 'load_device', None)
        if device is None or mm.is_device_cpu(device) or model.is_dynamic():
            self.decisions.append(dict(mode=self.residency, applied='tile estimate only (dynamic or CPU model)'))
            return shape
        _, tile_minimum = sh.estimate_memory(model, shape, conds)
        own = id(model.model)
        others = 0
        for patcher in mm.loaded_models():
            if patcher is not None and id(patcher.model) != own and patcher.current_loaded_device() == device:
                others += int(patcher.loaded_size())
        free = int(mm.get_free_memory(device)) + int(model.loaded_size())
        room = free + others
        reserved = int(mm.extra_reserved_memory())
        model_bytes = int(model.model_size())
        want = self.wanted_resident(model_bytes, tile_minimum, reserved, room)
        record = dict(mode=self.residency, cap_gb=self.max_resident_gb, model_gb=model_bytes / 2**30,
                      tile_minimum_gb=tile_minimum / 2**30, room_gb=room / 2**30, resident_target_gb=want / 2**30)
        target = room - want - reserved
        # Keeping the whole model is the tile estimate itself; never inflate it (that would evict other models).
        if want >= model_bytes or target <= tile_minimum:
            record['applied'] = 'tile estimate'
            self.decisions.append(record)
            return shape
        # Core's estimate is affine in the latent size: solve it for the target minimum.
        probe = list(shape)
        axis = -1 if len(shape) == 3 else -2
        probe[axis] = shape[axis] * 2
        _, doubled = sh.estimate_memory(model, probe, conds)
        per_unit = (doubled - tile_minimum) / shape[axis]
        if per_unit <= 0:
            record['applied'] = 'tile estimate (estimate does not grow with size)'
            self.decisions.append(record)
            return shape
        out = list(shape)
        out[axis] = shape[axis] + math.ceil((target - tile_minimum) / per_unit)
        record['applied'] = 'residency target'
        self.decisions.append(record)
        return out


def install_tile_memory_estimate(patcher, tile_h, tile_w, residency='auto', max_resident_gb=0.0):
    """Attach the OUTER_SAMPLE + PREPARE_SAMPLING pair to a ModelPatcher (a clone the caller owns)."""
    import comfy.patcher_extension
    mp = comfy.patcher_extension.WrappersMP
    estimate = TileMemoryEstimate(tile_h, tile_w, residency, max_resident_gb)
    for kind, fn in ((mp.OUTER_SAMPLE, estimate.outer_sample), (mp.PREPARE_SAMPLING, estimate.prepare_sampling)):
        patcher.remove_wrappers_with_key(kind, TILE_MEMORY_KEY)
        patcher.add_wrapper_with_key(kind, TILE_MEMORY_KEY, fn)
    return estimate


def remove_tile_memory_estimate(patcher):
    import comfy.patcher_extension
    mp = comfy.patcher_extension.WrappersMP
    for kind in (mp.OUTER_SAMPLE, mp.PREPARE_SAMPLING):
        patcher.remove_wrappers_with_key(kind, TILE_MEMORY_KEY)
