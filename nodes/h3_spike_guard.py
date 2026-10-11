"""Frame-streamed H3 latent spike correction.

Adapted from the submitted Talon Upscale Node fix V4 spike_guard_fn (GPL-3.0).
This modified integration retains the DaSiWa repository's GPL-3.0 license.

Checkpoint heuristic, not a universal artifact classifier: the signed 24-channel
burn direction was measured on four patches in two scenes (similarity .74-.89)
in the submitted Talon v4 experiment. Other channel counts use its RMS fallback.
Output is caller-owned and deliberately mutated in place, as in that experiment.
Statistics and correction stay float32; only one output frame is converted at once.
"""
import torch
import torch.nn.functional as F

BURN_DIRECTION = (0.021, -0.081, 0.017, 0.119, -0.009, -0.203, -0.023, 0.016,
                  -0.062, 0.02, -0.068, 0.045, -0.316, 0.004, -0.258, -0.278,
                  -0.029, 0.508, -0.045, 0.367, 0.001, 0.239, 0.465, 0.109)


def spike_guard_memory_bytes(source_shape, output_shape):
    """Conservative additional CPU working-set bound, excluding caller tensors.

    16 bytes/source element covers float32 source, coarse residual and frame
    statistics temporaries; 96 bytes/coarse cell covers scores, MAD/median
    workspace, morphology maps and pool indices. Padding gets a separate
    allowance. 32 bytes/output-frame element covers float32 conversion and
    bilinear correction, dtype cast (up to float64), comparisons and reduction
    workspaces. Frame-size, not clip-size, output accounting is intentional.
    A 1 MiB fixed allowance covers small statistics/allocator workspaces. This
    is a conservative tensor working-set estimate, not an RSS/allocator cap.
    """
    source_shape, output_shape = tuple(source_shape), tuple(output_shape)
    if (len(source_shape) != 5 or len(output_shape) != 5
            or any(int(n) != n or n <= 0 for n in source_shape + output_shape)
            or source_shape[:3] != output_shape[:3]):
        raise ValueError('spike guard requires matching positive B,C,T and five-dimensional latents')
    b, c, t, sh, sw = (int(n) for n in source_shape)
    oh, ow = (int(n) for n in output_shape[-2:])
    cells = b * t * sh * sw
    padded = b * t * (sh + 2) * (sw + 2)
    return 16 * c * cells + 96 * cells + 16 * padded + 32 * b * c * oh * ow + 64 * c + 1024**2


@torch.no_grad()
def spike_guard(source, upscaled, sensitivity=4.3, strength=1.0, mode='burn_direction'):
    """Return (the same output tensor, measured diagnostic text).

    ``burn_direction`` scores only the positive measured signature; ``both``
    scores its absolute value. Non-24-channel latents use unsigned RMS in
    either mode. Growth (.58 of the seed threshold, at most 96 steps), halo
    smoothing and two correction passes preserve the submitted algorithm.
    Cancellation propagates; already corrected frames remain caller-owned.
    """
    if mode not in ('burn_direction', 'both'):
        raise ValueError("spike guard mode must be 'burn_direction' or 'both'")
    if strength == 0:
        return upscaled, 'spike guard: off (strength=0); changed latent frames []; output footprint 0 pixels'
    from comfy import model_management as mm
    required = spike_guard_memory_bytes(source.shape, upscaled.shape)
    mm.throw_exception_if_processing_interrupted()
    available = mm.get_free_memory(torch.device('cpu'))
    if available < required:
        raise MemoryError(f'spike guard: CPU RAM required {required} bytes, available {int(available)} bytes '
                          f'(source={tuple(source.shape)}, output={tuple(upscaled.shape)}; additional working set)')
    src = source.to(device='cpu', dtype=torch.float32)
    b, c, t, sh, sw = src.shape
    oh, ow = upscaled.shape[-2:]
    sd = src.std(dim=(0, 2, 3, 4), keepdim=True).clamp_min_(1e-6)
    err = torch.empty_like(src)
    for i in range(t):
        mm.throw_exception_if_processing_interrupted()
        frame = upscaled[:, :, i].to(device='cpu', dtype=torch.float32)
        err[:, :, i] = F.interpolate(frame, size=(sh, sw), mode='area')
    err.sub_(src).div_(sd)
    shift = err.mean(dim=(0, 2, 3, 4), keepdim=True)
    score = torch.empty((b, 1, t, sh, sw), dtype=torch.float32, device='cpu')
    direction = torch.tensor(BURN_DIRECTION, dtype=torch.float32, device='cpu').view(1, c, 1, 1) if c == 24 else None
    for i in range(t):
        mm.throw_exception_if_processing_interrupted()
        residual = err[:, :, i] - shift[:, :, 0]
        if direction is not None:
            score[:, :, i] = (residual * direction).sum(dim=1, keepdim=True)
        else:
            score[:, :, i] = residual.square().mean(dim=1, keepdim=True).sqrt()
    del err, residual, frame
    if direction is not None:
        mm.throw_exception_if_processing_interrupted()
        centre = score.median()
        spread = (score - centre).abs().median() * 1.4826
        score.sub_(centre).div_(spread.clamp_min_(1e-6))
        if mode == 'both':
            score.abs_()
        median = 1.0
    else:
        median = float(score.median())
    peak = float(score.max())
    if median <= 0 or sensitivity <= 0:
        return upscaled, 'spike guard: nothing to compare; changed latent frames []; output footprint 0 pixels'
    high = sensitivity * median
    low = high * .58
    region = (score > high).float()
    allowed = (score > low).float()
    for _ in range(96):
        mm.throw_exception_if_processing_interrupted()
        grown = F.max_pool3d(region, 3, 1, 1) * allowed
        if torch.equal(grown, region):
            break
        region = grown
    cells = int(region.sum())
    if cells == 0:
        return upscaled, f'spike guard: clean (peak {peak / median:.1f}x; acts above {sensitivity:g}x); changed latent frames []; output footprint 0 pixels'
    soft = ((score - low) / max(1e-6, high - low)).clamp_(0, 1) * region
    weight = F.max_pool3d(soft, 3, 1, 1)
    weight = F.avg_pool3d(F.pad(weight, (1, 1, 1, 1, 0, 0), mode='replicate'), (1, 3, 3), 1)
    frames = []
    bounds = None
    changed_cells = 0
    for i in range(t):
        mm.throw_exception_if_processing_interrupted()
        w = weight[:, :, i]
        if not bool(w.any()):
            continue
        original = upscaled[:, :, i].to(device='cpu', dtype=torch.float32, copy=True)
        cur = original.clone()
        target = src[:, :, i] + shift[:, :, 0] * sd[:, :, 0]
        for _ in range(2):
            mm.throw_exception_if_processing_interrupted()
            fix = (target - F.interpolate(cur, size=(sh, sw), mode='area')) * w * float(strength)
            cur.add_(F.interpolate(fix, size=(oh, ow), mode='bilinear', align_corners=False))
        # Preserve original dtype bits outside the representable float32 change,
        # including float64 precision and signed zeros in untouched pixels.
        converted = torch.where((cur != original).to(upscaled.device),
                                cur.to(device=upscaled.device, dtype=upscaled.dtype), upscaled[:, :, i])
        changed = (converted != upscaled[:, :, i]).any(dim=1).cpu()
        if bool(changed.any()):
            frames.append(i)
            changed_cells += int(changed.sum())
            spatial = changed.any(dim=0)
            ys = spatial.any(dim=1).nonzero().flatten()
            xs = spatial.any(dim=0).nonzero().flatten()
            box = (int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1)
            bounds = box if bounds is None else (min(bounds[0], box[0]), max(bounds[1], box[1]), min(bounds[2], box[2]), max(bounds[3], box[3]))
            upscaled[:, :, i] = converted
        del original, cur, converted, changed
    if bounds is None:
        return upscaled, f'spike guard: detected {cells} cells; no representable change; changed latent frames []; output footprint 0 pixels'
    x0, x1, y0, y1 = (value * 16 for value in bounds)
    return upscaled, (f'spike guard: repaired {cells} detected cells; changed latent frames {frames}; '
                      f'output area x{x0}-{x1} y{y0}-{y1} (exclusive bounds); '
                      f'output footprint {changed_cells * 256} pixels (latent grid x16); '
                      f'peak {peak / median:.1f}x (acts above {sensitivity:g}x, strength={strength:g}, mode={mode})')
