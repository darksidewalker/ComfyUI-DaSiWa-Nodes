"""Automatic circular overlap morphing; uses ComfyUI's native interpolation models."""
import logging
import math
import os

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def _check_interrupt():
    import comfy.model_management
    comfy.model_management.throw_exception_if_processing_interrupted()


def _preview(images):
    h, w = images.shape[1:3]
    size = (max(2, round(h * min(1, 48 / max(h, w)))),
            max(2, round(w * min(1, 48 / max(h, w)))))
    rows = []
    for frame in images:
        _check_interrupt()
        if not torch.isfinite(frame).all():
            raise ValueError("Seamless Loop input contains NaN or infinity.")
        rows.append(F.interpolate(frame.movedim(-1, 0)[None].float(), size=size,
                                  mode="area").to('cpu')[0])
    return torch.stack(rows)


def _pair_metrics(a, b):
    mse = (a - b).square().mean()
    mu_a = F.avg_pool2d(a, 3, 1, 1)
    mu_b = F.avg_pool2d(b, 3, 1, 1)
    va = (F.avg_pool2d(a.square(), 3, 1, 1) - mu_a.square()).clamp_min(0)
    vb = (F.avg_pool2d(b.square(), 3, 1, 1) - mu_b.square()).clamp_min(0)
    cov = F.avg_pool2d(a * b, 3, 1, 1) - mu_a * mu_b
    ssim = (((2 * mu_a * mu_b + .01 ** 2) * (2 * cov + .03 ** 2)) /
            ((mu_a.square() + mu_b.square() + .01 ** 2) * (va + vb + .03 ** 2))).mean()
    edge = ((a[..., 1:] - a[..., :-1]) - (b[..., 1:] - b[..., :-1])).square().mean()
    edge += ((a[..., 1:, :] - a[..., :-1, :]) - (b[..., 1:, :] - b[..., :-1, :])).square().mean()
    return float(mse), float(ssim), float(edge)


def _analyze(preview):
    n = len(preview)
    if n < 6:
        return 0, n, 0, 0.0
    trim = int(n * .10)
    offsets = sorted({0, trim // 2, trim})
    best = None
    for start in offsets:
        for end_trim in offsets:
            _check_interrupt()
            end = n - end_trim
            length = end - start
            overlaps = sorted({max(3, min(64, round(length * fraction), length // 3))
                               for fraction in (.08, .12, .18, .25)})
            for k in overlaps:
                a, b = preview[end-k:end], preview[start:start+k]
                # Small, spatially uniform exposure differences are correctable; geometry is not.
                delta = (b.mean((-2, -1), keepdim=True) - a.mean((-2, -1), keepdim=True)).clamp(-.04, .04)
                mse, ssim, edge = _pair_metrics(a + delta, b)
                motion = float(((a[1:] - a[:-1]) - (b[1:] - b[:-1])).square().mean())
                step = (preview[start+1:end] - preview[start:end-1]).square().mean((1, 2, 3))
                typical = max(float(step.median()), 1e-6)
                cut = float(step.max()) / typical
                cost = mse + .012 * (1 - ssim) + .25 * edge + 2 * motion
                cost += .015 * ((n - length + k) / n) + .002 * min(cut / 100, 1)
                candidate = (cost, start, end, k, mse)
                if best is None or candidate < best:
                    best = candidate
    _, start, end, k, mse = best
    return start, end, k, -10 * math.log10(max(mse, 1e-12))


def _load_interpolator(model_name, shape):
    import folder_paths
    import comfy.model_management as mm
    from comfy_extras.nodes_frame_interpolation import FrameInterpolationModelLoader
    if not model_name.lower().endswith('.safetensors'):
        raise ValueError("Select a native RIFE/FILM .safetensors model from models/frame_interpolation.")
    folder_paths.get_full_path_or_raise('frame_interpolation', model_name)
    patcher = FrameInterpolationModelLoader.execute(model_name)[0]
    return _interpolator_from_patcher(patcher, shape)


def _interpolator_from_patcher(patcher, shape):
    import comfy.model_management as mm
    model = patcher.model
    dtype, device = patcher.model_dtype(), patcher.load_device
    h, w = shape[1:3]
    # Native FILM lacks pad_align, but its seven-level pyramid also needs multiples of 64.
    align = getattr(model, 'pad_align', 64)
    ph, pw = math.ceil(h / align) * align, math.ceil(w / align) * align
    mm.load_models_gpu([patcher], memory_required=model.memory_used_forward((1, ph, pw, 3), dtype))

    def interpolate(a, b, t):
        _check_interrupt()
        model = patcher.model  # Keep the native patcher alive for the entire interpolation call.
        x = F.pad(a.movedim(-1, 0)[None].to(device=device, dtype=dtype),
                  (0, pw-w, 0, ph-h), mode='replicate')
        y = F.pad(b.movedim(-1, 0)[None].to(device=device, dtype=dtype),
                  (0, pw-w, 0, ph-h), mode='replicate')
        # Symmetric inference reduces direction-dependent morph artifacts.
        forward = model(x, y, timestep=t)
        backward = model(y, x, timestep=1-t)
        result = ((forward + backward) * .5)[0, :, :h, :w].movedim(0, -1)
        if not torch.isfinite(result).all():
            raise RuntimeError("Frame interpolation produced non-finite pixels; check the selected checkpoint.")
        return result.clamp(0, 1).to(device=a.device, dtype=a.dtype)
    return interpolate


class DaSiWa_SeamlessLoop:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = "DaSiWa/video"
    DESCRIPTION = "Automatically selects and morphs a circular overlap. Arbitrary footage cannot be guaranteed artifact-free."

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        models = [name for name in folder_paths.get_filename_list('frame_interpolation')
                  if name.lower().endswith('.safetensors')]
        return {"required": {"images": ("IMAGE",)}, "optional": {
            "model_name": (models, {"tooltip": "Native RIFE or FILM safetensors in models/frame_interpolation."}),
            "exact_endpoint": ("BOOLEAN", {"socketless": True, "default": False, "tooltip": "Copy the first frame to the end: exact pixels, but one duplicate frame on repeat. Disable for continuous playback."}),
        }}

    @classmethod
    def IS_CHANGED(cls, images, model_name='', exact_endpoint=False):
        import folder_paths
        path = folder_paths.get_full_path('frame_interpolation', model_name)
        if path:
            stat = os.stat(path)
            return (stat.st_mtime_ns, stat.st_size)
        return model_name

    @torch.inference_mode()
    def execute(self, images, model_name='', exact_endpoint=False):
        if images.ndim != 4 or images.shape[-1] != 3 or images.shape[0] == 0 or min(images.shape[1:3]) < 1:
            raise ValueError("Seamless Loop expects a non-empty RGB IMAGE batch [frames, height, width, 3].")
        preview = _preview(images)
        if all(torch.equal(images[0], frame) for frame in images):
            return (images.clone(),)
        # Already closed footage should not be shortened or re-interpolated.
        if torch.equal(images[0], images[-1]):
            return (images.clone() if exact_endpoint else images[:-1].clone(),)
        start, end, k, psnr = _analyze(preview)
        import comfy.model_management as mm
        output_device = mm.intermediate_device()
        interpolate = _load_interpolator(model_name, images.shape)
        if not k:
            # Tiny batches have insufficient context for overlap analysis: bridge the two endpoints.
            k = max(3, len(images))
            output = torch.empty((len(images) + k + int(exact_endpoint), *images.shape[1:]),
                                 dtype=images.dtype, device=output_device)
            output[:len(images)] = images
            for j in range(k):
                _check_interrupt()
                output[len(images)+j] = interpolate(images[-1], images[0], (j+1)/(k+1))
            count = len(images) + k
        else:
            count = end - start - k
            output = torch.empty((count + int(exact_endpoint), *images.shape[1:]),
                                 dtype=images.dtype, device=output_device)
            middle = end - start - 2*k
            output[:middle] = images[start+k:end-k]
            from comfy.utils import ProgressBar
            progress = ProgressBar(k)
            for j in range(k):
                _check_interrupt()
                a, b = images[end-k+j], images[start+j]
                u = j / (k-1)
                t = u*u*(3-2*u)
                if j == 0:
                    frame = a
                elif j == k-1:
                    frame = b
                else:
                    # Bounded local exposure matching, with zero correction at both joins.
                    delta = (b.float().mean((0, 1)) - a.float().mean((0, 1))).clamp(-.04, .04)
                    weight = 4*t*(1-t)
                    aa = (a + delta.to(a.dtype)*(.5*weight)).clamp(0, 1)
                    bb = (b - delta.to(b.dtype)*(.5*weight)).clamp(0, 1)
                    frame = interpolate(aa, bb, t)
                output[middle+j] = frame
                progress.update(1)
        if exact_endpoint:
            output[-1].copy_(output[0])
        # The visible wrap lies in the untouched source, not at an interpolator endpoint.
        quality = _preview(output[:count])
        jumps = (quality.roll(-1, 0) - quality).square().mean((1, 2, 3))
        ratio = float(jumps.max()) / max(float(jumps.median()), 1e-8)
        log.info("[DaSiWa Loop] %d → %d frames; trim=%d/%d, overlap=%d, analysis PSNR=%.2f dB, temporal spike=%.1fx",
                 len(images), len(output), start, len(images)-end, k, psnr, ratio)
        if ratio > 12:
            log.warning("[DaSiWa Loop] Strong motion/scene discontinuity remains. Endpoint equality does not guarantee a visually seamless loop.")
        return (output,)
