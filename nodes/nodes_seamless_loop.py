"""Automatic circular overlap morphing with native RIFE/FILM or RTX interpolation."""
import contextlib
import inspect
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


# A beat-aligned loop may cost up to 25% more visually: small enough to stay seamless, large enough to reach a beat.
BEAT_VISUAL_TOLERANCE = 1.25
# Normalized autocorrelation height above its median; noise stays near 0.1 on clip-length audio, steady beats exceed 0.3.
BEAT_MIN_CONFIDENCE = 0.2


def _candidates(preview, end_offsets=None):
    n = len(preview)
    trim = int(n * .10)
    offsets = sorted({0, trim // 2, trim})
    result = []
    for start in offsets:
        for end_trim in (offsets if end_offsets is None else end_offsets):
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
                result.append((cost, start, end, k, mse))
    return result


def _analyze(preview, beat=None):
    """beat=(period_seconds, fps) biases the loop length to whole beats within BEAT_VISUAL_TOLERANCE."""
    n = len(preview)
    if n < 6:
        return 0, n, 0, 0.0
    best = min(_candidates(preview))
    if beat is not None:
        period, fps = beat

        def beat_error(candidate):
            length = (candidate[2] - candidate[1] - candidate[3]) / fps
            return abs(length - max(1, round(length / period)) * period) / period
        near = [c for c in _candidates(preview, range(int(n * .10) + 1)) if c[0] <= best[0] * BEAT_VISUAL_TOLERANCE]
        aligned = min(near, key=lambda c: (beat_error(c), c[0]))
        beats = (aligned[2] - aligned[1] - aligned[3]) / fps / period
        # Frame quantization leaves at most half a frame of beat error; anything above is not aligned.
        if beat_error(aligned) * period * fps <= .5 + 1e-9:
            log.info("[DaSiWa Loop] Beat-aware: %.1f BPM, loop %.2f beats (visual cost %.2fx best)",
                     60 / period, beats, aligned[0] / max(best[0], 1e-12))
            best = aligned
        else:
            log.info("[DaSiWa Loop] Beat-aware: %.1f BPM but no whole-beat loop within %.2fx visual cost "
                     "(closest %.2f beats); using Crossfade choice", 60 / period, BEAT_VISUAL_TOLERANCE, beats)
    _, start, end, k, mse = best
    return start, end, k, -10 * math.log10(max(mse, 1e-12))


def _estimate_tempo(waveform, sample_rate):
    """Returns (bpm, confidence) from spectral-flux onset autocorrelation over 60-200 BPM."""
    mono = waveform.float().mean((0, 1)).cpu()
    hop = max(1, round(sample_rate / 100))
    n_fft = 1 << (4 * hop - 1).bit_length()
    if mono.numel() < n_fft:
        return 0.0, 0.0
    spec = torch.stft(mono, n_fft, hop, window=torch.hann_window(n_fft), return_complex=True).abs()
    env = torch.log1p(100 * spec).diff(dim=1).clamp_min(0).sum(0)
    env = env - env.mean()
    rate = sample_rate / hop
    lo, hi = math.floor(rate * 60 / 200), math.ceil(rate * 60 / 60)
    if len(env) < 2 * hi:
        return 0.0, 0.0
    ac = torch.stack([(env[:len(env)-lag] * env[lag:]).sum() for lag in range(hi + 2)]) / max(float(env.square().sum()), 1e-12)
    lag = lo + int(ac[lo:hi+1].argmax())
    # Parabolic peak refinement: one-hop lag resolution alone is ~1.3% at 150 BPM.
    y0, y1, y2 = (float(v) for v in ac[lag-1:lag+2])
    shift = .5 * (y0 - y2) / (y0 - 2 * y1 + y2) if y0 - 2 * y1 + y2 < 0 else 0.0
    return 60 * rate / (lag + shift), y1 - float(ac[lo:hi+1].median())


def _take(wave, lo, length):
    """wave[..., lo:lo+length], zero-filled where the range leaves the source."""
    out = wave.new_zeros((*wave.shape[:-1], length))
    a, b = max(lo, 0), min(lo + length, wave.shape[-1])
    if b > a:
        out[..., a-lo:b-lo] = wave[..., a:b]
    return out


def _equal_power(tail, head):
    theta = torch.linspace(0, math.pi / 2, tail.shape[-1], dtype=torch.float32, device=tail.device)
    return (tail.float() * theta.cos() + head.float() * theta.sin()).to(tail.dtype)


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


@contextlib.contextmanager
def _load_rtx_interpolator(shape, mode, device_id):
    try:
        from . import nodes_rtx_frame_generation as fg
    except ImportError:
        import nodes_rtx_frame_generation as fg
    effect_class, modes, encodings, _ = fg._dasiwa_fg_sdk()
    if mode not in ('LOW', 'MEDIUM', 'HIGH'):
        raise ValueError("Unsupported RTX loop quality mode.")
    if isinstance(device_id, bool) or not isinstance(device_id, int) or device_id < 0:
        raise ValueError("rtx_device_id must be a nonnegative CUDA index.")
    if not torch.cuda.is_available() or device_id >= torch.cuda.device_count():
        raise RuntimeError("RTX Seamless Loop requires an available NVIDIA CUDA device.")
    device = torch.device(f'cuda:{device_id}')
    h, w = shape[1:3]
    with torch.cuda.device(device), fg._dasiwa_fg_stream(device):
        effect = effect_class(mode=getattr(modes, mode),
            automatic_shot_change_detection_enabled=False, device=device_id,
            **({'image_encoding': encodings.RGB8}
               if 'image_encoding' in inspect.signature(effect_class).parameters else {}))
        try:
            effect.input_width, effect.input_height = w, h
            effect.frame_multiplier = 2
            effect.load()

            def interpolate(a, b, t):
                _check_interrupt()
                if a.device.type == 'cuda' and a.device != device:
                    torch.cuda.synchronize(a.device)
                if b.device.type == 'cuda' and b.device != device:
                    torch.cuda.synchronize(b.device)
                x = fg._dasiwa_fg_prepare(a, device, False)
                y = fg._dasiwa_fg_prepare(b, device, False)
                torch.cuda.synchronize(device)
                forward = torch.empty_like(a)
                backward = torch.empty_like(a)
                result = effect.run_at_timestep(x, y, t, shot_change=False,
                                               non_blocking=False, stream_ptr=0)
                fg._dasiwa_fg_copy_pair_output(result, forward, device, False)
                _check_interrupt()
                result = effect.run_at_timestep(y, x, 1-t, shot_change=False,
                                               non_blocking=False, stream_ptr=0)
                fg._dasiwa_fg_copy_pair_output(result, backward, device, False)
                frame = (forward + backward) * .5
                if not torch.isfinite(frame).all():
                    raise RuntimeError("RTX loop interpolation produced non-finite pixels.")
                return frame.clamp(0, 1)

            yield interpolate
        finally:
            torch.cuda.synchronize(device)
            fg._dasiwa_fg_close_effect(effect)


class DaSiWa_SeamlessLoop:
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "execute"
    CATEGORY = "DaSiWa/video"
    DESCRIPTION = "Automatically selects and morphs a circular overlap. Arbitrary footage cannot be guaranteed artifact-free."

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        models = [name for name in folder_paths.get_filename_list('frame_interpolation')
                  if name.lower().endswith('.safetensors')]
        return {"required": {"images": ("IMAGE",)}, "optional": {
            "model_name": (models or ["(no native models)"], {"tooltip": "RIFE / FILM only: native safetensors in models/frame_interpolation. Ignored for RTX Frame Generation."}),
            "exact_endpoint": ("BOOLEAN", {"socketless": True, "default": False, "tooltip": "Copy the first frame to the end: exact pixels, but one duplicate frame on repeat. Disable for continuous playback."}),
            "interpolation_backend": (["RIFE / FILM", "RTX Frame Generation"], {"default": "RIFE / FILM", "tooltip": "RIFE / FILM: native checkpoint interpolation (default). RTX Frame Generation: NVIDIA VFX pair interpolation without a RIFE/FILM checkpoint; requires a supported NVIDIA GPU and SDK. Both morph only the loop transition, not the whole video's frame rate."}),
            "rtx_mode": (["LOW", "MEDIUM", "HIGH"], {"default": "HIGH", "tooltip": "RTX only: LOW favors speed, MEDIUM is the middle tier, HIGH favors quality. Ignored for RIFE / FILM."}),
            "rtx_device_id": ("INT", {"default": 0, "min": 0, "max": 8, "tooltip": "RTX only: CUDA device index, 0 is the first GPU. Ignored for RIFE / FILM. Output still follows ComfyUI's intermediate-device policy."}),
            "audio": ("AUDIO", {"tooltip": "Optional soundtrack of the same clip. It is trimmed and crossfaded with exactly the same frame mapping as the video (sample-exact) and returned on the audio output. Requires frame_rate."}),
            "frame_rate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.001, "tooltip": "Video frames per second; required when audio is connected (maps frame f to sample round(f × sample_rate / frame_rate)). Ignored without audio."}),
            "audio_loop": (["Crossfade", "Beat-aware"], {"default": "Crossfade", "tooltip": "Crossfade: audio follows the visual loop choice with an equal-power crossfade (default). Beat-aware: estimates the tempo and prefers a loop spanning whole beats if its visual cost stays within 1.25× of the best; falls back to Crossfade when the tempo is unclear. Ignored without audio."}),
        }}

    @classmethod
    def IS_CHANGED(cls, images, model_name='', exact_endpoint=False,
                   interpolation_backend='RIFE / FILM', rtx_mode='HIGH', rtx_device_id=0,
                   audio=None, frame_rate=0.0, audio_loop='Crossfade'):
        import folder_paths
        if interpolation_backend == 'RTX Frame Generation':
            return interpolation_backend
        path = folder_paths.get_full_path('frame_interpolation', model_name)
        if path:
            stat = os.stat(path)
            return (stat.st_mtime_ns, stat.st_size)
        return model_name

    @torch.inference_mode()
    def execute(self, images, model_name='', exact_endpoint=False,
                interpolation_backend='RIFE / FILM', rtx_mode='HIGH', rtx_device_id=0,
                audio=None, frame_rate=0.0, audio_loop='Crossfade'):
        if interpolation_backend not in ('RIFE / FILM', 'RTX Frame Generation'):
            raise ValueError("Unsupported Seamless Loop interpolation backend.")
        if audio_loop not in ('Crossfade', 'Beat-aware'):
            raise ValueError("Unsupported Seamless Loop audio_loop mode.")
        if images.ndim != 4 or images.shape[-1] != 3 or images.shape[0] == 0 or min(images.shape[1:3]) < 1:
            raise ValueError("Seamless Loop expects a non-empty RGB IMAGE batch [frames, height, width, 3].")
        n = len(images)
        wave = None
        if audio is not None:
            audio = {key: audio[key] for key in audio.keys()}  # mapping-like AUDIO (VHS LazyAudioMap)
            wave, sr = audio.get('waveform'), audio.get('sample_rate')
            if (not isinstance(wave, torch.Tensor) or wave.ndim != 3 or isinstance(sr, bool)
                    or not isinstance(sr, int) or sr <= 0):
                raise ValueError("Seamless Loop audio must be a ComfyUI AUDIO value with waveform [batch, channels, samples] and a positive integer sample_rate.")
            if not frame_rate or frame_rate <= 0:
                raise ValueError("Seamless Loop needs frame_rate (the video FPS) when audio is connected, to map frames to audio samples.")

            def sample(frame):
                return round(frame * sr / frame_rate)  # absolute index: rounding never accumulates

            def audio_out(*pieces):
                return {**audio, 'waveform': torch.cat(pieces, -1)}
            if wave.shape[-1] < sample(n):
                log.info("[DaSiWa Loop] audio is shorter than the video (%d < %d samples); missing samples are silence",
                         wave.shape[-1], sample(n))
        preview = _preview(images)
        if all(torch.equal(images[0], frame) for frame in images):
            return (images.clone(), None if wave is None else audio_out(wave.clone()))
        # Already closed footage should not be shortened or re-interpolated.
        if torch.equal(images[0], images[-1]):
            frames = images.clone() if exact_endpoint else images[:-1].clone()
            return (frames, None if wave is None else audio_out(_take(wave, 0, sample(len(frames)))))
        beat, tempo = None, ''
        if wave is not None and audio_loop == 'Beat-aware':
            bpm, confidence = _estimate_tempo(wave, sr)
            tempo = f"; tempo {bpm:.1f} BPM (confidence {confidence:.2f})"
            if confidence >= BEAT_MIN_CONFIDENCE:
                beat = (60 / bpm, frame_rate)
            else:
                log.info("[DaSiWa Loop] Beat-aware: tempo confidence %.2f < %.2f (%.1f BPM); using Crossfade choice",
                         confidence, BEAT_MIN_CONFIDENCE, bpm)
        start, end, k, psnr = _analyze(preview, beat)
        import comfy.model_management as mm
        output_device = mm.intermediate_device()
        backend = (_load_rtx_interpolator(images.shape, rtx_mode, rtx_device_id)
                   if interpolation_backend == 'RTX Frame Generation'
                   else contextlib.nullcontext(_load_interpolator(model_name, images.shape)))
        with backend as interpolate:
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
                if wave is not None:
                    span, bridge = sample(n), sample(count) - sample(n)
                    pieces = [_take(wave, 0, span), _equal_power(_take(wave, span - bridge, bridge), _take(wave, 0, bridge))]
                    wrap = 0
            else:
                count = end - start - k
                if wave is not None:
                    # Head ends exactly where the middle starts, so the wrap is sample-continuous; the tail
                    # start may differ from round((end-k)·sr/fps) by one sample of rounding.
                    wrap, head = sample(start + k), sample(start + k) - sample(start)
                    middle_samples = sample(count) - head
                    pieces = [_take(wave, wrap, middle_samples),
                              _equal_power(_take(wave, wrap + middle_samples, head), _take(wave, sample(start), head))]
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
        if wave is None:
            return (output, None)
        if exact_endpoint:
            # Mirrors the copied first frame: one frame of the audio that follows the wrap.
            pieces.append(_take(wave, wrap, sample(len(output)) - sample(count)))
        result = audio_out(*pieces)
        log.info("[DaSiWa Loop] audio %d → %d frames = %d → %d samples @ %d Hz / %.3f fps; source frames %d..%d → sample %d%s",
                 n, len(output), wave.shape[-1], result['waveform'].shape[-1], sr, frame_rate,
                 start, end, sample(start), tempo)
        return (output, result)
