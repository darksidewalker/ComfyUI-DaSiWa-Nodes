"""Seamless Loop audio/beat contracts (CPU, synthetic, fake interpolation)."""
import hashlib
import importlib.util
import json
import logging
import math
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'nodes'))
GOLDEN = ROOT / 'tests/fixtures/seamless_loop_golden.json'


def load_loop():
    spec = importlib.util.spec_from_file_location('nodes_seamless_loop', ROOT / 'nodes/nodes_seamless_loop.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lerp_loader(loop, monkeypatch):
    monkeypatch.setattr(loop, '_load_interpolator', lambda *args: lambda a, b, t: torch.lerp(a, b, t))


def frames(n, period, h=9, w=11, phase=0.0):
    """Deterministic moving texture (no RNG, so golden hashes survive torch upgrades)."""
    y, x = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing='ij')
    out = []
    for i in range(n):
        p = 2 * math.pi * (i / period + phase)
        out.append(torch.stack([.5 + .4 * torch.sin(x / 2 + p), .5 + .4 * torch.cos(y / 3 - p),
                                .5 + .3 * torch.sin((x + y) / 4 + 2 * p)], -1))
    return torch.stack(out)


def image_cases():
    """name -> (images, exact_endpoint), covering cases (a)-(d)."""
    closed = frames(17, 23.0)
    closed[-1] = closed[0]
    return {
        'static': (torch.full((10, 9, 11, 3), .25), False),
        'closed': (closed, False),
        'closed_exact': (closed, True),
        'short4': (frames(4, 7.0), False),
        'short4_exact': (frames(4, 7.0), True),
        'short2': (frames(2, 7.0), False),
        'overlap40': (frames(40, 13.0), False),
        'overlap40_exact': (frames(40, 13.0), True),
        'overlap57': (frames(57, 29.5, phase=.1), False),
        'linear30': (torch.linspace(0, 1, 30)[:, None, None, None].expand(30, 9, 11, 3).clone(), True),
    }


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def fingerprint(loop, monkeypatch):
    """Image outputs and _analyze choices; used to prove the no-audio path is unchanged."""
    lerp_loader(loop, monkeypatch)
    result = {}
    for name, (images, exact) in image_cases().items():
        out = loop.DaSiWa_SeamlessLoop().execute(images, 'rife.safetensors', exact)
        result[name] = {'shape': list(out[0].shape), 'sha256': digest(out[0]),
                        'analyze': list(loop._analyze(loop._preview(images))[:3])}
    return result


def audio_for(n, fps, sr, signal='sine', channels=2, batch=1, extra=0):
    s = round(n * sr / fps) + extra
    t = torch.arange(s, dtype=torch.float64) / sr
    if signal == 'sine':
        mono = torch.sin(2 * math.pi * 220 * t)
    else:
        mono = signal(t)
    wave = mono.float()[None, None].repeat(batch, channels, 1)
    wave[:, 1:] *= .5  # distinct channels must stay distinct
    return {'waveform': wave, 'sample_rate': sr}


def s_(f, sr, fps):
    return round(f * sr / fps)


# --- backward compatibility -------------------------------------------------

def test_no_audio_path_is_byte_identical_to_0_5_40(monkeypatch):
    loop = load_loop()
    assert fingerprint(loop, monkeypatch) == json.loads(GOLDEN.read_text())


def test_no_audio_returns_none_second_output(monkeypatch):
    loop = load_loop()
    lerp_loader(loop, monkeypatch)
    images, _ = image_cases()['overlap40']
    out = loop.DaSiWa_SeamlessLoop().execute(images, 'rife.safetensors')
    assert len(out) == 2 and out[1] is None
    assert loop.DaSiWa_SeamlessLoop.RETURN_TYPES == ('IMAGE', 'AUDIO')
    assert loop.DaSiWa_SeamlessLoop.RETURN_NAMES == ('images', 'audio')


def test_crossfade_audio_does_not_change_images(monkeypatch):
    loop = load_loop()
    lerp_loader(loop, monkeypatch)
    golden = json.loads(GOLDEN.read_text())
    for name, (images, exact) in image_cases().items():
        out, _ = loop.DaSiWa_SeamlessLoop().execute(images, 'rife.safetensors', exact,
                                                    audio=audio_for(len(images), 24, 48000), frame_rate=24.0)
        assert digest(out) == golden[name]['sha256'], name


def test_schema_new_inputs_are_optional_and_default_off(monkeypatch):
    loop = load_loop()
    import folder_paths
    monkeypatch.setattr(folder_paths, 'get_filename_list', lambda category: ['rife.safetensors'])
    schema = loop.DaSiWa_SeamlessLoop.INPUT_TYPES()
    assert set(schema['required']) == {'images'}
    keys = list(schema['optional'])
    assert keys[-3:] == ['audio', 'frame_rate', 'audio_loop']  # appended, legacy widget order kept
    assert schema['optional']['audio'][0] == 'AUDIO'
    assert schema['optional']['frame_rate'][1]['default'] == 0.0
    assert schema['optional']['audio_loop'][0] == ['Crossfade', 'Beat-aware']
    assert schema['optional']['audio_loop'][1]['default'] == 'Crossfade'


# --- audio mapping -----------------------------------------------------------

def test_audio_without_frame_rate_raises():
    loop = load_loop()
    images = frames(12, 7.0)
    with pytest.raises(ValueError, match='frame_rate'):
        loop.DaSiWa_SeamlessLoop().execute(images, 'missing.safetensors', audio=audio_for(12, 24, 48000))


@pytest.mark.parametrize('fps,sr', [(24.0, 48000), (25.0, 44100), (30000 / 1001, 48000)])
@pytest.mark.parametrize('name', ['closed', 'closed_exact', 'short4', 'short4_exact', 'short2',
                                  'overlap40', 'overlap40_exact', 'overlap57'])
def test_audio_length_is_sample_exact(monkeypatch, fps, sr, name):
    loop = load_loop()
    lerp_loader(loop, monkeypatch)
    images, exact = image_cases()[name]
    audio = audio_for(len(images), fps, sr, channels=2, batch=2)
    out, aout = loop.DaSiWa_SeamlessLoop().execute(images, 'rife.safetensors', exact, audio=audio, frame_rate=fps)
    wave = aout['waveform']
    assert wave.shape == (2, 2, s_(len(out), sr, fps))
    assert wave.dtype == audio['waveform'].dtype and aout['sample_rate'] == sr
    assert aout is not audio and torch.isfinite(wave).all()


def test_static_audio_is_returned_unchanged(monkeypatch):
    loop = load_loop()
    images, _ = image_cases()['static']
    audio = audio_for(10, 24, 48000, extra=123)  # already mismatched: still unchanged
    out, aout = loop.DaSiWa_SeamlessLoop().execute(images, 'missing.safetensors', audio=audio, frame_rate=24.0)
    assert torch.equal(aout['waveform'], audio['waveform']) and aout is not audio


@pytest.mark.parametrize('exact', [False, True])
def test_closed_audio_drops_last_frame(exact):
    loop = load_loop()
    images, _ = image_cases()['closed']
    audio = audio_for(len(images), 25, 44100)
    out, aout = loop.DaSiWa_SeamlessLoop().execute(images, 'missing.safetensors', exact, audio=audio, frame_rate=25.0)
    frames_out = len(images) if exact else len(images) - 1
    assert torch.equal(aout['waveform'], audio['waveform'][..., :s_(frames_out, 44100, 25)])


@pytest.mark.parametrize('exact', [False, True])
def test_overlap_audio_mapping_and_wrap_continuity(monkeypatch, exact, caplog):
    loop = load_loop()
    lerp_loader(loop, monkeypatch)
    fps, sr = 24.0, 48000
    images, _ = image_cases()['overlap40']
    audio = audio_for(len(images), fps, sr)
    with caplog.at_level(logging.INFO):
        out, aout = loop.DaSiWa_SeamlessLoop().execute(images, 'rife.safetensors', exact, audio=audio, frame_rate=fps)
    start, end, k, _ = loop._analyze(loop._preview(images))
    src, wave = audio['waveform'], aout['waveform']
    head_end = s_(start + k, sr, fps)
    middle = s_(end - k, sr, fps) - head_end
    # Middle audio is the untouched source starting at frame start+k.
    assert torch.equal(wave[..., :middle], src[..., head_end:head_end + middle])
    count = end - start - k
    # The crossfade ends exactly on the head, i.e. on the sample right before frame start+k.
    assert torch.allclose(wave[..., s_(count, sr, fps) - 1], src[..., head_end - 1], atol=1e-6)
    if exact:
        tail = wave[..., s_(count, sr, fps):]
        assert torch.equal(tail, src[..., head_end:head_end + tail.shape[-1]])
    else:
        # Repeating the output must be sample-continuous for a continuous signal.
        mono = wave[0, 0].double()
        joined = torch.cat([mono, mono])
        steps = (joined[1:] - joined[:-1]).abs()
        joint = steps[len(mono) - 1]
        assert joint <= steps.max() * 1.0001
        assert joint <= 2 * math.sin(math.pi * 220 / sr) + 1e-6  # no larger than a source sample step
    assert any('audio' in r.message and 'samples' in r.message for r in caplog.records)


def test_short_audio_is_zero_padded_and_logged(monkeypatch, caplog):
    loop = load_loop()
    lerp_loader(loop, monkeypatch)
    images, _ = image_cases()['overlap40']
    audio = audio_for(len(images), 24, 48000, extra=-4800)
    with caplog.at_level(logging.INFO):
        out, aout = loop.DaSiWa_SeamlessLoop().execute(images, 'rife.safetensors', audio=audio, frame_rate=24.0)
    assert aout['waveform'].shape[-1] == s_(len(out), 48000, 24)
    assert any('shorter' in r.message for r in caplog.records)


def test_bridge_audio_appends_equal_power_crossfade(monkeypatch):
    loop = load_loop()
    lerp_loader(loop, monkeypatch)
    fps, sr = 24.0, 48000
    images, _ = image_cases()['short4']
    audio = audio_for(4, fps, sr)
    out, aout = loop.DaSiWa_SeamlessLoop().execute(images, 'rife.safetensors', audio=audio, frame_rate=fps)
    src, wave = audio['waveform'], aout['waveform']
    n = s_(4, sr, fps)
    assert len(out) == 8 and torch.equal(wave[..., :n], src[..., :n])
    x = wave.shape[-1] - n
    assert torch.allclose(wave[..., n], src[..., n - x])  # starts as the tail
    assert torch.allclose(wave[..., -1], src[..., x - 1], atol=1e-6)  # ends as the head


# --- tempo / beat-aware ------------------------------------------------------

def clicks(bpm, seconds, sr=48000, offset=.05):
    s = round(seconds * sr)
    wave = torch.zeros(s)
    beat = 60 / bpm
    t = offset
    while t < seconds:
        i = round(t * sr)
        n = min(240, s - i)
        wave[i:i + n] += torch.hann_window(2 * n)[n:] * torch.sin(torch.arange(n) * .9)
        t += beat
    return wave


@pytest.mark.parametrize('bpm', [100, 120, 150])
def test_tempo_estimator(bpm):
    loop = load_loop()
    est, confidence = loop._estimate_tempo(clicks(bpm, 8.0)[None, None], 48000)
    assert abs(est - bpm) / bpm < .02
    assert confidence >= loop.BEAT_MIN_CONFIDENCE


def test_tempo_estimator_low_confidence_on_noise():
    loop = load_loop()
    g = torch.Generator().manual_seed(1)
    _, confidence = loop._estimate_tempo(torch.randn(1, 2, 48000 * 6, generator=g) * .1, 48000)
    assert confidence < loop.BEAT_MIN_CONFIDENCE


def beat_video():
    # Visual period 21 frames plus drift: best crossfade loop is 62 frames (5.17 beats at 120 BPM / 24 fps),
    # an 84-frame (7-beat) loop costs ~1.07x.
    return (frames(96, 21.0) + .3 * torch.linspace(0, 1, 96)[:, None, None, None]).clamp(0, 1)


def test_beat_aware_picks_whole_beats_within_visual_tolerance(monkeypatch, caplog):
    loop = load_loop()
    lerp_loader(loop, monkeypatch)
    fps, sr = 24.0, 48000
    images = beat_video()
    preview = loop._preview(images)
    s0, e0, k0, _ = loop._analyze(preview)
    period = .5
    assert ((e0 - s0 - k0) / fps / period) % 1 > .1, 'fixture: visual best must not be a beat multiple'
    wave = clicks(120, len(images) / fps + .1, sr)[:s_(len(images), sr, fps)]
    audio = {'waveform': wave[None, None].repeat(1, 2, 1), 'sample_rate': sr}
    with caplog.at_level(logging.INFO):
        out, aout = loop.DaSiWa_SeamlessLoop().execute(images, 'rife.safetensors', audio=audio,
                                                       frame_rate=fps, audio_loop='Beat-aware')
    loop_len = len(out)
    beats = loop_len / fps / period
    assert abs(beats - round(beats)) * period * fps <= .5 + 1e-9  # within frame quantization
    assert aout['waveform'].shape[-1] == s_(loop_len, sr, fps)
    costs = {(c[1], c[2], c[3]): c[0] for c in loop._candidates(preview, range(int(len(images) * .1) + 1))}
    best = min(loop._candidates(preview))[0]
    chosen = [key for key in costs if key[2] and key[1] - key[0] - key[2] == loop_len]
    assert any(costs[key] <= best * loop.BEAT_VISUAL_TOLERANCE for key in chosen)
    assert any('BPM' in r.message and 'beats' in r.message for r in caplog.records)


def test_beat_aware_low_confidence_falls_back_to_crossfade(monkeypatch, caplog):
    loop = load_loop()
    lerp_loader(loop, monkeypatch)
    fps, sr = 24.0, 48000
    images = beat_video()
    g = torch.Generator().manual_seed(3)
    audio = {'waveform': torch.randn(1, 2, s_(len(images), sr, fps), generator=g) * .1, 'sample_rate': sr}
    cross = loop.DaSiWa_SeamlessLoop().execute(images, 'rife.safetensors', audio=audio, frame_rate=fps)
    with caplog.at_level(logging.INFO):
        beat = loop.DaSiWa_SeamlessLoop().execute(images, 'rife.safetensors', audio=audio, frame_rate=fps,
                                                  audio_loop='Beat-aware')
    assert torch.equal(cross[0], beat[0])
    assert torch.equal(cross[1]['waveform'], beat[1]['waveform'])
    assert any('Crossfade' in r.message and 'confidence' in r.message for r in caplog.records)
