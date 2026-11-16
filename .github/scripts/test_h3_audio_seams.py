"""Optional seam smoothing contracts; synthetic PCM fixtures are not model output."""
from test_h3_clip_assembly import backend, request, make_video
import numpy as np
import pytest


@pytest.mark.parametrize("policy", [True, False, [], ["preserve"], {}, None, 1, "unknown", "Smooth-seams"])
def test_invalid_audio_policy_fails_cleanly(policy, tmp_path):
    data = request([dict(kind="media", filename="clip.mp4")])
    data["audio_policy"] = policy
    with pytest.raises(ValueError, match="audio_policy"):
        backend()._request(data)
    result = backend().preflight(data, input_root=tmp_path, output_root=tmp_path)
    assert not result["ok"] and "audio_policy" in result["issues"][0]


def test_synthetic_pcm_fades_only_internal_edges_without_mutation(monkeypatch, tmp_path):
    module = backend()
    make_video(tmp_path / "clip.mov", values=[40, 80, 120])
    data = request([dict(kind="media", filename="clip.mov")] * 3)
    data["audio_policy"] = "smooth-seams"
    plan, _, _ = module._plan(data, input_root=tmp_path, output_root=tmp_path)
    original = np.full((2, 2000), 0.75, dtype=np.float32)
    monkeypatch.setattr(module, "_audio_frames", lambda row, check: (original for _ in range(3)))
    audio = np.concatenate([np.concatenate(list(module._assembly_audio_frames(row, lambda: None)), axis=1)
                            for row in plan], axis=1)
    assert audio.shape == (2, 18000)
    assert np.all(original == 0.75)
    assert np.all(audio[:, :5760] == 0.75) and np.all(audio[:, -5760:] == 0.75)
    for seam in (6000, 12000):
        assert np.all(audio[:, seam-1:seam+1] == 0)
        assert np.max(np.abs(np.diff(audio[:, seam-241:seam+241], axis=1))) < 0.004
    assert np.all(audio[:, 6240:11760] == 0.75)
    data["audio_policy"] = "preserve"
    plan, _, _ = module._plan(data, input_root=tmp_path, output_root=tmp_path)
    assert all(block is original for row in plan for block in module._assembly_audio_frames(row, lambda: None))
    data.pop("audio_policy")
    default, _, _ = module._plan(data, input_root=tmp_path, output_root=tmp_path)
    assert default == plan



def test_smooth_seams_is_an_explicit_audio_policy():
    data = request([dict(kind="media", filename="clip.mp4")])
    data["audio_policy"] = "smooth-seams"
    assert backend()._request(data) is data


def make_contrasting_pcm(path, polarity):
    """Synthetic phase/DC discontinuity with actual H.264 video and lossless PCM."""
    import av
    from fractions import Fraction
    with av.open(str(path), "w", format="mov", options={"movie_timescale": "48000"}) as container:
        video = container.add_stream("libx264", rate=24)
        video.width = video.height = 64
        video.pix_fmt = "yuv420p"
        video.options = {"crf": "18", "bf": "0"}
        sound = container.add_stream("pcm_s16le", rate=48000)
        sound.layout = "stereo"
        for index in range(12):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), 80 + index, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 24)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode(None):
            container.mux(packet)
        signal = polarity * (0.6 + 0.1 * np.cos(np.arange(24000) * 2 * np.pi * 440 / 48000))
        pcm = np.repeat((signal * 32767).astype(np.int16), 2)[None, :]
        frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="stereo")
        frame.sample_rate, frame.pts, frame.time_base = 48000, 0, Fraction(1, 48000)
        for packet in sound.encode(frame):
            container.mux(packet)
        for packet in sound.encode(None):
            container.mux(packet)


@pytest.mark.parametrize("copy_video", [False, True])
def test_synthetic_phase_dc_jump_improves_after_actual_aac_encode(tmp_path, copy_video):
    import av
    import hashlib
    make_contrasting_pcm(tmp_path / "a.mov", 1)
    make_contrasting_pcm(tmp_path / "b.mov", -1)
    before = [hashlib.sha256((tmp_path / n).read_bytes()).hexdigest() for n in ("a.mov", "b.mov")]
    data = request([dict(kind="media", filename=n) for n in ("a.mov", "b.mov")])
    data["copy_video"] = copy_video
    audio, pixels, packets = {}, {}, {}
    for policy in ("preserve", "smooth-seams", "omitted"):
        if policy == "omitted":
            data.pop("audio_policy")
        else:
            data["audio_policy"] = policy
        result = backend().assemble(data, "synthetic", input_root=tmp_path, output_root=tmp_path, check=lambda: None)
        assert result["frames"] == 24 and result["mode"] == ("copy" if copy_video else "reencode")
        backend()._verify(result["filename"], 24, lambda: None)
        with av.open(result["filename"]) as container:
            audio[policy] = np.concatenate([f.to_ndarray() for f in container.decode(audio=0)], axis=1)
        with av.open(result["filename"]) as container:
            pixels[policy] = [f.to_ndarray(format="rgb24").tobytes() for f in container.decode(video=0)]
        with av.open(result["filename"]) as container:
            packets[policy] = [bytes(p) for p in container.demux(video=0) if p.size]
    assert np.array_equal(audio["preserve"], audio["omitted"])
    assert pixels["preserve"] == pixels["smooth-seams"] == pixels["omitted"]
    assert packets["preserve"] == packets["smooth-seams"] == packets["omitted"]
    seam = 24000
    jumps = {p: float(np.max(np.abs(np.diff(a[:, seam-240:seam+240], axis=1)))) for p, a in audio.items()}
    assert jumps["smooth-seams"] < jumps["preserve"] * 0.2, jumps
    assert audio["smooth-seams"].shape == audio["preserve"].shape
    # AAC's transform spreads local changes; exact edge/interior identity is a pre-encode contract.
    assert np.allclose(audio["smooth-seams"][:, :20000], audio["preserve"][:, :20000], atol=0.002)
    if copy_video:
        source_pixels, source_packets = [], []
        for name in ("a.mov", "b.mov"):
            with av.open(str(tmp_path / name)) as container:
                source_pixels.extend(f.to_ndarray(format="rgb24").tobytes() for f in container.decode(video=0))
            with av.open(str(tmp_path / name)) as container:
                source_packets.extend(bytes(p) for p in container.demux(video=0) if p.size)
        assert pixels["smooth-seams"] == source_pixels
        assert packets["smooth-seams"] == source_packets
    assert before == [hashlib.sha256((tmp_path / n).read_bytes()).hexdigest() for n in ("a.mov", "b.mov")]


@pytest.mark.parametrize("failure", ["cancel", "disk", "encoder", "verify", "copy_preflight", "copy_cancel"])
def test_smooth_seams_keeps_existing_failure_cleanup(tmp_path, monkeypatch, failure):
    import test_h3_clip_assembly as existing
    original = existing.request
    monkeypatch.setattr(existing, "request", lambda rows: {**original(rows), "audio_policy": "smooth-seams"})
    existing.test_failure_removes_only_own_temporary_file(tmp_path, monkeypatch, failure)


def test_single_clip_has_no_fades_and_wrapper_closes_decoder(tmp_path, monkeypatch):
    module = backend()
    make_video(tmp_path / "clip.mov", values=[40])
    data = {**request([dict(kind="media", filename="clip.mov")]), "audio_policy": "smooth-seams"}
    plan, _, _ = module._plan(data, input_root=tmp_path, output_root=tmp_path)
    closed = []
    block = np.ones((2, 2000), dtype=np.float32)
    def source(row, check):
        try:
            yield block
        finally:
            closed.append(True)
    monkeypatch.setattr(module, "_audio_frames", source)
    stream = module._assembly_audio_frames(plan[0], lambda: None)
    assert next(stream) is block
    stream.close()
    assert closed == [True]
