"""Model-free assembly contracts and real PyAV readback; no ComfyUI imports."""
import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for name, path in (("assembly_tests", ROOT), ("assembly_tests.nodes", ROOT / "nodes"),
                   ("assembly_tests.nodes.h3_continuity", ROOT / "nodes/h3_continuity")):
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


def backend():
    try:
        return importlib.import_module("assembly_tests.nodes.h3_continuity.assembly")
    except ModuleNotFoundError:
        pytest.fail("Assembly backend is not implemented")


class Store:
    def __init__(self, rows):
        self.rows = rows

    def metadata(self, session, clip):
        assert session == "s"
        return self.rows[clip]


def checkpoint(clip, parent, frames, export=None):
    return dict(schema=2 if export else 1, session="s", clip_id=clip,
                parent_id=parent, frames=frames, fps=24, width=64, height=64,
                mode_family="ref2va", status="ready", output_path=clip + ".mp4",
                export=export)


def test_trusted_output_path_may_contain_a_platform_drive_or_colon(tmp_path):
    import os
    path = tmp_path / ('take.mp4' if os.name == 'nt' else 'take:one.mp4')
    path.write_bytes(b'local output')
    assert backend()._local(str(path), tmp_path, absolute=True) == path
    with pytest.raises(ValueError):
        backend()._local(str(path), tmp_path)


def test_legacy_chain_uses_latest_full_export_once():
    rows = {"a": checkpoint("a", "", 124), "b": checkpoint("b", "a", 243),
            "c": checkpoint("c", "b", 362, dict(kind="segment", start_frame=243, frames=119))}
    selection = [dict(kind="checkpoint", session="s", clip_id=c) for c in rows]
    result = backend().resolve_checkpoints(selection, Store(rows))
    assert [(r["clip_id"], r["start_frame"], r["frames"]) for r in result] == [
        ("b", 0, 243), ("c", 243, 119)]
    assert rows["a"]["output_path"] == "a.mp4"


@pytest.mark.parametrize("change", ["branch", "cycle", "gap", "wrong_type", "canvas"])
def test_checkpoint_resolution_rejects_invalid_history(change):
    rows = {"a": checkpoint("a", "", 124), "b": checkpoint("b", "a", 243),
            "c": checkpoint("c", "b", 362, dict(kind="segment", start_frame=243, frames=119))}
    selected = ["a", "b", "c"]
    if change == "branch":
        rows["c"]["parent_id"] = "a"
    elif change == "cycle":
        rows["a"]["parent_id"] = "c"
    elif change == "gap":
        rows["c"]["export"]["start_frame"] = 244
    elif change == "wrong_type":
        rows["c"]["export"]["frames"] = True
    else:
        rows["c"]["width"] = 128
    with pytest.raises(ValueError):
        backend().resolve_checkpoints([dict(session="s", clip_id=c) for c in selected], Store(rows))


def request(rows):
    return dict(version=1, rows=rows, repair_transitions=False, audio_policy="preserve")


@pytest.mark.parametrize("data", [None, {}, request([]), request([{}]),
    {**request([]), "repair_transitions": True}, {**request([]), "version": True},
    request([dict(kind="media", filename="/etc/passwd")]),
    request([dict(kind="media", filename="http://host/a.mp4")]),
    request([dict(kind="media", filename="x", trim_start=float("nan"))]),
    request([dict(kind="media", filename="x", trim_start=True)]),
    request([dict(kind="media", filename="x")] * 101), pytest.param("x" * 65537, id="oversized")])
def test_preflight_rejects_untrusted_requests(data, tmp_path):
    result = backend().preflight(data, input_root=tmp_path, output_root=tmp_path)
    assert result["ok"] is False
    assert result["issues"]


import av
import numpy as np
from fractions import Fraction


def make_video(path, values=None, audio=True, audio_delay=0, fps=24, width=64,
               codec="pcm_s16le", video_offset=0, gap=0, colors=None, tail_impulse=False):
    values = values or [30 + i * 10 for i in range(12)]
    # MOV edit lists otherwise quantize delayed AAC timestamps to milliseconds on PyAV 18.
    options = {} if gap or codec == "libopus" else {"movie_timescale": "48000"}
    with av.open(str(path), "w", format="matroska" if gap or codec == "libopus" else "mov", options=options) as container:
        video = container.add_stream("libx264", rate=fps)
        video.width, video.height = width, 64
        video.pix_fmt = "yuv420p"
        for key, value in (colors or {}).items():
            setattr(video.codec_context, key, value)
        video.options = {"crf": "18", "bf": "2", "b-adapt": "0"}
        sound = container.add_stream(codec, rate=48000) if audio else None
        if sound:
            sound.layout = "stereo"
        for i, value in enumerate(values):
            frame = av.VideoFrame.from_ndarray(np.full((64, width, 3), value, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = i + video_offset, Fraction(1, fps)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode(None):
            container.mux(packet)
        if sound:
            samples = len(values) * 48000 // fps - audio_delay - gap
            signal = np.zeros((1, samples * 2), dtype=np.int16)
            signal[0, 2000:2020] = 24000  # ten-sample impulse in both channels
            if tail_impulse:
                signal[0, -400:] = 24000
            if gap:
                signal[0, 24000:24020] = 24000
            cuts = [(0, 6000), (6000, samples)] if gap else [(0, samples)]
            for begin, end in cuts:
                frame = av.AudioFrame.from_ndarray(signal[:, begin * 2:end * 2].copy(), format="s16", layout="stereo")
                frame.sample_rate = 48000
                frame.pts = begin + audio_delay + video_offset * 2000 + (gap if begin else 0)
                frame.time_base = Fraction(1, 48000)
                for packet in sound.encode(frame):
                    container.mux(packet)
            for packet in sound.encode(None):
                container.mux(packet)


def test_real_h264_sei_frames_decode_for_assembly(tmp_path):
    make_video(tmp_path / "sei.mov", values=[40, 80, 120], audio=False)
    data = request([dict(kind="media", filename="sei.mov")])
    plan, _, _ = backend()._plan(data, input_root=tmp_path, output_root=tmp_path)
    frames = list(backend()._video_frames(plan[0], lambda: None))
    assert frames[0].side_data["SEI_UNREGISTERED"].type.name == "SEI_UNREGISTERED"
    assert len(frames) == 3
    assert np.allclose([f.to_ndarray(format="rgb24").mean() for f in frames], [40, 80, 120], atol=3)


def test_real_assembly_streams_frames_and_audio_without_models(tmp_path, monkeypatch):
    make_video(tmp_path / "a.mov")
    make_video(tmp_path / "b.mov", values=[180] * 12, audio=False)
    data = request([dict(kind="media", filename="a.mov", trim_start=0.125, trim_end=0.5),
                    dict(kind="media", filename="b.mov")])
    info = backend().preflight(data, input_root=tmp_path, output_root=tmp_path)
    assert info["ok"], info
    assert (info["frames"], info["mode"]) == (21, "reencode")
    assert (info["rows"][0]["trim_start"], info["rows"][0]["trim_end"]) == (0.125, 0.5)
    original_import = __import__
    def no_models(name, *args, **kwargs):
        assert not name.startswith(("torch", "comfy", "transformers")), name
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr("builtins.__import__", no_models)
    result = backend().assemble(data, "joined", input_root=tmp_path, output_root=tmp_path,
                                check=lambda: None)
    output = Path(result["filename"])
    assert output.is_file()
    with av.open(str(output)) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 21
    markers = [round(float(f.to_ndarray(format="rgb24").mean())) for f in frames]
    assert np.allclose(markers[:9], np.arange(60, 150, 10), atol=3), markers
    assert np.allclose(markers[9:], 180, atol=3)
    with av.open(str(output)) as container:
        sound = container.streams.audio[0]
        assert round(sound.duration * sound.time_base * 48000) == 42000
        samples = np.concatenate([f.to_ndarray() for f in container.decode(audio=0)], axis=1)
    assert samples.shape[1] >= 42000
    assert np.max(np.abs(samples[:, 19000:41000])) < 0.001
    assert not list(tmp_path.glob("*.partial*"))


def test_preflight_rejects_mismatched_canvas_and_symlinks(tmp_path):
    make_video(tmp_path / "a.mov", audio=False)
    make_video(tmp_path / "b.mov", width=128, audio=False)
    result = backend().preflight(request([dict(kind="media", filename=n) for n in ("a.mov", "b.mov")]),
                                 input_root=tmp_path, output_root=tmp_path)
    assert not result["ok"]
    (tmp_path / "escape.mov").symlink_to("/etc/passwd")
    result = backend().preflight(request([dict(kind="media", filename="escape.mov")]),
                                 input_root=tmp_path, output_root=tmp_path)
    assert not result["ok"]


@pytest.mark.parametrize("codec,delay", [("pcm_s16le", 4800), ("aac", 0), ("aac", 4800), ("libopus", 0), ("libopus", 4800)])
def test_audio_impulse_positions_survive_delay_priming_and_join(tmp_path, codec, delay):
    make_video(tmp_path / "a.mov", audio_delay=delay, codec=codec)
    data = request([dict(kind="media", filename="a.mov")] * 2)
    result = backend().assemble(data, "impulse", input_root=tmp_path, output_root=tmp_path, check=lambda: None)
    with av.open(result["filename"]) as container:
        blocks = [f.to_ndarray() for f in container.decode(audio=0)]
    audio = np.concatenate(blocks, axis=1)[0]
    with av.open(str(tmp_path / "a.mov")) as source:
        first = next(source.decode(audio=0))
        # Matroska/Opus can quantize its origin to milliseconds; preserve the file's PTS.
        expected = round(first.pts * first.time_base * 48000) + 1000 if codec == "libopus" else delay + 1000
    for start in (0, 24000):
        peak = int(np.argmax(np.abs(audio[start:start + 24000])))
        assert abs(peak - expected) < 20, (peak, expected)


def test_output_node_contract_and_real_execution(tmp_path, monkeypatch):
    try:
        node = importlib.import_module("assembly_tests.nodes.h3_continuity.assembly_node").DaSiWaH3ClipAssemble
    except ModuleNotFoundError:
        pytest.fail("Assembly Output Node is not implemented")
    make_video(tmp_path / "a.mov", audio=False)
    import json
    assert node.OUTPUT_NODE is True
    assert node.RETURN_TYPES == ("STRING",)
    schema = node.INPUT_TYPES()
    assert list(schema["required"]) == ["assembly_data", "filename_prefix"]
    paths = types.ModuleType("folder_paths")
    paths.get_input_directory = paths.get_output_directory = lambda: str(tmp_path)
    monkeypatch.setitem(sys.modules, "folder_paths", paths)
    media = importlib.import_module("assembly_tests.nodes.h3_continuity.pyav_media")
    monkeypatch.setattr(media, "guard", lambda **kw: lambda: None)
    result = getattr(node(), node.FUNCTION)(json.dumps(request([dict(kind="media", filename="a.mov")])), "node")
    assert Path(result["result"][0]).is_file()
    event = result["ui"]["h3_clip_assembly"][0]
    assert (event["seconds"], event["mode"]) == (0.5, "reencode")
    assert event["preview"]["type"] == "output"


@pytest.mark.parametrize("export", [dict(kind="nonsense", start_frame=0, frames=124),
    dict(kind="full", start_frame=False, frames=124),
    dict(kind="segment", start_frame=0, frames=124.0)])
def test_export_coverage_types_are_authoritative(export):
    rows = {"a": checkpoint("a", "", 124, export)}
    with pytest.raises(ValueError):
        backend().resolve_checkpoints([dict(session="s", clip_id="a")], Store(rows))


@pytest.mark.parametrize("failure", ["cancel", "disk", "encoder", "verify", "copy_preflight", "copy_cancel"])
def test_failure_removes_only_own_temporary_file(tmp_path, monkeypatch, failure):
    import hashlib
    import shutil
    make_video(tmp_path / "a.mov", audio=False)
    original = hashlib.sha256((tmp_path / "a.mov").read_bytes()).hexdigest()
    (tmp_path / "existing.mp4").write_bytes(b"do not replace")
    data = request([dict(kind="media", filename="a.mov")])
    calls = 0
    def check():
        nonlocal calls
        calls += 1
        if failure == "cancel" and calls == 9:
            raise RuntimeError("cancelled")
    if failure == "disk":
        monkeypatch.setattr(shutil, "disk_usage", lambda _: types.SimpleNamespace(free=0))
    if failure == "encoder":
        helper = importlib.import_module("assembly_tests.nodes.helper_pyav_video")
        monkeypatch.setattr(helper, "_configure_video_stream", lambda *a, **k: (_ for _ in ()).throw(OSError("ENOSPC")))
    if failure == "verify":
        monkeypatch.setattr(backend(), "_verify", lambda *a: (_ for _ in ()).throw(ValueError("invalid readback")))
    if failure in ("copy_preflight", "copy_cancel"):
        data["copy_video"] = True
        def fail_copy_check(*args):
            raise ValueError("copy scan failed") if failure == "copy_preflight" else RuntimeError("cancelled")
        monkeypatch.setattr(backend(), "_copy_reason", fail_copy_check)
    with pytest.raises((RuntimeError, ValueError, OSError)):
        backend().assemble(data, "joined", input_root=tmp_path, output_root=tmp_path, check=check)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.mov", "existing.mp4"]
    assert hashlib.sha256((tmp_path / "a.mov").read_bytes()).hexdigest() == original
    assert (tmp_path / "existing.mp4").read_bytes() == b"do not replace"


def test_checkpoint_export_uses_only_selected_coverage_with_real_readback(tmp_path):
    make_video(tmp_path / "b.mov", values=[70] * 243, audio=False)
    make_video(tmp_path / "c.mov", values=[170] * 119, audio=False)
    rows = {"a": checkpoint("a", "", 124), "b": checkpoint("b", "a", 243),
            "c": checkpoint("c", "b", 362, dict(kind="segment", start_frame=243, frames=119))}
    rows["b"]["output_path"] = str(tmp_path / "b.mov")
    rows["c"]["output_path"] = str(tmp_path / "c.mov")
    data = request([dict(kind="checkpoint", session="s", clip_id=c) for c in rows])
    result = backend().assemble(data, "chain", Store(rows), input_root=tmp_path, output_root=tmp_path, check=lambda: None)
    with av.open(result["filename"]) as container:
        markers = [float(f.to_ndarray(format="rgb24").mean()) for f in container.decode(video=0)]
    assert len(markers) == 362
    assert np.allclose(markers[:243], 70, atol=3)
    assert np.allclose(markers[243:], 170, atol=3)


@pytest.mark.parametrize("prefix", ["../bad", "", "a/b", "/absolute", "x" * 81])
def test_preflight_validates_filename_prefix(prefix, tmp_path):
    make_video(tmp_path / "a.mov", audio=False)
    result = backend().preflight({**request([dict(kind="media", filename="a.mov")]), "filename_prefix": prefix},
                                 input_root=tmp_path, output_root=tmp_path)
    assert not result["ok"]


def test_schema2_missing_export_is_not_a_legacy_snapshot():
    row = checkpoint("a", "", 124)
    row["schema"] = 2
    with pytest.raises(ValueError):
        backend().resolve_checkpoints([dict(session="s", clip_id="a")], Store({"a": row}))


def test_internal_audio_gap_preserves_impulse_position(tmp_path):
    make_video(tmp_path / "gap.mov", gap=2400, video_offset=120)
    result = backend().assemble(request([dict(kind="media", filename="gap.mov")]), "gap",
                                input_root=tmp_path, output_root=tmp_path, check=lambda: None)
    with av.open(result["filename"]) as container:
        audio = np.concatenate([f.to_ndarray() for f in container.decode(audio=0)], axis=1)[0]
    assert np.max(np.abs(audio[6500:8000])) < 0.001
    peak = np.argmax(np.abs(audio[10000:20000])) + 10000
    assert abs(int(peak) - 14400) < 20


def test_real_sources_use_b_frames_and_negative_dts(tmp_path):
    make_video(tmp_path / "b.mov", values=[100] * 12, audio=False)
    with av.open(str(tmp_path / "b.mov")) as source:
        assert min(p.dts for p in source.demux(video=0) if p.dts is not None) < 0
    with av.open(str(tmp_path / "b.mov")) as source:
        assert any(f.pict_type == av.video.frame.PictureType.B for f in source.decode(video=0))


def test_assembly_preserves_explicit_sdr_color_tags(tmp_path):
    colors = dict(color_primaries=1, color_trc=1, colorspace=1, color_range=1)
    make_video(tmp_path / "color.mov", audio=False, colors=colors)
    result = backend().assemble(request([dict(kind="media", filename="color.mov")]), "color",
                                input_root=tmp_path, output_root=tmp_path, check=lambda: None)
    with av.open(result["filename"]) as output:
        ctx = output.streams.video[0].codec_context
        assert {k: getattr(ctx, k) for k in colors} == colors


@pytest.mark.parametrize("options", [dict(fps=30), dict(colors={"color_trc": 16}), dict(colors={"color_trc": 18})])
def test_preflight_rejects_unsupported_fps_or_hdr(tmp_path, options):
    make_video(tmp_path / "bad.mov", audio=False, **options)
    assert not backend().preflight(request([dict(kind="media", filename="bad.mov")]),
                                    input_root=tmp_path, output_root=tmp_path)["ok"]


def test_discontinuous_video_timestamps_fail_without_publishing(tmp_path):
    source = tmp_path / "vfr.mp4"
    with av.open(str(source), "w") as output:
        stream = output.add_stream("libx264", rate=24)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, 48)
        for i in range(12):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), 100, dtype=np.uint8), format="rgb24")
            frame.pts = 2 * i + (1 if i == 3 else 0)
            frame.time_base = Fraction(1, 48)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)
    data = request([dict(kind="media", filename="vfr.mp4")])
    with pytest.raises(ValueError, match="VFR|24-fps"):
        backend().assemble(data, "vfr", input_root=tmp_path, output_root=tmp_path, check=lambda: None)
    assert [p.name for p in tmp_path.iterdir()] == ["vfr.mp4"]


def test_checkpoint_frames_must_increase_along_parents():
    rows = {"a": checkpoint("a", "", 243), "b": checkpoint("b", "a", 124)}
    with pytest.raises(ValueError):
        backend().resolve_checkpoints([dict(session="s", clip_id="b")], Store(rows))


@pytest.mark.parametrize("audio", [False, True])
@pytest.mark.parametrize("actual", [12, 22, 24])
def test_imported_root_contains_padding_and_preserves_audio_timing(tmp_path, audio, actual, monkeypatch):
    import hashlib
    inputs, outputs = tmp_path / "input", tmp_path / "output"
    inputs.mkdir()
    outputs.mkdir()
    source = inputs / "raw.mov"
    values = [40 + i * 6 for i in range(actual)]
    make_video(source, values=values, width=128,
               audio=audio, audio_delay=4800 if audio else 0)
    # The native importer rounds upward to 17k+5, padding the beginning.
    total = max(5, actual)
    total += (5 - total) % 17
    leading = total - actual
    root = checkpoint("a", "", total)
    root["output_path"] = str(source)
    root["provenance"] = dict(kind="imported_video", filename="raw.mov",
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        normalization=dict(decoded_frames=actual, leading_pad_frames=leading, fit="contain",
                           audio="encoded" if audio else "encoded silence"))
    delta = checkpoint("b", "a", total + 17, dict(kind="segment", start_frame=total, frames=17))
    make_video(outputs / "delta.mov", values=[190] * 17, audio=False)
    delta["output_path"] = str(outputs / "delta.mov")
    store = Store(dict(a=root, b=delta))
    original_import = __import__
    def no_models(name, *args, **kwargs):
        assert not name.startswith(("torch", "comfy", "transformers")), name
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr("builtins.__import__", no_models)
    data = request([dict(kind="checkpoint", session="s", clip_id=c) for c in ("a", "b")])
    info = backend().preflight(data, store, input_root=inputs, output_root=outputs)
    assert info["ok"], info
    assert (info["frames"], info["width"], info["height"]) == (total + 17, 64, 64)
    result = backend().assemble(data, "imported", store, input_root=inputs,
                                output_root=outputs, check=lambda: None)
    with av.open(result["filename"]) as output:
        frames = [f.to_ndarray(format="rgb24") for f in output.decode(video=0)]
    assert len(frames) == total + 17
    centers = [float(f[20:44].mean()) for f in frames]
    assert np.allclose(centers[:leading + 1], 40, atol=3)
    assert np.allclose(centers[leading:total], values, atol=3)
    assert np.allclose(centers[total:], 190, atol=3)
    assert max(float(f[:12].mean()) for f in frames[:total]) < 3
    with av.open(result["filename"]) as output:
        stream = output.streams.audio[0]
        assert stream.duration * stream.time_base == Fraction(total + 17, 24)
        samples = np.concatenate([f.to_ndarray() for f in output.decode(audio=0)], axis=1)[0]
    if audio:
        expected = leading * 2000 + 4800 + 1000
        assert abs(int(np.argmax(np.abs(samples))) - expected) < 20
        assert np.max(np.abs(samples[:leading * 2000 + 4000])) < 0.001
    else:
        assert np.max(np.abs(samples)) < 0.001


def imported_fixture(tmp_path, *, fps=24):
    import hashlib
    inputs, outputs = tmp_path / "input", tmp_path / "output"
    inputs.mkdir()
    outputs.mkdir()
    source = inputs / "raw.mov"
    make_video(source, width=128, audio=False, fps=fps)
    root = checkpoint("a", "", 22)
    root.update(output_path="ignored-untrusted-output-path", provenance=dict(kind="imported_video",
        filename="raw.mov", sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        normalization=dict(decoded_frames=12, leading_pad_frames=10, fit="contain", audio="encoded silence")))
    store = Store(dict(a=root))
    data = request([dict(kind="checkpoint", session="s", clip_id="a")])
    return inputs, outputs, source, root, store, data


@pytest.mark.parametrize("change", ["sha", "digest_format", "escape", "absolute", "fit", "leading",
    "bool_count", "decoded", "audio", "canvas", "fps", "nonroot"])
def test_imported_root_rejects_stale_or_unsafe_settings(tmp_path, change):
    inputs, outputs, source, root, store, data = imported_fixture(tmp_path)
    provenance = root["provenance"]
    normalization = provenance["normalization"]
    if change == "sha":
        provenance["sha256"] = "0" * 64
    elif change == "digest_format":
        provenance["sha256"] = "not-a-sha"
    elif change in ("escape", "absolute"):
        provenance["filename"] = "../input/raw.mov" if change == "escape" else str(source)
    elif change == "fit":
        normalization["fit"] = "stretch"
    elif change == "leading":
        normalization["leading_pad_frames"] = 9
    elif change == "bool_count":
        normalization["decoded_frames"] = True
    elif change == "decoded":
        normalization.update(decoded_frames=13, leading_pad_frames=9)
    elif change == "audio":
        normalization["audio"] = "encoded"
    elif change == "canvas":
        root["width"] = 1000000
    elif change == "fps":
        root["fps"] = 30
    else:
        root["parent_id"] = "b"
        store.rows["b"] = checkpoint("b", "", 5)
    info = backend().preflight(data, store, input_root=inputs, output_root=outputs)
    assert not info["ok"], info
    with pytest.raises(ValueError):
        backend().assemble(data, "bad", store, input_root=inputs, output_root=outputs, check=lambda: None)
    assert not list(outputs.iterdir())


def test_imported_source_rejects_unsupported_rate_with_actionable_error(tmp_path):
    inputs, outputs, source, root, store, data = imported_fixture(tmp_path, fps=30)
    info = backend().preflight(data, store, input_root=inputs, output_root=outputs)
    assert not info["ok"]
    assert "24-fps" in info["issues"][0]


@pytest.mark.parametrize("change", ["metadata", "same_stat_content"])
def test_imported_root_rechecked_after_output_readback(tmp_path, monkeypatch, change):
    import os
    inputs, outputs, source, root, store, data = imported_fixture(tmp_path)
    verify = backend()._verify
    def changed(path, count, check):
        verify(path, count, check)
        if change == "metadata":
            root["provenance"]["normalization"]["fit"] = "stretch"
        else:
            stat = source.stat()
            content = bytearray(source.read_bytes())
            content[-1] ^= 1
            source.write_bytes(content)
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    monkeypatch.setattr(backend(), "_verify", changed)
    with pytest.raises(ValueError, match="changed"):
        backend().assemble(data, "changed", store, input_root=inputs, output_root=outputs, check=lambda: None)
    assert not list(outputs.iterdir())


def test_imported_metadata_changed_while_planning_is_rejected(tmp_path, monkeypatch):
    inputs, outputs, source, root, store, data = imported_fixture(tmp_path)
    metadata = store.metadata
    calls = 0
    def racing(session, clip):
        nonlocal calls
        calls += 1
        if calls == 2:
            root["provenance"]["normalization"]["fit"] = "stretch"
        return metadata(session, clip)
    monkeypatch.setattr(store, "metadata", racing)
    assert not backend().preflight(data, store, input_root=inputs, output_root=outputs)["ok"]


def test_malformed_import_provenance_is_an_actionable_preflight_failure(tmp_path):
    inputs, outputs, source, root, store, data = imported_fixture(tmp_path)
    root["provenance"] = ["invalid"]
    info = backend().preflight(data, store, input_root=inputs, output_root=outputs)
    assert not info["ok"]
    assert "provenance" in info["issues"][0]


@pytest.mark.parametrize("change", ["vfr", "different_origin", "truncated"])
def test_imported_preflight_decodes_and_rejects_bad_timeline(tmp_path, change):
    import hashlib
    inputs, outputs, source, root, store, data = imported_fixture(tmp_path)
    if change == "truncated":
        source.write_bytes(source.read_bytes()[:-300])
    elif change == "different_origin":
        # Audio precedes video; the importer uses container origin, not video origin.
        make_video(source, audio=True, audio_delay=-4800, gap=2400)
        root["provenance"]["normalization"]["audio"] = "encoded"
    else:
        with av.open(str(source), "w", format="mov") as output:
            stream = output.add_stream("libx264", rate=24)
            stream.width = stream.height = 64
            stream.pix_fmt = "yuv420p"
            stream.codec_context.time_base = Fraction(1, 48)
            for i in range(12):
                frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), 100, dtype=np.uint8), format="rgb24")
                frame.pts, frame.time_base = 2 * i + (1 if i == 3 else 0), Fraction(1, 48)
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode(None):
                output.mux(packet)
    root["provenance"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    info = backend().preflight(data, store, input_root=inputs, output_root=outputs)
    assert not info["ok"], info
    assert not list(outputs.iterdir())


def test_imported_root_schema2_without_export_is_verified_raw_source(tmp_path):
    inputs, outputs, source, root, store, data = imported_fixture(tmp_path)
    root["schema"] = 2
    assert backend().preflight(data, store, input_root=inputs, output_root=outputs)["ok"]


def test_imported_audio_respects_native_rounded_token_duration(tmp_path):
    import hashlib
    inputs, outputs, source, root, store, data = imported_fixture(tmp_path)
    make_video(source, values=[100] * 48, width=128, tail_impulse=True)
    total = max(5, 48)
    total += (5 - total) % 17
    root["frames"] = total
    root["provenance"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    root["provenance"]["normalization"].update(decoded_frames=48, leading_pad_frames=total - 48, audio="encoded")
    plan, _, _ = backend()._plan(data, store, inputs, outputs)
    # Video preflight records the shared AV origin; inspect exact pre-encoder samples.
    samples = np.concatenate(list(backend()._audio_frames(plan[0], lambda: None)), axis=1)
    assert samples.shape == (2, total * 2000)
    # 56 frames round to 93 H3 audio tokens: the final 400 output samples are silence.
    assert np.max(np.abs(samples[:, -300:])) < 0.001


@pytest.mark.parametrize("kind", ["full", "segment", "legacy"])
def test_checkpoint_temp_export_descriptor_is_authoritative(tmp_path, kind, monkeypatch):
    inputs, outputs, source, root, store, data = imported_fixture(tmp_path)
    temps = tmp_path / "temp"
    (temps / "nested").mkdir(parents=True)
    make_video(temps / "nested" / "take.mov", values=[170] * 17, audio=False)
    if kind == "segment":
        # The raw imported prefix remains in input; only the delta lives in temp.
        row = checkpoint("b", "a", 39, dict(kind="segment", start_frame=22, frames=17))
        store.rows["b"] = row
        data["rows"].append(dict(kind="checkpoint", session="s", clip_id="b"))
        expected = 39
    else:
        row = checkpoint("b", "", 17, None if kind == "legacy" else dict(kind="full", start_frame=0, frames=17))
        store.rows = dict(b=row)
        data["rows"] = [dict(kind="checkpoint", session="s", clip_id="b")]
        expected = 17
    row["output_path"] = str(temps / "nested" / "take.mov")
    if kind != "legacy":
        row["export"].update(type="temp", filename="take.mov", subfolder="nested")
    paths = types.ModuleType("folder_paths")
    paths.get_temp_directory = lambda: str(temps)
    monkeypatch.setitem(sys.modules, "folder_paths", paths)
    # Runtime roots come from folder_paths, never a client JSON field.
    data["temp_root"] = str(outputs)
    info = backend().preflight(data, store, input_root=inputs, output_root=outputs)
    assert info["ok"], info
    result = backend().assemble(data, "temp", store, input_root=inputs, output_root=outputs,
                                temp_root=temps, check=lambda: None)
    with av.open(result["filename"]) as output:
        markers = [float(f.to_ndarray(format="rgb24").mean()) for f in output.decode(video=0)]
    assert len(markers) == expected
    assert np.allclose(markers[-17:], 170, atol=3)


@pytest.mark.parametrize("change", ["wrong_type", "wrong_file", "escape", "absolute", "symlink", "missing_type"])
def test_checkpoint_export_descriptor_cannot_redirect_source(tmp_path, change):
    inputs, outputs, source, root, store, data = imported_fixture(tmp_path)
    temps = tmp_path / "temp"
    temps.mkdir()
    make_video(temps / "take.mov", values=[170] * 17, audio=False)
    make_video(outputs / "take.mov", values=[80] * 17, audio=False)
    row = checkpoint("b", "", 17, dict(kind="full", start_frame=0, frames=17,
        type="temp", filename="take.mov", subfolder=""))
    row["output_path"] = str(temps / "take.mov")
    store.rows = dict(b=row)
    data["rows"] = [dict(kind="checkpoint", session="s", clip_id="b")]
    if change == "wrong_type":
        row["export"]["type"] = "output"
    elif change == "wrong_file":
        row["export"]["filename"] = "other.mov"
    elif change == "escape":
        row["export"]["subfolder"] = "../temp"
    elif change == "absolute":
        row["export"]["filename"] = str(temps / "take.mov")
    elif change == "symlink":
        (temps / "link.mov").symlink_to(outputs / "take.mov")
        row["export"]["filename"] = "link.mov"
    else:
        del row["export"]["type"]
    info = backend().preflight(data, store, input_root=inputs, output_root=outputs, temp_root=temps)
    assert not info["ok"], info


def test_publication_fsync_uses_writable_handle(tmp_path, monkeypatch):
    import os
    make_video(tmp_path / "a.mov", audio=False)
    fsync = os.fsync
    def windows_fsync(fd):
        # A read-only descriptor is rejected by Windows' _commit.
        os.write(fd, b"")
        fsync(fd)
    monkeypatch.setattr(os, "fsync", windows_fsync)
    backend().assemble(request([dict(kind="media", filename="a.mov")]), "fsync",
                       input_root=tmp_path, output_root=tmp_path, check=lambda: None)


def test_later_full_snapshot_supersedes_incomplete_selected_prefix():
    rows = {"a": checkpoint("a", "", 124),
            "b": checkpoint("b", "a", 243, dict(kind="segment", start_frame=124, frames=119)),
            "c": checkpoint("c", "b", 362)}
    result = backend().resolve_checkpoints([dict(session="s", clip_id=c) for c in ("b", "c")], Store(rows))
    assert [(r["clip_id"], r["frames"]) for r in result] == [("c", 362)]


def video_packets(path):
    with av.open(str(path)) as container:
        return [bytes(p) for p in container.demux(video=0) if p.size]


def test_copy_video_is_lossless_and_reencodes_only_audio(tmp_path):
    make_video(tmp_path / "a.mov")
    make_video(tmp_path / "b.mov", values=[180] * 12, audio=False)
    data = {**request([dict(kind="media", filename="a.mov"), dict(kind="media", filename="b.mov")]), "copy_video": True}
    info = backend().preflight(data, input_root=tmp_path, output_root=tmp_path)
    assert info["ok"] and info["mode"] == "copy" and info["video_codec"] == "h264", info
    result = backend().assemble(data, "joined", input_root=tmp_path, output_root=tmp_path, check=lambda: None)
    assert result["mode"] == "copy"
    output = Path(result["filename"])
    assert video_packets(output) == video_packets(tmp_path / "a.mov") + video_packets(tmp_path / "b.mov")
    with av.open(str(output)) as container:
        frames = list(container.decode(video=0))
        assert [round(float(f.pts * f.time_base * 24)) for f in frames] == list(range(24))
    with av.open(str(output)) as container:
        picture = container.streams.video[0]  # issue #89: last packet duration must be one frame, not its source ticks
        assert picture.duration * picture.time_base == 1
        sound = container.streams.audio[0]
        assert sound.codec_context.name == "aac" and round(sound.duration * sound.time_base * 48000) == 48000
    assert not list(tmp_path.glob("*.partial*"))


@pytest.mark.parametrize("copy_video", [False, True])
@pytest.mark.parametrize("frames", [17, 22, 56])
def test_assembly_preserves_sample_exact_audio_presentation_duration(tmp_path, copy_video, frames):
    make_video(tmp_path / "source.mov", values=[70] * frames, audio=False)
    data = {**request([dict(kind="media", filename="source.mov")]), "copy_video": copy_video}
    result = backend().assemble(data, "exact", input_root=tmp_path, output_root=tmp_path, check=lambda: None)
    assert result["mode"] == ("copy" if copy_video else "reencode")
    with av.open(result["filename"]) as output:
        sound = output.streams.audio[0]
        assert sound.duration * sound.time_base * 48000 == frames * 2000


def test_copy_video_falls_back_with_a_reason(tmp_path):
    make_video(tmp_path / "a.mov")
    make_video(tmp_path / "b.mov")
    with av.open(str(tmp_path / "c.mkv"), "w") as container:  # same canvas, different codec
        video = container.add_stream("ffv1", rate=24)
        video.width, video.height, video.pix_fmt = 64, 64, "yuv420p"
        for i in range(12):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), 90, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, 24)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode(None):
            container.mux(packet)
    cases = {"different video codec": ["a.mov", "c.mkv"], "keyframe": ["a.mov", "b.mov"]}
    for reason, files in cases.items():
        rows = [dict(kind="media", filename=name) for name in files]
        if reason == "keyframe":
            rows[1]["trim_start"] = 0.25  # frame 6 is not a keyframe
        data = {**request(rows), "copy_video": True}
        info = backend().preflight(data, input_root=tmp_path, output_root=tmp_path)
        assert info["ok"] and info["mode"] == "reencode", info
        assert any(reason in note and "clip 2" in note for note in info["notes"]), info["notes"]
    plain = backend().preflight(request([dict(kind="media", filename="a.mov")] * 2), input_root=tmp_path, output_root=tmp_path)
    assert plain["mode"] == "reencode" and not any("copy" in n.lower() for n in plain["notes"])
    bad = backend().preflight({**request([dict(kind="media", filename="a.mov")]), "copy_video": "yes"}, input_root=tmp_path, output_root=tmp_path)
    assert bad["ok"] is False


def test_copy_failure_during_join_falls_back_to_reencode(tmp_path, monkeypatch):
    make_video(tmp_path / "a.mov")
    def broken(*args):
        raise ValueError("muxer refused packet")
    monkeypatch.setattr(backend(), "_write_copy", broken)
    data = {**request([dict(kind="media", filename="a.mov")] * 2), "copy_video": True}
    result = backend().assemble(data, "joined", input_root=tmp_path, output_root=tmp_path, check=lambda: None)
    assert result["mode"] == "reencode" and "muxer refused packet" in result["notes"][0]
    with av.open(result["filename"]) as container:
        assert container.streams.video[0].codec_context.name == "h264"
        assert sum(1 for _ in container.decode(video=0)) == 24
