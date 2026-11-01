"""Read-only checkpoint resolution and bounded, model-free clip assembly."""
from __future__ import annotations

import logging
import re
from fractions import Fraction

log = logging.getLogger(__name__)


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", value):
        raise ValueError("Invalid continuity session/clip ID.")
    return value


def resolve_checkpoints(rows, store):
    """Selected snapshots supersede prefixes; segments must cover the remainder."""
    session = _id(rows[0]["session"])
    selected = [_id(r["clip_id"]) for r in rows]
    if len(set(selected)) != len(selected) or any(r["session"] != session for r in rows):
        raise ValueError("Select one ordered continuation chain in one session.")
    chain, seen, current = [], set(), selected[-1]
    while current:
        if current in seen or len(seen) >= 1000:
            raise ValueError("Continuity parent cycle or chain limit exceeded.")
        seen.add(current)
        m = store.metadata(session, _id(current))
        if m.get("status") != "ready" or m.get("clip_id") != current or m.get("session") != session:
            raise ValueError("Continuity checkpoint is missing or not ready.")
        if chain and any(m.get(k) != chain[0].get(k) for k in ("width", "height", "fps", "mode_family")):
            raise ValueError("Continuity parent canvas, FPS or model family differs.")
        if type(m.get("frames")) is not int or m["frames"] <= 0:
            raise ValueError("Invalid logical frame count.")
        if chain and m["frames"] >= chain[-1]["frames"]:
            raise ValueError("Continuation frames must increase along the parent chain.")
        chain.append(m)
        current = m.get("parent_id", "")
        if not isinstance(current, str):
            raise ValueError("Invalid continuity parent ID.")
    chain.reverse()
    order = [m["clip_id"] for m in chain]
    if [c for c in order if c in selected] != selected:
        raise ValueError("Alternative takes or wrong order; not a continuation chain.")
    result, covered = [], 0
    snapshots = [i for i, m in enumerate(chain) if m["clip_id"] in selected
                 and ((isinstance(m.get("export"), dict) and m["export"].get("kind") == "full")
                      or (m.get("schema") == 1 and not m.get("export")))]
    for m in chain[snapshots[-1] if snapshots else 0:]:
        if m["clip_id"] not in selected:
            continue
        imported_root = (isinstance(m.get("provenance"), dict)
                         and m["provenance"].get("kind") == "imported_video"
                         and not m.get("parent_id") and m.get("export") is None)
        if m.get("schema") not in (1, 2) or (m.get("schema") == 2 and not isinstance(m.get("export"), dict) and not imported_root):
            raise ValueError("Schema-2 checkpoints require verified export coverage.")
        export = m.get("export") or {"kind": "full", "start_frame": 0, "frames": m["frames"]}
        start, count = export["start_frame"], export["frames"]
        if export.get("kind") not in {"full", "segment"} or type(start) is not int or type(count) is not int:
            raise ValueError("Invalid checkpoint export coverage types/kind.")
        if export["kind"] == "full":
            result, covered = [], 0
        if start != covered or count <= 0 or start + count != m["frames"]:
            raise ValueError("Incomplete export coverage; select the missing prefix/segments.")
        result.append({**m, "start_frame": start, "frames": count})
        covered += count
    return result


MAX_JSON = 65536
MAX_ROWS = 100
MODE = "reencode"
COPY_CODECS = {"h264", "hevc", "av1", "vp9"}  # codecs MP4 can carry as copied packets


def _request(data):
    import json
    if isinstance(data, str):
        if len(data.encode("utf-8")) > MAX_JSON:
            raise ValueError("Assembly request exceeds 64 KiB.")
        data = json.loads(data)
    if not isinstance(data, dict):
        raise ValueError("Assembly request must be an object.")
    if len(json.dumps(data, allow_nan=False).encode("utf-8")) > MAX_JSON:
        raise ValueError("Assembly request exceeds 64 KiB.")
    if type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("Unsupported assembly version.")
    if type(data.get("repair_transitions", False)) is not bool:
        raise ValueError("repair_transitions must be boolean.")
    if data.get("repair_transitions", False):
        raise ValueError("Join Clips does not render transitions; use the Transition Guide/Render nodes (Join clips dialog → H3 transitions).")
    if type(data.get("copy_video", False)) is not bool:
        raise ValueError("copy_video must be boolean.")
    if data.get("audio_policy", "preserve") != "preserve":
        raise ValueError("Only the preserve audio policy is supported.")
    prefix = data.get("filename_prefix", "H3_Joined")
    if not isinstance(prefix, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", prefix):
        raise ValueError("Output prefix must use 1–80 letters, numbers, underscores or hyphens.")
    rows = data.get("rows")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError("Select between 1 and 100 clips.")
    if any(not isinstance(r, dict) or r.get("kind") not in {"media", "checkpoint"} for r in rows):
        raise ValueError("Invalid assembly row kind.")
    if len({r["kind"] for r in rows}) != 1:
        raise ValueError("Do not mix continuity chains with independent media rows.")
    return data


def _roots(input_root, output_root):
    from pathlib import Path
    if input_root is None or output_root is None:
        import folder_paths
        input_root = input_root or folder_paths.get_input_directory()
        output_root = output_root or folder_paths.get_output_directory()
    return Path(input_root).resolve(), Path(output_root).resolve()


def _local(value, root, absolute=False):
    from pathlib import Path
    if not isinstance(value, str) or not value or len(value) > 1024 or "\x00" in value:
        raise ValueError("Invalid local media handle.")
    relative = Path(value.replace("\\", "/"))
    if ":" in value and not (absolute and relative.is_absolute()):
        raise ValueError("Invalid local media handle.")
    if ".." in relative.parts or (relative.is_absolute() and not absolute):
        raise ValueError("Media must stay inside its allowed directory.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Media file is missing or outside its allowed directory.")
    return path


def _number(value):
    import math
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Trims must be finite nonnegative seconds, not booleans.")
    return value


def _export_path(row, output_root, temp_root):
    from pathlib import Path
    export = row.get("export") or {}
    if any(k in export for k in ("type", "filename", "subfolder")):
        if export.get("type") not in ("output", "temp"):
            raise ValueError("Checkpoint export must name its output/temp directory type.")
        if export["type"] == "temp":
            if temp_root is None:
                import folder_paths
                temp_root = folder_paths.get_temp_directory()
            root = Path(temp_root).resolve()
        else:
            root = output_root
        filename, subfolder = export.get("filename"), export.get("subfolder", "")
        if (not isinstance(filename, str) or not filename or filename in (".", "..")
                or "/" in filename or "\\" in filename or not isinstance(subfolder, str)):
            raise ValueError("Invalid checkpoint export filename/subfolder descriptor.")
        relative = Path(subfolder.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or ":" in subfolder:
            raise ValueError("Checkpoint export descriptor must stay inside its allowed directory.")
        described = _local(str(root / relative / filename), root, absolute=True)
        path = _local(row.get("output_path"), root, absolute=True)
        if path != described:
            raise ValueError("Checkpoint output path differs from its saved export descriptor.")
        return path
    try:
        return _local(row.get("output_path"), output_root, absolute=True)
    except ValueError:
        # Legacy publishers saved an absolute path without an export descriptor.
        if temp_root is None:
            try:
                import folder_paths
                temp_root = folder_paths.get_temp_directory()
            except ImportError:
                raise ValueError("Checkpoint export is missing or outside output/temp directories.") from None
        return _local(row.get("output_path"), Path(temp_root).resolve(), absolute=True)


def _import_settings(row, input_root):
    """Only immutable imported roots may resolve a checkpoint into input/."""
    from .pyav_media import check_dimensions, MAX_SECONDS
    provenance = row.get("provenance") or {}
    if not isinstance(provenance, dict):
        raise ValueError("Invalid checkpoint provenance; re-import the source.")
    if provenance.get("kind") != "imported_video" or row.get("export"):
        return None
    if row.get("parent_id") or row.get("fps") != 24:
        raise ValueError("Imported assembly requires a 24-fps root checkpoint; re-import the source.")
    normalization = provenance.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("Imported normalization is missing; re-import the source.")
    actual, leading = (normalization.get(k) for k in ("decoded_frames", "leading_pad_frames"))
    if type(actual) is not int or not 1 <= actual <= MAX_SECONDS * 24 or type(leading) is not int:
        raise ValueError("Invalid imported normalization frame counts; re-import the source.")
    total = max(5, actual)
    total += (5 - total) % 17
    if leading != total - actual or row["frames"] != total or normalization.get("fit") != "contain":
        raise ValueError("Imported normalization/grid differs from checkpoint coverage; re-import the source.")
    size = (row.get("width"), row.get("height"))
    if any(type(n) is not int or n < 32 or n % 32 for n in size):
        raise ValueError("Imported target canvas must have edges divisible by 32.")
    check_dimensions(*size)
    digest = provenance.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Imported source SHA256 is missing or invalid; re-import the source.")
    return dict(path=_local(provenance.get("filename"), input_root), sha256=digest,
                normalization=dict(normalization), canvas=size, leading=leading)


def _plan(data, store=None, input_root=None, output_root=None, check=None, temp_root=None):
    from fractions import Fraction
    from .pyav_media import open_media, video_stream, MAX_SECONDS, guard
    check = check or guard(45)
    data = _request(data)
    input_root, output_root = _roots(input_root, output_root)
    rows = data["rows"]
    continuity = rows[0]["kind"] == "checkpoint"
    if continuity:
        if store is None:
            from .core import ClipStore
            store = ClipStore()
        rows = resolve_checkpoints(rows, store)
    plan, dimensions, total, colors = [], None, 0, None
    for row in rows:
        imported = _import_settings(row, input_root) if continuity else None
        if continuity:
            path = imported["path"] if imported else _export_path(row, output_root, temp_root)
            start, end = 0, row["frames"]
        else:
            path = _local(row.get("filename"), input_root)
            start = _number(row.get("trim_start", 0))
            end = row.get("trim_end")
            end = _number(end) if end is not None else None
        with open_media(path) as container:
            stream = video_stream(container)
            if stream.average_rate != 24:
                raise ValueError("Only 24-fps CFR sources are supported; normalize explicitly first.")
            ctx = stream.codec_context
            if ctx.format is None or ctx.format.name != "yuv420p" or ctx.color_trc in (16, 18) or ctx.color_primaries == 9:
                raise ValueError("Only 8-bit yuv420p SDR sources are supported; HDR/10-bit normalization is unavailable.")
            if stream.metadata.get("rotate", "0") != "0" or stream.sample_aspect_ratio not in (None, Fraction(0), Fraction(1)):
                raise ValueError("Normalize rotated or non-square-pixel sources explicitly first.")
            size = imported["canvas"] if imported else (stream.width, stream.height)
            if any(n % 2 for n in size) or (dimensions is not None and size != dimensions):
                raise ValueError("Source canvases differ; normalize explicitly before joining.")
            dimensions = size
            current_colors = {k: getattr(ctx, k) for k in ("color_primaries", "color_trc", "colorspace", "color_range")}
            # Untagged YUV range decodes as limited (MPEG); compare it as such.
            current_colors["color_range"] = current_colors["color_range"] or 1
            if colors is not None and colors != current_colors:
                raise ValueError("Source color properties differ; normalize explicitly first.")
            colors = current_colors
            duration = stream.duration * stream.time_base if stream.duration is not None else None
            if duration is None and "DURATION" in stream.metadata:
                hours, minutes, seconds = stream.metadata["DURATION"].split(":")
                duration = Fraction(hours) * 3600 + Fraction(minutes) * 60 + Fraction(seconds)
                duration -= (stream.start_time or 0) * stream.time_base
            count = stream.frames or (round(duration * 24) if duration else 0)
            if count <= 0 or count > MAX_SECONDS * 24:
                raise ValueError("Source must have a bounded video duration of at most one hour.")
            if continuity:
                if count != (imported["normalization"]["decoded_frames"] if imported else end):
                    raise ValueError("Checkpoint export frame coverage differs from its file.")
                if imported and imported["normalization"].get("audio") != ("encoded" if container.streams.audio else "encoded silence"):
                    raise ValueError("Imported normalization audio settings differ from source; re-import the source.")
            else:
                start = round(start * 24)
                end = count if end is None or end == 0 else round(end * 24)
            if not 0 <= start < end <= (count + imported["leading"] if imported else count):
                raise ValueError("Trim is empty or outside source video duration.")
            total += end - start
            if total > MAX_SECONDS * 24:
                raise ValueError("Assembly duration exceeds one hour.")
            stat = path.stat()
            plan.append(dict(path=path, start=start, end=end, source_frames=count, colors=colors,
                             identity=(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns),
                             has_audio=bool(container.streams.audio),
                             codec=(getattr(ctx.codec, "canonical_name", None) or ctx.codec.name,
                                    ctx.profile, bytes(ctx.extradata or b""))))
            if imported:
                from copy import deepcopy
                metadata = deepcopy(row)
                metadata.pop("start_frame")
                plan[-1].update(imported=imported, store=store, metadata=metadata)
        if imported:
            # Scan only imported roots: reject stale normalization before any output exists.
            for _ in _video_frames(plan[-1], check):
                pass
    return plan, dimensions, total


def _copy_reason(plan, check):
    """Why packet copy is impossible (None if possible). Demux only; records each row's video origin."""
    from .pyav_media import open_media, video_stream
    first = plan[0]["codec"]
    for number, row in enumerate(plan, 1):
        codec = row["codec"]
        if row.get("imported"):
            return f"clip {number} is an imported start video that needs normalization"
        if codec[0] != first[0]:
            return f"clip {number} uses a different video codec ({codec[0]} instead of {first[0]})"
        if codec[0] not in COPY_CODECS:
            return f"clip {number} uses {codec[0]} video, which cannot be copied into MP4"
        if codec != first:
            return f"clip {number} uses different encoder settings than clip 1"
        with open_media(row["path"]) as source:
            stream = video_stream(source)
            stamps, keys = [], set()
            tolerance = max(Fraction(1, 1000), stream.time_base)
            for packet in source.demux(stream):
                check()
                if not packet.size:
                    continue
                if packet.pts is None:
                    return f"clip {number} has video packets without timestamps"
                stamps.append((packet.pts * stream.time_base, packet.is_keyframe))
        if not stamps:
            return f"clip {number} has no video packets"
        origin = min(stamp for stamp, _ in stamps)
        indices = []
        for stamp, key in stamps:
            index = round((stamp - origin) * 24)
            if abs(stamp - origin - Fraction(index, 24)) > tolerance:
                return f"clip {number} is not constant 24 fps"
            indices.append(index)
            if key:
                keys.add(index)
        if sorted(indices) != list(range(row["source_frames"])):
            return f"clip {number} has missing or duplicate video frames"
        if row["start"] not in keys:
            return f"clip {number} does not start on a keyframe (trim Start to a keyframe or use the file start)"
        if row["end"] != row["source_frames"] and row["end"] not in keys:
            return f"clip {number} does not end on a keyframe (trim End to a keyframe or use the file end)"
        row["origin"] = origin
    return None


def _mode(data, plan, check):
    """('copy', None), ('reencode', reason) or ('reencode', None) when copy was not requested."""
    if not _request(data).get("copy_video", False):
        return MODE, None
    reason = _copy_reason(plan, check)
    return ("copy", None) if reason is None else (MODE, reason)


def _notes(mode, reason):
    if mode == "copy":
        return ["Video packets are copied without re-encoding (lossless); audio is re-encoded once to AAC so seams stay gap-free. No models are used."]
    return ([f"Copy video not possible: {reason}. Video is re-encoded instead."] if reason else []) + [
        "Video and audio are conventionally re-encoded; no models are used."]


def preflight(data, store=None, *, input_root=None, output_root=None, temp_root=None):
    """Bounded headers/metadata; imported roots also get streaming normalization readback."""
    try:
        from .pyav_media import guard
        plan, size, count = _plan(data, store, input_root, output_root, temp_root=temp_root)
        mode, reason = _mode(data, plan, guard(45))
        return dict(ok=True, issues=[], seconds=count / 24, frames=count, mode=mode,
                    width=size[0], height=size[1], fps=24,
                    video_codec=plan[0]["codec"][0] if mode == "copy" else "h264",
                    audio_codec="aac", audio_rate=48000, audio_channels=2,
                    notes=_notes(mode, reason) + [
                           "Trims snap to the nearest 24-fps frame; returned row trims show the effective cuts.",
                           "CFR timestamps and decode integrity are verified during the queue job."],
                    rows=[dict(frames=r["end"] - r["start"], seconds=(r["end"] - r["start"]) / 24,
                               trim_start=r["start"] / 24, trim_end=r["end"] / 24,
                               has_audio=r["has_audio"]) for r in plan])
    except (ValueError, TypeError, KeyError, OSError, OverflowError, RecursionError):
        # Do not echo arbitrary absolute paths or decoder diagnostics to HTTP clients.
        import sys
        error = sys.exc_info()[1]
        message = str(error) if isinstance(error, ValueError) else "Missing or invalid assembly source/settings."
        return dict(ok=False, issues=[message], seconds=0, frames=0, mode=MODE)


def _unchanged(row):
    path = row["path"]
    stat = path.stat()
    if path.resolve() != path or (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != row["identity"]:
        raise ValueError("Source media changed during assembly; select it again.")
    if row.get("imported"):
        import hashlib
        metadata = row["metadata"]
        if row["store"].metadata(metadata["session"], metadata["clip_id"]) != metadata:
            raise ValueError("Imported checkpoint normalization/settings changed during assembly; select it again.")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                row["check"]()
                digest.update(block)
        if digest.hexdigest() != row["imported"]["sha256"]:
            raise ValueError("Imported source SHA256 changed; select and re-import the source.")


def _video_frames(row, check):
    from fractions import Fraction
    from .pyav_media import open_media, video_stream, decoded, graph_for, drain
    row["check"] = check
    _unchanged(row)
    with open_media(row["path"]) as source:
        stream = video_stream(source)
        origin, count = None, 0
        imported = row.get("imported")
        graph, normalized = None, 0
        def ready():
            nonlocal normalized
            for image in drain(graph, check):
                normalized += 1
                if normalized > row["source_frames"]:
                    raise ValueError("Imported normalization produced excess frames; re-import the source.")
                image = image.reformat(format="yuv420p", src_color_range="JPEG",
                                       dst_color_range="JPEG" if row["colors"]["color_range"] == 2 else "MPEG")
                image.color_range = row["colors"]["color_range"]
                for _ in range(1 + (imported["leading"] if normalized == 1 else 0)):
                    check()
                    yield image
        for index, frame in enumerate(decoded(source, stream, check)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Source video has no usable presentation timestamps.")
            stamp = frame.pts * frame.time_base
            if origin is None:
                origin = stamp
                row["origin"] = origin
                # shortcut: shared video/container origin only; extend with timestamp fixtures.
                if imported and origin != Fraction(source.start_time or 0, 1_000_000):
                    raise ValueError("Imported video/container start times differ; normalize to zero-start 24-fps SDR and re-import.")
            tolerance = max(Fraction(1, 1000), frame.time_base)
            if abs(stamp - origin - Fraction(index, 24)) > tolerance:
                raise ValueError("VFR or discontinuous video timestamps; normalize explicitly first.")
            if frame.rotation != 0 or frame.side_data.get("DISPLAYMATRIX") is not None or frame.format.name != "yuv420p":
                raise ValueError("Source rotation/pixel format requires unsupported normalization.")
            if (frame.width, frame.height) != (stream.width, stream.height):
                raise ValueError("Source changes canvas during decoding.")
            count += 1
            if count > row["source_frames"]:
                raise ValueError("Source decoded frame count exceeds its declared duration.")
            if row["start"] <= index < row["end"]:
                if imported:
                    frame.pts -= round(origin / frame.time_base)
                    if graph is None:
                        width, height = imported["canvas"]
                        graph = graph_for(frame, [("fps", "24"),
                            ("scale", f"{width}:{height}:force_original_aspect_ratio=decrease"),
                            ("pad", f"{width}:{height}:(ow-iw)/2:(oh-ih)/2"),
                            ("setsar", "1"), ("format", "rgb24")])
                    graph.push(frame)
                    yield from ready()
                else:
                    yield frame
        if count != row["source_frames"]:
            raise ValueError("Source is truncated or its frame count is incorrect.")
        if imported:
            if graph is not None:
                graph.push(None)
                yield from ready()
            if normalized != count:
                raise ValueError("Imported normalization frame count differs; normalize to CFR and re-import.")
    _unchanged(row)


def _audio_frames(row, check):
    """Decode gap/priming-aware timestamps, trim, pad and emit 2000-sample blocks."""
    import av
    import numpy as np
    from .pyav_media import open_media, decoded, graph_for, drain
    samples = (row["end"] - row["start"]) * 2000
    if not row["has_audio"]:
        for _ in range(samples // 2000):
            check()
            yield np.zeros((2, 2000), dtype=np.float32)
        return
    _unchanged(row)
    with open_media(row["path"]) as source:
        stream = source.streams.audio[0]
        stream.codec_context.thread_count = 2
        graph, written, previous = None, 0, None
        for frame in decoded(source, stream, check):
            if frame.pts is None or frame.time_base is None or frame.samples > 1_048_576:
                raise ValueError("Source audio timestamps or frame size are invalid.")
            stamp = frame.pts * frame.time_base
            if previous is not None and stamp < previous:
                raise ValueError("Source audio timestamps run backwards.")
            previous = stamp
            frame.pts -= round(row["origin"] / frame.time_base)
            if graph is None:
                imported = row.get("imported")
                filters = ([("aresample", "32000:async=1:first_pts=0"),
                            ("adelay", f"{round(imported['leading'] / 24 * 32000)}S:all=1"),
                            ("apad", None),
                            ("atrim", f"end_sample={round(row['end'] / 24 * 40) * 800}"),
                            ("aformat", "sample_fmts=flt:sample_rates=32000:channel_layouts=stereo"),
                            ("aresample", "48000")] if imported else
                           [("aresample", "48000:async=1:first_pts=0:min_hard_comp=0.000001")])
                graph = graph_for(frame, filters + [
                    ("aformat", "sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"),
                    ("apad", None), ("atrim", f"start_sample={row['start'] * 2000}:end_sample={row['end'] * 2000}"),
                    ("asetpts", "N/SR/TB"), ("asetnsamples", "n=2000:p=0")])
            graph.push(frame)
            for audio in drain(graph, check):
                written += audio.samples
                yield audio.to_ndarray()
            if written >= samples:
                break
        if graph is not None and written < samples:
            graph.push(None)
            for audio in drain(graph, check):
                written += audio.samples
                yield audio.to_ndarray()
        if written != samples:
            raise ValueError("Source audio could not be aligned to the selected video interval.")
    _unchanged(row)


def _verify(path, count, check):
    from fractions import Fraction
    from .pyav_media import open_media, decoded, video_stream
    with open_media(path) as container:
        stream = video_stream(container)
        actual = 0
        for index, frame in enumerate(decoded(container, stream, check)):
            if frame.pts is None or abs(frame.pts * frame.time_base - Fraction(index, 24)) > Fraction(1, 48000):
                raise ValueError("Assembled video failed timestamp readback.")
            actual += 1
            if actual > count:
                raise ValueError("Assembled video has excess frames.")
        if actual != count:
            raise ValueError("Assembled video failed frame-count readback.")
    with open_media(path) as container:
        if len(container.streams.audio) != 1:
            raise ValueError("Assembled output has no audio timeline.")
        stream = container.streams.audio[0]
        samples = count * 2000
        if stream.duration is None or abs(stream.duration * stream.time_base * 48000 - samples) > 1:
            raise ValueError("Assembled audio failed presentation-duration readback.")
        end = 0
        for frame in decoded(container, stream, check):
            if frame.pts is None or frame.sample_rate != 48000:
                raise ValueError("Assembled audio failed timestamp readback.")
            start = round(frame.pts * frame.time_base * 48000)
            if start != end:
                raise ValueError("Assembled audio contains a timestamp gap.")
            end = start + frame.samples
            if end > samples + 1024:
                raise ValueError("Assembled audio has excess padding.")
        # AAC decoders can expose a final padded frame; MP4 duration trims it.
        if not samples <= end <= samples + 1024:
            raise ValueError("Assembled audio failed sample-coverage readback.")


def assemble(data, filename_prefix="H3_Joined", store=None, *, input_root=None, output_root=None, temp_root=None, check=None):
    """Execute in ComfyUI's queue, then publish only after full bounded readback."""
    import os
    import tempfile
    import uuid
    from pathlib import Path
    import av
    from .pyav_media import guard
    if not isinstance(filename_prefix, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", filename_prefix):
        raise ValueError("Output prefix must use 1–80 letters, numbers, underscores or hyphens.")
    check = check or guard(interrupt=True)
    check()
    plan, size, count = _plan(data, store, input_root, output_root, check, temp_root)
    mode, reason = _mode(data, plan, check)
    _, output_root = _roots(input_root, output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    final = output_root / (filename_prefix + "_" + uuid.uuid4().hex + ".mp4")
    handle = tempfile.NamedTemporaryFile(prefix=".h3-assembly-", suffix=".partial.mp4", dir=output_root, delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        if mode == "copy":
            try:
                _write_copy(temporary, plan, check, output_root)
                _verify(temporary, count, check)
            except (ValueError, av.FFmpegError) as error:
                # Lossless copy is an optimization: fall back to the proven re-encode path.
                log.warning("H3 Join: video copy failed (%s); re-encoding instead.", error)
                mode, reason = MODE, f"copy failed during joining ({error})"
        if mode != "copy":
            _write_reencode(temporary, plan, size, count, check, output_root)
            _verify(temporary, count, check)
        for row in plan:
            _unchanged(row)
        with temporary.open("rb+") as file:
            os.fsync(file.fileno())
        check()
        # Hard-link publication is atomic and never replaces any existing file.
        os.link(temporary, final)
        return dict(filename=str(final), seconds=count / 24, frames=count, mode=mode,
                    notes=_notes(mode, reason),
                    preview=dict(filename=final.name, subfolder="", type="output", format="video/mp4",
                                 width=size[0], height=size[1], fps=24))
    finally:
        temporary.unlink(missing_ok=True)


def _write_copy(temporary, plan, check, output_root):
    """Remux the selected video packets losslessly; audio still streams through one AAC encode."""
    import shutil
    from contextlib import closing
    import av
    from ..helper_pyav_video import _configure_audio_stream
    from .pyav_media import open_media, video_stream
    with av.open(str(temporary), "w", format="mp4", options={"movflags": "+faststart", "movie_timescale": "48000"}) as output:
        video = sound = None
        offset, last_dts, sample_offset = 0, None, 0
        for number, row in enumerate(plan, 1):
            _unchanged(row)
            frames = row["end"] - row["start"]
            with open_media(row["path"]) as source, closing(_audio_frames(row, check)) as audio:
                stream = video_stream(source)
                if video is None:
                    video = output.add_stream_from_template(stream, opaque=True)  # remux: no encoder needed (AV1 decodes via libdav1d)
                    video.time_base = Fraction(1, 24)
                    sound, _ = _configure_audio_stream(output, "aac", 48000, 2, "192k")
                copied = emitted = ahead = 0
                def emit(limit):
                    nonlocal emitted, sample_offset
                    while emitted < limit:
                        block = next(audio, None)
                        if block is None or block.shape != (2, 2000):
                            raise ValueError("Source audio did not produce one block per video frame.")
                        frame = av.AudioFrame.from_ndarray(block, format="fltp", layout="stereo")
                        frame.sample_rate = 48000
                        frame.pts, frame.time_base = sample_offset, Fraction(1, 48000)
                        for packet in sound.encode(frame):
                            output.mux(packet)
                        sample_offset += 2000
                        emitted += 1
                for packet in source.demux(stream):
                    check()
                    if not packet.size:
                        continue
                    index = round((packet.pts * stream.time_base - row["origin"]) * 24)
                    if not row["start"] <= index < row["end"]:
                        continue
                    if shutil.disk_usage(output_root).free < 64 * 1024**2:
                        raise ValueError("Insufficient output disk space (64 MiB reserve).")
                    dts = packet.dts if packet.dts is not None else packet.pts
                    packet.pts = offset + index - row["start"]
                    packet.dts = offset + round((dts * stream.time_base - row["origin"]) * 24) - row["start"]
                    if last_dts is not None and packet.dts <= last_dts:
                        raise ValueError(f"copied video timestamps would run backwards at clip {number}")
                    last_dts = packet.dts
                    packet.time_base, packet.stream = Fraction(1, 24), video
                    packet.duration = 1  # source ticks (e.g. 512 at 1/12288) would read as 512 frames on the last packet
                    output.mux(packet)
                    copied += 1
                    ahead = max(ahead, index - row["start"] + 1)
                    emit(ahead)  # keep audio interleaved with the video already written
                if copied != frames:
                    raise ValueError(f"clip {number} video packet count differs from preflight")
                emit(frames)
                if next(audio, None) is not None:
                    raise ValueError("Source audio exceeds its selected video duration.")
            offset += frames
        for packet in sound.encode(None):
            check()
            output.mux(packet)


def _write_reencode(temporary, plan, size, count, check, output_root):
    import shutil
    from contextlib import closing
    import av
    from ..helper_pyav_video import _configure_video_stream, _configure_audio_stream
    with av.open(str(temporary), "w", format="mp4", options={"movflags": "+faststart", "movie_timescale": "48000"}) as output:
        # Reuse the exporter configuration, but never its whole-waveform audio API.
        video = _configure_video_stream(output, "libx264", *size, 24, 8, 18)
        video.codec_context.thread_count = 2
        for key, value in plan[0]["colors"].items():
            setattr(video.codec_context, key, value)
        sound, _ = _configure_audio_stream(output, "aac", 48000, 2, "192k")
        index, sample_offset = 0, 0
        for row in plan:
            with closing(_video_frames(row, check)) as pictures, closing(_audio_frames(row, check)) as audio:
                for frame in pictures:
                    check()
                    if shutil.disk_usage(output_root).free < 64 * 1024**2:
                        raise ValueError("Insufficient output disk space (64 MiB reserve).")
                    frame.pts, frame.time_base = index, Fraction(1, 24)
                    frame.pict_type = av.video.frame.PictureType.NONE
                    for packet in video.encode(frame):
                        output.mux(packet)
                    block = next(audio, None)
                    if block is None or block.shape != (2, 2000):
                        raise ValueError("Source audio did not produce one block per video frame.")
                    audio_frame = av.AudioFrame.from_ndarray(block, format="fltp", layout="stereo")
                    audio_frame.sample_rate = 48000
                    audio_frame.pts, audio_frame.time_base = sample_offset, Fraction(1, 48000)
                    for packet in sound.encode(audio_frame):
                        output.mux(packet)
                    sample_offset += 2000
                    index += 1
                if next(audio, None) is not None:
                    raise ValueError("Source audio exceeds its selected video duration.")
        for stream in (video, sound):
            for packet in stream.encode(None):
                check()
                output.mux(packet)
    if index != count:
        raise ValueError("Assembly frame count differs from preflight.")


def media_info(filename, *, input_root=None):
    """Inspect input media without importing, normalizing or publishing it."""
    from pathlib import Path
    from .pyav_media import probe
    try:
        if input_root is None:
            import folder_paths
            input_root = folder_paths.get_input_directory()
        root = Path(input_root).resolve()
        path = _local(filename, root)
        if path != root / filename.replace("\\", "/"):
            raise ValueError("Media symlinks are not supported.")
        info = probe(path)
        return dict(ok=True, **{key: info[key] for key in ("seconds", "width", "height", "fps", "has_audio")})
    except (ValueError, TypeError, KeyError, OSError, OverflowError, RuntimeError):
        return dict(ok=False, issues=["Could not determine local video duration. Select a valid input video."])
