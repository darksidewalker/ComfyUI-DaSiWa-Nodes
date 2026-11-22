"""H3 transition contracts with real PyAV media; native H3 calls are recorded, not run."""
import itertools
import json
from fractions import Fraction
from types import SimpleNamespace

import av
import numpy as np
import pytest

from test_h3_segment_backend import importlib, torch, NestedTensor
from test_h3_clip_assembly import make_video, backend as assembly

transition = importlib.import_module('h3_segment_test.nodes.h3_continuity.transition')


def spec(**changes):
    data = dict(version=1, rows=[dict(kind='media', filename='a.mov', trim_start=0, trim_end=None),
                                 dict(kind='media', filename='b.mov', trim_start=0, trim_end=None)],
                context_frames=22, transition_frames=29, prompt='', filename_prefix='H3_Transition')
    data.update(changes)
    return json.dumps(data)


def clips(tmp_path, width=64):
    make_video(tmp_path / 'a.mov', values=[i * 8 for i in range(30)], width=width)
    make_video(tmp_path / 'b.mov', values=[250 - i * 8 for i in range(30)], width=width)


@pytest.mark.parametrize('change', [dict(transition_frames=17), dict(context_frames=6), dict(context_frames=4),
                                    dict(rows=[dict(kind='media', filename='a.mov')]),
                                    dict(rows=[dict(kind='checkpoint', session='s', clip_id='a')] * 2),
                                    dict(prompt='x' * 4001), dict(filename_prefix='../x')])
def test_spec_rejects_invalid_grid_rows_and_names(change):
    with pytest.raises(ValueError):
        transition.transition_spec(spec(**change))


def test_spec_window_satisfies_h3_grid():
    data = transition.transition_spec(spec(context_frames=39, transition_frames=12))
    assert data['window_frames'] == 90 and data['window_frames'] % 17 == 5
    assert data['guide_b_frame'] == 51


def test_edges_use_trimmed_tail_and_head_with_frame_audio(tmp_path):
    clips(tmp_path)
    data = transition.transition_spec(spec(rows=[dict(kind='media', filename='a.mov', trim_start=0, trim_end=29 / 24),
                                                 dict(kind='media', filename='b.mov', trim_start=1 / 24, trim_end=None)]))
    edges = transition.clip_edges(data, input_root=tmp_path, output_root=tmp_path)
    tail, head = edges['tail'], edges['head']
    assert tail.shape == head.shape == (22, 64, 64, 3) and tail.dtype == torch.float32
    assert [round(v) for v in (tail[:, 0, 0, 0] * 255).tolist()] == pytest.approx([i * 8 for i in range(7, 29)], abs=4)
    assert [round(v) for v in (head[:, 0, 0, 0] * 255).tolist()] == pytest.approx([250 - i * 8 for i in range(1, 23)], abs=4)
    for audio in (edges['tail_audio'], edges['head_audio']):
        assert audio['sample_rate'] == 48000 and audio['waveform'].shape == (1, 2, 22 * 2000)
    assert edges['canvas'] == (64, 64)


@pytest.fixture(autouse=True)
def deterministic_ram(monkeypatch):
    monkeypatch.setattr(transition.helper_batch_output, 'available_ram_bytes', lambda: 16 * 1024**3)
    monkeypatch.setattr(transition.helper_batch_output, 'total_ram_bytes', lambda: 32 * 1024**3)


@pytest.mark.parametrize('source', [(3840, 2160), (1280, 720), (20, 1000)])
def test_edges_resize_before_stack_with_native_cover_framing(monkeypatch, source):
    from comfy.utils import common_upscale
    # Large metadata, tiny pixel fixture: no source-resolution allocation.
    canvas = (64, 32)
    image_height, image_width = (100, 20) if source[0] < source[1] else (70, 110)
    image = np.arange(image_height * image_width * 3, dtype=np.uint8).reshape(image_height, image_width, 3)
    frame = av.VideoFrame.from_ndarray(image, format='rgb24')
    plan = [dict(start=0, end=7, colors={'color_range': 1})] * 2
    monkeypatch.setattr(transition.assembly, '_plan', lambda *a, **k: (plan, source, 14))
    monkeypatch.setattr(transition, 'generation_canvas', lambda size: canvas)
    monkeypatch.setattr(transition.assembly, '_video_frames', lambda *a: (f for f in [frame] * 7))
    monkeypatch.setattr(transition.assembly, '_audio_frames', lambda *a: (
        np.full((2, 2000), i, dtype=np.float32) for i in range(7)))
    expected = common_upscale(torch.from_numpy(image).float().div(255).movedim(-1, 0)[None],
                              *canvas, 'lanczos', 'center').movedim(1, -1)[0]
    stack = np.stack
    shapes = []
    def checked_stack(images, *args, **kwargs):
        shapes.append([im.shape for im in images])
        assert all(im.shape == (32, 64, 3) and im.dtype == np.uint8 for im in images)
        return stack(images, *args, **kwargs)
    monkeypatch.setattr(np, 'stack', checked_stack)
    edges = transition.clip_edges(dict(rows=[], context_frames=5))
    assert len(shapes) == 2
    assert edges['canvas'] == source and edges['colors'] == plan[0]['colors']
    for side, values in [('tail', range(2, 7)), ('head', range(5))]:
        assert edges[side].shape == (5, 32, 64, 3)
        assert torch.equal(edges[side][0], expected)
        audio = edges[side + '_audio']
        assert audio['sample_rate'] == 48000
        assert audio['waveform'][0, 0, ::2000].tolist() == list(values)


@pytest.mark.parametrize('available', [0, 8 * 1024**3 + 1,
    8 * 1024**3 + 22 * 1344 * 768 * 3 * 8 + 22 * 2 * 2000 * 4 * 2])
def test_edges_low_ram_rejects_before_decode_or_batch_allocation(monkeypatch, available):
    plan = [dict(start=0, end=22)] * 2
    monkeypatch.setattr(transition.assembly, '_plan', lambda *a, **k: (plan, (3840, 2160), 44))
    monkeypatch.setattr(transition.helper_batch_output, 'available_ram_bytes', lambda: available)
    def forbidden(*a, **k):
        pytest.fail('Low RAM must fail before decoding or allocating a batch')
    monkeypatch.setattr(transition.assembly, '_video_frames', forbidden)
    monkeypatch.setattr(np, 'stack', forbidden)
    monkeypatch.setattr(torch, 'from_numpy', forbidden)
    with pytest.raises(ValueError, match='RAM'):
        transition.clip_edges(dict(rows=[], context_frames=22))


def test_edges_rechecks_ram_before_second_clip(tmp_path, monkeypatch):
    clips(tmp_path)
    available = itertools.chain([16 * 1024**3], itertools.repeat(8 * 1024**3 + 1))
    monkeypatch.setattr(transition.helper_batch_output, 'available_ram_bytes', lambda: next(available))
    calls = []
    video_frames = transition.assembly._video_frames
    def recorded_frames(row, check):
        calls.append(row)
        return video_frames(row, check)
    monkeypatch.setattr(transition.assembly, '_video_frames', recorded_frames)
    monkeypatch.setattr(transition.helper_batch_output, 'unload_all_comfy_models',
                        lambda: pytest.fail('Transition must not reclaim models'))
    with pytest.raises(ValueError, match='RAM'):
        transition.clip_edges(transition.transition_spec(spec()), input_root=tmp_path, output_root=tmp_path)
    assert len(calls) == 1


def test_edges_reject_clip_shorter_than_context(tmp_path):
    clips(tmp_path)
    data = transition.transition_spec(spec(rows=[dict(kind='media', filename='a.mov', trim_start=0, trim_end=0.5),
                                                 dict(kind='media', filename='b.mov')]))
    with pytest.raises(ValueError, match='shorter'):
        transition.clip_edges(data, input_root=tmp_path, output_root=tmp_path)


def test_guide_builds_window_with_guides_at_both_ends(tmp_path, monkeypatch):
    clips(tmp_path)
    calls = []
    def apply_native(self, clip, vae, guide, audio_vae=None):
        calls.append(('native', guide))
        return [[torch.zeros(1), {}]], {'samples': NestedTensor((torch.zeros(1, 24, 22, 4, 4), torch.zeros(1, 32, 2, 122)))}
    class AddGuide:
        @classmethod
        def execute(cls, positive, latent, frame_idx, vae=None, audio_vae=None, image=None, audio=None):
            calls.append(('guide', frame_idx, tuple(image.shape), tuple(audio['waveform'].shape), vae, audio_vae))
            return [positive + [frame_idx]]
    monkeypatch.setattr(transition.MiniMaxH3DirectorGuide, '_apply_native', apply_native)
    monkeypatch.setattr(transition, '_native_node', lambda name: AddGuide)
    monkeypatch.setattr(transition, '_roots', lambda: (tmp_path, tmp_path))
    guide = dict(mode='REF2VA', width=1024, height=768, length=124, prompt='scene', resolved_prompt='scene',
                 prompt_blocks=[{'text': 'x'}], first_frame='a', last_frame='b', continuity='{"operation":"continue"}')
    positive, latent, context = transition.DaSiWaH3TransitionGuide().apply(
        'clip', 'vae', guide, spec(prompt='She turns to the door.'), 'audio_vae')
    native = calls[0][1]
    assert (native['width'], native['height']) == (64, 64)  # the clips' canvas, not the Director's
    assert native['length'] == 73 and native['first_frame'] is None and native['last_frame'] is None
    assert native['prompt_blocks'] == [] and 'continuity' not in native
    assert 'hidden context' in native['resolved_prompt'] and 'She turns to the door.' in native['resolved_prompt']
    assert calls[1:] == [('guide', 0, (22, 64, 64, 3), (1, 2, 44000), 'vae', 'audio_vae'),
                         ('guide', 51, (22, 64, 64, 3), (1, 2, 44000), 'vae', 'audio_vae')]
    assert positive[-2:] == [0, 51]
    assert context['window_frames'] == 73 and context['canvas'] == (64, 64)


def test_generation_canvas_follows_clips_on_the_h3_grid():
    assert transition.generation_canvas((512, 512)) == (512, 512)
    assert transition.generation_canvas((1280, 720)) == (1280, 704)
    assert transition.generation_canvas((20, 1000)) == (32, 992)
    assert transition.generation_canvas((1920, 1080)) == (1344, 768)  # H3 pixel cap; Render scales back


def test_guide_rejects_image_inpaint(tmp_path, monkeypatch):
    monkeypatch.setattr(transition, '_roots', lambda: (tmp_path, tmp_path))
    with pytest.raises(ValueError, match='video mode'):
        transition.DaSiWaH3TransitionGuide().apply('c', 'v', dict(mode='Image Inpaint', width=64, height=64), spec(), 'a')


def window(frames=73, height=64, width=64):
    pixels = torch.stack([torch.full((height, width, 3), i / 100) for i in range(frames)])
    audio_t = round(frames * 40 / 24)
    vae = SimpleNamespace(decode=lambda video: pixels[None])
    audio_vae = SimpleNamespace(decode=lambda audio: torch.linspace(-.5, .5, audio_t * 800).repeat(2, 1).T[None])
    latent = {'samples': NestedTensor((torch.zeros(1, 24, (frames - 5) // 17 * 5 + 2, height // 16, width // 16),
                                       torch.zeros(1, 32, 2, audio_t)))}
    return latent, vae, audio_vae


def render(tmp_path, monkeypatch, canvas=(64, 64), generated=(64, 64), colors=None):
    monkeypatch.setattr(transition, '_roots', lambda: (tmp_path, tmp_path))
    latent, vae, audio_vae = window(height=generated[1], width=generated[0])
    context = {**transition.transition_spec(spec()), 'canvas': canvas,
               'colors': colors or dict(color_primaries=1, color_trc=1, colorspace=1, color_range=1)}
    result = transition.DaSiWaH3TransitionRender().render(latent, vae, audio_vae, context)
    return result['ui']['h3_transition'][0]


def tagged_clip(path, colors):
    with av.open(str(path), 'w', format='matroska') as container:
        video = container.add_stream('ffv1', rate=24)
        video.width, video.height, video.pix_fmt = 64, 64, 'yuv420p'
        for key, value in colors.items():
            setattr(video.codec_context, key, value)
        for i in range(30):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), 100, dtype=np.uint8), format='rgb24').reformat(format='yuv420p')
            frame.pts, frame.time_base, frame.color_range = i, Fraction(1, 24), colors['color_range']
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode(None):
            container.mux(packet)


def test_bridge_joins_sources_tagged_limited_range_without_color_description(tmp_path, monkeypatch):
    colors = dict(color_primaries=2, color_trc=2, colorspace=2, color_range=1)
    for name in ('a.mkv', 'b.mkv'):
        tagged_clip(tmp_path / name, colors)
    with av.open(str(tmp_path / 'a.mkv')) as container:
        assert container.streams.video[0].codec_context.color_range == 1
    item = render(tmp_path, monkeypatch, colors=colors)
    rows = [dict(kind='media', filename=name) for name in ('a.mkv', item['filename'], 'b.mkv')]
    plan = assembly().preflight(dict(version=1, rows=rows), input_root=tmp_path, output_root=tmp_path)
    assert plan['ok'] is True, plan['issues']


def test_render_writes_only_unguided_middle_into_input(tmp_path, monkeypatch):
    item = render(tmp_path, monkeypatch)
    path = tmp_path / item['filename']
    assert item['filename'].startswith('h3_transitions/H3_Transition_') and item['frames'] == 29
    assert item['seconds'] == pytest.approx(29 / 24)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert stream.average_rate == 24 and stream.codec_context.format.name == 'yuv420p'
        assert stream.codec_context.color_primaries == 1 and stream.codec_context.color_range == 1
        values = [f.to_ndarray(format='rgb24')[0, 0, 0] for f in container.decode(video=0)]
    assert values == pytest.approx([round(i / 100 * 255) for i in range(22, 51)], abs=4)
    with av.open(str(path)) as container:
        sound = container.streams.audio[0]
        assert abs(sound.duration * sound.time_base - Fraction(29, 24)) < Fraction(1, 40)


@pytest.mark.parametrize('gap', [12, 29, 46, 63])
def test_bridge_audio_presentation_duration_is_sample_exact(tmp_path, monkeypatch, gap):
    """The MP4 movie timescale must not quantize the bridge's audio edit list to milliseconds.

    FFmpeg 7.1 (PyAV 18) writes the edit list duration in the movie timescale; at 1000 ticks/s
    a 29 or 46 frame bridge reads back 16 samples short and fails assembly's readback.
    """
    monkeypatch.setattr(transition, '_roots', lambda: (tmp_path, tmp_path))
    latent, vae, audio_vae = window(frames=2 * 22 + gap)
    context = {**transition.transition_spec(spec(transition_frames=gap)), 'canvas': (64, 64),
               'colors': dict(color_primaries=1, color_trc=1, colorspace=1, color_range=1)}
    item = transition.DaSiWaH3TransitionRender().render(latent, vae, audio_vae, context)['ui']['h3_transition'][0]
    with av.open(str(tmp_path / item['filename'])) as container:
        sound = container.streams.audio[0]
        assert abs(sound.duration * sound.time_base - Fraction(gap, 24)) <= Fraction(1, 48000)


def test_render_restores_source_canvas_by_cover_scaling(tmp_path, monkeypatch):
    item = render(tmp_path, monkeypatch, canvas=(96, 64), generated=(64, 64))
    with av.open(str(tmp_path / item['filename'])) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (96, 64)


def test_render_rejects_window_from_other_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(transition, '_roots', lambda: (tmp_path, tmp_path))
    latent, vae, audio_vae = window(frames=90)
    context = {**transition.transition_spec(spec()), 'canvas': (64, 64), 'colors': {}}
    with pytest.raises(ValueError, match='window'):
        transition.DaSiWaH3TransitionRender().render(latent, vae, audio_vae, context)


def test_rendered_bridge_joins_with_sources_model_free(tmp_path, monkeypatch):
    clips(tmp_path)
    with av.open(str(tmp_path / 'a.mov')) as container:
        ctx = container.streams.video[0].codec_context
        colors = {k: getattr(ctx, k) for k in ('color_primaries', 'color_trc', 'colorspace', 'color_range')}
    monkeypatch.setattr(transition, '_roots', lambda: (tmp_path, tmp_path))
    latent, vae, audio_vae = window()
    context = {**transition.transition_spec(spec()), 'canvas': (64, 64), 'colors': colors}
    bridge = transition.DaSiWaH3TransitionRender().render(latent, vae, audio_vae, context)['ui']['h3_transition'][0]
    rows = [dict(kind='media', filename=name) for name in ('a.mov', bridge['filename'], 'b.mov')]
    plan = assembly().preflight(dict(version=1, rows=rows), input_root=tmp_path, output_root=tmp_path)
    assert plan['ok'] is True, plan['issues']
    assert plan['frames'] == 30 + 29 + 30
