"""Real CPU PyAV readback at the publication boundary."""
from test_h3_segment_backend import core,latent,context,root,importlib
from pathlib import Path
from fractions import Fraction
import pytest
import av
import numpy as np


def media_file(path, frames=119, fps=24, audio_samples=None, video=True):
    with av.open(str(path),'w') as out:
        a = out.add_stream('aac' if path.suffix == '.mp4' else 'pcm_s16le',rate=32000)
        a.layout = 'stereo'
        if video:
            v = out.add_stream('libx264' if path.suffix == '.mp4' else 'ffv1',rate=fps)
            v.width = v.height = 64
            v.pix_fmt = 'yuv420p'
            for i in range(frames):
                frame = av.VideoFrame.from_ndarray(np.full((64,64,3),i%255,dtype=np.uint8),format='rgb24')
                frame.pts = i
                frame.time_base = Fraction(1,fps)
                for packet in v.encode(frame): out.mux(packet)
            for packet in v.encode(): out.mux(packet)
        count = audio_samples if audio_samples is not None else round(243*32000/24)-round(124*32000/24)
        for start in range(0,count,1024):
            size = min(1024,count-start)
            frame = av.AudioFrame.from_ndarray(np.zeros((1,size*2),dtype=np.int16),format='s16',layout='stereo')
            frame.sample_rate = 32000
            frame.pts = start
            frame.time_base = Fraction(1,32000)
            for packet in a.encode(frame): out.mux(packet)
        for packet in a.encode(): out.mux(packet)


def setup(tmp_path,monkeypatch):
    nodes = importlib.import_module('h3_segment_test.nodes.h3_continuity.nodes')
    store = core.ClipStore(tmp_path/'store')
    m = root(store)
    ticket = store.stage_delta(latent(141,9.),context(),m)
    monkeypatch.setattr(nodes,'ClipStore',lambda:store)
    import folder_paths
    monkeypatch.setattr(folder_paths,'get_output_directory',lambda:str(tmp_path))
    monkeypatch.setattr(folder_paths,'get_temp_directory',lambda:str(tmp_path/'temp'))
    return nodes,store,ticket


@pytest.mark.parametrize('case',['missing','duration','fps','audio_end','no_video','corrupt','cancel'])
def test_failed_publish_preserves_source_latest_and_staged(tmp_path,monkeypatch,case):
    nodes,store,ticket = setup(tmp_path,monkeypatch)
    path = tmp_path/'segment.mkv'
    if case == 'corrupt': path.write_bytes(b'not a video')
    elif case != 'missing': media_file(path,frames=118 if case == 'duration' else 119,fps=25 if case == 'fps' else 24,audio_samples=10000 if case == 'audio_end' else None,video=case != 'no_video')
    before = (store.session_dir('s')/'latest.json').read_bytes()
    if case == 'cancel':
        import comfy.model_management
        monkeypatch.setattr(comfy.model_management,'throw_exception_if_processing_interrupted',lambda: (_ for _ in ()).throw(RuntimeError('interrupted')))
    with pytest.raises(Exception): nodes.DaSiWaH3ContinuityPublish().publish(str(path),ticket)
    assert store.metadata('s',ticket['clip_id'],ready=False)['status'] == 'staged'
    assert (store.session_dir('s')/'latest.json').read_bytes() == before
    assert store.inspect('s','root')['status'] == 'ready'


def test_publish_metadata_failure_rolls_back_segment_ready(tmp_path,monkeypatch):
    nodes,store,ticket = setup(tmp_path,monkeypatch)
    path = tmp_path/'segment.mkv'
    media_file(path)
    original = core.atomic_json
    def fail_latest(path,data):
        if Path(path).name == 'latest.json': raise OSError('disk full')
        return original(path,data)
    monkeypatch.setattr(core,'atomic_json',fail_latest)
    with pytest.raises(OSError): nodes.DaSiWaH3ContinuityPublish().publish(str(path),ticket)
    assert store.metadata('s',ticket['clip_id'],ready=False)['status'] == 'staged'
    assert store.list_clips('s')['latest_id'] == 'root'


def test_publish_accepts_actual_h264_aac_priming_and_padding(tmp_path,monkeypatch):
    nodes,store,ticket = setup(tmp_path,monkeypatch)
    path = tmp_path/'segment.mp4'
    media_file(path)
    nodes.DaSiWaH3ContinuityPublish().publish(str(path),ticket)
    data = store.inspect('s',ticket['clip_id'])
    assert data['export']['audio']['samples'] == 158667
    assert data['export']['audio']['codec'] == 'aac'


def test_publish_accepts_audio_tail_padding_without_declared_duration(tmp_path,monkeypatch):
    # WebM/MKV report no audio stream duration, and AAC/Opus always pad the final
    # codec frame; one frame of undetectable tail padding must not fail the boundary.
    nodes,store,ticket = setup(tmp_path,monkeypatch)
    path = tmp_path/'segment.mkv'
    media_file(path,audio_samples=158667+107)
    nodes.DaSiWaH3ContinuityPublish().publish(str(path),ticket)
    assert store.inspect('s',ticket['clip_id'])['export']['audio']['samples'] == 158667


def test_publish_checks_segment_duration_not_timeline(tmp_path,monkeypatch):
    nodes,store,ticket = setup(tmp_path,monkeypatch)
    path = tmp_path/'segment.mkv'
    media_file(path)
    import sys
    from types import SimpleNamespace
    events = []
    monkeypatch.setitem(sys.modules,'server',SimpleNamespace(PromptServer=SimpleNamespace(instance=SimpleNamespace(send_sync=lambda name,data:events.append((name,data))))))
    result = nodes.DaSiWaH3ContinuityPublish().publish(str(path),ticket)
    assert events[0][1]['export_kind'] == 'segment'
    assert {'session','clip_id','status'} <= events[0][1].keys()
    assert '119 new frames' in result['result'][0] and '243 frames' in result['result'][0]
    data = store.inspect('s',ticket['clip_id'])
    assert data['frames'] == 243 and data['export']['frames'] == 119
    assert data['export']['kind'] == 'segment' and data['export']['start_frame'] == 124
    assert data['export']['audio']['samples'] == 158667
    full = store.stage(latent(243),dict(session='s',run_id='full',mode='REF2VA',resolved_prompt='full'))
    with pytest.raises(ValueError): nodes.DaSiWaH3ContinuityPublish().publish(str(path),full)
