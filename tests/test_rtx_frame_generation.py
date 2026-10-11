"""Contract tests: SDK stubs are not interpolation quality validation."""
import sys
from pathlib import Path
from types import SimpleNamespace
from enum import IntEnum
import contextlib

import pytest
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "ComfyUI"),
               str(Path(__file__).resolve().parents[1] / "nodes")]
import nodes_rtx_frame_generation as fg


class Encoding(IntEnum):
    RGB8 = 0
    RGB10A2 = 1


class Mode(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


class PairEffect:
    instances = []
    Mode = Mode

    def __init__(self, mode, automatic_shot_change_detection_enabled, device, image_encoding):
        self.settings = (mode, automatic_shot_change_detection_enabled, device, image_encoding)
        self.encoding = image_encoding
        self.calls = []
        self.buffer = None
        self.closed = False
        self.instances.append(self)

    def load(self):
        assert (self.input_width, self.input_height) == (8, 8)

    def _generate(self, previous, current, t, shot_change, non_blocking, stream_ptr):
        self.calls.append((t, shot_change, non_blocking, stream_ptr))
        if self.encoding == Encoding.RGB10A2:
            previous, current = fg._dasiwa_fg_unpack(previous), fg._dasiwa_fg_unpack(current)
        result = previous * (1-t) + current * t
        if self.encoding == Encoding.RGB10A2:
            result = fg._dasiwa_fg_pack(result)
        if self.buffer is None:
            self.buffer = torch.empty_like(result)
        self.buffer.copy_(result)
        return SimpleNamespace(image=self.buffer)

    def run(self, previous, current, *, frame_index, shot_change, non_blocking, stream_ptr):
        return self._generate(previous, current, frame_index/self.frame_multiplier,
                              shot_change, non_blocking, stream_ptr)

    def run_at_timestep(self, previous, current, timestep, *, shot_change, non_blocking, stream_ptr):
        return self._generate(previous, current, timestep, shot_change, non_blocking, stream_ptr)

    def close(self):
        self.closed = True
        if self.buffer is not None:
            self.buffer.zero_()  # borrowed outputs must already be owned


@pytest.fixture
def cpu_sdk(monkeypatch):
    PairEffect.instances = []
    monkeypatch.setattr(fg, '_dasiwa_fg_sdk', lambda: (PairEffect, Mode, Encoding, True))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'device', lambda *args: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: None)
    monkeypatch.setattr(fg, '_dasiwa_fg_stream', lambda *args: contextlib.nullcontext())
    original = fg._dasiwa_fg_prepare
    monkeypatch.setattr(fg, '_dasiwa_fg_prepare', lambda image, device, packed: original(image, 'cpu', packed))
    monkeypatch.setattr(fg, '_dasiwa_fg_allocate_output', lambda shape, dtype, *args, **kwargs: (torch.empty(shape, dtype=dtype), None))
    monkeypatch.setattr(fg, '_dasiwa_fg_interrupt', lambda: None)


@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('multiplier', [2, 3, 16])
def test_preserves_sources_count_owned_buffer_and_dtype(cpu_sdk, packed, multiplier):
    images = torch.stack([torch.full((8,8,4), v) for v in (.125, .5, .875)]).half()
    out, = fg.DaSiWa_RTX_FrameGeneration().dasiwa_rtx_generate_frames(images,
        multiplier=multiplier, image_encoding='10-bit RGB (RGB10A2)' if packed else '8-bit RGB',
        shot_change=True, automatic_shot_change_detection=False, mode='HIGH', device_id=1)
    assert out.shape == (2*multiplier+1,8,8,3)
    assert torch.equal(out[::multiplier], images[..., :3].to(out.dtype))
    assert out.dtype == (torch.float32 if packed else torch.float16)
    assert out[1].mean() < out[multiplier+1].mean()
    effect = PairEffect.instances[-1]
    assert effect.closed
    assert effect.settings[:3] == (Mode.HIGH, False, 1)
    assert len(effect.calls) == 2*(multiplier-1)
    assert all(call[1:] == (True, False, 0) for call in effect.calls)


def test_timestep_order_and_warning(cpu_sdk, monkeypatch):
    logs=[]
    monkeypatch.setattr(fg, '_dasiwa_fg_log', lambda *a: logs.append(a))
    images=torch.stack([torch.full((8,8,3), v) for v in (0., .5, 1.)])
    out,=fg.DaSiWa_RTX_FrameGeneration().dasiwa_rtx_generate_frames(images,
        generation_type='specific timestep',timestep=.25,multiplier=16)
    assert out.shape[0]==5
    assert torch.equal(out[::2],images)
    assert out[1,0,0,0]==.125 and out[3,0,0,0]==.625
    assert logs and 'nonuniform' in logs[0][1]
    assert [c[0] for c in PairEffect.instances[-1].calls]==[.25,.25]


@pytest.mark.parametrize('kwargs', [{'multiplier':1},{'multiplier':17},{'multiplier':2.5},
    {'multiplier':True},{'timestep':float('nan')},{'timestep':0.},{'mode':'ULTRA'},
    {'generation_type':'bad'},{'image_encoding':'HDR'},{'device_id':-1}])
def test_invalid_requests(kwargs):
    with pytest.raises(ValueError):
        fg.DaSiWa_RTX_FrameGeneration().dasiwa_rtx_generate_frames(torch.zeros(2,8,8,3),**kwargs)


def test_one_frame_no_cuda_or_sdk(monkeypatch):
    monkeypatch.setattr(fg, '_dasiwa_fg_sdk', lambda: pytest.fail('passthrough must not import SDK'))
    images=torch.ones(1,8,8,4).half()
    out,=fg.DaSiWa_RTX_FrameGeneration().dasiwa_rtx_generate_frames(images)
    assert torch.equal(out,images[..., :3]) and out.dtype==torch.float16


def test_missing_sdk_and_import_free_schema(monkeypatch):
    monkeypatch.setitem(sys.modules,'nvvfx',SimpleNamespace())
    with pytest.raises(RuntimeError,match='lacks VideoFrameGeneration'):
        fg.DaSiWa_RTX_FrameGeneration().dasiwa_rtx_generate_frames(torch.zeros(2,8,8,3))
    assert fg.DaSiWa_RTX_FrameGeneration.INPUT_TYPES()['optional']['image_encoding'][0]==['8-bit RGB']


def test_saved_packed_request_rejected(cpu_sdk, monkeypatch):
    monkeypatch.setattr(fg, '_dasiwa_fg_sdk', lambda: (PairEffect,Mode,Encoding,False))
    assert 'image_encoding' in fg.DaSiWa_RTX_FrameGeneration.INPUT_TYPES()['optional']
    with pytest.raises(RuntimeError,match='RGB10A2'):
        fg.DaSiWa_RTX_FrameGeneration().dasiwa_rtx_generate_frames(torch.zeros(2,8,8,3),image_encoding='10-bit RGB (RGB10A2)')


def test_interruption_closes_and_removes_failed_mmap(cpu_sdk, monkeypatch, tmp_path):
    path=tmp_path/'output.mmap'
    path.write_bytes(b'')
    monkeypatch.setattr(fg,'_dasiwa_fg_allocate_output',lambda shape,dtype,*a,**kw:(torch.empty(shape,dtype=dtype),str(path)))
    def interrupt():
        raise RuntimeError('interrupted')
    monkeypatch.setattr(fg,'_dasiwa_fg_interrupt',interrupt)
    with pytest.raises(RuntimeError,match='interrupted'):
        fg.DaSiWa_RTX_FrameGeneration().dasiwa_rtx_generate_frames(torch.zeros(2,8,8,3))
    assert PairEffect.instances[-1].closed and not path.exists()


def test_fg_budget_forwarding(cpu_sdk, monkeypatch):
    seen={}
    def alloc(shape,dtype,device,**kwargs):
        seen.update(kwargs)
        assert shape==(3,8,8,3)
        return torch.empty(shape,dtype=dtype),None
    monkeypatch.setattr(fg,'_dasiwa_fg_allocate_output',alloc)
    fg.DaSiWa_RTX_FrameGeneration().dasiwa_rtx_generate_frames(torch.zeros(2,8,8,3),auto_unload_models=False,use_mmap=True)
    assert seen['allow_mmap'] is True and seen['auto_unload_models'] is False
    assert seen['compute_device']==torch.device('cuda:0')
    assert seen['working_reserve_bytes']>=1024**3+8*8*72


def test_registration_abi_and_names():
    import nodes_rtx_upscaler_refiner as old
    assert old.NODE_CLASS_MAPPINGS['DaSiWa_RTX_UpscalerRefiner'] is old.DaSiWa_RTX_UpscalerRefiner
    assert old.NODE_DISPLAY_NAME_MAPPINGS['DaSiWa_RTX_UpscalerRefiner']=='RTX Upscaler & Refiner'
    assert fg.NODE_DISPLAY_NAME_MAPPINGS['DaSiWa_RTX_FrameGeneration']=='RTX Frame Generation'
    assert fg.DaSiWa_RTX_FrameGeneration.FUNCTION=='dasiwa_rtx_generate_frames'
    entry=(Path(__file__).resolve().parents[1]/'__init__.py').read_text()
    assert '"DaSiWa_RTX_FrameGeneration": DaSiWa_RTX_FrameGeneration' in entry
    assert '"DaSiWa_RTX_FrameGeneration": "RTX Frame Generation"' in entry
