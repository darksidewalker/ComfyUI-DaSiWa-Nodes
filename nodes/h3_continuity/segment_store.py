"""Schema-2 immutable AV deltas. Only requested safetensors slices are loaded."""
import os
import shutil
import time
import uuid

import torch
from comfy.nested_tensor import NestedTensor
from safetensors import safe_open
from safetensors.torch import save_file

from .segment_policy import delta_layout
from .vendor.continuation_nodes import validate_h3_av_window, _calculate_guided_continuation_layout


def file_stamp(path):
    stat = path.stat()
    # shortcut: size/mtime detect accidental rewrites without reading old payloads;
    # authenticated archival verification would require writer-time content hashes.
    return [stat.st_size, stat.st_mtime_ns]


def stage_delta(store, sampled, context, source_metadata):
    from .core import atomic_json, VERSION, mode_family
    source = store.inspect(context['session'], context['source_id'])
    if source != source_metadata:
        raise ValueError('Continuity source changed after preparation.')
    layout = delta_layout(source['frames'], context['extension_frames'])
    native = _calculate_guided_continuation_layout(source['frames'], context['overlap_frames'], context['extension_frames'])
    video, audio, frames = validate_h3_av_window(sampled, name='sampled continuation')
    if not video.is_floating_point() or not audio.is_floating_point():
        raise ValueError('Sampled continuation streams must be floating point.')
    if frames != native.window_frame_count or audio.shape[-1] != native.overlap_audio_t + native.new_audio_t:
        raise ValueError('Sampled continuation does not match the global window layout.')
    if (video.shape[4]*16, video.shape[3]*16) != (source['width'], source['height']) or mode_family(context['mode']) != source['mode_family']:
        raise ValueError('Stage the source-resolution continuation before upscaling.')
    path = store.member(context['session'], context['source_id'], 'delta.safetensors' if source.get('schema') == 2 else 'latent.safetensors')
    stamp = file_stamp(path)
    if 'source_file_stamp' in context and context['source_file_stamp'] != stamp:
        raise ValueError('Continuity source changed after sampling preparation.')
    with safe_open(str(path), framework='pt', device='cpu') as file:
        vd, ad = file.get_slice('video').get_dtype(), file.get_slice('audio').get_dtype()
    dtypes = {'F16':torch.float16,'BF16':torch.bfloat16,'F32':torch.float32,'F64':torch.float64}
    tensors = {'video':video[:,:,native.overlap_video_t:].detach().to(device='cpu',dtype=dtypes[vd]).clone().contiguous(),
               'audio':audio[...,native.overlap_audio_t:].detach().to(device='cpu',dtype=dtypes[ad]).clone().contiguous()}
    session, clip = context['session'], uuid.uuid4().hex
    directory = store.clip_dir(session, clip)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        save_file(tensors, str(directory / 'delta.safetensors.tmp'))
        import comfy.model_management
        comfy.model_management.throw_exception_if_processing_interrupted()
        if stamp != file_stamp(path):
            raise ValueError('Continuity source changed while writing the segment.')
        os.replace(directory / 'delta.safetensors.tmp', directory / 'delta.safetensors')
        data = {**source, **layout, 'schema':2, 'storage':'delta', 'package_version':VERSION,
                'session':session, 'clip_id':clip, 'parent_id':context['source_id'], 'status':'staged',
                'created_ns':time.time_ns(), 'seconds':layout['frames']/24,
                'prompt':context['resolved_prompt'], 'idea':context.get('idea',''),
                'overlap_frames':context['overlap_frames'], 'extension_frames':context['extension_frames'],
                'provenance':context.get('provenance',{}), 'thumbnails':[], 'export':None,
                'video_dtype':vd, 'audio_dtype':ad,
                'parent_file_stamp':stamp}
        for key in ('output_path','completed_ns','latent_bytes','preview_warning'):
            data.pop(key,None)
        atomic_json(directory / 'clip.json',data)
    except BaseException:
        shutil.rmtree(directory)
        raise
    return {'session':session,'clip_id':clip}


def load_range(store, session, clip, video_start, video_end, audio_start, audio_end):
    metadata = store.inspect(session, clip)
    bounds = ((video_start,video_end,(metadata['frames']-5)//17*5+2),
              (audio_start,audio_end,round(metadata['frames']*40/24)))
    if any(type(x) is not int for pair in bounds for x in pair) or any(not 0 <= start < end <= total for start,end,total in bounds):
        raise ValueError('Invalid half-open continuity token range.')
    pieces = {'video':[], 'audio':[]}
    remaining = {'video':[video_start,video_end], 'audio':[audio_start,audio_end]}
    current, seen = clip,set()
    while any(start < end for start,end in remaining.values()):
        if current in seen or not current:
            raise ValueError('Missing or cyclic continuity parent range.')
        seen.add(current)
        data = store.metadata(session,current)
        delta = data.get('schema') == 2
        path = store.member(session,current,'delta.safetensors' if delta else 'latent.safetensors')
        with safe_open(str(path),framework='pt',device='cpu') as file:
            for name, axis in (('video',2),('audio',3)):
                start,end = remaining[name]
                origin = data[name+'_token_start'] if delta else 0
                length = file.get_slice(name).get_shape()[axis]
                lo,hi = max(start,origin),min(end,origin+length)
                if lo < hi:
                    index = [slice(None)] * (axis+1)
                    index[axis] = slice(lo-origin,hi-origin)
                    pieces[name].insert(0,file.get_slice(name)[tuple(index)].clone())
                    remaining[name][1] = lo
        current = data.get('parent_id','') if delta else ''
    return {'samples':NestedTensor((torch.cat(pieces['video'],2),torch.cat(pieces['audio'],-1)))}


def verify_segment_export(path, metadata):
    from fractions import Fraction
    import comfy.model_management
    from .pyav_media import open_media, video_stream
    expected = metadata['segment_frames']
    start = metadata['segment_start_frame']
    stamp = file_stamp(path)
    with open_media(path) as media:
        video = video_stream(media)
        if video is None or Fraction(video.average_rate or 0) != 24:
            raise ValueError('Segment export requires a video stream at exactly 24 fps; interpolation, trim and pingpong are not supported.')
        count = 0
        for frame in media.decode(video):
            comfy.model_management.throw_exception_if_processing_interrupted()
            if count >= expected or frame.pts is None or abs(float(frame.pts*frame.time_base)-count/24) > max(float(frame.time_base),1e-6):
                raise ValueError('Segment video frame count or presentation timestamps differ from the saved interval.')
            count += 1
        if count != expected:
            raise ValueError('Segment video frame count differs from the saved interval.')
        video_info = dict(codec=video.codec_context.name,pix_fmt=video.codec_context.pix_fmt,
                          width=video.codec_context.width,height=video.codec_context.height)
    with open_media(path) as media:
        streams = list(media.streams.audio)
        if len(streams) != 1:
            raise ValueError('Segment export must contain one nonempty synchronized stereo audio stream.')
        stream = streams[0]
        rate = stream.codec_context.sample_rate
        if not rate or stream.codec_context.layout.nb_channels != 2:
            raise ValueError('Segment export requires stereo audio with a valid sample rate.')
        samples = 0
        max_frame = 0
        wanted = round((start+expected)*rate/24)-round(start*rate/24)
        for frame in media.decode(stream):
            comfy.model_management.throw_exception_if_processing_interrupted()
            if frame.pts is None or abs(float(frame.pts*frame.time_base)-samples/rate) > max(float(frame.time_base),1/rate):
                raise ValueError('Segment audio has a displaced start, timestamp gap or overlap.')
            samples += frame.samples
            max_frame = max(max_frame,frame.samples)
            if samples > wanted + max_frame:
                raise ValueError('Segment audio exceeds its global sample boundary.')
        declared = float(stream.duration*stream.time_base)*rate if stream.duration is not None else None
        # WebM/MKV report no audio stream duration, and AAC/Opus always pad the
        # final codec frame; one frame of padding the container cannot signal is not a boundary error.
        padded = (samples > wanted and samples-wanted < max_frame and (declared is None or abs(declared-wanted) <= 1))
        if samples != wanted and not padded:
            raise ValueError(f'Segment audio ends at {samples} samples; expected {wanted} at {rate} Hz.')
        audio_info = dict(codec=stream.codec_context.name,sample_rate=rate,channels=2,samples=wanted)
    if stamp != file_stamp(path):
        raise ValueError('Export changed during verification; checkpoint remains staged.')
    return dict(kind='segment',start_frame=start,frames=expected,fps=[24,1],
                width=video_info['width'],height=video_info['height'],video=video_info,audio=audio_info)


def inspect_delta(store, session, clip, ready=True):
    leaf = store.metadata(session,clip,ready)
    current, seen, child = clip, set(), None
    while current:
        if current in seen:
            raise ValueError('Continuity parent cycle.')
        seen.add(current)
        data = store.metadata(session,current, ready if current == clip else True)
        if child is not None:
            parent_path = store.member(session,current,'delta.safetensors' if data.get('schema') == 2 else 'latent.safetensors')
            if child.get('parent_file_stamp') != file_stamp(parent_path):
                raise ValueError('Continuity parent checkpoint was replaced or modified.')
            if any(data[k] != child[k] for k in ('width','height','mode_family','fps')):
                raise ValueError('Continuity parent canvas or model family differs.')
            if child['segment_start_frame'] != data['frames']:
                raise ValueError('Continuity delta has a gap or overlapping parent interval.')
        if data.get('schema') != 2:
            store.inspect(session,current)
            break
        layout = delta_layout(data['segment_start_frame'],data['segment_frames'])
        if data.get('storage') != 'delta' or any(data.get(k) != v or type(data.get(k)) is not int for k,v in layout.items()):
            raise ValueError('Invalid global delta boundaries.')
        path = store.member(session,current,'delta.safetensors')
        with safe_open(str(path),framework='pt',device='cpu') as file:
            if set(file.keys()) != {'video','audio'}:
                raise ValueError('Expected video and audio delta streams.')
            v,a = file.get_slice('video'),file.get_slice('audio')
            if v.get_shape() != [1,24,data['video_token_count'],data['height']//16,data['width']//16] or a.get_shape() != [1,32,2,data['audio_token_count']]:
                raise ValueError('Delta shape differs from its global boundaries.')
            if v.get_dtype() != data.get('video_dtype') or a.get_dtype() != data.get('audio_dtype'):
                raise ValueError('Delta dtype differs from its immutable source format.')
            if v.get_dtype() not in {'F16','BF16','F32','F64'} or a.get_dtype() not in {'F16','BF16','F32','F64'}:
                raise ValueError('Delta streams must be floating point.')
        if not data.get('parent_id'):
            raise ValueError('Delta parent is missing.')
        child, current = data, data['parent_id']
    return {**leaf,'latent_bytes':store.member(session,clip,'delta.safetensors').stat().st_size}
