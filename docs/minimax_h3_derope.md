# MiniMax H3 De-rope

H3 smears whip-fast motion (spins, sword arcs, fast reversals): one latent token covers four frames, and four very different poses do not fit into one token. De-rope fixes that after generation. It holds (repeats) the frames of the hot motion spans so the model gets room, re-renders the slowed clip video-to-video at partial denoise, then drops the held frames again. No extra model, no training.

Ported and condensed from MATLOWAI's [ComfyUI-MAINodes Motion Lab](https://github.com/matlowai/ComfyUI-MAINodes/blob/main/docs/MOTION_LAB.md) (GPL-3.0-or-later). Hold planning, frame holding, audio stretching and recovery are bit-identical to Motion Lab (pinned by `tests/test_h3_derope.py`).

## Graph

```
baseline latent + decoded frames (+ audio)
        -> De-rope Prepare (oracle, hold, VAE encode, audio stretch + encode, AV init)
        -> SamplerCustomAdvanced (BasicScheduler -> SplitSigmasDenoise = the inject fraction)
        -> VAEDecode -> De-rope Recover -> combine with the BASELINE audio
```

## Nodes

**MiniMaxH3 • De-rope Prepare** — inputs: baseline `images`, the baseline H3 `samples` they were decoded from, `vae`, `preset`, `frame_rate`; optional `audio` + `audio_vae`. Optional `conditioning`: its frame-indexed H3 keyframes (first/last frame) move to the first copy of their frame on the slowed clock, so an FL2VA last frame still pins the recovered last frame; timed multi-frame/audio keyframes are dropped and counted in the report. `keep_keyframes` off drops all keyframes, for continuations whose keyframes index the sampled clip rather than the decoded timeline. Outputs the AV `latent` for the second pass, the `hold_map`, a one-line `report` and the remapped `conditioning`.

| preset | hot quantile | peak hold | effect |
|---|---|---|---|
| balanced | 0.75 | x4 | Motion Lab default |
| max quality | 0.70 | x4 | wider slowed spans, more cost |
| economy | 0.85 | x3 | tighter spans, cheaper |

Inter-peak valleys up to 8 tokens are bridged and hold edges ramp by one step per token (Motion Lab's production rules). With audio wired, the baseline track is stretched onto the slowed clock (pitch preserved) and seeds the audio rows at 0.5, so the second pass performs the dialogue slowly and the lips stay in sync after recovery.

**MiniMaxH3 • De-rope Recover** — keeps the first frame of every hold group; exact frame selection, no resampling. Use the baseline audio for the result.

## Second pass

Motion Lab's measured settings: a turbo LoRA at 6 total steps with inject 0.48 (about three executed steps), or the base model at 25 total steps with inject 0.70 for the keeper. Do not run a turbo LoRA at its full distilled step count in a partial pass. Cost scales with the slowed length, not the clip length: a 5 s action clip often re-renders as 10–13 s of frames.

Limits: H3 only (17k+5 frames). The baseline must be the latent the frames were decoded from.
