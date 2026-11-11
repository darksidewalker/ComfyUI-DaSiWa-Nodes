# MiniMax H3 De-rope

H3 smears whip-fast motion (spins, sword arcs, fast reversals): one latent token covers four frames, and four very different poses do not fit into one token. De-rope fixes that after generation. It holds (repeats) the frames of the hot motion spans so the model gets room, re-renders the slowed clip video-to-video at partial denoise, then drops the held frames again. No extra model, no training.

Ported and condensed from MATLOWAI's [ComfyUI-MAINodes Motion Lab](https://github.com/matlowai/ComfyUI-MAINodes/blob/main/docs/MOTION_LAB.md) (GPL-3.0-or-later). Hold planning, frame holding, audio stretching and default (background preservation OFF) recovery are bit-identical to Motion Lab (pinned by `tests/test_h3_derope.py`). Optional static-background compositing is a DaSiWa extension.

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

### Optional static-background preservation

Recover appends two optional inputs; existing required sockets and output 0 `IMAGE` are unchanged:

| Input | Default | Purpose |
|---|---|---|
| `baseline` (`IMAGE`) | unconnected | Original decoded frames **before Prepare**, at the original clock; identical **T,H,W,C** to the recovered result, not the held second-pass length. |
| `keep_background` (`BOOLEAN`) | **false** | ON composites static baseline detail around the second pass's motion/changed regions. |

Wire the same baseline image batch into Prepare `images` and Recover `baseline`. Keep the second-pass decode on Recover `images` and Prepare's hold map on `hold_map`. With the switch absent/OFF, Recover executes the old exact selection, without reading/validating the baseline or computing masks. With ON, both batches must be nonempty floating RGB/RGBA images, finite in [0,1], with identical recovered shape; missing or mismatched baseline is an error. The result retains the recovered dtype and CPU device, and neither input is modified.

The mask unions baseline previous/next-frame motion with spatially coherent, meaningful recovered-minus-baseline changes. A wider structural scale and removal of the median colour shift prevent ordinary VAE softening/colour drift from activating every background pixel. Dilation covers both old and new limb locations, feathering softens the outside boundary, a three-frame union bridges threshold dips, and gradual release reduces mask chatter. Broad camera motion or scene-wide structural changes fall back to the complete second pass rather than overlaying incompatible scenes.

This is a conservative, model-free heuristic, **not segmentation or optical-flow alignment**. It cannot promise sharpness restoration during camera motion, a substantially reimagined background, or sub-threshold low-contrast changes. Existing inference blur inside the repaired subject is not removed by compositing. Leave OFF when the full rerender is the preferred look. Same-clock inputs must also depict the same scene; matching dimensions alone cannot verify provenance.

CPU processing adds only a few frame-sized RGB/mask intermediates beyond the normal output batch, not another full-video float RGB copy. Validation and processing loops check ComfyUI interruption. The ComfyUI/Python log reports operation, raw/feathered coverage and fallback count. There is no model load, GPU resampling, frame retiming or audio socket: **continue using the untouched baseline audio**.

## Second pass

Motion Lab's measured settings: a turbo LoRA at 6 total steps with inject 0.48 (about three executed steps), or the base model at 25 total steps with inject 0.70 for the keeper. Do not run a turbo LoRA at its full distilled step count in a partial pass. Cost scales with the slowed length, not the clip length: a 5 s action clip often re-renders as 10–13 s of frames.

Limits: H3 only (17k+5 frames). The baseline must be the latent the frames were decoded from.
