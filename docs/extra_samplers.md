# beta57 scheduler

The pack adds the scheduler `beta57` to ComfyUI's scheduler list (KSampler, KSampler Advanced, BasicScheduler and every node that reads that list). No node to wire: pick it in the scheduler dropdown.

`beta57` is core `beta_scheduler` with alpha 0.5 and beta 0.7 (core `beta` uses 0.6/0.6); it spends more steps at high noise. It works with every sampler; core `res_multistep` + `beta57` is the closest core equivalent of the popular RES4LYF "res + beta57" recipes (RES4LYF's own `res_2s` adds SDE noise and two model calls per step, so images differ for the same seed).

Own implementation in `nodes/extra_samplers.py`; no RES4LYF code and no extra dependency. If RES4LYF (or another pack) already registered `beta57`, the pack leaves it alone.

Verification: `tests/test_extra_samplers.py` (registered once, sigmas exactly equal to core `beta_scheduler(alpha=0.5, beta=0.7)`). A real SDXL CPU run with `beta57` produced a clean, fully denoised image.
