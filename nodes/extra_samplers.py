"""Extra scheduler choice ``beta57``: core ``beta_scheduler`` with alpha 0.5 / beta 0.7.

Own implementation (the name RES4LYF made common, no RES4LYF code). An already
registered ``beta57`` (e.g. RES4LYF installed) is left untouched.
"""
from functools import partial

import comfy.samplers as samplers


def register():
    if "beta57" not in samplers.SCHEDULER_HANDLERS:
        samplers.SCHEDULER_HANDLERS["beta57"] = samplers.SchedulerHandler(partial(samplers.beta_scheduler, alpha=0.5, beta=0.7))
        samplers.SCHEDULER_NAMES.append("beta57")  # same list object as KSampler.SCHEDULERS
