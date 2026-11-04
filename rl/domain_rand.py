"""Per-environment parameter sampling (spec sections 3 and 4.1).

The patched env draws its traced ``DynParams`` at every reset through
``Param.dyn_param_sampler``. This module builds that sampler from
``car_params`` and the ``Param`` used for phase-1 training and evaluation.
"""

from __future__ import annotations

import functools
from typing import Callable, Mapping

import chex
import jax
import jax.numpy as jnp

from f1tenth_gym_jax.envs.utils import DYN_PARAM_NAMES, DynParams, Param

from . import car_params as cp


def make_dyn_param_sampler(
    sigma: Mapping[str, float] = cp.SIGMA,
    clip_sigmas: float = cp.CLIP_SIGMAS,
    fixed_wheelbase: bool = True,
) -> Callable[[chex.PRNGKey, DynParams], DynParams]:
    """Multiplicative factor N(1, sigma) per parameter, clipped to +-clip_sigmas*sigma.

    With ``fixed_wheelbase`` the CG moves along the car instead of the
    wheelbase changing: lf is sampled and lr = (lf + lr)_nominal - lf_sampled.
    """
    sigma = dict(sigma)

    def sampler(key: chex.PRNGKey, nominal: DynParams) -> DynParams:
        keys = jax.random.split(key, len(DYN_PARAM_NAMES))
        out = {}
        for k, name in zip(keys, DYN_PARAM_NAMES):
            base = getattr(nominal, name)
            z = jnp.clip(jax.random.normal(k, jnp.shape(base)), -clip_sigmas, clip_sigmas)
            out[name] = base * (1.0 + sigma.get(name, 0.0) * z)
        if fixed_wheelbase:
            out["lr"] = nominal.lf + nominal.lr - out["lf"]
        return DynParams(**out)

    return sampler


# One shared sampler object, so every env's Param hashes the same way.
DEFAULT_SAMPLER = make_dyn_param_sampler()


@functools.lru_cache(maxsize=None)
def make_param(map_name: str, randomize: bool = True, model: str = "st", **overrides) -> Param:
    """The phase-1 ``Param`` for one map.

    ``randomize`` turns on per-reset vehicle-parameter sampling. Sensor noise,
    delay and steering offset live in the wrappers, so the env's own scan noise
    is switched off here.
    """
    ref_from_rear_axle = cp.NOMINAL["lr"] if model in ("st", "st_smooth") else 0.0
    kwargs = dict(
        map_name=map_name,
        # dynamics
        model=model,
        integrator="rk4",
        timestep=cp.TIMESTEP,
        timestep_ratio=cp.TIMESTEP_RATIO,
        longitudinal_action_type="velocity",
        steering_action_type="steeringangle",
        s_min=cp.S_MIN,
        s_max=cp.S_MAX,
        sv_min=cp.SV_MIN,
        sv_max=cp.SV_MAX,
        v_min=cp.V_MIN,
        v_max=cp.V_MAX,
        a_max=cp.A_MAX,
        v_switch=cp.V_SWITCH,
        width=cp.WIDTH,
        length=cp.LENGTH,
        **{name: cp.NOMINAL[name] for name in DYN_PARAM_NAMES},
        dyn_param_sampler=DEFAULT_SAMPLER if randomize else None,
        # sensing
        produce_scans=True,
        num_beams=cp.NUM_BEAMS,
        fov=cp.LIDAR_FOV,
        max_range=cp.MAX_RANGE,
        scan_origin_offset=cp.scan_origin_offset(ref_from_rear_axle),
        scan_noise_std=0.0,
        # episode
        collision_on=True,
        observe_others=False,
        max_num_laps=cp.MAX_LAPS,
        max_steps=cp.MAX_STEPS,
        reward_type="progress",
    )
    kwargs.update(overrides)
    return Param(**kwargs)
