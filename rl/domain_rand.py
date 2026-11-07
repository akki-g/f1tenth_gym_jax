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


@functools.lru_cache(maxsize=None)
def _friction_sampler(mu_sigma: float):
    return make_dyn_param_sampler({**cp.SIGMA, "mu": mu_sigma})


def friction_overrides(friction, randomize: bool) -> dict:
    """make_param overrides for a friction range (lo, hi): nominal mu at the
    midpoint and, with ``randomize``, a sampler whose clipped factor reaches
    lo and hi (other parameters as car_params). Empty range: no overrides."""
    if not friction:
        return {}
    lo, hi = (float(x) for x in friction)
    if not 0.0 < lo < hi:
        raise ValueError(f"friction range must satisfy 0 < lo < hi, got {friction}")
    mid = (lo + hi) / 2.0
    out = {"mu": mid}
    if randomize:
        out["dyn_param_sampler"] = _friction_sampler(round((hi - lo) / (2.0 * cp.CLIP_SIGMAS * mid), 9))
    return out


def steer_overrides(steer_asym: bool) -> dict:
    """No overrides: the env's steering limits are the car's servo end stops (car_params), and an
    amplified command is clipped there, as on the car."""
    return {}


# Param.max_num_laps used to switch the env's own lap termination off.
ENV_LAPS_OFF = 1_000_000

# One shared sampler object, so every env's Param hashes the same way.
DEFAULT_SAMPLER = make_dyn_param_sampler()


def reset_ey_scale(map_name: str) -> float:
    """Scale for the env's reset offset so the car starts clear of the walls on
    narrow maps: (narrowest half-width - RESET_CLEARANCE) / RESET_EY_MAX,
    clipped to [0, 1]. 1.0 when the map's centerline file is not in the map
    dir (e.g. Spielberg, which ships with the env and is wide)."""
    import os
    import pathlib

    import numpy as np

    root = os.environ.get("F1TENTH_GYM_JAX_MAP_DIR")
    path = pathlib.Path(root) / map_name / f"{map_name}_centerline.csv" if root else None
    if path is None or not path.exists():
        return 1.0
    widths = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)[:, 2:4]
    return float(np.clip((widths.min() - cp.RESET_CLEARANCE) / cp.RESET_EY_MAX, 0.0, 1.0))


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
        controller="classic",  # matches the ROS sim (decision D3)
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
        # laps are counted from progress in Phase1Wrapper; the env's winding
        # counter is unreliable on non-star-shaped tracks, so keep it out of reach
        max_num_laps=ENV_LAPS_OFF,
        max_steps=cp.MAX_STEPS,
        reset_ey_scale=reset_ey_scale(map_name),
        reward_type="progress",
    )
    kwargs.update(overrides)
    return Param(**kwargs)
