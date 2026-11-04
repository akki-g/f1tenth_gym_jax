# other
from functools import partial
from typing import Callable, Optional

import chex

# jax
import jax
import jax.numpy as jnp

# local
from .utils import DynParams, Param


def _rhs(f: Callable, params: Param, dyn_params: Optional[DynParams]) -> Callable:
    # Only pass dyn_params when given, so two-argument dynamics still work.
    if dyn_params is None:
        return lambda x: f(x, params)
    return lambda x: f(x, params, dyn_params)


@partial(jax.jit, static_argnums=[0, 2])
def integrate_rk4(
    f: Callable,
    x_and_u: chex.Array,
    params: Param,
    dyn_params: Optional[DynParams] = None,
) -> chex.Array:
    rhs = _rhs(f, params, dyn_params)
    for _ in range(params.timestep_ratio):
        k1 = rhs(x_and_u)
        k2_state = x_and_u + params.timestep * (k1 / 2)
        k2 = rhs(k2_state)
        k3_state = x_and_u + params.timestep * (k2 / 2)
        k3 = rhs(k3_state)
        k4_state = x_and_u + params.timestep * k3
        k4 = rhs(k4_state)
        # dynamics integration
        x_and_u = x_and_u + params.timestep * (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        x_and_u = x_and_u.at[4].set(
            jnp.arctan2(jnp.sin(x_and_u[4]), jnp.cos(x_and_u[4]))
        )
    return x_and_u


@partial(jax.jit, static_argnums=[0, 2])
def integrate_euler(
    f: Callable,
    x_and_u: chex.Array,
    params: Param,
    dyn_params: Optional[DynParams] = None,
) -> chex.Array:
    rhs = _rhs(f, params, dyn_params)
    for _ in range(params.timestep_ratio):
        dstate = rhs(x_and_u)
        x_and_u = x_and_u + params.timestep * dstate
        x_and_u = x_and_u.at[4].set(
            jnp.arctan2(jnp.sin(x_and_u[4]), jnp.cos(x_and_u[4]))
        )
    return x_and_u
