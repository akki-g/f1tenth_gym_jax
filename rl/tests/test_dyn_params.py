"""Spec section 6 unit checks for the per-env vehicle-parameter patch."""

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym_jax.envs import F110Env
from rl import car_params as cp
from rl.domain_rand import make_param

N_ENVS = 64


def _bounds(name):
    lo = cp.NOMINAL[name] * (1 - cp.CLIP_SIGMAS * cp.SIGMA[name])
    hi = cp.NOMINAL[name] * (1 + cp.CLIP_SIGMAS * cp.SIGMA[name])
    return lo, hi


def test_randomization_varies_per_env_dynamics():
    """Reset 64 envs: mu and mass differ across them and stay within +-2 sigma.

    (The delay part of this check is in test_wrappers.py.)
    """
    env = F110Env(1, make_param("Spielberg", randomize=True))
    keys = jax.random.split(jax.random.PRNGKey(0), N_ENVS)
    _, state = jax.vmap(env.reset)(keys)
    dp = state.dyn_params
    for name in ("mu", "m", "C_Sf", "C_Sr", "lf", "h", "I"):
        vals = np.asarray(getattr(dp, name)).ravel()
        assert vals.shape == (N_ENVS,)
        # ~4.6% of draws hit the +-2 sigma clip and coincide there
        assert len(np.unique(vals)) >= 0.8 * N_ENVS, f"{name} barely varies across envs"
        lo, hi = _bounds(name)
        assert vals.min() >= lo - 1e-6 and vals.max() <= hi + 1e-6, name
        assert vals.std() > 0.25 * cp.SIGMA[name] * cp.NOMINAL[name], name
    # lr follows lf so the wheelbase stays fixed (see results note, deviation D2)
    wheelbase = np.asarray(dp.lf + dp.lr).ravel()
    np.testing.assert_allclose(wheelbase, cp.NOMINAL["lf"] + cp.NOMINAL["lr"], atol=1e-6)
    lr = np.asarray(dp.lr).ravel()
    assert lr.min() >= cp.NOMINAL["lr"] - (cp.NOMINAL["lf"] - _bounds("lf")[0]) - 1e-6
    assert len(np.unique(lr)) >= 0.8 * N_ENVS


def test_nominal_without_randomization():
    env = F110Env(1, make_param("Spielberg", randomize=False))
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    _, state = jax.vmap(env.reset)(keys)
    for name, nominal in cp.NOMINAL.items():
        np.testing.assert_allclose(np.asarray(getattr(state.dyn_params, name)), nominal, rtol=1e-6)


def _hard_turn(env, state, mu, speed=5.0, steps=50):
    x = state.cartesian_states.at[0, 3].set(speed)
    s = state.replace(
        cartesian_states=x,
        last_cartesian_states=x,
        dyn_params=state.dyn_params.replace(mu=jnp.array([mu])),
    )
    action = {"agent_0": jnp.array([cp.S_MAX, speed])}  # env order: [steer, speed]
    step = jax.jit(env.step_env)
    traj = []
    for _ in range(steps):
        _, s, *_ = step(jax.random.PRNGKey(1), s, action)
        traj.append(np.asarray(s.cartesian_states[0]))
    return np.array(traj)


def test_parameters_change_the_physics():
    """mu=0.8 vs mu=1.3, same constant hard turn: trajectories diverge, low mu slips more."""
    env = F110Env(1, make_param("Spielberg", randomize=False, produce_scans=False, collision_on=False))
    _, state = env.reset(jax.random.PRNGKey(0))
    low, high = _hard_turn(env, state, 0.8), _hard_turn(env, state, 1.3)

    assert np.linalg.norm(low[-1, :2] - high[-1, :2]) > 0.5

    # state: [x, y, delta, v, psi, psi_dot, beta]; skip the steering transient
    beta_low, beta_high = np.abs(low[10:, 6]).mean(), np.abs(high[10:, 6]).mean()
    assert beta_low > 1.5 * beta_high

    def yaw_rate_deficit(tr):  # kinematic yaw rate minus actual: understeer
        kin = tr[:, 3] * np.tan(tr[:, 2]) / cp.WHEELBASE
        return np.mean(kin[10:] - tr[10:, 5])

    assert yaw_rate_deficit(low) > yaw_rate_deficit(high) > 0


def test_no_recompile_on_reset(compile_counter):
    """Reset + step 100 times with fresh parameter draws: one compile each, no more."""
    env = F110Env(1, make_param("Spielberg", randomize=True))
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    action = {"agent_0": jnp.array([0.1, 2.0])}

    key = jax.random.PRNGKey(0)
    mus = []
    for i in range(100):
        key, k_reset, k_step = jax.random.split(key, 3)
        _, state = reset(k_reset)
        _, state, *_ = step(k_step, state, action)
        mus.append(float(state.dyn_params.mu[0]))
        if i == 0:
            first_call = len(compile_counter.messages)

    assert first_call > 0, "compile logging did not capture anything"
    later = compile_counter.messages[first_call:]
    assert later == [], f"recompiled after the first call: {later}"
    assert reset._cache_size() == 1 and step._cache_size() == 1
    assert len(set(mus)) > 90  # parameters really changed between resets
