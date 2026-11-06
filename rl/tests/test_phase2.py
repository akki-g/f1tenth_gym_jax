"""Phase 2 (1v1) checks from the recurrent-policy spec, "Validation before training"."""

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym_jax.envs import F110Env
from rl import car_params as cp
from rl import obs as ob
from rl.domain_rand import make_param


def _two_car_env():
    return F110Env(2, make_param("OpenBox", randomize=False, scan_other_agents=True))


def _set_poses(env, state, poses):
    cs = state.cartesian_states
    for i, (x, y, yaw) in enumerate(poses):
        cs = cs.at[i].set(jnp.array([x, y, 0.0, 0.0, yaw, 0.0, 0.0]))
    return state.replace(cartesian_states=cs)


def test_opponent_visible_in_scan(test_map_dir):
    """A car 2.0 m dead ahead reads 2.0 - scan offset - half its length on the front beams (1 cm)."""
    env = _two_car_env()
    _, st = env.reset(jax.random.PRNGKey(0))
    st = _set_poses(env, st, [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)])
    scans = np.asarray(env._scan(st, jax.random.PRNGKey(1)).scans)
    front = ob.NUM_BEAMS // 2
    assert abs(float(env.beam_offsets[front])) < 1e-6
    want = 2.0 - env.params.scan_origin_offset - cp.LENGTH / 2
    for b in (front - 1, front, front + 1):  # +-1 deg still hits the 0.31 m wide rear face
        assert abs(scans[0, b] - want) < 0.01, (b, scans[0, b], want)
    # the opponent sees the ego behind it? no: it faces away; turn it round and it sees the ego's front
    st2 = _set_poses(env, st, [(0.0, 0.0, 0.0), (2.0, 0.0, np.pi)])
    scans2 = np.asarray(env._scan(st2, jax.random.PRNGKey(1)).scans)
    assert abs(scans2[1, front] - (2.0 - env.params.scan_origin_offset - cp.LENGTH / 2)) < 0.01


def test_car_outside_fov_changes_no_beam(test_map_dir):
    env = _two_car_env()
    env_map_only = F110Env(2, make_param("OpenBox", randomize=False, scan_other_agents=False))
    _, st = env.reset(jax.random.PRNGKey(0))
    st = _set_poses(env, st, [(0.0, 0.0, 0.0), (-3.0, 0.0, 0.0)])  # dead behind: the 90 deg blind sector
    with_agents = np.asarray(env._scan(st, jax.random.PRNGKey(1)).scans[0])
    map_only = np.asarray(env_map_only._scan(st, jax.random.PRNGKey(1)).scans[0])
    np.testing.assert_array_equal(with_agents, map_only)
    # and far away in view: nothing within max range either
    st = _set_poses(env, st, [(0.0, 0.0, 0.0), (25.0, 0.0, 0.0)])
    np.testing.assert_array_equal(np.asarray(env._scan(st, jax.random.PRNGKey(1)).scans[0]), map_only)
