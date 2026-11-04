"""Spec section 6 unit checks for the wrappers (delay, offsets, noise, obs)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from f1tenth_gym_jax.envs import F110Env
from rl import car_params as cp
from rl import obs as ob
from rl.domain_rand import make_param
from rl.tests.conftest import FINE_HALF, FINE_WALL
from rl.wrappers import Phase1Wrapper, delay_action, lidar_noise

N_ENVS = 64


@pytest.fixture(scope="module")
def train_env():
    return Phase1Wrapper(F110Env(1, make_param("Spielberg", randomize=True)), randomize=True)


def test_randomization_varies_per_env(train_env):
    """Reset 64 envs: mu, mass and delay differ across them and stay within +-2 sigma."""
    keys = jax.random.split(jax.random.PRNGKey(0), N_ENVS)
    _, ws = jax.vmap(train_env.reset)(keys)
    dp = ws.env_state.dyn_params
    for name in ("mu", "m"):
        vals = np.asarray(getattr(dp, name)).ravel()
        lo = cp.NOMINAL[name] * (1 - cp.CLIP_SIGMAS * cp.SIGMA[name])
        hi = cp.NOMINAL[name] * (1 + cp.CLIP_SIGMAS * cp.SIGMA[name])
        assert vals.min() >= lo - 1e-6 and vals.max() <= hi + 1e-6
        assert len(np.unique(vals)) >= 0.8 * N_ENVS
    k = np.asarray(ws.delay_k)
    assert set(k.tolist()) == {0, 1, 2}
    off = np.asarray(ws.steer_offset)
    assert np.all(np.abs(off) <= cp.STEER_OFFSET_MAX) and len(np.unique(off)) == N_ENVS


def test_randomization_resampled_on_auto_reset(train_env):
    """Per-episode draws change when an episode ends inside step()."""
    obs, ws = train_env.reset(jax.random.PRNGKey(1))
    ws = ws.replace(env_state=ws.env_state.replace(step=cp.MAX_STEPS - 1))  # force timeout
    _, ws2, _, done, _ = train_env.step(jax.random.PRNGKey(2), ws, jnp.array([-0.5, 0.0]))
    assert bool(done)
    assert float(ws2.env_state.dyn_params.mu[0]) != float(ws.env_state.dyn_params.mu[0])
    assert float(ws2.steer_offset) != float(ws.steer_offset)
    assert int(ws2.ep_length) == 0


def test_delay_buffer_unit():
    buf = jnp.zeros((2, 2))
    cmds = [jnp.array([float(i), 10.0 + i]) for i in range(5)]
    executed = []
    for c in cmds:
        e, buf = delay_action(buf, 2, c)
        executed.append(e)
    np.testing.assert_allclose(executed[0], [0, 0])
    np.testing.assert_allclose(executed[1], [0, 0])
    for t in range(2, 5):
        np.testing.assert_allclose(executed[t], cmds[t - 2])


def test_delay_works(train_env):
    """With k = 2, the executed action equals the action from 2 steps earlier."""
    _, ws = train_env.reset(jax.random.PRNGKey(3))
    ws = ws.replace(delay_k=jnp.array(2), steer_offset=jnp.array(0.0))
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, size=(12, 2)).astype(np.float32)
    actions[:, 0] = rng.uniform(-1, -0.6, size=12)  # slow, so the episode survives
    commands, executed = [], []
    key = jax.random.PRNGKey(4)
    for a in actions:
        key, k = jax.random.split(key)
        _, ws, _, done, info = train_env.step(k, ws, jnp.asarray(a))
        assert not bool(done)
        speed, steer = ob.action_to_command(a)
        commands.append(np.array([steer, speed]))
        executed.append(np.asarray(info["executed_command"]))
    np.testing.assert_allclose(executed[0], [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(executed[1], [0.0, 0.0], atol=1e-6)
    for t in range(2, len(actions)):
        np.testing.assert_allclose(executed[t], commands[t - 2], atol=1e-5)


def test_steering_offset_applied(train_env):
    _, ws = train_env.reset(jax.random.PRNGKey(5))
    ws = ws.replace(delay_k=jnp.array(0), steer_offset=jnp.array(0.015))
    _, _, _, _, info = train_env.step(jax.random.PRNGKey(6), ws, jnp.array([-0.8, 0.5]))
    speed, steer = ob.action_to_command(np.array([-0.8, 0.5]))
    np.testing.assert_allclose(np.asarray(info["executed_command"]), [steer + 0.015, speed], atol=1e-6)


def test_lidar_noise_statistics():
    ranges = jnp.full((200_000,), 5.0)
    noisy = np.asarray(lidar_noise(jax.random.PRNGKey(0), ranges))
    dropped = noisy == cp.MAX_RANGE
    assert abs(dropped.mean() - cp.DROPOUT_PROB) < 0.002
    assert abs(noisy[~dropped].std() - cp.RANGE_NOISE_STD) < 0.002
    assert abs(noisy[~dropped].mean() - 5.0) < 0.001
    near = np.asarray(lidar_noise(jax.random.PRNGKey(1), jnp.full((1000,), 0.01)))
    assert near.min() >= 0.0


def test_scan_offset(test_map_dir):
    """Facing a wall at a known distance, the front beam reads that distance minus d (1 cm)."""
    p = make_param("FineBox", randomize=False, collision_on=False)
    d = cp.LIDAR_OFFSET - cp.NOMINAL["lr"]  # st model: state x, y is the CG
    assert abs(p.scan_origin_offset - d) < 1e-9
    assert abs(d - 0.1037) < 1e-4
    env = F110Env(1, p)
    _, st = env.reset(jax.random.PRNGKey(0))
    wall_x = FINE_HALF - FINE_WALL  # inner face of the +x wall
    front = int(np.argmin(np.abs(np.asarray(env.beam_offsets))))
    assert abs(float(env.beam_offsets[front])) < 1e-6
    for x_cg in (0.3, 1.237, 2.0):
        x = st.cartesian_states.at[0].set(jnp.array([x_cg, 0.21, 0.0, 0.0, 0.0, 0.0, 0.0]))
        scanned = env._scan(st.replace(cartesian_states=x), jax.random.PRNGKey(1))
        reading = float(scanned.scans[0, front])
        assert abs(reading - (wall_x - x_cg - d)) < 0.01, (x_cg, reading)
    # sanity: without the offset the same pose reads d further
    env0 = F110Env(1, make_param("FineBox", randomize=False, collision_on=False, scan_origin_offset=0.0))
    x = st.cartesian_states.at[0].set(jnp.array([1.237, 0.21, 0.0, 0.0, 0.0, 0.0, 0.0]))
    r0 = float(env0._scan(st.replace(cartesian_states=x), jax.random.PRNGKey(1)).scans[0, front])
    assert abs(r0 - (wall_x - 1.237)) < 0.01


def test_observation_shape_and_range(train_env):
    """Shape (330,), scan values in [0, 1], stacked frames update in order."""
    obs, ws = train_env.reset(jax.random.PRNGKey(7))
    assert obs.shape == (ob.OBS_DIM,) == (330,)
    nb = ob.NUM_BEAMS
    frames = np.asarray(obs[: 3 * nb]).reshape(3, nb)
    np.testing.assert_array_equal(frames[0], frames[1])  # reset fills the stack
    np.testing.assert_array_equal(frames[0], frames[2])
    np.testing.assert_allclose(np.asarray(obs[-2:]), ob.RESET_ACTION)
    key = jax.random.PRNGKey(8)
    prev = frames
    for i in range(5):
        key, k = jax.random.split(key)
        a = jnp.array([-0.6, 0.2 * (-1) ** i])
        obs, ws, _, done, _ = train_env.step(k, ws, a)
        assert not bool(done)
        o = np.asarray(obs)
        assert o.shape == (330,)
        frames = o[: 3 * nb].reshape(3, nb)
        assert frames.min() >= 0.0 and frames.max() <= 1.0
        np.testing.assert_array_equal(frames[1], prev[0])  # previous newest moved back
        np.testing.assert_array_equal(frames[2], prev[1])
        assert not np.array_equal(frames[0], prev[0])  # fresh noisy scan
        np.testing.assert_allclose(o[3 * nb], float(ws.env_state.cartesian_states[0, 3]) / ob.V_CMD_MAX, rtol=1e-6)
        np.testing.assert_allclose(o[-2:], np.asarray(a), atol=1e-7)
        prev = frames


def test_reward_terms(train_env):
    """Progress dominates when driving forward; collision costs 1."""
    from rl.wrappers import step_reward

    r = step_reward(jnp.array(0.5), 50.0, jnp.array([0.0, 0.1]), jnp.array([0.0, 0.0]), jnp.array(False))
    np.testing.assert_allclose(float(r), 10 * 0.5 / 50 - 0.01 * 0.1, rtol=1e-6)
    r = step_reward(jnp.array(0.0), 50.0, jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]), jnp.array(True))
    np.testing.assert_allclose(float(r), -1.0)


def test_eval_mode_is_deterministic():
    env = Phase1Wrapper(F110Env(1, make_param("Spielberg", randomize=False)), randomize=False)
    o1, ws = env.reset(jax.random.PRNGKey(9))
    assert int(ws.delay_k) == 0 and float(ws.steer_offset) == 0.0
    a = jnp.array([-0.5, 0.0])
    o2, *_ = env.step(jax.random.PRNGKey(10), ws, a)
    o3, *_ = env.step(jax.random.PRNGKey(11), ws, a)
    np.testing.assert_array_equal(np.asarray(o2), np.asarray(o3))  # no sensor noise
    with pytest.raises(ValueError):
        Phase1Wrapper(F110Env(1, make_param("Spielberg", randomize=False)), randomize=True)


def test_laps_counted_from_progress(train_env):
    """The episode ends once accumulated centerline progress reaches max_laps."""
    _, ws = train_env.reset(jax.random.PRNGKey(12))
    moving = ws.env_state.cartesian_states.at[0, 3].set(3.0)  # 3 m/s: ~0.12 m per step
    ws = ws.replace(env_state=ws.env_state.replace(cartesian_states=moving),
                    ep_progress=jnp.array(train_env.max_laps - 1e-4, jnp.float32), delay_k=jnp.array(0))
    _, ws2, _, done, info = train_env.step(jax.random.PRNGKey(13), ws, jnp.array([-0.3, 0.0]))
    assert bool(done) and bool(info["lap_complete"]) and not bool(info["collision"])
    _, ws3, _, done, info = train_env.step(jax.random.PRNGKey(13), ws.replace(ep_progress=jnp.array(0.5, jnp.float32)), jnp.array([-0.3, 0.0]))
    assert not bool(done) and not bool(info["lap_complete"])
    assert train_env.env.params.max_num_laps > 1000  # env's winding counter is off
