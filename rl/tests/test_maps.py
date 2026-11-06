"""Map lists, the narrow-track generator and the multi-map env (spec 4.3)."""

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import distance_transform_edt as edt

from rl import maps as M
from rl import obs as ob

ALL_GEN = M.TRAIN_GEN_SEEDS + M.HELDOUT_GEN_SEEDS


def point_in_polygon(poly, p) -> bool:
    """Even-odd ray casting (a closed polygon given as an (N, 2) vertex array)."""
    x, y = poly[:, 0], poly[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    crosses = (y > p[1]) != (yn > p[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at = x + (p[1] - y) * (xn - x) / (yn - y)
    return bool(np.count_nonzero(crosses & (p[0] < x_at)) % 2)


def test_point_in_polygon():
    square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    assert point_in_polygon(square, (1.0, 1.0))
    assert not point_in_polygon(square, (3.0, 1.0))
    assert not point_in_polygon(square, (1.0, -0.5))
    u = np.array([[0, 0], [3, 0], [3, 3], [2, 3], [2, 1], [1, 1], [1, 3], [0, 3]], float)
    assert point_in_polygon(u, (0.5, 2.0)) and not point_in_polygon(u, (1.5, 2.0))


@pytest.fixture(scope="module")
def prepared():
    map_dir = M.use_map_dir(M.DEFAULT_MAP_DIR)
    missing = [
        m for m in M.TRAIN_MAPS + M.HELDOUT_MAPS
        if m != "Spielberg" and not (map_dir / m).exists()
    ]
    if missing:
        pytest.fail(f"maps missing: {missing}; run `python -m rl.maps prepare` first")
    return map_dir


def test_lists():
    assert len(M.TRAIN_RACETRACKS) == 8 and len(M.TRAIN_GEN_SEEDS) == 8
    assert len(M.HELDOUT_RACETRACKS) == 3 and len(M.HELDOUT_GEN_SEEDS) == 3
    assert len(set(M.TRAIN_MAPS)) == 16 and len(set(M.HELDOUT_MAPS)) == 6
    assert not set(M.TRAIN_MAPS) & set(M.HELDOUT_MAPS)
    cfg = M.map_lists_config()
    assert cfg["train_gen_seeds"] == list(M.TRAIN_GEN_SEEDS)


@pytest.mark.parametrize("seed", ALL_GEN)
def test_generated_track_is_reproducible_and_narrow(prepared, seed):
    g = M.generate_duct_track(seed)
    assert g is not None
    name = M.gen_name(seed)
    img = np.array(Image.open(prepared / name / f"{name}.png"))[::-1] > 128
    np.testing.assert_array_equal(img, g["free"])  # committed map == code output
    cl = np.loadtxt(prepared / name / f"{name}_centerline.csv", delimiter=",")
    np.testing.assert_allclose(cl, g["centerline"], atol=1e-5)

    assert M.GEN_WIDTH[0] <= g["width"] <= M.GEN_WIDTH[1]
    assert M.GEN_LENGTH[0] <= g["length"] <= M.GEN_LENGTH[1]
    assert g["min_radius"] >= M.GEN_MIN_RADIUS
    # measured corridor width at the centerline matches the nominal width
    free = g["free"].copy()
    lo = np.array(g["origin"])
    rc = ((g["centerline"][:, [1, 0]] - lo[[1, 0]]) / g["resolution"]).astype(int)
    # distance to the nearest wall pixel, ignoring free space outside the walls
    wall_dist = edt(free) * g["resolution"]
    measured = 2 * wall_dist[rc[:, 0], rc[:, 1]]
    assert abs(np.median(measured) - g["width"]) < 2 * g["resolution"]


def test_winding_point_inside_every_loop(prepared):
    """The env counts laps around a point beside s = 0; it must lie inside the loop."""
    from f1tenth_gym_jax.envs.track import Track

    from f1tenth_gym_jax.envs import F110Env

    from rl.domain_rand import make_param

    for name in M.TRAIN_MAPS + M.HELDOUT_MAPS:
        env = F110Env(1, make_param(name, randomize=False, produce_scans=False))
        t = env.track
        s = np.linspace(0, float(t.centerline.s[-1]), 600, endpoint=False)
        loop = np.array([t.centerline.calc_position(si) for si in s])
        assert point_in_polygon(loop, np.asarray(env.winding_point)), name


def _pure_pursuit_lap(name, speed=2.5, lookahead=1.0, steps=2000):
    from f1tenth_gym_jax.envs import F110Env

    from rl import car_params as cp
    from rl.domain_rand import make_param

    env = F110Env(1, make_param(name, randomize=False, produce_scans=False, max_steps=steps + 10))
    _, st = env.reset(jax.random.PRNGKey(0))

    @jax.jit
    def step(st):
        s = st.frenet_states[0, 0]
        target = env.track.vmap_frenet_to_cartesian_jax(jnp.array([[s + lookahead, 0.0, 0.0]]))[0]
        x, y, psi = st.cartesian_states[0, 0], st.cartesian_states[0, 1], st.cartesian_states[0, 4]
        alpha = jnp.arctan2(target[1] - y, target[0] - x) - psi
        steer = jnp.arctan2(2 * cp.WHEELBASE * jnp.sin(alpha), lookahead)
        _, st, _, dones, _ = env.step_env(jax.random.PRNGKey(0), st, {"agent_0": jnp.array([steer, speed])})
        return st, dones["__all__"]

    s0, progress = float(st.frenet_states[0, 0]), 0.0
    for t in range(steps):
        prev = float(st.frenet_states[0, 0])
        st, done = step(st)
        ds = float(st.frenet_states[0, 0]) - prev
        L = float(env.track_length)
        progress += ds - L * np.round(ds / L)  # unwrap the seam
        if bool(done) or progress >= 1.0:
            break
    return bool(st.collisions[0]), progress, t + 1


@pytest.mark.parametrize("seed", ALL_GEN)
def test_generated_track_drivable(prepared, seed):
    """Pure pursuit on the centerline drives a full lap without touching a wall."""
    collided, progress, steps = _pure_pursuit_lap(M.gen_name(seed))
    assert not collided and progress >= 1.0, (collided, progress, steps)


def test_multimap_env_shapes(prepared):
    env = M.MultiMapEnv(M.TRAIN_MAPS, num_envs=32, randomize=True)
    assert env.envs_per_map == 2
    obs, states = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert obs.shape == (32, ob.OBS_DIM)
    step = jax.jit(env.step)
    a = jnp.tile(jnp.array([-0.5, 0.0]), (32, 1))
    obs, states, r, d, info = step(jax.random.PRNGKey(1), states, a)
    assert obs.shape == (32, ob.OBS_DIM) and r.shape == (32,) and d.shape == (32,)
    np.testing.assert_array_equal(np.asarray(info["map_id"]), np.repeat(np.arange(16), 2))
    with pytest.raises(ValueError):
        M.MultiMapEnv(M.TRAIN_MAPS, num_envs=30)
