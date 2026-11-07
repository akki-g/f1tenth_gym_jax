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
ALL_LAB = M.LAB_TRAIN_SEEDS + M.LAB_HELDOUT_SEEDS + M.LAB_UNSEEN_SEEDS


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


# ---------------------------------------------------------------------------
# Lab-size tracks
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def lab_prepared():
    map_dir = M.use_map_dir(M.DEFAULT_MAP_DIR)
    M.prepare_maps(map_dir, M.LAB_TRAIN_MAPS + M.LAB_HELDOUT_MAPS + M.LAB_UNSEEN_MAPS)
    return map_dir


def test_lab_lists():
    assert len(M.LAB_TRAIN_SEEDS) == 24 and len(M.LAB_HELDOUT_SEEDS) == 6 and len(M.LAB_UNSEEN_SEEDS) == 6
    assert len(set(M.LAB_FT_TRAIN_MAPS)) == 32 and 2048 % len(M.LAB_FT_TRAIN_MAPS) == 0
    assert not set(M.LAB_FT_TRAIN_MAPS) & set(M.LAB_FT_HELDOUT_MAPS)
    assert not set(M.LAB_UNSEEN_MAPS) & set(M.LAB_FT_TRAIN_MAPS + M.LAB_FT_HELDOUT_MAPS)
    assert M.map_lists_config()["lab_train_seeds"] == list(M.LAB_TRAIN_SEEDS)


@pytest.mark.parametrize("seed", ALL_LAB)
def test_lab_track_in_spec(lab_prepared, seed):
    """Reproducible; length, mean width, narrowest/widest point and corner radius in
    range; the written map matches the generator and the car fits everywhere."""
    g, g2 = M.generate_lab_track(seed), M.generate_lab_track(seed)
    assert g is not None and np.array_equal(g["free"], g2["free"])
    assert M.LAB_LENGTH[0] <= g["length"] <= M.LAB_LENGTH[1]
    assert M.LAB_WIDTH[0] <= g["width"] <= M.LAB_WIDTH[1]
    assert M.LAB_MIN_WIDTH - 1e-9 <= g["min_width"] and g["max_width"] <= M.LAB_MAX_WIDTH + 1e-9
    assert g["min_radius"] >= M.LAB_MIN_RADIUS
    name = M.lab_name(seed)
    img = np.asarray(Image.open(lab_prepared / name / f"{name}.png"))[::-1] > 127  # row 0 = bottom
    np.testing.assert_array_equal(img, g["free"])
    clearance = edt(g["free"]) * g["resolution"]
    c = g["centerline"]
    ij = np.round((c[:, :2] - np.asarray(g["origin"])) / g["resolution"] - 0.5).astype(int)
    assert clearance[ij[:, 1], ij[:, 0]].min() >= 0.35  # half the car (0.155 m) with room to spare


def test_lab_tracks_vary():
    gs = [M.generate_lab_track(s) for s in M.LAB_TRAIN_SEEDS]
    turns = [np.sign(np.sum(M._curvature_closed(g["centerline"][:, :2]))) for g in gs]
    assert turns.count(1.0) >= 6 and turns.count(-1.0) >= 6  # both directions
    assert np.median([g["max_width"] - g["min_width"] for g in gs]) > 0.3  # walls not parallel


@pytest.mark.parametrize("seed", ALL_LAB)
def test_lab_track_drivable(lab_prepared, seed):
    """Pure pursuit on the centerline drives a full lap without touching a wall."""
    collided, progress, steps = _pure_pursuit_lap(M.lab_name(seed), speed=1.5, lookahead=0.6)
    assert not collided and progress >= 1.0, (collided, progress, steps)


def test_lab_multimap_env_with_rollover_and_friction(lab_prepared):
    from rl import car_params as cp
    from rl.domain_rand import friction_overrides

    maps = M.LAB_TRAIN_MAPS[:4]
    env = M.MultiMapEnv(maps, num_envs=16, randomize=True, wrapper_kwargs={"rollover": True, "max_laps": 5,
                                                                         "frame_stack": 1, "max_delay": 6},
                        **friction_overrides(cp.LAB_FRICTION, True))
    obs, states = jax.jit(env.reset)(jax.random.PRNGKey(0))
    mu = np.concatenate([np.asarray(s.env_state.dyn_params.mu).ravel() for s in states])
    assert mu.min() >= cp.LAB_FRICTION[0] - 1e-6 and mu.max() <= cp.LAB_FRICTION[1] + 1e-6 and mu.std() > 0.02
    delays = np.concatenate([np.asarray(s.delay_k) for s in states])
    assert delays.max() <= 6
    obs, states, r, d, info = jax.jit(env.step)(jax.random.PRNGKey(1), states, jnp.tile(jnp.array([-0.5, 0.0]), (16, 1)))
    assert "rollover" in info and info["lat_accel"].shape == (16,)


# ---------------------------------------------------------------------------
# Room tracks and the last race's track
# ---------------------------------------------------------------------------
ALL_ROOM = M.ROOM_TRAIN_SEEDS + M.ROOM_HELDOUT_SEEDS + M.ROOM_UNSEEN_SEEDS


@pytest.fixture(scope="module")
def room_prepared():
    map_dir = M.use_map_dir(M.DEFAULT_MAP_DIR)
    M.prepare_maps(map_dir, M.ROOM_TRAIN_MAPS + M.ROOM_HELDOUT_MAPS + M.ROOM_UNSEEN_MAPS)
    return map_dir


def test_room_and_fine_tune_lists():
    assert len(M.LAB_FT_TRAIN_MAPS) == 32 and len(set(M.LAB_FT_TRAIN_MAPS)) == 32
    assert set(M.LAST_RACE_MAPS) <= set(M.LAB_FT_TRAIN_MAPS)
    assert all(n.startswith(("LastRace2", "RaceLike", "Trap")) for n in M.LAB_FT_TRAIN_MAPS + M.LAB_FT_HELDOUT_MAPS)
    assert not set(M.LAB_FT_TRAIN_MAPS) & set(M.LAB_FT_HELDOUT_MAPS + M.LAB_EVAL_UNSEEN_MAPS)


@pytest.mark.parametrize("name", [M.room_name(s) for s in ALL_ROOM])
def test_room_track_in_spec_and_drivable(room_prepared, name):
    g = M.generate_room_track(int(name[4:]))
    assert g is not None and M.ROOM_LENGTH[0] <= g["length"] <= M.ROOM_LENGTH[1]
    assert g["min_width"] >= M.ROOM_MIN_WIDTH and g["min_radius"] >= M.ROOM_MIN_RADIUS
    img = np.asarray(Image.open(room_prepared / name / f"{name}.png"))[::-1] > 127
    np.testing.assert_array_equal(img, g["free"])
    assert g["centerline"][:, 2].min() >= 0.4  # the car (0.155 m half-width) fits everywhere on the centerline
    collided, progress, steps = _pure_pursuit_lap(name, speed=1.5, lookahead=0.6)
    assert not collided and progress >= 1.0, (name, collided, progress, steps)


def test_last_race_both_directions():
    fwd, rev = M.last_race_track(), M.last_race_track(reverse=True)
    turn = lambda g: np.sign(np.sum(M._curvature_closed(g["centerline"][:, :2])))  # noqa: E731
    assert turn(fwd) == -turn(rev) and np.array_equal(fwd["free"], rev["free"])



# ---------------------------------------------------------------------------
# The fine-tune's tracks: the last race (both ways), race-like and speed-trap tracks
# ---------------------------------------------------------------------------
FT_ALL = tuple(dict.fromkeys(M.LAB_FT_TRAIN_MAPS + M.LAB_FT_HELDOUT_MAPS + M.LAB_EVAL_UNSEEN_MAPS))


def _ft_gen(name):
    if name.startswith("LastRace2"):
        return M.last_race_track(reverse=name.endswith("Rev"))
    if name.startswith("RaceLike"):
        return M.generate_race_like_track(int(name[8:]))
    return M.generate_trap_track(int(name[4:]))


@pytest.fixture(scope="module")
def ft_prepared():
    map_dir = M.use_map_dir(M.DEFAULT_MAP_DIR)
    M.prepare_maps(map_dir, FT_ALL)
    return map_dir


@pytest.mark.parametrize("name", FT_ALL)
def test_fine_tune_track_in_spec_and_drivable(ft_prepared, name):
    """Written map matches the generator; corridor >= 0.9 m (trap straights <= 3 m); the car fits on the
    centerline; pure pursuit on the centerline laps it without touching a wall."""
    g = _ft_gen(name)
    assert g is not None
    img = np.asarray(Image.open(ft_prepared / name / f"{name}.png"))[::-1] > 127
    np.testing.assert_array_equal(img, g["free"])
    assert g["min_width"] >= 0.9 - 1e-6
    if name.startswith("Trap"):
        assert 2.0 <= g["trap_width"] <= 3.0 and g["max_width"] <= 3.0 + 1e-6
        # one wall pushes out on the trap: at its widest point the pushed side is the wider one by far
        wl, wr = g["wall_offsets"].T
        i = int(np.argmax(wl + wr))
        pushed, other = (wl[i], wr[i]) if g["trap_side"] > 0 else (wr[i], wl[i])
        assert pushed - other >= 0.4, (pushed, other)
    else:
        assert g["min_clearance"] >= 0.27 and g["min_radius"] >= 0.75
    collided, progress, steps = _pure_pursuit_lap(name, speed=1.5, lookahead=0.6, steps=3000)
    assert not collided and progress >= 1.0, (name, collided, progress, steps)



def test_accel_spread_over_maps(lab_prepared):
    maps = M.LAB_TRAIN_MAPS[:4]
    env = M.MultiMapEnv(maps, num_envs=8, randomize=True, accel_range=(5.5, 6.5),
                        wrapper_kwargs={"frame_stack": 1})
    a = [w.env.params.a_max for w in env.wrappers]
    assert all(5.5 <= x <= 6.5 for x in a) and len(set(a)) == len(a)
