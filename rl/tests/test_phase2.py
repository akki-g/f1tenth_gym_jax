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


# ---------------------------------------------------------------- race wrapper
from rl import maps as M  # noqa: E402


def _race(name, randomize=False, **kw):
    from rl.race import RaceWrapper

    M.use_map_dir(M.DEFAULT_MAP_DIR)
    return RaceWrapper(F110Env(2, make_param(name, randomize=randomize, scan_other_agents=True)),
                       randomize=randomize, frame_stack=1, **kw)


def test_starts_inside_the_track_on_every_training_map():
    """Side-by-side starts share s, each car a quarter-width off the centerline;
    staggered starts put the opponent 1-5 m ahead; nobody starts in a wall or the other car."""
    from rl.race import OPP_SOLO

    for name in M.TRAIN_MAPS:
        w = _race(name)
        _, rs = jax.vmap(w.reset)(jax.random.split(jax.random.PRNGKey(0), 64))
        es = rs.ego.env_state
        present = np.asarray(rs.opp_type) != OPP_SOLO
        fr = np.asarray(es.frenet_states)
        gap = np.asarray(rs.start_gap)
        side = present & (np.abs(gap) < 1e-6)
        stag = present & ~side
        assert side.any() and stag.any(), name
        q = np.asarray(jax.vmap(w.local_half_width)(jnp.asarray(fr[:, 0, 0]))) / 2
        np.testing.assert_allclose(np.abs(fr[side, 0, 1]), q[side], atol=1e-5)
        np.testing.assert_allclose(fr[side, 0, 1], -fr[side, 1, 1], atol=1e-5)  # opposite sides
        assert np.all((gap[stag] >= 1.0 - 1e-4) & (gap[stag] <= 5.0 + 1e-4)), name
        checked = jax.vmap(w.env._collisions)(es.replace(collisions=jnp.zeros_like(es.collisions)))
        assert not np.asarray(checked.collisions)[present].any(), name
        assert not np.asarray(jax.vmap(w.wall_buffer_hit)(es, rs.ego.margin))[present].any(), name


def test_privileged_inputs_never_reach_the_actor():
    from rl.networks import CnnGru
    from rl.race import PRIV_DIM

    model = CnnGru(priv_dim=PRIV_DIM)
    obs = jax.random.uniform(jax.random.PRNGKey(0), (8, ob.OBS_DIM + PRIV_DIM))
    params = model.init(jax.random.PRNGKey(1), jnp.zeros((1, 128)), obs[:1])
    h0 = jax.random.normal(jax.random.PRNGKey(2), (8, 128))
    n = ob.NUM_BEAMS
    obs2 = obs.at[:, n : n + PRIV_DIM].set(jax.random.normal(jax.random.PRNGKey(3), (8, PRIV_DIM)) * 5)
    h_a, mean_a, _, v_a = model.apply(params, h0, obs)
    h_b, mean_b, _, v_b = model.apply(params, h0, obs2)
    np.testing.assert_array_equal(np.asarray(h_a), np.asarray(h_b))
    np.testing.assert_array_equal(np.asarray(mean_a), np.asarray(mean_b))
    assert np.abs(np.asarray(v_a - v_b)).max() > 1e-4  # the critic does read them


def test_no_recompile_when_opponent_type_changes(test_map_dir):
    w = _race("Duct1002", randomize=True)
    step = jax.jit(jax.vmap(w.step))
    obs, rs = jax.vmap(w.reset)(jax.random.split(jax.random.PRNGKey(0), 16))
    rs = jax.tree.map(lambda x: x.astype(x.dtype), rs)  # strong types, as the trainer's init does
    types = set()
    for t in range(120):  # episodes end and reset with new opponent types inside the jitted step
        obs, rs, r, d, info = step(jax.random.split(jax.random.PRNGKey(t), 16), rs, jnp.zeros((16, 2)))
        types |= set(np.asarray(rs.opp_type).tolist())
    assert step._cache_size() == 1
    assert len(types) >= 3, types


def test_follow_the_gap_port_steers_into_the_gap():
    from rl.race import follow_the_gap

    n = ob.NUM_BEAMS
    ang = np.linspace(-135, 135, n)
    ranges = np.full(n, 1.0)
    ranges[(ang > 20) & (ang < 70)] = 8.0  # the only opening, on the left
    steer, speed = np.asarray(follow_the_gap(jnp.asarray(ranges), 1.0))
    assert steer > 0.05 and speed in (1.5, 2.5, 4.0)
    steer_r, _ = np.asarray(follow_the_gap(jnp.asarray(ranges[::-1].copy()), 1.0))
    assert abs(steer_r + steer) < 1e-6  # mirror image
    # speed bands scale with the opponent's speed factor (1.5 / 2.5 / 4.0 m/s by steering)
    _, v = np.asarray(follow_the_gap(jnp.asarray(ranges), 0.8))
    assert min(abs(v - 0.8 * b) for b in (1.5, 2.5, 4.0)) < 1e-5


def test_parked_solo_opponent_is_invisible_and_harmless(test_map_dir):
    from rl.race import OPP_SOLO

    w = _race("OpenBox", opponents=False)  # every episode solo
    obs, rs = w.reset(jax.random.PRNGKey(0))
    assert int(rs.opp_type) == OPP_SOLO
    single = F110Env(1, make_param("OpenBox", randomize=False))
    st1 = rs.ego.env_state.replace(cartesian_states=rs.ego.env_state.cartesian_states[:1])
    np.testing.assert_allclose(np.asarray(rs.ego.env_state.scans[0]),
                               np.asarray(single._scan(st1, jax.random.PRNGKey(5)).scans[0]), atol=1e-6)
    for t in range(30):
        obs, rs, r, d, info = w.step(jax.random.PRNGKey(t), rs, jnp.array([-0.5, 0.0]))
        assert not bool(info["car_contact"])
    assert np.allclose(np.asarray(rs.ego.env_state.cartesian_states[1, :2]), [1.0e4, 1.0e4])


# ---------------------------------------------------------------- race training
import pytest  # noqa: E402
from flax.traverse_util import flatten_dict  # noqa: E402


@pytest.fixture(scope="module")
def race_trainer():
    from rl.train_ppo import Config, make_trainer, race_model

    M.use_map_dir(M.DEFAULT_MAP_DIR)
    cfg = Config(arch="cnn_gru", frame_stack=1, race=True, num_envs=4, num_steps=32, seq_len=16,
                 num_minibatches=2, total_timesteps=4 * 32 * 6, critic_warmup=2, snapshot_pool=3,
                 train_maps=("Duct1002", "Duct1003"), heldout_maps=("Duct1077",))
    env = M.MultiMapEnv(cfg.train_maps, cfg.num_envs, wrapper_kwargs=cfg.wrapper_kwargs(), num_beams=cfg.num_beams,
                        race=dict(opponents=True, gru_dim=cfg.gru_dim, snapshot_model=race_model(cfg)))
    init, update, model, internals = make_trainer(cfg, env)
    return cfg, init, update, internals


def _split(params):
    flat = flatten_dict(params, sep="/")
    critic = {k: np.asarray(v) for k, v in flat.items() if "critic_" in k}
    rest = {k: np.asarray(v) for k, v in flat.items() if "critic_" not in k}
    return critic, rest


def test_race_update_compiles_once_with_warmup(race_trainer):
    cfg, init, update, internals = race_trainer
    r = jax.jit(init)(jax.random.PRNGKey(0))
    assert jax.tree.leaves(r.opp_pool["params"])[0].shape[0] == cfg.snapshot_pool + 1
    c0, a0 = _split(r.actor.params)
    r, m, ep = update(r)  # update 0: warmup, only the critic head moves
    c1, a1 = _split(r.actor.params)
    assert all(np.array_equal(a0[k], a1[k]) for k in a0), "actor/trunk changed during critic warmup"
    assert any(not np.array_equal(c0[k], c1[k]) for k in c0), "critic head did not train"
    r, m, ep = update(r)
    r, m, ep = update(r)  # update 2: warmup over
    _, a3 = _split(r.actor.params)
    assert any(not np.array_equal(a1[k], a3[k]) for k in a1), "actor never trained after warmup"
    assert update._cache_size() == 1
    m = jax.device_get(m)
    assert {"race/win_rate", "race/car_contact_rate", "race/overtakes"} <= set(m)
    assert all(np.isfinite(float(v)) for v in m.values())


def test_init_from_phase1c_checkpoint(race_trainer, tmp_path):
    from rl.train_ppo import Config, add_snapshot, load_init_from, make_trainer, save_checkpoint

    cfg, init, update, internals = race_trainer
    c1 = Config(arch="cnn_gru", frame_stack=1, num_envs=4, num_steps=32, seq_len=16, num_minibatches=2,
                total_timesteps=4 * 32 * 2, train_maps=cfg.train_maps, heldout_maps=cfg.heldout_maps)
    env1 = M.MultiMapEnv(c1.train_maps, c1.num_envs, wrapper_kwargs=c1.wrapper_kwargs(), num_beams=c1.num_beams)
    init1, update1, *_ = make_trainer(c1, env1)
    r1, *_ = update1(jax.jit(init1)(jax.random.PRNGKey(3)))
    path = tmp_path / "gru1c.safetensors"
    save_checkpoint(path, r1, c1)

    r = jax.jit(init)(jax.random.PRNGKey(0))
    r, copied = load_init_from(path, r, internals.init_pool)
    src = flatten_dict(r1.actor.params, sep="/")
    dst = flatten_dict(r.actor.params, sep="/")
    assert copied and all(np.array_equal(np.asarray(src[k]), np.asarray(dst[k])) for k in copied)
    assert all("critic_" not in k for k in copied)
    assert dst["params/critic_hidden/kernel"].shape[0] == 128 + 17  # new head: [h, privileged]
    np.testing.assert_array_equal(np.asarray(r.normalizer.mean), np.asarray(r1.normalizer.mean))
    pool0 = jax.tree.map(lambda x: x[0], r.opp_pool["params"])
    assert all(np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(jax.tree.leaves(pool0), jax.tree.leaves(r.actor.params)))
    pool = add_snapshot(r.opp_pool, 2, jax.tree.map(jnp.zeros_like, r.actor.params), r.normalizer)
    assert float(jnp.abs(jax.tree.leaves(jax.tree.map(lambda x: x[2], pool["params"]))[0]).max()) == 0.0
    assert float(jnp.abs(jax.tree.leaves(jax.tree.map(lambda x: x[1], pool["params"]))[0]).max()) > 0.0
    r2, *_ = update(r)  # the race update runs from the loaded state
    assert int(r2.update) == 1


def test_race_evaluator_runs_every_setting(race_trainer):
    from rl.race_eval import DEFAULT_SETTINGS, make_race_evaluator, snapshot_pool

    cfg, init, update, internals = race_trainer
    r = jax.jit(init)(jax.random.PRNGKey(0))
    evaluate = make_race_evaluator(["Duct1077"], episodes=2, max_steps=30, model_cfg=cfg.model_cfg())
    rep = evaluate(r.actor.params, r.normalizer, snapshot_pool(r.actor.params, r.normalizer))
    assert set(rep["summary"]) == {name for name, _, _ in DEFAULT_SETTINGS}
    starts = {(c["opponent"], c["start"]) for c in rep["cases"]}
    assert ("follow_gap_0.9", "side_by_side") in starts and ("follow_gap_0.9", "staggered") in starts
    assert all(0.0 <= c["win_rate"] <= 1.0 for c in rep["cases"])


# ---------------------------------------------------------------------------
# Racing line and slow pursuit opponents (lab tracks)
# ---------------------------------------------------------------------------
def _line_lap(name, speed_scale, v_cap, steps=4000):
    """Drive agent 0 with the race pursuit opponent on the map's racing line; returns
    (collided, laps, mean speed)."""
    from f1tenth_gym_jax.envs import F110Env
    from rl import maps as M
    from rl import race as RC
    from rl.domain_rand import make_param

    M.use_map_dir(M.DEFAULT_MAP_DIR)
    M.prepare_maps(M.DEFAULT_MAP_DIR, (name,)) if name.startswith("Lab") else None
    env = F110Env(1, make_param(name, randomize=False, produce_scans=False, max_steps=steps + 10))
    line = tuple(jnp.asarray(a) for a in RC.racing_line(name, float(env.track_length)))
    _, st = env.reset(jax.random.PRNGKey(0))

    @jax.jit
    def step(st):
        cmd = RC.pure_pursuit(env, st.cartesian_states[0], st.frenet_states[0, 0], speed_scale, line=line, v_cap=v_cap)
        _, st, rew, dones, _ = env.step_env(jax.random.PRNGKey(0), st, {"agent_0": cmd})
        return st, rew["agent_0"], dones["__all__"]

    progress, speeds = 0.0, []
    for t in range(steps):
        st, r, done = step(st)
        progress += float(r) / float(env.track_length)
        speeds.append(float(st.cartesian_states[0, 3]))
        if bool(done) or progress >= 1.0:
            break
    return bool(st.collisions[0]), progress, float(np.mean(speeds[25:]))


def test_racing_line_inside_corridor_and_shorter():
    from rl import maps as M
    from rl import race as RC

    M.use_map_dir(M.DEFAULT_MAP_DIR)
    M.prepare_maps(M.DEFAULT_MAP_DIR, M.LAB_TRAIN_MAPS[:3])
    for name in M.LAB_TRAIN_MAPS[:3] + ("Duct1002",):
        d = M.DEFAULT_MAP_DIR / name
        cl = np.loadtxt(d / f"{name}_centerline.csv", delimiter=",", comments="#")
        s, ey = RC.racing_line(name, 10.0)
        bound = np.minimum(cl[:, 2], cl[:, 3]) - RC.LINE_CLEARANCE
        assert np.all(np.abs(ey) <= np.maximum(bound, 0.0) + 1e-5), name
        t = np.roll(cl[:, :2], -1, 0) - np.roll(cl[:, :2], 1, 0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        line = cl[:, :2] + np.c_[-t[:, 1], t[:, 0]] * ey[:, None]
        length = lambda p: np.sum(np.hypot(*np.diff(np.vstack([p, p[:1]]), axis=0).T))  # noqa: E731
        assert length(line) < length(cl[:, :2]), name
        assert np.abs(ey).max() > 0.05, name  # it does leave the centerline


@pytest.mark.parametrize("name", ["Lab2004", "Lab2021", "Lab2506", "Duct1002"])
def test_slow_and_normal_pursuit_lap_on_racing_line(name):
    collided, laps, v = _line_lap(name, 1.0, 1.0)  # slow car: 1 m/s cap
    assert not collided and laps >= 1.0 and 0.8 <= v <= 1.05, (name, collided, laps, v)
    collided, laps, v = _line_lap(name, 1.0, 100.0)  # curvature-limited speed
    assert not collided and laps >= 1.0 and v > 1.2, (name, collided, laps, v)
