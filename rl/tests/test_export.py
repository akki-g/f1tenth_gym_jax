"""Export (spec section 7.1): NumPy forward pass matches the JAX actor's
tanh(mean) within 1e-5 on 1000 observations; the car's copy is verbatim."""

import importlib.util
import json
import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl import obs as ob

HERE = pathlib.Path(__file__).resolve().parent
ROBORACER = pathlib.Path(os.environ.get("ROBORACER_ROS2_DIR", HERE.parents[2] / "roboracer_ros2"))
CAR_POLICY = ROBORACER / "src" / "rl_policy" / "rl_policy" / "numpy_policy.py"


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """A checkpoint with non-trivial weights and normalizer (one PPO update)."""
    from rl import maps as M
    from rl.train_ppo import Config, make_trainer, save_checkpoint

    M.use_map_dir(M.DEFAULT_MAP_DIR)
    cfg = Config(num_envs=2, num_steps=16, total_timesteps=64, num_minibatches=2,
                 train_maps=("Duct1002", "Duct1003"), heldout_maps=("Duct1077",))
    env = M.MultiMapEnv(cfg.train_maps, cfg.num_envs, wrapper_kwargs=cfg.wrapper_kwargs(), num_beams=cfg.num_beams)
    init, update, *_ = make_trainer(cfg, env)
    r = jax.jit(init)(jax.random.PRNGKey(0))
    r, *_ = update(r)
    # make the output layer non-trivial (the 0.01-scale init keeps it near 0)
    p = r.actor.params
    p["params"]["mlp"]["Dense_2"]["kernel"] = jax.random.normal(jax.random.PRNGKey(5), p["params"]["mlp"]["Dense_2"]["kernel"].shape)
    r = r._replace(actor=r.actor.replace(params=p))
    path = tmp_path_factory.mktemp("ckpt") / "ckpt.safetensors"
    save_checkpoint(path, r, cfg)
    return path, r


def _load_car_policy_module():
    if not CAR_POLICY.exists():
        pytest.fail(f"car copy of numpy_policy.py not found at {CAR_POLICY} (set ROBORACER_ROS2_DIR)")
    spec = importlib.util.spec_from_file_location("car_numpy_policy", CAR_POLICY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_numpy_matches_jax_actor(checkpoint, tmp_path):
    from rl.export_policy import export
    from rl.networks import Actor, deterministic_action
    from rl.numpy_policy import NumpyPolicy

    path, runner = checkpoint
    out = tmp_path / "policy.npz"
    config = export(path, out)
    assert config["beam_indices"] == list(range(0, 1081, 4)) and config["arch"] == "mlp"
    assert config["v_cmd_max"] == ob.V_CMD_MAX and config["steer_max"] == ob.S_MAX
    assert config["policy_rate_hz"] == 25.0 and config["frame_stack"] == 3
    assert len(config["checkpoint_sha256"]) == 64 and config["train_config_hash"]

    rng = np.random.default_rng(0)
    obs = np.concatenate([
        rng.uniform(0, 1, size=(1000, config["obs_dim"] - 3)),  # scans
        rng.uniform(0, 1, size=(1000, 1)),            # speed / v_cmd_max
        rng.uniform(-1, 1, size=(1000, 2)),           # previous action
    ], axis=1).astype(np.float32)
    mean_j, _ = Actor().apply(runner.actor.params, runner.normalizer(jnp.asarray(obs)))
    a_jax = np.asarray(deterministic_action(mean_j))
    assert np.abs(a_jax).max() > 0.3  # weights are not trivially ~0
    # the same JAX actor in float64: the NumPy forward pass is float64, and at
    # 816 inputs float32 rounding alone is ~1e-5, so the 1e-5 gate is checked here
    from jax.experimental import enable_x64

    with enable_x64():
        p64 = jax.tree.map(lambda x: jnp.asarray(x, jnp.float64), runner.actor.params)
        n = runner.normalizer
        n64 = type(n)(*(jnp.asarray(v, jnp.float64) for v in (n.mean, n.var, n.count)))
        mean64, _ = Actor().apply(p64, n64(jnp.asarray(obs, jnp.float64)))
        a_jax64 = np.asarray(deterministic_action(mean64))

    for policy in (NumpyPolicy(out), _load_car_policy_module().NumpyPolicy(out)):
        a_np = policy(obs)
        assert a_np.shape == (1000, 2)
        assert np.max(np.abs(a_np - a_jax64)) < 1e-5
        assert np.max(np.abs(a_np - a_jax)) < 3e-5  # float32 training actor: rounding only
        single = policy(obs[0])
        assert single.shape == (2,) and np.max(np.abs(single - a_jax64[0])) < 1e-5


def test_car_numpy_policy_is_verbatim():
    _load_car_policy_module()
    assert CAR_POLICY.read_bytes() == (HERE.parent / "numpy_policy.py").read_bytes()


def test_npz_has_no_pickles(checkpoint, tmp_path):
    from rl.export_policy import export

    out = tmp_path / "p.npz"
    export(checkpoint[0], out)
    with np.load(out, allow_pickle=False) as d:
        assert set(d.files) >= {"W0", "b0", "W1", "b1", "W2", "b2", "log_std", "obs_mean", "obs_std", "config"}
        json.loads(str(d["config"]))
        assert d["obs_mean"].shape == (816,) and d["W0"].shape == (816, 256)  # 271 beams x stack 3 + 3


def test_deeper_mlp_exports_and_matches(tmp_path):
    """A 3-hidden-layer, 512-wide MLP (sweep variant) exports and runs in both NumPy copies."""
    from jax.experimental import enable_x64

    from rl import maps as M
    from rl.export_policy import export
    from rl.networks import Actor, deterministic_action
    from rl.numpy_policy import NumpyPolicy
    from rl.train_ppo import Config, make_trainer, save_checkpoint

    M.use_map_dir(M.DEFAULT_MAP_DIR)
    cfg = Config(num_envs=2, num_steps=16, total_timesteps=64, num_minibatches=2, mlp_layers=3, hidden_dim=512,
                 optimizer="adamw", weight_decay=1e-4, train_maps=("Duct1002", "Duct1003"), heldout_maps=("Duct1077",))
    env = M.MultiMapEnv(cfg.train_maps, cfg.num_envs, wrapper_kwargs=cfg.wrapper_kwargs(), num_beams=cfg.num_beams)
    init, update, *_ = make_trainer(cfg, env)
    r, *_ = update(jax.jit(init)(jax.random.PRNGKey(0)))
    p = r.actor.params
    assert set(p["params"]["mlp"]) == {"Dense_0", "Dense_1", "Dense_2", "Dense_3"}
    p["params"]["mlp"]["Dense_3"]["kernel"] = jax.random.normal(jax.random.PRNGKey(5), p["params"]["mlp"]["Dense_3"]["kernel"].shape)
    r = r._replace(actor=r.actor.replace(params=p))
    ckpt = tmp_path / "deep.safetensors"
    save_checkpoint(ckpt, r, cfg)
    out = tmp_path / "deep.npz"
    config = export(ckpt, out)
    obs = np.random.default_rng(1).uniform(0, 1, size=(200, config["obs_dim"])).astype(np.float32)
    with enable_x64():
        p64 = jax.tree.map(lambda x: jnp.asarray(x, jnp.float64), r.actor.params)
        n = r.normalizer
        n64 = type(n)(*(jnp.asarray(v, jnp.float64) for v in (n.mean, n.var, n.count)))
        ref = np.asarray(deterministic_action(Actor(512, layers=3).apply(p64, n64(jnp.asarray(obs, jnp.float64)))[0]))
    for policy in (NumpyPolicy(out), _load_car_policy_module().NumpyPolicy(out)):
        assert len(policy.layers) == 4
        assert np.max(np.abs(policy(obs) - ref)) < 1e-5
