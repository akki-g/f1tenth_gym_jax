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
    init, update, *_ = make_trainer(cfg, M.MultiMapEnv(cfg.train_maps, cfg.num_envs))
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
    assert config["beam_indices"] == list(range(0, 1081, 10))
    assert config["v_cmd_max"] == ob.V_CMD_MAX and config["steer_max"] == ob.S_MAX
    assert config["policy_rate_hz"] == 25.0 and config["frame_stack"] == 3
    assert len(config["checkpoint_sha256"]) == 64 and config["train_config_hash"]

    rng = np.random.default_rng(0)
    obs = np.concatenate([
        rng.uniform(0, 1, size=(1000, 327)),          # scans
        rng.uniform(0, 1, size=(1000, 1)),            # speed / v_cmd_max
        rng.uniform(-1, 1, size=(1000, 2)),           # previous action
    ], axis=1).astype(np.float32)
    mean_j, _ = Actor().apply(runner.actor.params, runner.normalizer(jnp.asarray(obs)))
    a_jax = np.asarray(deterministic_action(mean_j))
    assert np.abs(a_jax).max() > 0.3  # weights are not trivially ~0

    for policy in (NumpyPolicy(out), _load_car_policy_module().NumpyPolicy(out)):
        a_np = policy(obs)
        assert a_np.shape == (1000, 2)
        assert np.max(np.abs(a_np - a_jax)) < 1e-5
        single = policy(obs[0])
        assert single.shape == (2,) and np.max(np.abs(single - a_jax[0])) < 1e-5


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
        assert d["obs_mean"].shape == (330,) and d["W0"].shape == (330, 256)
