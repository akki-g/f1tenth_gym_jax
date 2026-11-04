"""Policy distribution, normalizer, GAE and checkpoint round trip."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl import maps as M
from rl.networks import (
    NORM_SLICE,
    Actor,
    Normalizer,
    deterministic_action,
    log_prob_from_pre_tanh,
    policy_distribution,
    sample_action,
)


def test_tanh_gaussian_log_prob_matches_distrax():
    key = jax.random.PRNGKey(0)
    mean = jax.random.normal(key, (512, 2)) * 0.5
    log_std = jnp.array([-0.5, -0.5])
    a, u, logp = sample_action(jax.random.PRNGKey(1), mean, log_std)
    assert a.shape == (512, 2) and jnp.all(jnp.abs(a) < 1)
    np.testing.assert_allclose(a, jnp.tanh(u))
    dist = policy_distribution(mean, log_std)
    ok = jnp.all(jnp.abs(u) < 3, axis=-1)  # away from saturation, distrax's own path is exact
    np.testing.assert_allclose(logp[ok], dist.log_prob(a)[ok], rtol=1e-4, atol=1e-4)
    # stable where tanh saturates (distrax's atanh path is not)
    big = jnp.array([[12.0, -15.0]])
    assert jnp.isfinite(log_prob_from_pre_tanh(policy_distribution(big, log_std), big)).all()


def test_actor_init_and_deterministic_action():
    actor = Actor()
    params = actor.init(jax.random.PRNGKey(0), jnp.zeros((1, 330)))
    np.testing.assert_allclose(params["params"]["log_std"], [-0.5, -0.5])
    mean, log_std = actor.apply(params, jnp.ones((3, 330)))
    assert mean.shape == (3, 2) and jnp.all(jnp.abs(mean) < 0.5)  # 0.01-scale output init
    np.testing.assert_allclose(deterministic_action(mean), jnp.tanh(mean))


def test_normalizer_tracks_mean_std_of_non_scan_inputs():
    rng = np.random.default_rng(0)
    batches = [rng.normal([0.3, -0.2, 0.1], [0.1, 0.5, 0.2], size=(1000, 3)).astype(np.float32) for _ in range(5)]
    n = Normalizer.create()
    for b in batches:
        n = n.update(jnp.asarray(b))
    allx = np.concatenate(batches)
    np.testing.assert_allclose(n.mean, allx.mean(0), atol=1e-4)
    np.testing.assert_allclose(n.var, allx.var(0), rtol=1e-3)
    obs = jnp.asarray(rng.uniform(0, 1, size=(4, 330)).astype(np.float32))
    normed = n(obs)
    np.testing.assert_array_equal(normed[:, : NORM_SLICE.start], obs[:, : NORM_SLICE.start])  # scans untouched
    mean, std = n.full_mean_std()
    np.testing.assert_allclose(normed, jnp.clip((obs - mean) / std, -10, 10), atol=1e-6)


def test_gae_cuts_bootstrap_on_done():
    from rl.train_ppo import compute_gae

    value = jnp.array([[1.0], [1.0], [1.0]])
    reward = jnp.array([[1.0], [1.0], [1.0]])
    done = jnp.array([[False], [True], [False]])
    adv, targets = compute_gae(reward, value, done, jnp.array([10.0]), gamma=0.5, lam=1.0)
    # t=2: 1 + 0.5*10 - 1 = 5 ; t=1 (done): 1 - 1 = 0 ; t=0: 1 + 0.5*1 - 1 + 0.5*0 = 0.5
    np.testing.assert_allclose(np.asarray(adv).ravel(), [0.5, 0.0, 5.0])
    np.testing.assert_allclose(np.asarray(targets).ravel(), [1.5, 1.0, 6.0])


@pytest.fixture(scope="module")
def tiny_trainer():
    from rl.train_ppo import Config, make_trainer

    M.use_map_dir(M.DEFAULT_MAP_DIR)
    cfg = Config(num_envs=2, num_steps=8, total_timesteps=64, num_minibatches=2,
                 train_maps=("Duct1002", "Duct1003"), heldout_maps=("Duct1077",))
    env = M.MultiMapEnv(cfg.train_maps, cfg.num_envs)
    init, update, *_ = make_trainer(cfg, env)
    return cfg, init, update


def test_update_runs_and_compiles_once(tiny_trainer):
    cfg, init, update = tiny_trainer
    r = jax.jit(init)(jax.random.PRNGKey(0))
    for _ in range(3):
        r, metrics, episodes = update(r)
    assert update._cache_size() == 1
    assert int(r.update) == 3 and int(r.actor.step) == 3 * cfg.update_epochs * cfg.num_minibatches
    assert all(np.isfinite(float(v)) for v in jax.device_get(metrics).values())


def test_checkpoint_round_trip(tiny_trainer, tmp_path):
    from rl.train_ppo import load_checkpoint, restore_runner, save_checkpoint

    cfg, init, update = tiny_trainer
    r = jax.jit(init)(jax.random.PRNGKey(0))
    r, *_ = update(r)
    path = tmp_path / "ckpt.safetensors"
    save_checkpoint(path, r, cfg, {"best_score": 0.25})

    fresh = jax.jit(init)(jax.random.PRNGKey(99))
    restored = restore_runner(path, fresh)
    for a, b in zip(jax.tree.leaves((r.actor, r.critic, r.normalizer, r.update)),
                    jax.tree.leaves((restored.actor, restored.critic, restored.normalizer, restored.update))):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    r2, *_ = update(restored)  # optimizer state is usable after restore
    assert int(r2.update) == 2

    ck = load_checkpoint(path)
    assert ck["update"] == 1 and ck["extra"]["best_score"] == 0.25
    assert ck["config"]["num_envs"] == cfg.num_envs
    for a, b in zip(jax.tree.leaves(ck["actor_params"]), jax.tree.leaves(r.actor.params)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_config_hash_ignores_bookkeeping():
    from rl.train_ppo import Config

    a = Config()
    assert a.hash() == Config(runs_root="elsewhere", wandb_mode="disabled", time_limit=10).hash()
    assert a.hash() != Config(num_envs=1024).hash()
    assert Config().num_updates == 200_000_000 // (2048 * 256)
