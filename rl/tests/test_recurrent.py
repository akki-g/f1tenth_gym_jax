"""Phase-1c CNN+GRU and recurrent PPO checks (spec "Validation before training")."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from rl import maps as M
from rl import obs as ob
from rl.networks import CONV_LAYERS, Actor, CnnGru, gru_unroll


@pytest.fixture(scope="module")
def gru_trainer():
    """Tiny cnn_gru trainer. Episodes time out every 10 steps, so the 16-step
    chunks contain resets in the middle, not just at their start."""
    from rl.train_ppo import Config, make_trainer

    M.use_map_dir(M.DEFAULT_MAP_DIR)
    cfg = Config(arch="cnn_gru", frame_stack=1, num_envs=4, num_steps=32, seq_len=16, num_minibatches=2,
                 total_timesteps=4 * 32 * 3, train_maps=("Duct1002", "Duct1003"), heldout_maps=("Duct1077",))
    env = M.MultiMapEnv(cfg.train_maps, cfg.num_envs, wrapper_kwargs=cfg.wrapper_kwargs(),
                        num_beams=cfg.num_beams, max_steps=10)
    init, update, model, internals = make_trainer(cfg, env)
    return cfg, init, update, internals


def test_config_rules():
    from rl.train_ppo import Config

    with pytest.raises(ValueError):
        Config(arch="cnn_gru", frame_stack=3).validate()  # the GRU takes one frame
    with pytest.raises(ValueError):
        Config(arch="cnn_gru", frame_stack=1, num_steps=256, seq_len=60).validate()
    Config(arch="cnn_gru", frame_stack=1).validate()
    # spec sizes: 2048 envs x 256 steps / 64 = 8192 sequences, 32 minibatches of 256
    c = Config(arch="cnn_gru", frame_stack=1)
    assert c.num_envs * c.num_steps // c.seq_len // c.num_minibatches == 256


def test_edge_padding():
    """An obstacle at beam 0: the first conv equals a VALID conv over the scan
    padded by hand with repeated edge values (never zeros)."""
    model = CnnGru()
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 128)), jnp.zeros((1, ob.OBS_DIM)))
    scan = jnp.full((ob.NUM_BEAMS,), 0.8).at[0].set(0.05).at[-1].set(0.3)
    got = model.apply(params, scan, method=CnnGru.conv1)
    _, k, st = CONV_LAYERS[0]
    total = (-(-ob.NUM_BEAMS // st) - 1) * st + k - ob.NUM_BEAMS  # 4 for 271 beams, k5 s2
    left, right = total // 2, total - total // 2
    by_hand = jnp.concatenate([jnp.full((left,), scan[0]), scan, jnp.full((right,), scan[-1])])
    want = model.apply(params, by_hand[:, None], method=lambda m, x: m.convs[0](x))
    assert got.shape == want.shape == (136, 16)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=1e-6)
    zeros = jnp.concatenate([jnp.zeros((left,)), scan, jnp.zeros((right,))])
    zero_out = model.apply(params, zeros[:, None], method=lambda m, x: m.convs[0](x))
    assert np.abs(np.asarray(zero_out[0]) - np.asarray(got[0])).max() > 1e-4  # zero padding would differ
    # 271 -> 136 -> 68 -> 34 -> 17 through the four layers
    feats = model.apply(params, jnp.zeros((ob.OBS_DIM,)), method=CnnGru.encode)
    assert feats.shape == (128 + 3,)


def test_resets_zero_h_only_for_finished_envs(gru_trainer):
    cfg, init, update, internals = gru_trainer
    r = jax.jit(init)(jax.random.PRNGKey(0))
    h = jax.random.normal(jax.random.PRNGKey(1), r.h.shape)
    first = jnp.array([True, False, True, False])
    r2, t = jax.jit(internals.env_step)(r._replace(h=h, first=first), None)
    h_in = np.asarray(t.h_in)
    assert np.all(h_in[[0, 2]] == 0.0)
    np.testing.assert_array_equal(h_in[[1, 3]], np.asarray(h)[[1, 3]])
    np.testing.assert_array_equal(np.asarray(r2.first), np.asarray(t.done))  # next step's resets


def test_update0_recompute_matches_rollout(gru_trainer):
    """At update 0 the loss's rerun of CNN+GRU from the stored chunk-start states
    reproduces the rollout's states, values and log-probs; the PPO ratio is 1."""
    from rl.networks import log_prob_from_pre_tanh, policy_distribution

    cfg, init, update, internals = gru_trainer
    r = jax.jit(init)(jax.random.PRNGKey(2))
    rollout = jax.jit(lambda r: jax.lax.scan(internals.env_step, r, None, length=cfg.num_steps))
    _, traj = rollout(r)
    seqs = jax.tree.map(internals.to_sequences, traj._replace(info={}))
    first = np.asarray(seqs.first)
    assert first[:, 1:].any(), "no reset inside a chunk: the test would not exercise it"
    mean, log_std, value, h_in = jax.jit(internals.recompute)(r.actor.params, seqs)
    np.testing.assert_allclose(np.asarray(h_in), np.asarray(seqs.h_in), atol=1e-5)
    np.testing.assert_allclose(np.asarray(value), np.asarray(seqs.value), atol=1e-5)
    logp = log_prob_from_pre_tanh(policy_distribution(mean, log_std), seqs.pre_tanh)
    np.testing.assert_allclose(np.asarray(logp), np.asarray(seqs.log_prob), atol=1e-5)
    np.testing.assert_allclose(np.exp(np.asarray(logp - seqs.log_prob)), 1.0, atol=1e-5)
    # the first minibatch (what the first gradient step sees) as well
    mb0 = jax.tree.map(lambda x: x[0], internals.minibatches(seqs, jax.random.PRNGKey(3)))
    _, _, v0, _ = jax.jit(internals.recompute)(r.actor.params, mb0)
    np.testing.assert_allclose(np.asarray(v0), np.asarray(mb0.value), atol=1e-5)


def test_minibatches_hold_whole_in_order_chunks(gru_trainer):
    cfg, init, update, internals = gru_trainer
    T, N, L = cfg.num_steps, cfg.num_envs, cfg.seq_len
    code = jnp.arange(T)[:, None] * 1000 + jnp.arange(N)[None, :]  # value = 1000 t + env
    mbs = np.asarray(internals.minibatches(internals.to_sequences(code), jax.random.PRNGKey(4)))
    n_seq = N * T // L
    assert mbs.shape == (cfg.num_minibatches, n_seq // cfg.num_minibatches, L)
    seen = set()
    for seq in mbs.reshape(-1, L):
        env, t0 = int(seq[0] % 1000), int(seq[0] // 1000)
        np.testing.assert_array_equal(seq % 1000, env)  # one env
        np.testing.assert_array_equal(seq // 1000, t0 + np.arange(L))  # consecutive, in time order
        assert t0 % L == 0  # aligned 64-step (here 16-step) chunks
        seen.add((env, t0))
    assert len(seen) == n_seq  # every chunk exactly once


def test_gru_update_compiles_once(gru_trainer):
    cfg, init, update, internals = gru_trainer
    r = jax.jit(init)(jax.random.PRNGKey(0))
    for _ in range(3):
        r, metrics, episodes = update(r)
    assert update._cache_size() == 1
    assert int(r.update) == 3 and int(r.actor.step) == 3 * cfg.update_epochs * cfg.num_minibatches
    m = jax.device_get(metrics)
    assert all(np.isfinite(float(v)) for v in m.values())
    assert float(m["gru_h_norm"]) > 0.0 and r.critic is None


def test_gru_checkpoint_round_trip(gru_trainer, tmp_path):
    from rl.train_ppo import load_checkpoint, restore_runner, save_checkpoint

    cfg, init, update, internals = gru_trainer
    r = jax.jit(init)(jax.random.PRNGKey(0))
    r, *_ = update(r)
    path = tmp_path / "gru.safetensors"
    save_checkpoint(path, r, cfg)
    restored = restore_runner(path, jax.jit(init)(jax.random.PRNGKey(9)))
    for a, b in zip(jax.tree.leaves((r.actor, r.normalizer, r.update)),
                    jax.tree.leaves((restored.actor, restored.normalizer, restored.update))):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert load_checkpoint(path)["config"]["arch"] == "cnn_gru"
    r2, *_ = update(restored)
    assert int(r2.update) == 2


# ------------------------------------------------------------------ toy recall
RECALL_T = 31  # signal at t = 0, report it at t = 30


def _recall_batch(key, batch):
    k_s, k_scan = jax.random.split(key)
    s = jnp.where(jax.random.bernoulli(k_s, 0.5, (batch,)), 1.0, -1.0)
    obs = jax.random.uniform(k_scan, (RECALL_T, batch, ob.OBS_DIM))  # noise scans
    obs = obs.at[:, :, ob.NUM_BEAMS:].set(0.0)
    obs = obs.at[0, :, ob.NUM_BEAMS].set(s)  # the signal, in the speed slot, at t = 0 only
    return obs, s


def _train(loss_fn, params, steps, lr=3e-3, done=None):
    """Adam on fresh batches; ``done(params)`` checked every 100 steps stops early.
    Long-range recall sits on a plateau first (about 500 steps here), then drops."""
    tx = optax.adam(lr)
    opt = tx.init(params)

    @jax.jit
    def step(params, opt, key):
        loss, g = jax.value_and_grad(loss_fn)(params, key)
        upd, opt = tx.update(g, opt)
        return optax.apply_updates(params, upd), opt, loss

    key = jax.random.PRNGKey(0)
    for i in range(1, steps + 1):
        key, k = jax.random.split(key)
        params, opt, _ = step(params, opt, k)
        if done is not None and i % 100 == 0 and done(params):
            break
    return params


def test_toy_recall_gru_remembers_mlp3_cannot():
    """A sign shown at t = 0 must be reported at t = 30: the GRU learns it
    (> 95%), the 3-frame MLP stays near chance."""
    batch = 64
    gru = CnnGru()
    p_gru = gru.init(jax.random.PRNGKey(1), jnp.zeros((1, 128)), jnp.zeros((1, ob.OBS_DIM)))

    def gru_out(params, obs):
        first = jnp.zeros(obs.shape[:2], bool).at[0].set(True)  # one episode per sequence
        mean, *_ = gru_unroll(gru, params, jnp.zeros((obs.shape[1], 128)), obs, first)
        return jnp.tanh(mean[-1, :, 0])

    def gru_loss(params, key):
        obs, s = _recall_batch(key, batch)
        return jnp.mean((gru_out(params, obs) - s) ** 2)

    mlp = Actor(256)
    stack3 = lambda obs: jnp.concatenate([obs[-1], obs[-2], obs[-3]], axis=-1)  # noqa: E731  frames 30, 29, 28
    p_mlp = mlp.init(jax.random.PRNGKey(2), jnp.zeros((1, 3 * ob.OBS_DIM)))

    def mlp_loss(params, key):
        obs, s = _recall_batch(key, batch)
        return jnp.mean((jnp.tanh(mlp.apply(params, stack3(obs))[0][:, 0]) - s) ** 2)

    obs, s = _recall_batch(jax.random.PRNGKey(123), 1000)
    gru_acc = jax.jit(lambda p: jnp.mean(jnp.sign(gru_out(p, obs)) == s))
    p_gru = _train(gru_loss, p_gru, 1500, done=lambda p: float(gru_acc(p)) > 0.97)
    p_mlp = _train(mlp_loss, p_mlp, 500)
    acc_gru = float(gru_acc(p_gru))
    acc_mlp = float(jnp.mean(jnp.sign(jnp.tanh(mlp.apply(p_mlp, stack3(obs))[0][:, 0])) == s))
    assert acc_gru > 0.95, acc_gru
    assert 0.35 < acc_mlp < 0.65, acc_mlp


def test_gru_evaluator_runs(gru_trainer):
    """evaluate.py drives a cnn_gru policy (state carried, reset at episode start)."""
    from rl.evaluate import make_evaluator

    cfg, init, update, internals = gru_trainer
    r = jax.jit(init)(jax.random.PRNGKey(0))
    evaluate = make_evaluator(["Duct1077"], episodes_per_map=2, max_steps=40, model_cfg=cfg.model_cfg(),
                              wrapper_kwargs=cfg.wrapper_kwargs())
    rep = evaluate(r.actor.params, r.normalizer)
    assert rep["per_map"]["Duct1077"]["episodes"] == 2
    assert 0.0 <= rep["completion_rate"] <= 1.0
