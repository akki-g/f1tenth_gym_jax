"""Record 100 car-format observation inputs from the sim (run once).

A 1081-beam, 270 deg scan env (the Hokuyo's beam layout) drives a slow
weave on Spielberg. Each record: the full 1081-beam scan (with ~0.5% of
beams replaced by +inf / NaN like the real driver's no-return values),
the speed, and the previous action. Saved to rl/tests/data/obs_inputs.npz.

    uv run python -m rl.tests.make_obs_fixture
"""

import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym_jax.envs import F110Env
from rl import obs as ob
from rl.domain_rand import make_param

OUT = pathlib.Path(__file__).parent / "data" / "obs_inputs.npz"


def main():
    env = F110Env(1, make_param("Spielberg", randomize=False, num_beams=ob.CAR_SCAN_BEAMS, scan_noise_std=0.01))
    _, st = env.reset(jax.random.PRNGKey(42))
    rng = np.random.default_rng(42)
    step = jax.jit(env.step_env)
    scans, speeds, prev_actions = [], [], []
    prev = np.array(ob.RESET_ACTION, dtype=np.float32)
    for t in range(100):
        a = np.array([-0.7 + 0.1 * rng.standard_normal(), 0.3 * np.sin(t / 7.0)], dtype=np.float32)
        a = np.clip(a, -1, 1)
        speed, steer = ob.action_to_command(a)
        _, st, _, done, _ = step(jax.random.PRNGKey(t), st, {"agent_0": jnp.array([steer, speed])})
        scan = np.asarray(st.scans[0], dtype=np.float32).copy()
        bad = rng.random(scan.shape) < 0.005
        scan[bad] = np.where(rng.random(bad.sum()) < 0.7, np.inf, np.nan)
        scans.append(scan)
        speeds.append(float(st.cartesian_states[0, 3]))
        prev_actions.append(prev)
        prev = a
    np.savez_compressed(OUT, scans=np.array(scans), speeds=np.array(speeds, np.float32), prev_actions=np.array(prev_actions))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
