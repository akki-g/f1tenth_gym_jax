"""Parity with the classic f1tenth_gym that the ROS sim runs (spec section 6).

Same map, same pose, nominal parameters, no noise / delay / offset, the same
500-step action sequence (straights, a slalom, a hard turn) at 25 Hz.
Pass: position error < 0.05 m and heading error < 0.05 rad over the first
200 steps.

The JAX side is driven through Phase1Wrapper with policy actions (a_v, a_s);
the classic side gets [steering_angle, speed] exactly as f1tenth_gym_ros's
bridge passes an AckermannDriveStamped, so action ordering and scaling are
part of the test. The classic gym is loaded straight from its source tree
(pinned to the commit the roboracer Docker image clones) without its gym /
pyglet package __init__; that is the same Simulator.step the bridge calls.
"""

import importlib
import os
import pathlib
import subprocess
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl import car_params as cp
from rl import obs as ob

CLASSIC_REPO = "https://github.com/f1tenth/f1tenth_gym.git"
CLASSIC_COMMIT = "4fdb9c7e6f"  # f1tenth_gym main, 2025-06-09 (cloned by roboracer_ros2's Dockerfile)
CLASSIC_DIR = pathlib.Path(
    os.environ.get("F1TENTH_GYM_CLASSIC_DIR", pathlib.Path(__file__).resolve().parents[1] / "third_party" / "f1tenth_gym")
)
N_STEPS, GATE_STEPS = 500, 200
POS_TOL, HEAD_TOL = 0.05, 0.05


def _classic_base_classes():
    pytest.importorskip("numba", reason="install the 'parity' dependency group: uv sync --group parity")
    if not (CLASSIC_DIR / "gym" / "f110_gym").exists():
        CLASSIC_DIR.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(["git", "clone", "-q", CLASSIC_REPO, str(CLASSIC_DIR)], check=True)
            subprocess.run(["git", "-C", str(CLASSIC_DIR), "checkout", "-q", CLASSIC_COMMIT], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            pytest.fail(f"could not fetch the classic f1tenth_gym into {CLASSIC_DIR}: {exc}")
    envs_dir = CLASSIC_DIR / "gym" / "f110_gym" / "envs"
    for name, path in (("f110_gym", envs_dir.parent), ("f110_gym.envs", envs_dir)):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(path)]
            sys.modules[name] = mod
    return importlib.import_module("f110_gym.envs.base_classes")


def action_sequence():
    """Policy actions (a_v, a_s), 25 Hz: ramp + straight, slalom, hard turn, fast slalom."""
    t = np.arange(N_STEPS) * cp.POLICY_DT
    speed = np.zeros(N_STEPS)
    steer = np.zeros(N_STEPS)
    speed[:100] = np.minimum(3.0, 0.5 + t[:100])
    speed[100:250] = 3.0
    steer[100:250] = 0.3 * np.sin(2 * np.pi * (t[100:250] - t[100]) / 2.0)
    speed[250:350] = 2.5
    steer[250:350] = cp.S_MAX
    speed[350:] = 4.0
    steer[350:] = 0.15 * np.sin(2 * np.pi * (t[350:] - t[350]) / 1.5)
    return np.asarray(ob.command_to_action(speed, steer))


def test_parity_with_classic_gym(test_map_dir):
    from f1tenth_gym_jax.envs import F110Env

    from rl.domain_rand import make_param
    from rl.wrappers import Phase1Wrapper

    bc = _classic_base_classes()
    actions = action_sequence()

    # JAX: phase-1 env, nominal, through the wrapper
    p = make_param("OpenBox", randomize=False, max_steps=10_000, max_num_laps=100)
    w = Phase1Wrapper(F110Env(1, p), randomize=False)
    _, ws = w.reset_at(jax.random.PRNGKey(0), 0.0)
    start = np.asarray(ws.env_state.cartesian_states[0, [0, 1, 4]], dtype=np.float64)
    step = jax.jit(w.step)
    jax_traj = []
    for a in actions:
        _, ws, _, done, _ = step(jax.random.PRNGKey(1), ws, jnp.asarray(a))
        assert not bool(done)
        s = np.asarray(ws.env_state.cartesian_states[0])
        jax_traj.append([s[0], s[1], s[4]])
    jax_traj = np.array(jax_traj)

    # classic: same params, map and pose; the bridge's [steer, speed] at 100 Hz
    params = {**cp.NOMINAL, "s_min": cp.S_MIN, "s_max": cp.S_MAX, "sv_min": cp.SV_MIN, "sv_max": cp.SV_MAX,
              "v_switch": cp.V_SWITCH, "a_max": cp.A_MAX, "v_min": cp.V_MIN, "v_max": cp.V_MAX,
              "width": cp.WIDTH, "length": cp.LENGTH}
    sim = bc.Simulator(params, 1, 12345, time_step=cp.TIMESTEP, integrator=bc.Integrator.RK4)
    sim.set_map(str(test_map_dir / "OpenBox" / "OpenBox.yaml"), ".png")
    sim.reset(start[None])
    classic_traj = []
    for a in actions:
        speed, steer = ob.action_to_command(a)
        for _ in range(cp.TIMESTEP_RATIO):
            sim.step(np.array([[float(steer), float(speed)]]))
        s = sim.agents[0].state
        assert not sim.agents[0].in_collision
        classic_traj.append([s[0], s[1], s[4]])
    classic_traj = np.array(classic_traj)

    pos_err = np.hypot(*(jax_traj[:, :2] - classic_traj[:, :2]).T)
    head_err = np.abs(np.angle(np.exp(1j * (jax_traj[:, 2] - classic_traj[:, 2]))))
    print(f"\nparity: first {GATE_STEPS} steps max pos {pos_err[:GATE_STEPS].max():.2e} m, "
          f"head {head_err[:GATE_STEPS].max():.2e} rad; all {N_STEPS}: {pos_err.max():.2e} m, {head_err.max():.2e} rad")
    assert pos_err[:GATE_STEPS].max() < POS_TOL
    assert head_err[:GATE_STEPS].max() < HEAD_TOL
    # the car really moved through the manoeuvres
    assert np.hypot(*(jax_traj[-1, :2] - start[:2])) > 10.0
    assert np.ptp(np.unwrap(jax_traj[:, 2])) > 1.0
