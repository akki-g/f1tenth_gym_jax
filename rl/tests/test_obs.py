"""Spec section 6 checks for the shared observation code and beam geometry."""

import importlib.util
import math
import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from f1tenth_gym_jax.envs import F110Env
from rl import car_params as cp
from rl import obs as ob
from rl.domain_rand import make_param

HERE = pathlib.Path(__file__).resolve().parent
ROBORACER = pathlib.Path(
    os.environ.get("ROBORACER_ROS2_DIR", HERE.parents[2] / "roboracer_ros2")
)
CAR_OBS = ROBORACER / "src" / "rl_policy" / "rl_policy" / "obs.py"

# urg_node's LaserScan for the UST-10LX: -135 deg .. +135 deg in 0.25 deg steps.
HOKUYO_ANGLE_MIN = -2.356194496154785
HOKUYO_ANGLE_INCREMENT = 0.004363323096185923


def _load_car_obs():
    if not CAR_OBS.exists():
        pytest.fail(f"car copy of obs.py not found at {CAR_OBS} (set ROBORACER_ROS2_DIR)")
    spec = importlib.util.spec_from_file_location("car_obs", CAR_OBS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pipeline(mod, scans_1081, speeds, prev_actions, xp=np, num_beams=ob.NUM_BEAMS, frame_stack=ob.FRAME_STACK):
    """Run a recorded sequence through obs.py the way the car node does."""
    out, stack = [], None
    for scan, speed, prev in zip(scans_1081, speeds, prev_actions):
        s = mod.normalize_scan(mod.subsample_scan(scan, num_beams, xp=xp), xp=xp)
        stack = mod.init_stack(s, frame_stack, xp=xp) if stack is None else mod.push_scan(stack, s, xp=xp)
        out.append(np.asarray(mod.build_obs(stack, speed, prev, xp=xp)))
    return np.stack(out)


def test_obs_constants_match_car_params():
    assert ob.NUM_BEAMS == cp.NUM_BEAMS == 271
    assert ob.CAR_SCAN_BEAMS == cp.CAR_SCAN_BEAMS and ob.BEAM_STRIDE == cp.BEAM_STRIDE == 4
    assert ob.FOV == cp.LIDAR_FOV and ob.MAX_RANGE == cp.MAX_RANGE
    assert ob.FRAME_STACK == cp.FRAME_STACK == 1 and ob.OBS_DIM == cp.OBS_DIM == 274
    assert ob.V_CMD_MAX == cp.V_CMD_MAX and ob.S_MAX == min(-cp.S_MIN, cp.S_MAX)  # the policy's range: the smaller full lock
    np.testing.assert_array_equal(ob.CAR_BEAM_INDICES, np.arange(0, 1081, 4))
    assert ob.obs_dim(3) == 816 and ob.obs_dim(8) == 2171  # the MLP baselines
    # phase-1 layout still expressible: 109 beams, stride 10, stack 3 -> 330
    np.testing.assert_array_equal(ob.beam_indices(109), np.arange(0, 1081, 10))
    assert ob.obs_dim(3, 109) == 330
    with pytest.raises(ValueError):
        ob.beam_stride(270)  # does not land on beam 1080


def test_shared_observation_code():
    """The sim and car copies of obs.py give identical output on 100 recorded inputs."""
    car = _load_car_obs()
    assert CAR_OBS.read_bytes() == (HERE.parent / "obs.py").read_bytes(), "car obs.py is not a verbatim copy"
    rec = np.load(HERE / "data" / "obs_inputs.npz")
    scans, speeds, prev = rec["scans"], rec["speeds"], rec["prev_actions"]
    assert scans.shape == (100, 1081)
    assert np.isinf(scans).any() and np.isnan(scans).any()

    for num_beams, frame_stack in ((271, 1), (271, 3), (271, 8), (109, 3)):  # GRU, MLP-3, MLP-8, phase 1
        sim_out = _pipeline(ob, scans, speeds, prev, num_beams=num_beams, frame_stack=frame_stack)
        car_out = _pipeline(car, scans, speeds, prev, num_beams=num_beams, frame_stack=frame_stack)
        n_scan = frame_stack * num_beams
        assert sim_out.shape == (100, ob.obs_dim(frame_stack, num_beams))
        np.testing.assert_array_equal(sim_out, car_out)
        assert np.isfinite(sim_out).all()
        assert sim_out[:, :n_scan].min() >= 0.0 and sim_out[:, :n_scan].max() <= 1.0

        # the sim traces the same functions with jax.numpy
        jax_out = _pipeline(ob, jnp.asarray(scans), jnp.asarray(speeds), jnp.asarray(prev), xp=jnp,
                            num_beams=num_beams, frame_stack=frame_stack)
        np.testing.assert_allclose(jax_out, sim_out, atol=1e-7)


@pytest.mark.parametrize("num_beams, spacing_deg", [(271, 1.0), (109, 2.5)])
def test_beam_angle_match(num_beams, spacing_deg):
    """Sim beam angles equal the car's subsampled angles (0, stride, ..., 1080) within 1e-4 rad
    (271 beams: indices 0, 4, ..., 1080; the phase-1 109-beam layout too)."""
    env = F110Env(1, make_param("Spielberg", num_beams=num_beams))
    sim = np.asarray(env.beam_offsets, dtype=np.float64)
    car = HOKUYO_ANGLE_MIN + ob.beam_indices(num_beams) * HOKUYO_ANGLE_INCREMENT
    assert sim.shape == car.shape == (num_beams,)
    assert np.max(np.abs(sim - car)) < 1e-4
    np.testing.assert_allclose(np.diff(sim), math.radians(spacing_deg), atol=1e-6)
    if num_beams == cp.NUM_BEAMS:
        np.testing.assert_allclose(cp.beam_angles_sim(), car, atol=1e-4)


def test_action_mapping():
    speed, steer = ob.action_to_command(np.array([-1.0, 0.0]))
    assert speed == 0.0 and steer == 0.0
    speed, steer = ob.action_to_command(np.array([1.0, -1.0]))
    assert np.isclose(speed, ob.V_CMD_MAX) and np.isclose(steer, -ob.S_MAX)
    speed, steer = ob.action_to_command(np.array([3.0, 2.0]))  # clipped
    assert np.isclose(speed, ob.V_CMD_MAX) and np.isclose(steer, ob.S_MAX)
    a = np.array([0.3, -0.4], dtype=np.float32)
    np.testing.assert_allclose(ob.command_to_action(*ob.action_to_command(a)), a, atol=1e-6)


def test_normalize_scan_edge_cases():
    r = ob.normalize_scan(np.array([np.inf, np.nan, -np.inf, -1.0, 0.0, 5.0, 12.0]))
    np.testing.assert_allclose(r, [1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 1.0])
    with pytest.raises(ValueError):
        ob.subsample_scan(np.zeros(1080))
