"""Phase-1 observation builder, shared verbatim by training and the car.

This file is copied byte-for-byte into roboracer_ros2
(src/rl_policy/rl_policy/obs.py); a test checks the two copies agree. It
must only import NumPy at module level: the car has no JAX. Every function
takes ``xp`` (numpy or jax.numpy) so the sim can trace the same code.

Observation (spec section 4.5), 330 values:
    [scan_t, scan_{t-1}, scan_{t-2}]  3 x 109 beams, ranges clipped to
                                      [0, MAX_RANGE] and divided by MAX_RANGE
    speed / V_CMD_MAX                 1 value, longitudinal speed
    previous action (a_v, a_s)        2 values, as executed (sent on)

Action: a = (a_v, a_s) in [-1, 1]^2 (speed first).
    speed command    = (a_v + 1) / 2 * V_CMD_MAX
    steering command = a_s * S_MAX
"""

import numpy as np

NUM_BEAMS = 109  # policy beams, 2.5 deg apart over 270 deg
CAR_SCAN_BEAMS = 1081  # Hokuyo UST-10LX: 0.25 deg over 270 deg
BEAM_STRIDE = 10
CAR_BEAM_INDICES = np.arange(0, CAR_SCAN_BEAMS, BEAM_STRIDE)  # 0, 10, ..., 1080
FOV = 4.7124  # rad
MAX_RANGE = 10.0  # m
FRAME_STACK = 3
V_CMD_MAX = 8.0  # m/s
S_MAX = 0.4189  # rad
OBS_DIM = FRAME_STACK * NUM_BEAMS + 1 + 2  # 330
# Previous action after a reset / activation: stopped (speed 0), wheels straight.
RESET_ACTION = (-1.0, 0.0)


def subsample_scan(ranges, xp=np):
    """The car's 1081-beam scan -> the 109 policy beams (indices 0, 10, ..., 1080)."""
    ranges = xp.asarray(ranges)
    if ranges.shape[-1] != CAR_SCAN_BEAMS:
        raise ValueError(f"expected {CAR_SCAN_BEAMS} beams, got {ranges.shape[-1]}")
    return ranges[..., ::BEAM_STRIDE]


def normalize_scan(ranges, xp=np):
    """Ranges in metres -> [0, 1]. NaN and +inf (no return) count as max range."""
    ranges = xp.asarray(ranges, dtype=xp.float32)
    ranges = xp.nan_to_num(ranges, nan=MAX_RANGE, posinf=MAX_RANGE, neginf=0.0)
    return xp.clip(ranges, 0.0, MAX_RANGE) / MAX_RANGE


def init_stack(scan_norm, xp=np):
    """Frame stack after a reset: every frame is the first scan. Newest first."""
    scan_norm = xp.asarray(scan_norm, dtype=xp.float32)
    return xp.stack([scan_norm] * FRAME_STACK)


def push_scan(stack, scan_norm, xp=np):
    """Add the newest normalized scan, drop the oldest. Newest first."""
    scan_norm = xp.asarray(scan_norm, dtype=xp.float32)
    return xp.concatenate([scan_norm[None], stack[:-1]], axis=0)


def build_obs(stack, speed, prev_action, xp=np):
    """Concatenate the stack, normalized speed and previous action -> (330,)."""
    speed = xp.asarray(speed, dtype=xp.float32).reshape(1)
    prev_action = xp.asarray(prev_action, dtype=xp.float32).reshape(2)
    return xp.concatenate(
        [xp.asarray(stack, dtype=xp.float32).reshape(-1), speed / V_CMD_MAX, prev_action]
    )


def reset_action(xp=np):
    return xp.asarray(RESET_ACTION, dtype=xp.float32)


def action_to_command(action, v_cmd_max=V_CMD_MAX, s_max=S_MAX, xp=np):
    """Policy action (a_v, a_s) in [-1, 1] -> (speed m/s, steering angle rad)."""
    action = xp.clip(xp.asarray(action, dtype=xp.float32), -1.0, 1.0)
    speed = (action[..., 0] + 1.0) / 2.0 * v_cmd_max
    steer = action[..., 1] * s_max
    return speed, steer


def command_to_action(speed, steer, v_cmd_max=V_CMD_MAX, s_max=S_MAX, xp=np):
    """Inverse of action_to_command, clipped to [-1, 1] (for clamped commands)."""
    a_v = 2.0 * xp.asarray(speed, dtype=xp.float32) / v_cmd_max - 1.0
    a_s = xp.asarray(steer, dtype=xp.float32) / s_max
    return xp.clip(xp.stack([a_v, a_s], axis=-1), -1.0, 1.0)
