"""Observation builder, shared verbatim by training and the car.

This file is copied byte-for-byte into roboracer_ros2
(src/rl_policy/rl_policy/obs.py); a test checks the two copies agree. It
must only import NumPy at module level: the car has no JAX. Every function
takes ``xp`` (numpy or jax.numpy) so the sim can trace the same code.

Observation, frame_stack x num_beams + 3 values:
    [scan_t, ..., scan_{t-k+1}]  frame_stack scans of num_beams, ranges clipped
                                 to [0, MAX_RANGE] and divided by MAX_RANGE
    speed / V_CMD_MAX            1 value, longitudinal speed
    previous action (a_v, a_s)   2 values, as sent on

The layout is set per policy and recorded in its exported config. The module
constants are the phase-1c defaults: 271 beams (every 4th of the UST-10LX's
1081, 1.0 deg apart), frame stack 1 (the CNN+GRU carries history), 274
values. The MLP baselines use stacks of 3 and 8; phase-1 policies used 109
beams (every 10th) and a stack of 3.

Action: a = (a_v, a_s) in [-1, 1]^2 (speed first).
    speed command    = (a_v + 1) / 2 * V_CMD_MAX
    steering command = a_s * S_MAX
"""

import numpy as np

CAR_SCAN_BEAMS = 1081  # Hokuyo UST-10LX: 0.25 deg over 270 deg
BEAM_STRIDE = 4
NUM_BEAMS = (CAR_SCAN_BEAMS - 1) // BEAM_STRIDE + 1  # 271 policy beams, 1.0 deg apart
CAR_BEAM_INDICES = np.arange(0, CAR_SCAN_BEAMS, BEAM_STRIDE)  # 0, 4, ..., 1080
FOV = 4.7124  # rad
MAX_RANGE = 10.0  # m
FRAME_STACK = 1
V_CMD_MAX = 8.0  # m/s
S_MAX = 0.4189  # rad
NUM_NON_SCAN = 3  # speed and the previous action, after the scans
OBS_DIM = FRAME_STACK * NUM_BEAMS + NUM_NON_SCAN  # 274
# Previous action after a reset / activation: stopped (speed 0), wheels straight.
RESET_ACTION = (-1.0, 0.0)


def obs_dim(frame_stack=FRAME_STACK, num_beams=NUM_BEAMS):
    return frame_stack * num_beams + NUM_NON_SCAN


def beam_stride(num_beams=NUM_BEAMS):
    """Car-scan stride for ``num_beams`` policy beams spanning beams 0..1080."""
    if num_beams < 2 or (CAR_SCAN_BEAMS - 1) % (num_beams - 1):
        raise ValueError(f"{num_beams} beams do not land on the car's {CAR_SCAN_BEAMS}-beam grid")
    return (CAR_SCAN_BEAMS - 1) // (num_beams - 1)


def beam_indices(num_beams=NUM_BEAMS):
    """Car scan indices of the policy beams: 0, stride, ..., 1080."""
    return np.arange(0, CAR_SCAN_BEAMS, beam_stride(num_beams))


def subsample_scan(ranges, num_beams=NUM_BEAMS, xp=np):
    """The car's 1081-beam scan -> the policy beams (indices 0, stride, ..., 1080)."""
    ranges = xp.asarray(ranges)
    if ranges.shape[-1] != CAR_SCAN_BEAMS:
        raise ValueError(f"expected {CAR_SCAN_BEAMS} beams, got {ranges.shape[-1]}")
    return ranges[..., :: beam_stride(num_beams)]


def normalize_scan(ranges, xp=np):
    """Ranges in metres -> [0, 1]. NaN and +inf (no return) count as max range."""
    ranges = xp.asarray(ranges, dtype=xp.float32)
    ranges = xp.nan_to_num(ranges, nan=MAX_RANGE, posinf=MAX_RANGE, neginf=0.0)
    return xp.clip(ranges, 0.0, MAX_RANGE) / MAX_RANGE


def init_stack(scan_norm, frame_stack=FRAME_STACK, xp=np):
    """Frame stack after a reset: every frame is the first scan. Newest first."""
    scan_norm = xp.asarray(scan_norm, dtype=xp.float32)
    return xp.stack([scan_norm] * frame_stack)


def push_scan(stack, scan_norm, xp=np):
    """Add the newest normalized scan, drop the oldest. Newest first."""
    scan_norm = xp.asarray(scan_norm, dtype=xp.float32)
    return xp.concatenate([scan_norm[None], stack[:-1]], axis=0)


def build_obs(stack, speed, prev_action, xp=np):
    """Concatenate the stack, normalized speed and previous action."""
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
