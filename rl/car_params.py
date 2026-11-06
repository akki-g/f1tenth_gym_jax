"""Phase-1 vehicle, sensor and timing parameters.

Every number here comes from the tables in section 3 of docs/rl_phase1_spec.md
(roboracer_ros2). Values the spec marks *measure* are estimates; replace them
with measurements, retrain and compare.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Vehicle: randomized physical parameters (nominal, multiplicative sigma).
# Factor ~ N(1, sigma), clipped to +-CLIP_SIGMAS * sigma.
# ---------------------------------------------------------------------------
NOMINAL = {
    "m": 3.45,  # mass, kg (estimate; measure)
    "lf": 0.1587,  # CG to front axle, m (gym split scaled to 0.33 m; measure)
    "lr": 0.1713,  # CG to rear axle, m (same)
    "h": 0.074,  # CG height, m (gym)
    "I": 0.0435,  # yaw inertia, kg m^2 (gym, scaled by mass)
    "C_Sf": 4.718,  # front cornering stiffness, 1/rad (gym)
    "C_Sr": 5.4562,  # rear cornering stiffness, 1/rad (gym)
    "mu": 1.10,  # friction (carpet estimate; measure)
}
SIGMA = {
    "m": 0.05,
    "lf": 0.05,
    "lr": 0.05,
    "h": 0.15,
    "I": 0.20,
    "C_Sf": 0.20,
    "C_Sr": 0.20,
    "mu": 0.10,
}
CLIP_SIGMAS = 2.0

# ---------------------------------------------------------------------------
# Vehicle: fixed parameters.
# ---------------------------------------------------------------------------
S_MIN, S_MAX = -0.4189, 0.4189  # steering limit, rad (gym; measure full lock)
SV_MIN, SV_MAX = -3.2, 3.2  # steering rate, rad/s (gym, car max_servo_speed)
V_MIN, V_MAX = -5.0, 20.0  # speed, m/s (gym; phase-1 decision)
A_MAX = 9.51  # acceleration, m/s^2 (gym; phase-1 decision)
V_SWITCH = 7.319  # m/s (gym)
WIDTH, LENGTH = 0.31, 0.58  # footprint, m (gym; measure with bumper and cage)
WHEELBASE = 0.33  # m (car config vesc.yaml)

# ---------------------------------------------------------------------------
# Sensing and timing.
# ---------------------------------------------------------------------------
LIDAR_FOV = 4.7124  # rad, 270 deg (Hokuyo UST-10LX)
NUM_BEAMS = 109  # policy beams, 2.5 deg apart
CAR_SCAN_BEAMS = 1081  # beams in the car's full scan
BEAM_STRIDE = 10  # car beams used: indices 0, 10, ..., 1080
MAX_RANGE = 10.0  # m; clip and normalize to [0, 1]
RANGE_NOISE_STD = 0.03  # m, Gaussian (spec repeatability)
DROPOUT_PROB = 0.01  # fraction of beams set to max range (estimate)
LIDAR_OFFSET = 0.275  # m, LiDAR ahead of the rear axle (sim.yaml; measure)
TIMESTEP = 0.01  # s, physics step (gym)
TIMESTEP_RATIO = 4  # physics steps per policy step -> 25 Hz
POLICY_DT = TIMESTEP * TIMESTEP_RATIO
POLICY_RATE_HZ = 1.0 / POLICY_DT
MAX_DELAY_STEPS = 2  # action delay 0..2 policy steps (0-80 ms), uniform per episode
STEER_OFFSET_MAX = 0.02  # rad, uniform +- per episode

# ---------------------------------------------------------------------------
# Action, observation and episode (spec sections 4.4-4.6).
# ---------------------------------------------------------------------------
V_CMD_MAX = 8.0  # m/s; policy speed command range [0, V_CMD_MAX]
FRAME_STACK = 3
OBS_DIM = FRAME_STACK * NUM_BEAMS + 1 + 2  # 330
EPISODE_SECONDS = 90.0
MAX_STEPS = int(round(EPISODE_SECONDS / POLICY_DT))  # 2250
MAX_LAPS = 2
PROGRESS_SCALE = 10.0  # reward per lap of progress
SMOOTHNESS_COEF = 0.01  # penalty on |a_t - a_{t-1}|_1
COLLISION_PENALTY = 1.0

# Episode ends added after the first Newton run (results note, D22). Each one
# ends the episode with COLLISION_PENALTY; none changes what the car runs.
# Wall buffer: the collision box is inflated by this much per side. The race
# walls are round air ducts and the LiDAR scans one plane, so it sees them a
# few cm further away than their widest point; bumpers stick out too.
WALL_MARGIN = 0.05  # m per side (2 in)
WALL_MARGIN_NOISE = 0.30  # randomized episodes: WALL_MARGIN * U(0.7, 1.3)
# Wrong way: heading more than 90 deg off the track direction for ~0.5 s.
WRONG_WAY_STEPS = 13
# Stall: below STALL_SPEED for 2 s (e.g. pinned or stopped against a wall).
STALL_SPEED = 0.2  # m/s
STALL_STEPS = 50


def scan_origin_offset(ref_from_rear_axle: float) -> float:
    """Forward shift of the scan origin from the dynamics reference point.

    The single-track models ("st", "st_smooth") integrate the CG, which sits
    ``lr`` ahead of the rear axle; the kinematic model ("ks") integrates the
    rear axle (0).
    """
    return LIDAR_OFFSET - ref_from_rear_axle


def beam_angles_sim() -> list[float]:
    """Nominal policy beam angles: NUM_BEAMS evenly over [-FOV/2, FOV/2]."""
    step = LIDAR_FOV / (NUM_BEAMS - 1)
    return [-LIDAR_FOV / 2 + i * step for i in range(NUM_BEAMS)]


def beam_angles_car(
    angle_min: float = -LIDAR_FOV / 2,
    angle_increment: float = math.radians(0.25),
) -> list[float]:
    """Angles of the car's subsampled beams (indices 0, 10, ..., 1080)."""
    return [
        angle_min + i * angle_increment
        for i in range(0, CAR_SCAN_BEAMS, BEAM_STRIDE)
    ]
