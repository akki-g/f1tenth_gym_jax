"""Phase-1 env wrapper (spec section 4.2): delay, steering offset, LiDAR noise,
frame stack, observation and reward around the patched single-agent F110Env.

Stages, innermost first, as the spec orders them:

1. Action delay      executed = [a_t, ..., a_{t-K}][k], k ~ U{0, ..., max_delay} per episode
                     (max_delay 4 in phase 1c, 2 in phase 1)
2. Steering offset   + U(-0.02, 0.02) rad per episode on the executed steering
3. Scan origin       inside the env: Param.scan_origin_offset (lidar - lr)
4. LiDAR noise       N(0, 0.03 m), clip [0, 10], 1% of beams set to 10 m
5. Obs and reward    obs.py (shared with the car) and
                     r = 10 * dprogress/lap - 0.01 * |a_t - a_{t-1}|_1 - 1[collision]

Episode ends (each with the collision penalty): env collision; the wall
buffer (the collision box inflated by ``wall_margin`` per side, randomized
+-30% per episode); wrong way (heading > 90 deg off the track direction for
``wrong_way_steps``); stall (speed < STALL_SPEED for ``stall_steps``). They use
the sim's exact pose and track position; nothing changes on the car.

Laps are counted from accumulated centerline progress (``max_laps``, 2 for
training) and end the episode here. The env's own winding-number lap
counter is switched off (make_param sets a huge max_num_laps): on tracks
that are not star-shaped around its winding point it reports a lap after as
little as half a lap, depending on the start (results note, D9).

The policy acts in a = (a_v, a_s) in [-1, 1]. The previous action in the
observation is the policy's (clipped) action as sent on, before delay and
steering offset: the car's node only knows what it published, not what the
actuators do with it (see the results note, D6).

``randomize=False`` gives the evaluation setting: nominal vehicle parameters
(the env's Param must also be built with randomize=False), no delay, no
steering offset, no LiDAR noise.
"""

from __future__ import annotations

from functools import partial
from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from f1tenth_gym_jax.envs import F110Env
from f1tenth_gym_jax.envs.collision_models import collision_map, get_vertices
from f1tenth_gym_jax.envs.utils import State

from . import car_params as cp
from . import obs as ob


@struct.dataclass
class WrapperState:
    env_state: State
    delay_buffer: chex.Array  # [max_delay, 2] previous env commands, newest first
    delay_k: chex.Array  # int, this episode's delay in policy steps
    steer_offset: chex.Array  # rad, this episode's steering offset
    stack: chex.Array  # [frame_stack, num_beams] normalized scans, newest first
    prev_action: chex.Array  # [2] previous policy action (a_v, a_s)
    ep_return: chex.Array
    ep_length: chex.Array
    ep_speed_sum: chex.Array
    ep_progress: chex.Array  # laps of progress so far
    margin: chex.Array  # m per side, this episode's wall buffer
    wrong_way_count: chex.Array  # consecutive steps pointing the wrong way
    stall_count: chex.Array  # consecutive steps below STALL_SPEED


def delay_action(buffer, k, command):
    """Stage 1. Returns (executed command, new buffer). buffer: newest first."""
    history = jnp.concatenate([command[None], buffer], axis=0)
    return history[k], history[:-1]


def add_steer_offset(command, offset):
    """Stage 2. Env command order is [steer, speed]."""
    return command.at[0].add(offset)


def lidar_noise(key, ranges, noise_std=cp.RANGE_NOISE_STD, dropout=cp.DROPOUT_PROB):
    """Stage 4. Gaussian range noise, clip to [0, max range], random dropouts."""
    k_noise, k_drop = jax.random.split(key)
    noisy = ranges + noise_std * jax.random.normal(k_noise, ranges.shape)
    noisy = jnp.clip(noisy, 0.0, cp.MAX_RANGE)
    drop = jax.random.bernoulli(k_drop, dropout, ranges.shape)
    return jnp.where(drop, cp.MAX_RANGE, noisy)


def step_reward(progress_m, track_length, action, prev_action, collided):
    """Stage 5 reward."""
    smooth = jnp.sum(jnp.abs(action - prev_action))
    return (
        cp.PROGRESS_SCALE * progress_m / track_length
        - cp.SMOOTHNESS_COEF * smooth
        - cp.COLLISION_PENALTY * collided.astype(jnp.float32)
    )


class Phase1Wrapper:
    """Single-agent, array-in/array-out wrapper; auto-resets on done."""

    NUM_AGENTS = 1  # rl.race.RaceWrapper: 2 (ego is agent 0)

    def __init__(
        self,
        env: F110Env,
        randomize: bool = True,
        max_laps: int = cp.MAX_LAPS,
        wall_margin: float = cp.WALL_MARGIN,
        wall_margin_noise: float = cp.WALL_MARGIN_NOISE,
        wrong_way_steps: int = cp.WRONG_WAY_STEPS,
        stall_steps: int = cp.STALL_STEPS,
        frame_stack: int = ob.FRAME_STACK,
        max_delay: int = cp.MAX_DELAY_STEPS,
    ):
        p = env.params
        if env.num_agents != self.NUM_AGENTS:
            raise ValueError(f"{type(self).__name__} needs an env with {self.NUM_AGENTS} agent(s).")
        if (p.longitudinal_action_type, p.steering_action_type) != ("velocity", "steeringangle"):
            raise ValueError("Phase 1 uses velocity+steeringangle control.")
        if not p.produce_scans or abs(p.fov - ob.FOV) > 1e-9:
            raise ValueError(f"Env must produce scans over {ob.FOV} rad.")
        ob.beam_stride(p.num_beams)  # raises unless the beams land on the car's grid
        if frame_stack < 1 or max_delay < 0:
            raise ValueError("need frame_stack >= 1 and max_delay >= 0")
        if randomize != (p.dyn_param_sampler is not None):
            raise ValueError("randomize must match whether the env samples vehicle params.")
        if p.max_num_laps <= max_laps:
            raise ValueError("the env's winding lap counter must be off (max_num_laps >> max_laps)")
        if wall_margin < 0 or not 0 <= wall_margin_noise < 1:
            raise ValueError("need wall_margin >= 0 and 0 <= wall_margin_noise < 1")
        if wrong_way_steps < 0 or stall_steps < 0:
            raise ValueError("wrong_way_steps and stall_steps must be >= 0 (0 turns the check off)")
        self._env = env
        self.randomize = randomize
        self.max_laps = max_laps
        self.wall_margin = float(wall_margin)
        self.wall_margin_noise = float(wall_margin_noise)
        self.wrong_way_steps = int(wrong_way_steps)
        self.stall_steps = int(stall_steps)
        # wall-buffer check on a window of the map around the car, sized to
        # cover the largest inflated footprint (same test as the env's
        # collision_map, without scanning every wall pixel of the map)
        res = float(env.track.resolution)
        m_max = self.wall_margin * (1.0 + self.wall_margin_noise)
        reach = ((p.length / 2 + m_max) ** 2 + (p.width / 2 + m_max) ** 2) ** 0.5
        self._win = 2 * (int(reach / res) + 2) + 1
        occ = np.asarray(env.track.occ_map) == 0.0  # occupied, as in F110Env._set_pixelcenters
        if occ.shape[0] < self._win or occ.shape[1] < self._win:
            raise ValueError("map smaller than the wall-buffer window")
        self._wall_px = jnp.asarray(occ)
        self.track_length = env.track_length
        self.num_beams = int(p.num_beams)
        self.frame_stack = int(frame_stack)
        self.max_delay = int(max_delay)
        self.obs_dim = ob.obs_dim(self.frame_stack, self.num_beams)
        self.action_dim = 2

    @property
    def env(self) -> F110Env:
        return self._env

    # ------------------------------------------------------------------ reset
    def _episode_draws(self, key):
        """(delay k, steering offset, wall margin) for one episode."""
        if not self.randomize:
            return jnp.array(0, dtype=jnp.int32), jnp.array(0.0), jnp.array(self.wall_margin, jnp.float32)
        k_delay, k_off, k_margin = jax.random.split(key, 3)
        k = jax.random.randint(k_delay, (), 0, self.max_delay + 1)
        off = jax.random.uniform(k_off, (), minval=-cp.STEER_OFFSET_MAX, maxval=cp.STEER_OFFSET_MAX)
        n = self.wall_margin_noise
        margin = self.wall_margin * jax.random.uniform(k_margin, (), minval=1.0 - n, maxval=1.0 + n)
        return k, off, margin.astype(jnp.float32)

    def wall_buffer_hit(self, env_state, margin):
        """True when a wall pixel centre lies inside the collision box inflated
        by ``margin`` per side (the env's own check uses margin 0). Same test
        as collision_map, restricted to the pixels around the car."""
        env, p = self._env, self._env.params
        pose = env_state.cartesian_states[0, jnp.array([0, 1, 4])]
        verts = get_vertices(pose, p.length + 2.0 * margin, p.width + 2.0 * margin)
        res, ox, oy = env.track.resolution, env.track.ox, env.track.oy
        h, w = self._wall_px.shape
        half = self._win // 2
        r0 = jnp.clip(jnp.floor((pose[1] - oy) / res).astype(jnp.int32) - half, 0, h - self._win)
        c0 = jnp.clip(jnp.floor((pose[0] - ox) / res).astype(jnp.int32) - half, 0, w - self._win)
        wall = jax.lax.dynamic_slice(self._wall_px, (r0, c0), (self._win, self._win))
        idx = jnp.arange(self._win)
        cy, cx = jnp.meshgrid((r0 + idx) * res + oy + res / 2, (c0 + idx) * res + ox + res / 2, indexing="ij")
        centers = jnp.stack([cx, cy], axis=-1).reshape(-1, 2)
        centers = jnp.where(wall.reshape(-1, 1), centers, 1e9)  # free pixels can never be inside
        return collision_map(verts[None], centers)[0]

    def _sense(self, key, env_state):
        ranges = env_state.scans[0]
        if self.randomize:
            ranges = lidar_noise(key, ranges)
        return ob.normalize_scan(ranges, xp=jnp)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, WrapperState]:
        k_env, k_ep, k_noise = jax.random.split(key, 3)
        _, env_state = self._env.reset(k_env)
        delay_k, steer_offset, margin = self._episode_draws(k_ep)
        stack = ob.init_stack(self._sense(k_noise, env_state), self.frame_stack, xp=jnp)
        prev_action = ob.reset_action(xp=jnp)
        state = WrapperState(
            env_state=env_state,
            delay_buffer=jnp.zeros((self.max_delay, 2)),
            delay_k=delay_k,
            steer_offset=steer_offset,
            stack=stack,
            prev_action=prev_action,
            ep_return=jnp.zeros((), jnp.float32),
            ep_length=jnp.zeros((), jnp.int32),
            ep_speed_sum=jnp.zeros((), jnp.float32),
            ep_progress=jnp.zeros((), jnp.float32),
            margin=margin,
            wrong_way_count=jnp.zeros((), jnp.int32),
            stall_count=jnp.zeros((), jnp.int32),
        )
        speed = env_state.cartesian_states[0, 3]
        return ob.build_obs(stack, speed, prev_action, xp=jnp), state

    def reset_at(self, key: chex.PRNGKey, s: chex.Array) -> Tuple[chex.Array, WrapperState]:
        """Reset with the car on the centerline at arc length ``s``, heading
        along the track, speed 0 (fixed evaluation start points)."""
        k_reset, k_scan, k_noise = jax.random.split(key, 3)
        _, state = self.reset(k_reset)
        env = self._env
        frenet = jnp.array([[s, 0.0, 0.0]])
        pose = env.track.vmap_frenet_to_cartesian_jax(frenet)
        x = jnp.zeros_like(state.env_state.cartesian_states).at[:, jnp.array([0, 1, 4])].set(pose)
        es = state.env_state.replace(
            cartesian_states=x,
            last_cartesian_states=x,
            frenet_states=frenet,
            last_frenet_states=frenet,
            prev_winding_vector=x[:, :2] - env.winding_point,
        )
        es = env._scan(es, k_scan)
        stack = ob.init_stack(self._sense(k_noise, es), self.frame_stack, xp=jnp)
        state = state.replace(env_state=es, stack=stack)
        return ob.build_obs(stack, x[0, 3], state.prev_action, xp=jnp), state

    # ------------------------------------------------------------------- step
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, key: chex.PRNGKey, state: WrapperState, action: chex.Array
    ) -> Tuple[chex.Array, WrapperState, chex.Array, chex.Array, Dict[str, chex.Array]]:
        k_env, k_noise, k_reset = jax.random.split(key, 3)
        action = jnp.clip(action, -1.0, 1.0)
        speed_cmd, steer_cmd = ob.action_to_command(action, xp=jnp)
        command = jnp.stack([steer_cmd, speed_cmd])  # env order: [steer, speed]

        # 1-2. actuation
        executed, delay_buffer = delay_action(state.delay_buffer, state.delay_k, command)
        executed = add_steer_offset(executed, state.steer_offset)

        # env (3. scan origin offset is applied in the env's _scan)
        _, env_state, rewards, dones, _ = self._env.step_env(
            k_env, state.env_state, {"agent_0": executed}
        )

        # 4-5. sensing, observation, reward
        stack = ob.push_scan(state.stack, self._sense(k_noise, env_state), xp=jnp)
        speed = env_state.cartesian_states[0, 3]
        obs = ob.build_obs(stack, speed, action, xp=jnp)
        contact = env_state.collisions[0]
        buffer_hit = self.wall_buffer_hit(env_state, state.margin)
        collided = contact | buffer_hit
        # wrong way: heading error (wrapped to [-pi, pi]) beyond 90 deg
        wrong_way_count = jnp.where(
            jnp.cos(env_state.frenet_states[0, 2]) < 0.0, state.wrong_way_count + 1, 0
        ).astype(jnp.int32)
        wrong_way = (self.wrong_way_steps > 0) & (wrong_way_count >= self.wrong_way_steps)
        stall_count = jnp.where(speed < cp.STALL_SPEED, state.stall_count + 1, 0).astype(jnp.int32)
        stall = (self.stall_steps > 0) & (stall_count >= self.stall_steps)
        failed = collided | wrong_way | stall
        progress_m = rewards["agent_0"]
        reward = step_reward(progress_m, self.track_length, action, state.prev_action, failed)

        ep_return = state.ep_return + reward
        ep_length = state.ep_length + 1
        ep_speed_sum = state.ep_speed_sum + speed
        ep_progress = state.ep_progress + progress_m / self.track_length
        lap_complete = ep_progress >= self.max_laps
        done = dones["__all__"] | lap_complete | failed
        cont = state.replace(
            env_state=env_state,
            delay_buffer=delay_buffer,
            stack=stack,
            prev_action=action,
            ep_return=ep_return,
            ep_length=ep_length,
            ep_speed_sum=ep_speed_sum,
            ep_progress=ep_progress,
            wrong_way_count=wrong_way_count,
            stall_count=stall_count,
        )

        dp = env_state.dyn_params
        info = {
            "executed_command": executed,
            "returned_episode": done,
            "episode_return": ep_return,
            "episode_length": ep_length,
            "episode_mean_speed": ep_speed_sum / ep_length,
            "episode_progress": ep_progress,
            "lap_complete": lap_complete,
            "collision": collided,  # wall contact or inside the wall buffer
            "contact": contact,  # the env's own collision (no buffer)
            "wall_buffer": buffer_hit & ~contact,
            "wrong_way": wrong_way,
            "stall": stall,
            "margin": state.margin,
            "timeout": env_state.step >= self._env.params.max_steps,
            "speed": speed,
            "delay_k": state.delay_k,
            "steer_offset": state.steer_offset,
            **{f"param_{n}": getattr(dp, n)[0] for n in ("mu", "m", "C_Sf", "C_Sr", "lf", "h", "I")},
        }

        # auto-reset
        reset_obs, reset_state = self.reset(k_reset)
        state = jax.tree.map(lambda r, c: jnp.where(done, r, c), reset_state, cont)
        obs = jnp.where(done, reset_obs, obs)
        return obs, state, reward, done, info
