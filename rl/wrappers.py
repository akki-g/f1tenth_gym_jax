"""Phase-1 env wrapper (spec section 4.2): delay, steering offset, LiDAR noise,
frame stack, observation and reward around the patched single-agent F110Env.

Stages, innermost first, as the spec orders them:

1. Action delay      executed = [a_t, a_{t-1}, a_{t-2}][k], k ~ U{0, 1, 2} per episode
2. Steering offset   + U(-0.02, 0.02) rad per episode on the executed steering
3. Scan origin       inside the env: Param.scan_origin_offset (lidar - lr)
4. LiDAR noise       N(0, 0.03 m), clip [0, 10], 1% of beams set to 10 m
5. Obs and reward    obs.py (shared with the car) and
                     r = 10 * dprogress/lap - 0.01 * |a_t - a_{t-1}|_1 - 1[collision]

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
from flax import struct

from f1tenth_gym_jax.envs import F110Env
from f1tenth_gym_jax.envs.utils import State

from . import car_params as cp
from . import obs as ob


@struct.dataclass
class WrapperState:
    env_state: State
    delay_buffer: chex.Array  # [MAX_DELAY_STEPS, 2] previous env commands, newest first
    delay_k: chex.Array  # int, this episode's delay in policy steps
    steer_offset: chex.Array  # rad, this episode's steering offset
    stack: chex.Array  # [FRAME_STACK, NUM_BEAMS] normalized scans, newest first
    prev_action: chex.Array  # [2] previous policy action (a_v, a_s)
    ep_return: chex.Array
    ep_length: chex.Array
    ep_speed_sum: chex.Array
    ep_progress: chex.Array  # laps of progress so far


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

    def __init__(self, env: F110Env, randomize: bool = True):
        p = env.params
        if env.num_agents != 1:
            raise ValueError("Phase1Wrapper is single-agent.")
        if (p.longitudinal_action_type, p.steering_action_type) != ("velocity", "steeringangle"):
            raise ValueError("Phase 1 uses velocity+steeringangle control.")
        if not p.produce_scans or p.num_beams != ob.NUM_BEAMS:
            raise ValueError(f"Env must produce {ob.NUM_BEAMS}-beam scans.")
        if randomize != (p.dyn_param_sampler is not None):
            raise ValueError("randomize must match whether the env samples vehicle params.")
        self._env = env
        self.randomize = randomize
        self.track_length = env.track_length
        self.obs_dim = ob.OBS_DIM
        self.action_dim = 2

    @property
    def env(self) -> F110Env:
        return self._env

    # ------------------------------------------------------------------ reset
    def _episode_draws(self, key):
        if not self.randomize:
            return jnp.array(0, dtype=jnp.int32), jnp.array(0.0)
        k_delay, k_off = jax.random.split(key)
        k = jax.random.randint(k_delay, (), 0, cp.MAX_DELAY_STEPS + 1)
        off = jax.random.uniform(k_off, (), minval=-cp.STEER_OFFSET_MAX, maxval=cp.STEER_OFFSET_MAX)
        return k, off

    def _sense(self, key, env_state):
        ranges = env_state.scans[0]
        if self.randomize:
            ranges = lidar_noise(key, ranges)
        return ob.normalize_scan(ranges, xp=jnp)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, WrapperState]:
        k_env, k_ep, k_noise = jax.random.split(key, 3)
        _, env_state = self._env.reset(k_env)
        delay_k, steer_offset = self._episode_draws(k_ep)
        stack = ob.init_stack(self._sense(k_noise, env_state), xp=jnp)
        prev_action = ob.reset_action(xp=jnp)
        state = WrapperState(
            env_state=env_state,
            delay_buffer=jnp.zeros((cp.MAX_DELAY_STEPS, 2)),
            delay_k=delay_k,
            steer_offset=steer_offset,
            stack=stack,
            prev_action=prev_action,
            ep_return=jnp.array(0.0),
            ep_length=jnp.array(0, dtype=jnp.int32),
            ep_speed_sum=jnp.array(0.0),
            ep_progress=jnp.array(0.0),
        )
        speed = env_state.cartesian_states[0, 3]
        return ob.build_obs(stack, speed, prev_action, xp=jnp), state

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
        done = dones["__all__"]

        # 4-5. sensing, observation, reward
        stack = ob.push_scan(state.stack, self._sense(k_noise, env_state), xp=jnp)
        speed = env_state.cartesian_states[0, 3]
        obs = ob.build_obs(stack, speed, action, xp=jnp)
        collided = env_state.collisions[0]
        progress_m = rewards["agent_0"]
        reward = step_reward(progress_m, self.track_length, action, state.prev_action, collided)

        ep_return = state.ep_return + reward
        ep_length = state.ep_length + 1
        ep_speed_sum = state.ep_speed_sum + speed
        ep_progress = state.ep_progress + progress_m / self.track_length
        cont = state.replace(
            env_state=env_state,
            delay_buffer=delay_buffer,
            stack=stack,
            prev_action=action,
            ep_return=ep_return,
            ep_length=ep_length,
            ep_speed_sum=ep_speed_sum,
            ep_progress=ep_progress,
        )

        dp = env_state.dyn_params
        info = {
            "executed_command": executed,
            "returned_episode": done,
            "episode_return": ep_return,
            "episode_length": ep_length,
            "episode_mean_speed": ep_speed_sum / ep_length,
            "episode_progress": ep_progress,
            "lap_complete": env_state.num_laps[0] >= self._env.params.max_num_laps,
            "collision": collided,
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
