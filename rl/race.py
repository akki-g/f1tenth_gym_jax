"""Phase 2 (1v1): two-car race wrapper, opponent policies and race reward.

Recurrent-policy spec, "Phase 2: 1v1 model". Agent 0 is the ego and keeps
phase 1's whole pipeline (delay, steering offset, LiDAR noise, wall buffer,
wrong-way and stall ends, progress + smoothness reward). Agent 1 is the
opponent, drawn per episode:

    solo        (25%) parked far off the map: the phase-1c setup
    snapshot    (40%) a past ego GRU, run in JAX with its own hidden state
                      (pool passed to step(); falls back to pure pursuit when absent)
    follow-gap  (20%) JAX port of the team's follow-the-gap, speed x U(0.6, 1.0)
    pursuit     (15%) centerline pure pursuit, speed x U(0.6, 1.0)

Starts: side by side (same s, each car a quarter of the local track width
off the centerline; the width is measured from the map, since track files
list a nominal one) or staggered (opponent 1-5 m ahead). The env casts both cars' scans against
each other (Param.scan_other_agents).

Race reward on top of phase 1's: +0.5 / -0.5 when the ego passes / is passed
(race position, 0.5 m hysteresis), +2 / -2 at the episode's end for being
ahead / behind (opponent episodes only); contact with the opponent ends the
episode with the collision penalty, like a wall.

The observation is [scan, privileged (PRIV_DIM), speed, previous action]: the
actor reads only the scan and the last 3 values (phase-1c layout), the critic
also reads the privileged block (networks.CnnGru, priv_dim).
"""

from __future__ import annotations

import pathlib
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from f1tenth_gym_jax.envs import F110Env
from f1tenth_gym_jax.envs.collision_models import collision, get_vertices
from f1tenth_gym_jax.envs.laser_models import _distance_at

from . import car_params as cp
from . import obs as ob
from .wrappers import Phase1Wrapper, WrapperState, add_steer_offset, delay_action, step_reward

OPP_SOLO, OPP_SNAPSHOT, OPP_FTG, OPP_PURSUIT = 0, 1, 2, 3
OPP_NAMES = ("solo", "snapshot", "follow_gap", "pursuit")
OPP_MIX = (0.25, 0.40, 0.20, 0.15)
SPEED_SCALE = (0.6, 1.0)
STAGGER_M = (1.0, 5.0)
PARK_XY = (1.0e4, 1.0e4)  # far beyond max range of any scan, off every map
OVERTAKE_REWARD = 0.5
FINISH_REWARD = 2.0
PASS_HYSTERESIS_M = 0.5
PRIV_DIM = 17  # opponent (ds, dey, dpsi, v), ego (vx, vy, yaw rate, ey), mu, m, C_Sf, C_Sr, delay k, type one-hot (4)


# ---------------------------------------------------------------- opponents
def pure_pursuit(env: F110Env, car: chex.Array, s: chex.Array, speed_scale, lookahead=1.5, v_max=6.0, a_lat=4.0):
    """Centerline follower: [steer, speed]; speed limited by the curvature ahead."""
    target = env.track.vmap_frenet_to_cartesian_jax(jnp.array([[s + lookahead, 0.0, 0.0]]))[0]
    x, y, psi = car[0], car[1], car[4]
    alpha = jnp.arctan2(target[1] - y, target[0] - x) - psi
    steer = jnp.clip(jnp.arctan2(2.0 * cp.WHEELBASE * jnp.sin(alpha), lookahead), -cp.S_MAX, cp.S_MAX)
    ks = jax.vmap(env.track.centerline.calc_curvature_jax)(s + jnp.array([0.0, 1.0, 2.0, 3.0]))
    kappa = jnp.max(jnp.abs(ks))
    speed = speed_scale * jnp.minimum(v_max, jnp.sqrt(a_lat / jnp.maximum(kappa, 1e-3)))
    return jnp.array([steer, speed])


def follow_the_gap(ranges: chex.Array, speed_scale, fov_deg=180.0, bubble_deg=15.0, best_window_deg=50.0,
                   max_dist=10.0, gain=0.5, max_steer=0.349066):
    """JAX port of blaze_racer's follow_the_gap planner on a 270 deg scan of any
    beam count (its index parameters, tuned for 0.25 deg beams, are converted
    to angles: bubble 60 beams = 15 deg, aim window 200 beams = 50 deg; its
    3-beam smoothing is below one 1-deg beam and is dropped). Returns
    [steer, speed]: 1.5 / 2.5 / 4.0 m/s by steering band, times speed_scale."""
    n = ranges.shape[0]
    deg = 270.0 / (n - 1)
    half = int(round(fov_deg / 2 / deg))
    c = n // 2
    r = jnp.clip(ranges[c - half : c + half + 1], 0.0, max_dist)
    idx = jnp.arange(r.shape[0])
    r = jnp.where(jnp.abs(idx - jnp.argmin(r)) <= int(round(bubble_deg / deg)), 0.0, r)  # safety bubble
    free = r > 0.0
    run = jax.lax.scan(lambda carry, f: ((carry + 1) * f, (carry + 1) * f), 0, free.astype(jnp.int32))[1]
    end = jnp.argmax(run)  # longest free run ends here
    length = run[end]
    start = end - length + 1
    w = max(int(round(best_window_deg / deg)), 1)
    avg = jnp.convolve(r, jnp.ones(w) / w, mode="same")
    in_gap = (idx >= start) & (idx <= end)
    best = jnp.where(length < w, (start + end) // 2, jnp.argmax(jnp.where(in_gap, avg, -jnp.inf)))
    best = jnp.where(length > 0, best, half)  # nothing free: straight ahead
    steer = jnp.clip(jnp.radians((best - half) * deg) * gain, -max_steer, max_steer)
    a = jnp.abs(steer)
    speed = jnp.where(a > 0.1396, 1.5, jnp.where(a > 0.0698, 2.5, 4.0)) * speed_scale
    return jnp.array([steer, jnp.where(length > 0, speed, 0.0)])


def half_width(map_name: str) -> float:
    """Median half-width of a track from its centerline file (w_left, w_right)."""
    from f1tenth_gym_jax.envs.track.utils import find_track_dir

    d = find_track_dir(map_name)
    cl = np.loadtxt(pathlib.Path(d) / f"{d.name}_centerline.csv", delimiter=",", comments="#")
    return float(np.median(cl[:, 2] + cl[:, 3]) / 2)


# ---------------------------------------------------------------- wrapper
@struct.dataclass
class RaceState:
    ego: WrapperState  # phase-1 bookkeeping; ego.env_state holds both cars
    opp_type: chex.Array  # int32, OPP_*
    opp_speed_scale: chex.Array
    opp_snapshot: chex.Array  # int32 index into the snapshot pool
    opp_h: chex.Array  # [gru_dim] snapshot opponent's GRU state
    opp_first: chex.Array  # bool, the snapshot's next step starts its episode
    opp_prev_action: chex.Array  # [2]
    opp_progress: chex.Array  # laps since the start
    start_gap: chex.Array  # m, opponent ahead of the ego at the start (track direction)
    position: chex.Array  # int32: +1 ego ahead, -1 behind, 0 not yet decided (hysteresis)
    overtakes: chex.Array  # int32, net passes this episode


class RaceWrapper(Phase1Wrapper):
    """Two-car env (ego agent 0, opponent agent 1) with phase-1's ego pipeline."""

    NUM_AGENTS = 2

    def __init__(self, env: F110Env, randomize: bool = True, opponents: bool = True, gru_dim: int = 128,
                 snapshot_model=None, fixed_opponent=None, fixed_speed_scale=None, fixed_side_by_side=None,
                 **kwargs):
        """``fixed_*`` (evaluation): use this opponent type (OPP_*), speed factor and
        start mode in every episode instead of drawing them."""
        super().__init__(env, randomize, **kwargs)
        if not env.params.scan_other_agents:
            raise ValueError("the race env must cast scans against the other car (scan_other_agents=True)")
        if self.frame_stack != 1:
            raise ValueError("the race wrapper feeds the CNN+GRU: frame_stack 1")
        self.opponents = opponents
        self.gru_dim = gru_dim
        self.snapshot_model = snapshot_model  # networks.CnnGru (race format) for snapshot opponents
        self.fixed_opponent = fixed_opponent
        self.fixed_speed_scale = fixed_speed_scale
        self.fixed_side_by_side = fixed_side_by_side
        self.obs_dim = ob.obs_dim(1, self.num_beams) + PRIV_DIM

    # ------------------------------------------------------------ helpers
    def _build_obs(self, stack, speed, prev_action, priv):
        base = ob.build_obs(stack, speed, prev_action, xp=jnp)
        n = self.num_beams
        return jnp.concatenate([base[:n], priv, base[n:]])

    def _signed_gap(self, s_opp, s_ego):
        """Opponent ahead of the ego along the track, m, wrapped to [-L/2, L/2)."""
        d = s_opp - s_ego
        return d - self.track_length * jnp.round(d / self.track_length)

    def privileged(self, rs: RaceState, env_state=None):
        es = rs.ego.env_state if env_state is None else env_state
        ego, opp = es.cartesian_states[0], es.cartesian_states[1]
        fe, fo = es.frenet_states[0], es.frenet_states[1]
        present = rs.opp_type != OPP_SOLO
        dpsi = jnp.arctan2(jnp.sin(opp[4] - ego[4]), jnp.cos(opp[4] - ego[4]))
        opp_part = jnp.where(present, jnp.array([self._signed_gap(fo[0], fe[0]) / 10.0, (fo[1] - fe[1]),
                                                 dpsi / jnp.pi, opp[3] / ob.V_CMD_MAX]), 0.0)
        v, beta = ego[3], ego[6]
        dp = es.dyn_params
        own = jnp.array([v * jnp.cos(beta) / ob.V_CMD_MAX, v * jnp.sin(beta) / 2.0, ego[5] / 5.0, fe[1]])
        params = jnp.array([dp.mu[0] / cp.NOMINAL["mu"], dp.m[0] / cp.NOMINAL["m"],
                            dp.C_Sf[0] / cp.NOMINAL["C_Sf"], dp.C_Sr[0] / cp.NOMINAL["C_Sr"]])
        k = jnp.array([rs.ego.delay_k / max(self.max_delay, 1)], jnp.float32)
        onehot = jax.nn.one_hot(rs.opp_type, 4)
        return jnp.concatenate([opp_part, own, params, k, onehot]).astype(jnp.float32)

    def _opp_action(self, key, rs: RaceState, opp_params):
        es = rs.ego.env_state
        car, s = es.cartesian_states[1], es.frenet_states[1, 0]
        pursuit = pure_pursuit(self._env, car, s, rs.opp_speed_scale)
        ftg = follow_the_gap(es.scans[1], rs.opp_speed_scale)
        parked = jnp.zeros(2)
        h, prev = rs.opp_h, rs.opp_prev_action
        if opp_params is not None and self.snapshot_model is not None:
            params = jax.tree.map(lambda x: x[rs.opp_snapshot], opp_params["params"])
            norm = jax.tree.map(lambda x: x[rs.opp_snapshot], opp_params["normalizer"])
            # the snapshot is an ego policy (race format): zeros in its critic-only slot
            o = self._build_obs(ob.normalize_scan(es.scans[1], xp=jnp)[None], car[3], prev, jnp.zeros(PRIV_DIM))
            h_in = jnp.where(rs.opp_first, 0.0, h)
            h_new, mean, _, _ = self.snapshot_model.apply(params, h_in[None], norm(o[None]))
            a = jnp.tanh(mean[0])
            speed, steer = ob.action_to_command(a, xp=jnp)
            snapshot = jnp.array([steer, speed])
            h = h_new[0]
        else:
            snapshot = pursuit  # no pool yet
            a = ob.command_to_action(pursuit[1], pursuit[0], xp=jnp)
        cmd = jnp.select([rs.opp_type == OPP_SOLO, rs.opp_type == OPP_SNAPSHOT, rs.opp_type == OPP_FTG],
                         [parked, snapshot, ftg], pursuit)
        new_prev = jnp.where(rs.opp_type == OPP_SNAPSHOT, a, ob.command_to_action(cmd[1], cmd[0], xp=jnp))
        return cmd, h, new_prev

    # ------------------------------------------------------------ reset
    def local_half_width(self, s):
        """Free distance from the centerline point at s to the nearest wall, m."""
        env = self._env
        xy = env.track.vmap_frenet_to_cartesian_jax(jnp.array([[s, 0.0, 0.0]]))[0]
        return _distance_at(xy[0], xy[1], env.orig_x, env.orig_y, env.orig_c, env.orig_s,
                            env.height, env.width, env.resolution, jnp.asarray(env.distance_transform))

    def _place(self, key, env_state, opp_type):
        """Side-by-side or staggered start; the solo opponent is parked off the map."""
        k_s, k_mode, k_side, k_gap, k_ey = jax.random.split(key, 5)
        L = self.track_length
        s0 = jax.random.uniform(k_s) * L
        side_by_side = jax.random.bernoulli(k_mode, 0.5)
        if self.fixed_side_by_side is not None:
            side_by_side = jnp.array(bool(self.fixed_side_by_side))
        sign = jnp.where(jax.random.bernoulli(k_side, 0.5), 1.0, -1.0)
        q = self.local_half_width(s0) / 2  # a quarter of the local width
        gap = jax.random.uniform(k_gap, minval=STAGGER_M[0], maxval=STAGGER_M[1])
        ey = jax.random.uniform(k_ey, (2,), minval=-0.25 * q, maxval=0.25 * q)
        s_opp = jnp.where(side_by_side, s0, s0 + gap)
        ey_ego = jnp.where(side_by_side, sign * q, ey[0])
        ey_opp = jnp.where(side_by_side, -sign * q, ey[1])
        frenet = jnp.array([[s0, ey_ego, 0.0], [jnp.mod(s_opp, L), ey_opp, 0.0]])
        poses = self._env.track.vmap_frenet_to_cartesian_jax(frenet)
        solo = opp_type == OPP_SOLO
        poses = poses.at[1, :2].set(jnp.where(solo, jnp.array(PARK_XY), poses[1, :2]))
        x = jnp.zeros_like(env_state.cartesian_states).at[:, jnp.array([0, 1, 4])].set(poses)
        es = env_state.replace(cartesian_states=x, last_cartesian_states=x, frenet_states=frenet,
                               last_frenet_states=frenet, prev_winding_vector=x[:, :2] - self._env.winding_point)
        start_gap = jnp.where(solo, 0.0, self._signed_gap(frenet[1, 0], frenet[0, 0]))
        return es, start_gap

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        k_base, k_opp, k_place, k_scan, k_noise, k_scale, k_snap = jax.random.split(key, 7)
        _, base = super().reset(k_base)  # ego draws and bookkeeping; both cars' vehicle params
        p = jnp.array(OPP_MIX if self.opponents else (1.0, 0.0, 0.0, 0.0))
        opp_type = jax.random.choice(k_opp, 4, p=p).astype(jnp.int32)
        if self.fixed_opponent is not None:
            opp_type = jnp.array(self.fixed_opponent, jnp.int32)
        es, start_gap = self._place(k_place, base.env_state, opp_type)
        es = self._env._scan(es, k_scan)
        stack = ob.init_stack(self._sense(k_noise, es), self.frame_stack, xp=jnp)
        ego = base.replace(env_state=es, stack=stack)
        rs = RaceState(
            ego=ego,
            opp_type=opp_type,
            opp_speed_scale=(jax.random.uniform(k_scale, minval=SPEED_SCALE[0], maxval=SPEED_SCALE[1])
                             if self.fixed_speed_scale is None else jnp.array(self.fixed_speed_scale, jnp.float32)),
            opp_snapshot=jax.random.randint(k_snap, (), 0, 1 << 20).astype(jnp.int32),
            opp_h=jnp.zeros((self.gru_dim,)),
            opp_first=jnp.array(True),
            opp_prev_action=ob.reset_action(xp=jnp),
            opp_progress=jnp.zeros((), jnp.float32),
            start_gap=start_gap.astype(jnp.float32),
            position=jnp.zeros((), jnp.int32),
            overtakes=jnp.zeros((), jnp.int32),
        )
        obs = self._build_obs(stack, es.cartesian_states[0, 3], ego.prev_action, self.privileged(rs))
        return obs, rs

    def reset_at(self, key, s):
        raise NotImplementedError("race evaluation sets starts through reset()")

    # ------------------------------------------------------------ step
    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: chex.PRNGKey, rs: RaceState, action: chex.Array, opp_params=None):
        k_env, k_noise, k_reset, k_opp = jax.random.split(key, 4)
        st = rs.ego
        if opp_params is not None:  # snapshot index into this pool's size
            n_pool = jax.tree.leaves(opp_params["params"])[0].shape[0]
            rs = rs.replace(opp_snapshot=rs.opp_snapshot % n_pool)
        action = jnp.clip(action, -1.0, 1.0)
        speed_cmd, steer_cmd = ob.action_to_command(action, xp=jnp)
        command = jnp.stack([steer_cmd, speed_cmd])
        executed, delay_buffer = delay_action(st.delay_buffer, st.delay_k, command)
        executed = add_steer_offset(executed, st.steer_offset)
        opp_cmd, opp_h, opp_prev = self._opp_action(k_opp, rs, opp_params)

        _, es, rewards, _, _ = self._env.step_env(k_env, st.env_state, {"agent_0": executed, "agent_1": opp_cmd})
        # the solo opponent stays parked off the map (speed 0 already; this pins it exactly)
        parked = rs.opp_type == OPP_SOLO
        cs = es.cartesian_states.at[1].set(jnp.where(parked, st.env_state.cartesian_states[1], es.cartesian_states[1]))
        es = es.replace(cartesian_states=cs)

        stack = ob.push_scan(st.stack, self._sense(k_noise, es), xp=jnp)
        speed = es.cartesian_states[0, 3]
        verts = jax.vmap(partial(get_vertices, length=self._env.params.length, width=self._env.params.width))(
            es.cartesian_states[:, jnp.array([0, 1, 4])])
        car_contact = collision(jnp.concatenate([verts[0], verts[1]], axis=-1)) & (rs.opp_type != OPP_SOLO)
        contact = es.collisions[0] | car_contact
        buffer_hit = self.wall_buffer_hit(es, st.margin)
        collided = contact | buffer_hit
        wrong_way_count = jnp.where(jnp.cos(es.frenet_states[0, 2]) < 0.0, st.wrong_way_count + 1, 0).astype(jnp.int32)
        wrong_way = (self.wrong_way_steps > 0) & (wrong_way_count >= self.wrong_way_steps)
        stall_count = jnp.where(speed < cp.STALL_SPEED, st.stall_count + 1, 0).astype(jnp.int32)
        stall = (self.stall_steps > 0) & (stall_count >= self.stall_steps)
        failed = collided | wrong_way | stall
        progress_m = rewards["agent_0"]
        reward = step_reward(progress_m, self.track_length, action, st.prev_action, failed)

        # race position: opponent ahead by start_gap + its progress - ours (m)
        ep_progress = st.ep_progress + progress_m / self.track_length
        opp_progress = rs.opp_progress + rewards["agent_1"] / self.track_length
        lead = (ep_progress - opp_progress) * self.track_length - rs.start_gap  # > 0: ego ahead
        new_pos = jnp.where(lead > PASS_HYSTERESIS_M, 1, jnp.where(lead < -PASS_HYSTERESIS_M, -1, rs.position))
        present = rs.opp_type != OPP_SOLO
        passed = present & (rs.position == -1) & (new_pos == 1)
        was_passed = present & (rs.position == 1) & (new_pos == -1)
        reward = reward + OVERTAKE_REWARD * (passed.astype(jnp.float32) - was_passed.astype(jnp.float32))

        ep_return = st.ep_return + reward
        ep_length = st.ep_length + 1
        lap_complete = ep_progress >= self.max_laps
        timeout = es.step >= self._env.params.max_steps
        done = failed | lap_complete | timeout
        finish = present & done & ~car_contact
        reward = reward + FINISH_REWARD * finish * jnp.sign(lead)
        ep_return = ep_return + FINISH_REWARD * finish * jnp.sign(lead)

        ego = st.replace(env_state=es, delay_buffer=delay_buffer, stack=stack, prev_action=action,
                         ep_return=ep_return, ep_length=ep_length, ep_speed_sum=st.ep_speed_sum + speed,
                         ep_progress=ep_progress, wrong_way_count=wrong_way_count, stall_count=stall_count)
        cont = rs.replace(ego=ego, opp_h=opp_h, opp_first=jnp.array(False), opp_prev_action=opp_prev,
                          opp_progress=opp_progress, position=new_pos,
                          overtakes=rs.overtakes + passed.astype(jnp.int32) - was_passed.astype(jnp.int32))
        obs = self._build_obs(stack, speed, action, self.privileged(cont))

        dp = es.dyn_params
        info = {
            "executed_command": executed,
            "returned_episode": done,
            "episode_return": ep_return,
            "episode_length": ep_length,
            "episode_mean_speed": ego.ep_speed_sum / ep_length,
            "episode_progress": ep_progress,
            "lap_complete": lap_complete,
            "collision": collided,
            "contact": contact,
            "car_contact": car_contact,
            "wall_buffer": buffer_hit & ~contact,
            "wrong_way": wrong_way,
            "stall": stall,
            "rollover": jnp.zeros((), bool),  # phase 2 does not model tipping
            "lat_accel": jnp.zeros((), jnp.float32),
            "margin": st.margin,
            "timeout": timeout,
            "speed": speed,
            "delay_k": st.delay_k,
            "steer_offset": st.steer_offset,
            "opp_type": rs.opp_type,
            "race_lead": lead,
            "won": present & done & (lead > 0) & ~car_contact,
            "overtakes": cont.overtakes,
            **{f"param_{n}": getattr(dp, n)[0] for n in ("mu", "m", "C_Sf", "C_Sr", "lf", "h", "I")},
        }
        reset_obs, reset_state = self.reset(k_reset)
        rs_out = jax.tree.map(lambda r, c: jnp.where(done, r, c), reset_state, cont)
        obs = jnp.where(done, reset_obs, obs)
        return obs, rs_out, reward, done, info
