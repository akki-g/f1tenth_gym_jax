"""Map handling for phase 1 (spec section 4.3).

Maps use the layout ``Track.from_track_name`` expects:
``<map dir>/<Name>/<Name>.yaml``, ``<Name>.png`` and ``<Name>_centerline.csv``.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
from PIL import Image

MAP_DIR_ENV = "F1TENTH_GYM_JAX_MAP_DIR"


def use_map_dir(path: os.PathLike) -> pathlib.Path:
    """Point the env's map search (and downloads) at ``path``."""
    path = pathlib.Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    os.environ[MAP_DIR_ENV] = str(path)
    return path


def write_map(
    map_dir: os.PathLike,
    name: str,
    free: np.ndarray,
    resolution: float,
    origin: tuple[float, float],
    centerline: np.ndarray,
) -> pathlib.Path:
    """Write one map.

    free: bool [H, W], row 0 at the *bottom* (y = origin_y), True = drivable.
    centerline: [N, 4] rows of (x, y, w_left, w_right) in metres, open loop
    (the last point is not a copy of the first).
    """
    track_dir = pathlib.Path(map_dir) / name
    track_dir.mkdir(parents=True, exist_ok=True)
    img = np.where(free, 255, 0).astype(np.uint8)[::-1]  # image row 0 is the top
    Image.fromarray(img).save(track_dir / f"{name}.png")
    (track_dir / f"{name}.yaml").write_text(
        f"image: {name}.png\n"
        f"resolution: {resolution}\n"
        f"origin: [{origin[0]}, {origin[1]}, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.45\nfree_thresh: 0.196\n"
    )
    np.savetxt(
        track_dir / f"{name}_centerline.csv",
        np.asarray(centerline, dtype=float),
        delimiter=",",
        header="x_m,y_m,w_tr_left_m,w_tr_right_m",
        fmt="%.6f",
    )
    return track_dir


# ---------------------------------------------------------------------------
# Narrow, duct-style generated tracks
# ---------------------------------------------------------------------------
GEN_RESOLUTION = 0.05  # m/px
GEN_WALL = 0.15  # m, wall thickness (thin walls keep collision checks cheap)
GEN_MARGIN = 1.0  # m of free space around the outer wall
GEN_WIDTH = (1.5, 2.5)  # m, corridor width, uniform per track
GEN_LENGTH = (40.0, 80.0)  # m, centerline length, uniform per track
GEN_MIN_RADIUS = 1.0  # m, tightest centerline corner allowed
GEN_CLEARANCE = 0.3  # m, extra gap required between non-adjacent corridor walls
GEN_RAD_SCALE = 0.5  # loop radius vs the example generator (shorter loops, same turn rate)


def _carracing_centerline(rng: np.random.RandomState, rad_scale: float = 1.0):
    """Closed centerline from examples/random_trackgen.py:create_track().

    Same checkpoint-chasing algorithm (adapted from the OpenAI Gym CarRacing
    generator; MIT, Auckley, O'Kelly, Sinha, Zheng), drawing from ``rng``
    instead of the global NumPy state. ``rad_scale`` shrinks the loop radius
    relative to the fixed turn rate, which gives a shorter loop for the same
    tightest corner. Returns [N, 2] in generator units, or None if the loop
    does not close.
    """
    import math

    CHECKPOINTS = 16
    SCALE = 6.0
    TRACK_RAD = 900 / SCALE * rad_scale
    TRACK_DETAIL_STEP = 21 / SCALE
    TRACK_TURN_RATE = 0.31

    start_alpha = 0.0
    checkpoints = []
    for c in range(CHECKPOINTS):
        alpha = 2 * math.pi * c / CHECKPOINTS + rng.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
        rad = rng.uniform(TRACK_RAD / 3, TRACK_RAD)
        if c == 0:
            alpha = 0
            rad = 1.5 * TRACK_RAD
        if c == CHECKPOINTS - 1:
            alpha = 2 * math.pi * c / CHECKPOINTS
            start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
            rad = 1.5 * TRACK_RAD
        checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))

    x, y, beta = 1.5 * TRACK_RAD, 0, 0
    dest_i = 0
    laps = 0
    track = []
    no_freeze = 2500
    visited_other_side = False
    while True:
        alpha = math.atan2(y, x)
        if visited_other_side and alpha > 0:
            laps += 1
            visited_other_side = False
        if alpha < 0:
            visited_other_side = True
            alpha += 2 * math.pi
        while True:
            failed = True
            while True:
                dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                if alpha <= dest_alpha:
                    failed = False
                    break
                dest_i += 1
                if dest_i % len(checkpoints) == 0:
                    break
            if not failed:
                break
            alpha -= 2 * math.pi
            continue
        r1x, r1y = math.cos(beta), math.sin(beta)
        p1x, p1y = -r1y, r1x
        dest_dx, dest_dy = dest_x - x, dest_y - y
        proj = r1x * dest_dx + r1y * dest_dy
        while beta - alpha > 1.5 * math.pi:
            beta -= 2 * math.pi
        while beta - alpha < -1.5 * math.pi:
            beta += 2 * math.pi
        prev_beta = beta
        proj *= SCALE
        if proj > 0.3:
            beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
        if proj < -0.3:
            beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
        x += p1x * TRACK_DETAIL_STEP
        y += p1y * TRACK_DETAIL_STEP
        track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
        if laps > 4:
            break
        no_freeze -= 1
        if no_freeze == 0:
            break

    i1, i2 = -1, -1
    i = len(track)
    while True:
        i -= 1
        if i == 0:
            return None
        pass_through_start = track[i][0] > start_alpha and track[i - 1][0] <= start_alpha
        if pass_through_start and i2 == -1:
            i2 = i
        elif pass_through_start and i1 == -1:
            i1 = i
            break
    track = track[i1 : i2 - 1]
    first_beta = track[0][1]
    glue = np.sqrt(
        np.square(math.cos(first_beta) * (track[0][2] - track[-1][2]))
        + np.square(math.sin(first_beta) * (track[0][3] - track[-1][3]))
    )
    if glue > TRACK_DETAIL_STEP:
        return None
    return np.array([(px, py) for (_, _, px, py) in track])


def _resample_closed(xy: np.ndarray, spacing: float) -> np.ndarray:
    closed = np.vstack([xy, xy[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(round(s[-1] / spacing)), 8)
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.c_[np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])]


def _smooth_closed(xy: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    pad = window
    out = []
    for k in range(2):
        v = np.concatenate([xy[-pad:, k], xy[:, k], xy[:pad, k]])
        out.append(np.convolve(v, kernel, mode="same")[pad:-pad])
    return np.c_[out[0], out[1]]


def _curvature_closed(xy: np.ndarray) -> np.ndarray:
    d1 = (np.roll(xy, -1, 0) - np.roll(xy, 1, 0)) / 2.0
    d2 = np.roll(xy, -1, 0) - 2 * xy + np.roll(xy, 1, 0)
    num = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    return num / np.maximum(np.hypot(d1[:, 0], d1[:, 1]) ** 3, 1e-12)


def generate_duct_track(seed: int):
    """One narrow track from ``seed``, or None if it fails the checks.

    Returns dict(free, resolution, origin, centerline, width, length, min_radius).
    """
    from scipy.spatial import cKDTree

    rng = np.random.RandomState(seed)
    raw = _carracing_centerline(rng, GEN_RAD_SCALE)
    if raw is None:
        return None
    width = rng.uniform(*GEN_WIDTH)
    length = rng.uniform(*GEN_LENGTH)

    # light smoothing of the generator's 3.5-unit tiles (a moving average
    # tightens bends, so keep it short), then scale to the target length
    xy = _resample_closed(raw, 1.0)
    xy = _smooth_closed(xy, 5)
    xy = _resample_closed(xy, 1.0)
    perimeter = np.sum(np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T))
    xy = xy * (length / perimeter)
    xy = _resample_closed(xy, 0.1)  # 10 cm centerline spacing
    xy = xy - xy.mean(axis=0)

    min_radius = 1.0 / np.max(np.abs(_curvature_closed(xy)))
    if min_radius < GEN_MIN_RADIUS:
        return None

    # corridors must not merge: points far apart along the track must be far apart in space
    n = len(xy)
    spacing = length / n
    tree = cKDTree(xy)
    need = width + 2 * GEN_WALL + GEN_CLEARANCE
    for i, j in tree.query_pairs(need):
        along = abs(i - j) * spacing
        along = min(along, length - along)
        if along > np.pi * need:  # not just neighbours on the same bend
            return None

    # render: free inside the corridor and outside the walls, walls between
    lo = xy.min(axis=0) - width / 2 - GEN_WALL - GEN_MARGIN
    hi = xy.max(axis=0) + width / 2 + GEN_WALL + GEN_MARGIN
    res = GEN_RESOLUTION
    w_px, h_px = np.ceil((hi - lo) / res).astype(int)
    cx = lo[0] + (np.arange(w_px) + 0.5) * res
    cy = lo[1] + (np.arange(h_px) + 0.5) * res
    gx, gy = np.meshgrid(cx, cy)  # row 0 = bottom (y = lo[1])
    dense = _resample_closed(xy, res / 2)
    dist, _ = cKDTree(dense).query(np.c_[gx.ravel(), gy.ravel()])
    dist = dist.reshape(h_px, w_px)
    wall = (dist > width / 2) & (dist <= width / 2 + GEN_WALL)
    free = ~wall

    # Start the centerline on a gentle bend towards the loop's inside, away
    # from straights. (Originally for the env's winding point, which the fork
    # now places by loop orientation; kept so the committed maps reproduce.)
    k = _curvature_closed(xy)
    turn = np.sign(np.sum(k))  # +1 for a counter-clockwise loop
    k_smooth = np.convolve(np.concatenate([k[-20:], k, k[:20]]), np.ones(41) / 41, mode="same")[20:-20]
    k_smooth = turn * k_smooth
    candidates = np.where(k_smooth > 0.02)[0]
    if len(candidates) == 0:
        return None
    start = int(candidates[np.argmin(np.abs(k_smooth[candidates] - 0.05))])
    xy = np.roll(xy, -start, axis=0)

    centerline = np.c_[xy, np.full(n, width / 2), np.full(n, width / 2)]
    return dict(
        free=free,
        resolution=res,
        origin=(float(lo[0]), float(lo[1])),
        centerline=centerline,
        width=float(width),
        length=float(length),
        min_radius=float(min_radius),
    )


# ---------------------------------------------------------------------------
# Lab-size tracks: short loops of air-duct walls, like the lab's carpet tracks
# ---------------------------------------------------------------------------
LAB_LENGTH = (10.0, 20.0)  # m, centerline length, uniform per track
LAB_WIDTH = (1.0, 1.7)  # m, mean corridor width, uniform per track
LAB_MIN_WIDTH = 0.9  # m, narrowest point anywhere
LAB_MAX_WIDTH = 2.0  # m, widest point anywhere
LAB_WALL = 0.30  # m, wall thickness (the ducts' diameter)
LAB_MIN_RADIUS = 0.8  # m, tightest centerline corner (full lock turns at ~0.75 m)
LAB_CLEARANCE = 0.1  # m, gap between the walls of corridors that run side by side
LAB_WANDER = 0.15  # m, amplitude of each smooth wave in a wall's drift
LAB_BUMPS = (0, 3)  # local bulges per wall (inclusive)
LAB_BUMP_AMP = (0.10, 0.35)  # m, into or out of the corridor
LAB_BUMP_SIGMA = (0.25, 0.9)  # m, along the track
LAB_SQUARENESS = (2.0, 4.0)  # superellipse exponent of the base loop: 2 ellipse, 4 rounded rectangle
LAB_CORNER_WIDTH = (0.9, 1.1)  # m: corridors pinch to this in bends (the lab tracks are tight in corners)
LAB_PINCH_CURVATURE = (0.3, 0.8)  # 1/m: no pinch below, full pinch above (radius 3.3 m -> 1.25 m)


def _blob_centerline(rng: np.random.RandomState, length: float) -> np.ndarray:
    """Closed star-shaped loop: a superellipse (ellipse to rounded rectangle,
    i.e. straights and room-corner turns) times 1 + sum_k a_k cos(k theta + phi_k)
    (bulges and dents), stretched and rotated at random, run clockwise or
    counter-clockwise, scaled to ``length``; 5 cm spacing."""
    th = np.linspace(0.0, 2 * np.pi, 1440, endpoint=False)
    p = rng.uniform(*LAB_SQUARENESS)
    base = (np.abs(np.cos(th)) ** p + np.abs(np.sin(th)) ** p) ** (-1.0 / p)
    wobble = np.ones_like(th)
    for k in range(2, 2 + rng.randint(2, 6)):
        wobble += rng.uniform(0.0, 0.30 / k**0.7) * np.cos(k * th + rng.uniform(0, 2 * np.pi))
    r = base * wobble
    xy = np.c_[r * np.cos(th) * rng.uniform(1.0, 1.8), r * np.sin(th)]
    a = rng.uniform(0, 2 * np.pi)
    xy = xy @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    if rng.rand() < 0.5:
        xy = xy[::-1]
    xy = _resample_closed(xy, 0.01)
    perimeter = np.sum(np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T))
    return _resample_closed(xy * (length / perimeter), 0.05)


def _wall_drift(rng: np.random.RandomState, s: np.ndarray, length: float) -> np.ndarray:
    """One wall's offset from the nominal half-width along arc length ``s``:
    1-3 smooth periodic waves plus a few local bulges (periodic in s)."""
    out = np.zeros_like(s)
    for _ in range(rng.randint(1, 4)):
        cycles = max(1, int(round(length / rng.uniform(2.0, 6.0))))
        out += rng.uniform(0.0, LAB_WANDER) * np.sin(2 * np.pi * cycles * s / length + rng.uniform(0, 2 * np.pi))
    for _ in range(rng.randint(LAB_BUMPS[0], LAB_BUMPS[1] + 1)):
        d = np.abs(s - rng.uniform(0, length))
        d = np.minimum(d, length - d)
        amp = rng.uniform(*LAB_BUMP_AMP) * rng.choice([-1.0, 1.0])
        out += amp * np.exp(-0.5 * (d / rng.uniform(*LAB_BUMP_SIGMA)) ** 2)
    return out


def generate_lab_track(seed: int):
    """One lab-size track from ``seed``, or None if it fails the checks.

    The two walls are placed independently around a blob centerline: each
    drifts and bulges on its own, so the width varies along the track and one
    wall can curve while the other runs straight. Written centerline: the
    corridor's midline, with its local half-width on both sides.

    Returns dict(free, resolution, origin, centerline, width, length,
    min_radius, min_width, max_width).
    """
    from scipy.spatial import cKDTree

    rng = np.random.RandomState(seed)
    length = rng.uniform(*LAB_LENGTH)
    xy = _blob_centerline(rng, length)
    k = _curvature_closed(xy)  # signed: + turns left along the loop
    min_radius = 1.0 / np.max(np.abs(k))
    if min_radius < LAB_MIN_RADIUS:
        return None

    n = len(xy)
    s = np.arange(n) * (length / n)
    tangent = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.c_[-tangent[:, 1], tangent[:, 0]]  # points left
    width = rng.uniform(*LAB_WIDTH)
    # corridors pinch in bends: from the mean width toward the corner width as the bend tightens
    k_abs = np.convolve(np.abs(np.concatenate([k[-10:], k, k[:10]])), np.ones(21) / 21, mode="same")[10:-10]
    pinch = np.clip((k_abs - LAB_PINCH_CURVATURE[0]) / (LAB_PINCH_CURVATURE[1] - LAB_PINCH_CURVATURE[0]), 0.0, 1.0)
    nominal = width + (np.minimum(rng.uniform(*LAB_CORNER_WIDTH), width) - width) * pinch
    w_left = nominal / 2 + _wall_drift(rng, s, length)
    w_right = nominal / 2 + _wall_drift(rng, s, length)
    # the wall on the inside of a bend must stay short of the bend's centre
    with np.errstate(divide="ignore"):
        inner_max = 0.85 / np.abs(k)
    w_left = np.where(k > 0, np.minimum(w_left, inner_max), w_left)
    w_right = np.where(k < 0, np.minimum(w_right, inner_max), w_right)
    excess = np.maximum(w_left + w_right - LAB_MAX_WIDTH, 0.0)
    w_left, w_right = w_left - excess / 2, w_right - excess / 2
    deficit = np.maximum(LAB_MIN_WIDTH - (w_left + w_right), 0.0)
    w_left, w_right = w_left + deficit / 2, w_right + deficit / 2
    if np.any((k > 0) & (w_left > inner_max)) or np.any((k < 0) & (w_right > inner_max)):
        return None
    if np.min(np.minimum(w_left, w_right)) < 0.2:  # keep the corridor around the centerline
        return None

    # corridors must not merge: far-apart (along the track) points need two walls and a gap between them
    reach = np.maximum(w_left, w_right)
    tree = cKDTree(xy)
    for i, j in tree.query_pairs(2 * reach.max() + 2 * LAB_WALL + LAB_CLEARANCE):
        along = abs(i - j) * (length / n)
        along = min(along, length - along)
        need = reach[i] + reach[j] + 2 * LAB_WALL + LAB_CLEARANCE
        if along > np.pi * need and np.hypot(*(xy[i] - xy[j])) < need:
            return None

    # render: a pixel is wall when its offset from the nearest centerline point
    # lies in the wall band beyond that point's left or right face
    res = GEN_RESOLUTION
    pad = reach.max() + LAB_WALL + GEN_MARGIN
    lo, hi = xy.min(axis=0) - pad, xy.max(axis=0) + pad
    w_px, h_px = np.ceil((hi - lo) / res).astype(int)
    gx, gy = np.meshgrid(lo[0] + (np.arange(w_px) + 0.5) * res, lo[1] + (np.arange(h_px) + 0.5) * res)
    pix = np.c_[gx.ravel(), gy.ravel()]
    _, j = tree.query(pix)
    e = np.einsum("ij,ij->i", pix - xy[j], normal[j])
    wall = ((e > w_left[j]) & (e <= w_left[j] + LAB_WALL)) | ((e < -w_right[j]) & (e >= -w_right[j] - LAB_WALL))
    free = ~wall.reshape(h_px, w_px)

    mid = xy + normal * ((w_left - w_right) / 2)[:, None]
    half = (w_left + w_right) / 2
    start = int(np.argmin(np.abs(k)))  # start on the straightest part
    mid, half = np.roll(mid, -start, axis=0), np.roll(half, -start)
    keep = slice(None, None, 2)  # 10 cm centerline spacing
    centerline = np.c_[mid[keep], half[keep], half[keep]]
    return dict(
        free=free,
        resolution=res,
        origin=(float(lo[0]), float(lo[1])),
        centerline=centerline,
        width=float(width),
        length=float(length),
        min_radius=float(min_radius),
        min_width=float(np.min(w_left + w_right)),
        max_width=float(np.max(w_left + w_right)),
    )


def lab_name(seed: int) -> str:
    return f"Lab{seed}"


# ---------------------------------------------------------------------------
# Room tracks: a duct wall around a room with an island inside (the race layout)
# ---------------------------------------------------------------------------
ROOM_SIZE = ((5.0, 11.0), (4.0, 8.0))  # m, room (x, y) extent of the outer wall's inner face
ROOM_CORNER = (1.5, 3.0)  # m, corner radius (round enough that the turnarounds stay tight diagonally)
ROOM_GAP = ((0.9, 1.4), (0.9, 3.0))  # m, wall-face gap beside the island: at its ends (the turnarounds,
# tight like the lab tracks' corners) and along its sides (straights, up to open areas)
ISLAND_MIN = (1.5, 0.4)  # m, smallest island length and width
ROOM_MIN_GAP = 0.9  # m, narrowest corridor along a ray from the island centre (wall face to wall face)
ROOM_MIN_WIDTH = 0.85  # m, narrowest corridor in the rendered map (twice the clearance on the centerline)
ROOM_LENGTH = (12.0, 34.0)  # m, centerline length
ROOM_MIN_RADIUS = 0.8  # m


def _periodic_noise(rng, th, amp, waves=(2, 6)):
    out = np.zeros_like(th)
    for k in range(waves[0], waves[1] + 1):
        out += rng.uniform(0, amp) / k * np.cos(k * th + rng.uniform(0, 2 * np.pi))
    return out


def _dent(th, center, width, depth):
    d = np.angle(np.exp(1j * (th - center)))
    return depth * np.exp(-0.5 * (d / width) ** 2)


def build_room_track(room_r, island_r, centre, reverse=False):
    """Render a room track from the outer wall's inner face and the island's outer face,
    both as radii over angle (720 samples) around ``centre`` (inside the island).
    Corridor centerline: halfway between the faces along each ray, smoothed; its
    half-width: the clearance to the nearest wall in the rendered map. Returns the
    generate_*_track dict, or None if the corridor is too narrow or too tight."""
    from scipy.ndimage import distance_transform_edt

    th = np.linspace(0.0, 2 * np.pi, len(room_r), endpoint=False)
    if np.min(room_r - island_r) < ROOM_MIN_GAP:
        return None
    mid_r = (room_r + island_r) / 2
    xy = centre + np.c_[mid_r * np.cos(th), mid_r * np.sin(th)]
    xy = _smooth_closed(_resample_closed(xy, 0.05), 9)
    if reverse:
        xy = xy[::-1]
    xy = _resample_closed(xy, 0.1)
    length = float(np.sum(np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T)))
    min_radius = 1.0 / np.max(np.abs(_curvature_closed(_smooth_closed(xy, 3))))

    res = GEN_RESOLUTION
    reach = room_r.max() + LAB_WALL + GEN_MARGIN
    lo, hi = centre - reach, centre + reach
    w_px, h_px = np.ceil((hi - lo) / res).astype(int)
    gx, gy = np.meshgrid(lo[0] + (np.arange(w_px) + 0.5) * res, lo[1] + (np.arange(h_px) + 0.5) * res)
    rho = np.hypot(gx - centre[0], gy - centre[1])
    phi = np.mod(np.arctan2(gy - centre[1], gx - centre[0]), 2 * np.pi)
    r_room = np.interp(phi, th, room_r, period=2 * np.pi)
    r_isl = np.interp(phi, th, island_r, period=2 * np.pi)
    # radial bands approximate the walls' 0.30 m thickness (the faces are near-convex)
    wall = ((rho >= r_room) & (rho <= r_room + LAB_WALL)) | ((rho <= r_isl) & (rho >= r_isl - LAB_WALL))
    free = ~wall
    corridor = (rho < r_room) & (rho > r_isl)
    clear = distance_transform_edt(corridor) * res  # distance to the nearest wall face, m
    ij = np.clip(np.round((xy - lo) / res - 0.5).astype(int), 0, [w_px - 1, h_px - 1])
    half = clear[ij[:, 1], ij[:, 0]]
    k = _curvature_closed(xy)
    start = int(np.argmin(np.abs(_smooth_closed(np.c_[k, k], 15)[:, 0])))
    xy, half = np.roll(xy, -start, axis=0), np.roll(half, -start)
    return dict(free=free, resolution=res, origin=(float(lo[0]), float(lo[1])),
                centerline=np.c_[xy, half, half], width=float(2 * np.median(half)), length=length,
                min_radius=float(min_radius), min_width=float(2 * half.min()), max_width=float(2 * half.max()))


def generate_room_track(seed: int):
    """One room track from ``seed``, or None if it fails the checks: a rounded-rectangle
    room (5-11 x 4-8 m) whose duct wall wanders and may dent inward, around an
    off-centre elongated island (also wandering, maybe with a bulge), so the corridor
    runs from narrow passages to wide open areas; clockwise or not."""
    rng = np.random.RandomState(seed)
    th = np.linspace(0.0, 2 * np.pi, 720, endpoint=False)
    W, H = rng.uniform(*ROOM_SIZE[0]), rng.uniform(*ROOM_SIZE[1])
    if not 1.0 <= W / H <= 2.2:
        return None
    corner = min(rng.uniform(*ROOM_CORNER), H / 2 - 0.1, W / 2 - 0.1)
    # island: a superellipse filling the room minus a gap to each wall (so the corridor runs from
    # ~0.9 m passages to ~3 m open areas), slightly rotated; its centre is the polar origin
    gl, gr = rng.uniform(*ROOM_GAP[0], size=2)
    gb, gt = rng.uniform(*ROOM_GAP[1], size=2)
    Li, Wi = W - gl - gr, H - gb - gt
    if Li < ISLAND_MIN[0] or Wi < ISLAND_MIN[1]:
        return None
    Wi = min(Wi, rng.uniform(ISLAND_MIN[1], max(Wi, ISLAND_MIN[1] + 1e-3)))  # often thinner than the room allows
    ang = rng.uniform(-0.35, 0.35)
    off = np.array([(gl - gr) / 2, (gb - gt) / 2])
    p = rng.uniform(2.0, 4.0)
    a = th - ang
    island = (np.abs(np.cos(a) / (Li / 2)) ** p + np.abs(np.sin(a) / (Wi / 2)) ** p) ** (-1.0 / p)
    island *= 1.0 + _periodic_noise(rng, th, 0.08)
    if rng.rand() < 0.5:  # a bulge on one side
        island += _dent(th, rng.uniform(0, 2 * np.pi), rng.uniform(0.2, 0.5), rng.uniform(0.2, 0.6))
    # room face, seen from the island centre: rays from (off) to the rounded rectangle centred at 0
    d = np.c_[np.cos(th), np.sin(th)]
    t = np.linspace(0.0, 20.0, 4001)[:, None, None]
    pts = off[None, None] + t * d[None]  # [steps, 720, 2]
    qx = np.maximum(np.abs(pts[..., 0]) - (W / 2 - corner), 0.0)
    qy = np.maximum(np.abs(pts[..., 1]) - (H / 2 - corner), 0.0)
    inside = (np.abs(pts[..., 0]) <= W / 2) & (np.abs(pts[..., 1]) <= H / 2) & (np.hypot(qx, qy) <= corner)
    room = t[np.argmax(~inside, axis=0), 0, 0]
    room += _periodic_noise(rng, th, 0.25)
    if rng.rand() < 0.5:  # a dent pushed in from the outer wall
        room -= _dent(th, rng.uniform(0, 2 * np.pi), rng.uniform(0.15, 0.4), rng.uniform(0.3, 0.9))
    g = build_room_track(room, island, off, reverse=rng.rand() < 0.5)
    if g is None or not ROOM_LENGTH[0] <= g["length"] <= ROOM_LENGTH[1] or g["min_radius"] < ROOM_MIN_RADIUS:
        return None
    if g["min_width"] < ROOM_MIN_WIDTH:  # walls oblique to the rays can pinch the corridor below the gap
        return None
    return g


def room_name(seed: int) -> str:
    return f"Room{seed}"


# ---------------------------------------------------------------------------
# Duct layouts: every wall is a 0.30 m duct along a polyline (closed loops and free-ended walls)
# ---------------------------------------------------------------------------
def _seg(p0, p1, step=0.02):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(int(np.ceil(np.hypot(*(p1 - p0)) / step)), 1)
    return p0 + np.linspace(0.0, 1.0, n + 1)[:, None] * (p1 - p0)


def _arc(c, r, a0, a1, step=0.02):
    n = max(int(np.ceil(abs(a1 - a0) * r / step)), 2)
    a = np.linspace(a0, a1, n + 1)
    return np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a)]


def _rounded_rect(x0, y0, x1, y1, r):
    """Closed polyline of a rounded rectangle, counter-clockwise from the bottom edge."""
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    return np.vstack([
        _seg((x0 + r, y0), (x1 - r, y0)), _arc((x1 - r, y0 + r), r, -np.pi / 2, 0.0),
        _seg((x1, y0 + r), (x1, y1 - r)), _arc((x1 - r, y1 - r), r, 0.0, np.pi / 2),
        _seg((x1 - r, y1), (x0 + r, y1)), _arc((x0 + r, y1 - r), r, np.pi / 2, np.pi),
        _seg((x0, y1 - r), (x0, y0 + r)), _arc((x0 + r, y0 + r), r, np.pi, 1.5 * np.pi)])


def _periodic_spline(points, spacing=0.1):
    from scipy.interpolate import CubicSpline

    p = np.vstack([points, points[:1]])
    t = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    dense = CubicSpline(t, p, bc_type="periodic")(np.linspace(0.0, t[-1], int(t[-1] / 0.01), endpoint=False))
    return _resample_closed(dense, spacing)


def _rugged_wall(poly, rng, amp, closed):
    """A duct line made uneven: moved along its normal by 3 smooth waves (0.6-3 m wavelength, up
    to ``amp`` each) and 0-2 local kinks (1-2 x amp); an open wall's offset fades to 0 over its
    last 0.5 m at both ends, so walls stay attached where they meet."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(poly, axis=0).T))])
    length = s[-1]
    t = np.gradient(poly, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    normal = np.c_[-t[:, 1], t[:, 0]]
    off = np.zeros(len(poly))
    for _ in range(3):
        wl, a, ph = rng.uniform(0.6, 3.0), rng.uniform(0.0, amp), rng.uniform(0, 2 * np.pi)
        off += a * np.sin(2 * np.pi * (max(1, round(length / wl)) * s / length if closed else s / wl) + ph)
    for _ in range(rng.randint(0, 3)):
        d = np.abs(s - rng.uniform(0, length))
        d = np.minimum(d, length - d) if closed else d
        off += rng.choice([-1.0, 1.0]) * rng.uniform(1.0, 2.0) * amp * np.exp(-0.5 * (d / rng.uniform(0.15, 0.4)) ** 2)
    if not closed:
        off *= np.clip(np.minimum(s, length - s) / 0.5, 0.0, 1.0)
    return poly + normal * off[:, None]


def build_duct_track(walls, waypoints, reverse=False, rugged=0.0, rng=None, closed=None):
    """Render ducts (polylines along the ducts' centre lines; 0.30 m thick) and a centerline: a
    periodic spline through ``waypoints``, starting at waypoint 0 (``reverse``: same start, other
    direction). ``rugged`` > 0: every wall made uneven by up to that much (_rugged_wall; ``closed``
    flags which walls are loops). Half-width: the clearance to the nearest wall at each centerline point."""
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import cKDTree

    if rugged > 0:
        closed = closed or [True] * len(walls)
        walls = [_rugged_wall(w, rng, rugged, c) for w, c in zip(walls, closed)]
    pts = np.vstack(walls)
    res = GEN_RESOLUTION
    lo, hi = pts.min(axis=0) - LAB_WALL - GEN_MARGIN, pts.max(axis=0) + LAB_WALL + GEN_MARGIN
    w_px, h_px = np.ceil((hi - lo) / res).astype(int)
    gx, gy = np.meshgrid(lo[0] + (np.arange(w_px) + 0.5) * res, lo[1] + (np.arange(h_px) + 0.5) * res)
    d, _ = cKDTree(pts).query(np.c_[gx.ravel(), gy.ravel()])
    free = (d > LAB_WALL / 2).reshape(h_px, w_px)
    xy = _periodic_spline(np.asarray(waypoints, float))
    if reverse:
        xy = np.vstack([xy[:1], xy[:0:-1]])
    return _measure_track(free, lo, res, xy)


def _measure_track(free, lo, res, xy):
    """The generate_*_track dict for a rendered map ``free`` (origin ``lo``, resolution ``res``)
    and a closed centerline ``xy``: corridor widths across along each normal, clearance, radius."""
    from scipy.ndimage import distance_transform_edt

    h_px, w_px = free.shape
    # corridor cross-section: march along the normal to the first wall on each side
    t = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    normal = np.c_[-t[:, 1], t[:, 0]]  # left
    steps = np.arange(0.0, 4.0, res / 2)

    def reach(sign):
        q = xy[:, None, :] + sign * steps[None, :, None] * normal[:, None, :]
        ij = np.round((q - lo) / res - 0.5).astype(int)
        inside = (ij[..., 0] >= 0) & (ij[..., 0] < w_px) & (ij[..., 1] >= 0) & (ij[..., 1] < h_px)
        hit = ~free[np.clip(ij[..., 1], 0, h_px - 1), np.clip(ij[..., 0], 0, w_px - 1)] | ~inside
        return steps[np.argmax(hit, axis=1)]

    w_left, w_right = reach(1.0), reach(-1.0)
    width = w_left + w_right
    length = float(np.sum(np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T)))
    min_radius = 1.0 / np.max(np.abs(_curvature_closed(_smooth_closed(xy, 3))))
    clear = distance_transform_edt(free) * res - res / 2
    ij = np.clip(np.round((xy - lo) / res - 0.5).astype(int), 0, [w_px - 1, h_px - 1])
    return dict(free=free, resolution=res, origin=(float(lo[0]), float(lo[1])), centerline=np.c_[xy, w_left, w_right],
                width=float(np.median(width)), length=length, min_radius=float(min_radius),
                min_width=float(width.min()), max_width=float(width.max()), p95_width=float(np.percentile(width, 95)),
                min_clearance=float(clear[ij[:, 1], ij[:, 0]].min()))


def race_layout(W_gaps, island, c1, c2, pocket, hook_gap, g_top, corner, hook=True, spur=True, mirror=False):
    """The last race's layout family (last_track.png): a rounded-rectangle room; a capsule island
    in its lower part whose left side carries on up as a hook that curls right to a free end; a
    spur from the right wall between them. The track runs up the right side, left under the spur,
    U-turns around its end, right between spur and hook, U-turns around the hook's end, left along
    the top, down the left side and round the bottom. All sizes are wall face to wall face (m):
    W_gaps = (left, right, bottom) corridors, island = (width, straight length), c1 / c2 the
    channels under / over the spur, pocket the turnaround beside the spur's end, hook_gap the
    turnaround beside the hook's end, g_top the top corridor. Returns (walls, waypoints), the
    waypoints counter-clockwise from the right side beside the island's top (``mirror``: hook
    on the right, clockwise)."""
    T = LAB_WALL / 2
    g_l, g_r, g_b = W_gaps
    w_i, straight = island
    r_i = w_i / 2
    W = g_l + w_i + g_r
    yi_top = g_b + w_i + straight  # island's top face
    y_s = yi_top + c1 + T  # spur centre line
    y_h = (y_s + T + c2 + T) if spur else (yi_top + c1 + T)  # hook centre line
    H = (y_h + T + g_top) if hook else (y_s + T + g_top)
    outer = _rounded_rect(-T, -T, W + T, H + T, corner + T)
    xc = g_l + r_i
    walls = [outer, _rounded_rect(g_l + T, g_b + T, W - g_r - T, yi_top - T, r_i - T)]
    x_v = g_l + T  # the hook's vertical part continues the island's left side
    x_he = W - hook_gap - T  # hook end (centre)
    x_se = g_l + LAB_WALL + pocket + T  # spur end (centre)
    if hook:
        r_h = 0.5
        y0 = yi_top - r_i  # where the island's straight left side ends
        walls.append(np.vstack([_seg((x_v, y0), (x_v, y_h - r_h)), _arc((x_v + r_h, y_h - r_h), r_h, np.pi, np.pi / 2),
                                _seg((x_v + r_h, y_h), (x_he, y_h))]))
    if spur:
        walls.append(_seg((W + T, y_s), (x_se, y_s)))
    xr, xl, yb = W - g_r / 2, g_l / 2, g_b / 2
    y_top = H - g_top / 2
    m = 0.4  # the U-turns enter and leave their channels this far from the outer walls (a driver's line)

    def u_turn(cx, cy, r, left):
        a = np.radians([-90, -135, 180, 135, 90]) if left else np.radians([-90, -45, 0, 45, 90])
        return [(cx + r * np.cos(x), cy + r * np.sin(x)) for x in a]

    pts = [(xr, yi_top - 1.0)]
    if spur:  # a quarter-turn (open floor beside the island's top) into the channel under the spur
        rj = min(1.0, max(0.6, xr - (x_se + 0.5)))
        cj = (xr - rj, yi_top + m - rj)
        pts += [(cj[0] + rj * np.cos(a), cj[1] + rj * np.sin(a)) for a in np.radians([0, 45])]
    if spur:
        top_of_turn = (y_h - T - m) if hook else (y_s + T + 1.0)
        r1 = min((top_of_turn - (yi_top + m)) / 2, pocket - 0.35)
        pts += u_turn(x_se - T, y_s, r1, left=True)  # around the spur's end
        if hook:
            r2 = min(y_h - (y_s + T + m), H - m - y_h, hook_gap - 0.35)
            pts += u_turn(x_he + T, y_h, r2, left=False)  # around the hook's end
    elif hook:
        r2 = min(y_h - (yi_top + m), H - m - y_h, hook_gap - 0.35)
        pts += [(xr, y_h - r2)] + u_turn(x_he + T, y_h, r2, left=False)[1:]
    else:
        pts += [(xr, H - corner)]
    pts += [(xc, y_top), (xl + 0.3 * corner, H - 0.3 * corner - g_top / 2), (xl, H - corner - g_top / 2),
            (xl, (yi_top + g_b) / 2), (xl, g_b + 0.3 * corner), (xc - 0.3 * w_i, yb), (xc + 0.3 * w_i, yb),
            (xr, g_b + 0.3 * corner), (xr, (yi_top + g_b) / 2)]
    if mirror:
        walls = [np.c_[W - w[:, 0], w[:, 1]] for w in walls]
        pts = [(W - x, y) for x, y in pts]
    return walls, pts


def last_race_track(reverse: bool = False):
    """The previous race's track from its layout sketch (last_track.png, not to scale), sized so
    every corridor is 0.9-2.0 m: a 6.6 x 11.7 m room, a 3.2 m wide island, 1.2 m channels under
    and over the spur, 1.3 / 1.75 m turnarounds at the spur's and the hook's ends. Starts on the
    right side beside the island's top (the sketch's start line)."""
    walls, pts = race_layout((1.5, 1.9, 1.8), (3.2, 2.2), 1.2, 1.2, 1.3, 1.75, 1.5, 2.4)
    return build_duct_track(walls, pts, reverse=reverse)


LAST_RACE_MAPS = ("LastRace2", "LastRace2Rev")
RACE_LIKE_CORRIDOR = (0.9, 2.0)  # m, narrowest / widest corridor of the race-like tracks


def generate_race_like_track(seed: int):
    """A track like the last race's (race_layout with random sizes; mirrored or not, either
    direction; usually spur and hook, sometimes only one), or None if the corridor narrows
    below 0.9 m, the centerline turns tighter than 0.75 m or comes within 0.27 m of a wall,
    or the lap is not 14-40 m."""
    rng = np.random.RandomState(seed)
    variant = rng.choice(3, p=(0.6, 0.2, 0.2))
    hook, spur = variant != 2, variant != 1
    gaps = (rng.uniform(1.0, 1.6), rng.uniform(1.1, 1.8), rng.uniform(1.1, 1.7))  # on the narrow side of 0.9-2.0 m
    c1, c2 = rng.uniform(1.05, 1.4, size=2)  # wide enough for U-turns the car can make (radius >= ~0.8 m)
    pocket, hook_gap = rng.uniform(1.2, 1.6), rng.uniform(1.3, 1.8)
    g_top, corner = rng.uniform(1.0, 1.6), rng.uniform(1.5, 2.4)
    # the island is wide enough that the hook reaches 0.8-1.6 m past the spur's end
    reach = (0.8 + hook_gap + pocket + 2 * LAB_WALL - gaps[1]) if (spur and hook) else 1.2
    island = (float(np.clip(reach + rng.uniform(0.0, 0.8), 1.2, 3.6)), rng.uniform(1.0, 5.5))
    mirror, reverse = rng.rand() < 0.5, rng.rand() < 0.5
    rugged = rng.uniform(0.03, 0.07) if rng.rand() < 0.6 else 0.0  # most tracks: uneven duct walls
    W = sum(gaps[:2]) + island[0]
    if spur and hook and (W - hook_gap - LAB_WALL / 2) - (gaps[0] + LAB_WALL + pocket + LAB_WALL / 2) < 0.8:
        return None  # the hook must reach past the spur's end
    walls, pts = race_layout(gaps, island, c1, c2, pocket, hook_gap, g_top, corner, hook, spur, mirror)
    closed = [True, True] + [False] * (len(walls) - 2)

    def ok(g):
        # every corridor is drawn within 0.9-2.0 m above; check the narrowest point as rendered (where a
        # corridor meets a turn, the diagonal across it is wider, as on the real track); the centerline
        # keeps half the car, the wall buffer and 5 cm from every wall; turns the car can make; a lab lap
        return (g["min_width"] >= RACE_LIKE_CORRIDOR[0] and g["min_clearance"] >= 0.27
                and g["min_radius"] >= 0.75 and 14.0 <= g["length"] <= 40.0)

    # uneven walls: the drawn amplitude, then gentler ones, before falling back to straight walls
    for amp in ((rugged, 0.7 * rugged, 0.5 * rugged) if rugged > 0 else ()) + (0.0,):
        g = build_duct_track(walls, pts, reverse=reverse, rugged=amp, rng=np.random.RandomState(seed + 7919),
                             closed=closed)
        if ok(g):
            g["rugged"] = float(amp)
            return g
    return None


# ---------------------------------------------------------------------------
# Hand-drawn tracks (the team's sketches, digitized: walls, centerline, scale in drawn_tracks.npz)
# ---------------------------------------------------------------------------
DRAWN_FILE = pathlib.Path(__file__).resolve().parent / "drawn_tracks.npz"
DRAWN_BASES = ("Drawn1", "Drawn2", "Drawn3", "Drawn4")  # IMG_0055, then IMG_0056 top / middle / bottom
# per drawing: the extra scale that brings its narrowest neck to ~0.85-0.95 m (the open parts, which
# would come out 3-7 m wide, are capped at DRAWN_MAX_WIDTH); variant letter: (scale, mirrored),
# a-c train, h held-out
DRAWN_BASE_SCALE = {"Drawn1": 1.6, "Drawn2": 1.13, "Drawn3": 1.8, "Drawn4": 1.35}
DRAWN_MAX_WIDTH = 2.0  # m, the corridor cap (the open parts of the sketches are much wider)
DRAWN_VARIANTS = {"a": (1.0, False), "b": (1.1, True), "c": (1.2, False), "h": (1.05, True)}


def drawn_track(name: str):
    """A track from a hand drawing, e.g. "Drawn3b": the drawn lines become 0.30 m ducts (the
    outer wall's pen-lift gaps were bridged when digitizing), scaled so the corridor's typical
    width is ~1.45 m and its narrowest 10% >= 0.85 m, times DRAWN_BASE_SCALE (narrowest neck
    ~0.85-0.95 m) and the variant's scale; mirrored for some variants; corridor capped at
    DRAWN_MAX_WIDTH. Centerline: the path around the island through the corridor's middle."""
    from scipy import ndimage as ndi

    base, var = name[:-1], name[-1]
    k, mirror = DRAWN_VARIANTS[var]
    with np.load(DRAWN_FILE) as z:
        shape = tuple(int(v) for v in z[f"{base}_shape"])
        walls = np.unpackbits(z[f"{base}_walls"])[: shape[0] * shape[1]].reshape(shape).astype(bool)
        path = z[f"{base}_path"].astype(float)
        scale = float(z[f"{base}_scale"]) * DRAWN_BASE_SCALE[base] * k
        ink_half = float(z[f"{base}_ink_half"])
    if mirror:
        walls = walls[:, ::-1]
        path[:, 1] = shape[1] - 1 - path[:, 1]
    res = GEN_RESOLUTION
    duct = ndi.distance_transform_edt(~walls) * scale <= max(LAB_WALL / 2 - ink_half * scale, res / 2)
    H, W = shape
    pad = LAB_WALL + GEN_MARGIN
    lo = np.array([-pad, -pad])
    w_px, h_px = np.ceil((np.array([W * scale, H * scale]) + 2 * pad) / res).astype(int)
    gx, gy = np.meshgrid(lo[0] + (np.arange(w_px) + 0.5) * res, lo[1] + (np.arange(h_px) + 0.5) * res)
    col, row = gx / scale, H - gy / scale  # drawing pixels (row 0 at the top of the photo)
    inside = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    free = ~(inside & duct[np.clip(row.astype(int), 0, H - 1), np.clip(col.astype(int), 0, W - 1)])
    xy = np.c_[path[:, 1] * scale, (H - path[:, 0]) * scale]
    xy = _resample_closed(_smooth_closed(_resample_closed(xy, 0.05), 31), 0.1)  # 1.5 m smoothing: a progress reference
    # The sketches' corridors vary a lot in width; scaled so the necks are drivable, the open parts
    # come out 3-7 m wide. Cap them at DRAWN_MAX_WIDTH: keep the corridor within half of it of the
    # centerline, then wall in what is left with a LAB_WALL band (narrow parts keep the drawn walls).
    dense = _resample_closed(xy, res / 2)
    on_line = np.zeros_like(free)
    ij = np.clip(np.round((dense - lo) / res - 0.5).astype(int), 0, [w_px - 1, h_px - 1])
    on_line[ij[:, 1], ij[:, 0]] = True
    labels, _ = ndi.label(free & (ndi.distance_transform_edt(~on_line) * res <= DRAWN_MAX_WIDTH / 2 - res))
    corridor = labels == labels[ij[0, 1], ij[0, 0]]
    free = corridor | (ndi.distance_transform_edt(~corridor) * res > LAB_WALL)
    g = _measure_track(free, lo, res, xy)
    g["scale"] = scale
    return g


DRAWN_TRAIN_MAPS = tuple(f"{b}{v}" for b in DRAWN_BASES for v in "abc")
DRAWN_HELDOUT_MAPS = ("Drawn1h", "Drawn3h")


def race_like_name(seed: int) -> str:
    return f"RaceLike{seed}"  # seeds 6000+: wider, straight walls; 20000+: narrower, mostly rugged


# ---------------------------------------------------------------------------
# Speed-trap tracks: lab-size loops with one long straight that opens up (2.0-3.0 m)
# ---------------------------------------------------------------------------
TRAP_LENGTH = (14.0, 26.0)  # m
TRAP_STRETCH = (2.0, 3.0)  # length / width of the base loop: long straights
TRAP_WIDTH = (0.9, 1.4)  # m, mean corridor width off the trap
TRAP_STRAIGHT_WIDTH = (2.0, 3.0)  # m, the trap straight's width
TRAP_MIN_STRAIGHT = 3.0  # m
TRAP_SQUARENESS = (4.0, 6.0)  # superellipse exponent: squarer corners than the lab loops
TRAP_MIN_RADIUS = 0.75  # m: corners as tight as the car turns


def generate_trap_track(seed: int):
    """One speed-trap track, or None if it fails the checks: an elongated loop (superellipse,
    rounded rectangle-ish, mild bulges) 14-26 m long; corners pinch to 0.9-1.1 m like the lab
    loops; on the longest straight the outer wall pushes out so the corridor reaches U(2.0, 3.0) m,
    with ramps over 30% of the straight (1-2 m), while the inner wall stays in line; elsewhere U(1.0, 1.6) m with
    walls that wander a little. Starts at the beginning of the trap straight."""
    from scipy.spatial import cKDTree

    rng = np.random.RandomState(seed)
    length = rng.uniform(*TRAP_LENGTH)
    th = np.linspace(0.0, 2 * np.pi, 1440, endpoint=False)
    p = rng.uniform(*TRAP_SQUARENESS)
    r = (np.abs(np.cos(th)) ** p + np.abs(np.sin(th)) ** p) ** (-1.0 / p)
    for k_h in (2, 3):
        r *= 1.0 + rng.uniform(0.0, 0.12 / k_h**0.7) * np.cos(k_h * th + rng.uniform(0, 2 * np.pi))
    xy = np.c_[r * np.cos(th) * rng.uniform(*TRAP_STRETCH), r * np.sin(th)]
    a = rng.uniform(0, 2 * np.pi)
    xy = xy @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    if rng.rand() < 0.5:
        xy = xy[::-1]
    xy = _resample_closed(xy, 0.01)
    xy = _resample_closed(xy * (length / np.sum(np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T))), 0.05)
    k = _curvature_closed(xy)
    min_radius = 1.0 / np.max(np.abs(k))
    if min_radius < TRAP_MIN_RADIUS:
        return None
    n = len(xy)
    s = np.arange(n) * (length / n)
    tangent = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.c_[-tangent[:, 1], tangent[:, 0]]
    width = rng.uniform(*TRAP_WIDTH)
    k_abs = np.convolve(np.abs(np.concatenate([k[-10:], k, k[:10]])), np.ones(21) / 21, mode="same")[10:-10]
    pinch = np.clip((k_abs - LAB_PINCH_CURVATURE[0]) / (LAB_PINCH_CURVATURE[1] - LAB_PINCH_CURVATURE[0]), 0.0, 1.0)
    nominal = width + (np.minimum(rng.uniform(*LAB_CORNER_WIDTH), width) - width) * pinch
    # the trap: the longest straight run (|curvature| < 0.08 1/m), widened with 1 m cosine ramps
    straight = k_abs < 0.08
    if straight.all() or not straight.any():
        return None
    shift = int(np.argmin(straight))  # start the scan on a curved point so runs do not wrap
    st = np.roll(straight, -shift)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], st.astype(int), [0]])))
    runs = edges.reshape(-1, 2)
    i0, i1 = max(runs, key=lambda r_: r_[1] - r_[0])
    if (i1 - i0) * (length / n) < TRAP_MIN_STRAIGHT:
        return None
    pos = (np.arange(n) - i0) * (length / n)
    run_len = (i1 - i0) * (length / n)
    ramp = np.clip(np.minimum(pos, run_len - pos) / np.clip(0.3 * run_len, 1.0, 2.0), 0.0, 1.0)
    weight = np.where((pos >= 0) & (pos <= run_len), 0.5 - 0.5 * np.cos(np.pi * ramp), 0.0)
    weight = np.roll(weight, shift)
    trap_w = rng.uniform(*TRAP_STRAIGHT_WIDTH)
    extra = np.maximum(width + (trap_w - width) * weight - nominal, 0.0)  # the trap's added width
    side = -np.sign(np.sum(k))  # the outer wall pushes out (+: left; a counter-clockwise loop's outside is on its right)
    w_left = nominal / 2 + 0.5 * _wall_drift(rng, s, length) + extra * (side > 0)
    w_right = nominal / 2 + 0.5 * _wall_drift(rng, s, length) + extra * (side < 0)
    with np.errstate(divide="ignore"):
        inner_max = 0.85 / np.abs(k)
    w_left = np.where(k > 0, np.minimum(w_left, inner_max), w_left)
    w_right = np.where(k < 0, np.minimum(w_right, inner_max), w_right)
    excess = np.maximum(w_left + w_right - TRAP_STRAIGHT_WIDTH[1], 0.0)
    w_left, w_right = w_left - excess / 2, w_right - excess / 2
    deficit = np.maximum(LAB_MIN_WIDTH - (w_left + w_right), 0.0)
    w_left, w_right = w_left + deficit / 2, w_right + deficit / 2
    if np.any((k > 0) & (w_left > inner_max)) or np.any((k < 0) & (w_right > inner_max)):
        return None
    reach = np.maximum(w_left, w_right)
    tree = cKDTree(xy)
    for i, j in tree.query_pairs(2 * reach.max() + 2 * LAB_WALL + LAB_CLEARANCE):
        along = abs(i - j) * (length / n)
        along = min(along, length - along)
        need = reach[i] + reach[j] + 2 * LAB_WALL + LAB_CLEARANCE
        if along > np.pi * need and np.hypot(*(xy[i] - xy[j])) < need:
            return None
    res = GEN_RESOLUTION
    pad = reach.max() + LAB_WALL + GEN_MARGIN
    lo, hi = xy.min(axis=0) - pad, xy.max(axis=0) + pad
    w_px, h_px = np.ceil((hi - lo) / res).astype(int)
    gx, gy = np.meshgrid(lo[0] + (np.arange(w_px) + 0.5) * res, lo[1] + (np.arange(h_px) + 0.5) * res)
    pix = np.c_[gx.ravel(), gy.ravel()]
    _, j = tree.query(pix)
    e = np.einsum("ij,ij->i", pix - xy[j], normal[j])
    wall = ((e > w_left[j]) & (e <= w_left[j] + LAB_WALL)) | ((e < -w_right[j]) & (e >= -w_right[j] - LAB_WALL))
    free = ~wall.reshape(h_px, w_px)
    mid = xy + normal * ((w_left - w_right) / 2)[:, None]
    half = (w_left + w_right) / 2
    start = (int(i0) + shift) % n  # the trap's start
    mid, half = np.roll(mid, -start, axis=0), np.roll(half, -start)
    keep = slice(None, None, 2)
    return dict(free=free, resolution=res, origin=(float(lo[0]), float(lo[1])),
                centerline=np.c_[mid[keep], half[keep], half[keep]], width=float(width), length=float(length),
                min_radius=float(min_radius), min_width=float(np.min(w_left + w_right)),
                max_width=float(np.max(w_left + w_right)), trap_width=float(trap_w), trap_length=float(run_len),
                trap_side=float(side), wall_offsets=np.c_[w_left, w_right])


def trap_name(seed: int) -> str:
    return f"Trap{seed}"  # seeds 9000+: 1.0-1.6 m off the trap; 50000+: 0.9-1.4 m, sharper corners


# ---------------------------------------------------------------------------
# Fixed train / held-out lists (spec 4.3). Held-out maps are never trained on.
# ---------------------------------------------------------------------------
# Generator seeds: the first 11 seeds >= 1000 that pass generate_duct_track's
# checks (width U(1.5, 2.5) m, length U(40, 80) m, corner radius >= 1 m).
TRAIN_GEN_SEEDS = (1002, 1003, 1008, 1015, 1039, 1045, 1064, 1070)
HELDOUT_GEN_SEEDS = (1077, 1112, 1142)
TRAIN_RACETRACKS = (
    "Austin",
    "BrandsHatch",
    "Budapest",
    "Catalunya",
    "Hockenheim",
    "Montreal",
    "Monza",
    "Silverstone",
)
HELDOUT_RACETRACKS = ("Spielberg", "Oschersleben", "MoscowRaceway")


def gen_name(seed: int) -> str:
    return f"Duct{seed}"


TRAIN_MAPS = TRAIN_RACETRACKS + tuple(gen_name(s) for s in TRAIN_GEN_SEEDS)
HELDOUT_MAPS = HELDOUT_RACETRACKS + tuple(gen_name(s) for s in HELDOUT_GEN_SEEDS)
assert not set(TRAIN_MAPS) & set(HELDOUT_MAPS)

# Lab-size tracks (generate_lab_track, with pinched corners): the first valid seeds from 4000 (train),
# 4500 (held-out: checkpoint selection) and 4900 (never-seen: final evaluation). (Seeds 2000+ were the
# generator before corner pinching; new seeds keep their written maps from being reused.)
LAB_TRAIN_SEEDS = (4005, 4010, 4015, 4025, 4036, 4037, 4040, 4046, 4048, 4049, 4050, 4051,
                   4052, 4061, 4068, 4071, 4075, 4076, 4084, 4086, 4093, 4095, 4114, 4118)
LAB_HELDOUT_SEEDS = (4501, 4503, 4506, 4507, 4511, 4516)
LAB_UNSEEN_SEEDS = (4901, 4902, 4908, 4909, 4912, 4914)
LAB_TRAIN_MAPS = tuple(lab_name(s) for s in LAB_TRAIN_SEEDS)
LAB_HELDOUT_MAPS = tuple(lab_name(s) for s in LAB_HELDOUT_SEEDS)
LAB_UNSEEN_MAPS = tuple(lab_name(s) for s in LAB_UNSEEN_SEEDS)
# Room tracks (generate_room_track, tight turnarounds): first valid seeds from 5000 / 5500 / 5900
# (3000+ were the version with wide turnarounds).
ROOM_TRAIN_SEEDS = (5017, 5021, 5027, 5032, 5047, 5051, 5061, 5080, 5095, 5097, 5119, 5123)
ROOM_HELDOUT_SEEDS = (5519, 5546, 5547)
ROOM_UNSEEN_SEEDS = (5913, 5914, 5922, 5950)
ROOM_TRAIN_MAPS = tuple(room_name(s) for s in ROOM_TRAIN_SEEDS)
ROOM_HELDOUT_MAPS = tuple(room_name(s) for s in ROOM_HELDOUT_SEEDS)
ROOM_UNSEEN_MAPS = tuple(room_name(s) for s in ROOM_UNSEEN_SEEDS)
# Race-like tracks (generate_race_like_track) and speed-trap tracks (generate_trap_track): first valid
# seeds from 6000 / 6500 / 6900 and 9000 / 9500 / 9900 (train / held-out / never-seen; trap seeds
# 7000+ widened both walls, 8000+ one wall on a random side).
RACE_LIKE_TRAIN_SEEDS = (20010, 20012, 20013, 20021, 20024, 20047, 20050, 20059, 20065, 20070, 20088, 20102, 20113, 20114, 20140, 20183, 20186, 20195, 20198, 20235, 20251, 20273)
RACE_LIKE_HELDOUT_SEEDS = (30002, 30011, 30016, 30022, 30033)
RACE_LIKE_UNSEEN_SEEDS = (40000, 40002, 40013, 40064)
TRAP_TRAIN_SEEDS = (50023, 50055, 50104, 50110, 50141, 50144, 50158, 50178)
TRAP_HELDOUT_SEEDS = (60049, 60060, 60064)
TRAP_UNSEEN_SEEDS = (70027, 70038)
RACE_LIKE_TRAIN_MAPS = tuple(race_like_name(s) for s in RACE_LIKE_TRAIN_SEEDS)
RACE_LIKE_HELDOUT_MAPS = tuple(race_like_name(s) for s in RACE_LIKE_HELDOUT_SEEDS)
RACE_LIKE_UNSEEN_MAPS = tuple(race_like_name(s) for s in RACE_LIKE_UNSEEN_SEEDS)
TRAP_TRAIN_MAPS = tuple(trap_name(s) for s in TRAP_TRAIN_SEEDS)
TRAP_HELDOUT_MAPS = tuple(trap_name(s) for s in TRAP_HELDOUT_SEEDS)
TRAP_UNSEEN_MAPS = tuple(trap_name(s) for s in TRAP_UNSEEN_SEEDS)
# Lab fine-tune (32 maps, 64 envs each): the last race's track both ways, 12 tracks like it, the 4 hand
# drawings in 3 variants each, and 6 speed-trap loops. Selection: held-out race-like, drawn and trap tracks.
LAB_FT_TRAIN_MAPS = LAST_RACE_MAPS + RACE_LIKE_TRAIN_MAPS[:12] + DRAWN_TRAIN_MAPS + TRAP_TRAIN_MAPS[:6]
LAB_FT_HELDOUT_MAPS = RACE_LIKE_HELDOUT_MAPS[:3] + DRAWN_HELDOUT_MAPS + TRAP_HELDOUT_MAPS
LAB_EVAL_UNSEEN_MAPS = RACE_LIKE_UNSEEN_MAPS + TRAP_UNSEEN_MAPS
assert not set(LAB_FT_TRAIN_MAPS) & set(LAB_FT_HELDOUT_MAPS + LAB_EVAL_UNSEEN_MAPS)
assert not set(LAB_FT_HELDOUT_MAPS) & set(LAB_EVAL_UNSEEN_MAPS)
assert len(LAB_FT_TRAIN_MAPS) == 32

DEFAULT_MAP_DIR = pathlib.Path(__file__).resolve().parent / "maps"


def _sha256(path: pathlib.Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_maps(map_dir: os.PathLike = DEFAULT_MAP_DIR, maps=TRAIN_MAPS + HELDOUT_MAPS) -> dict:
    """Write generated tracks, download racetracks, and return a manifest.

    Run this once on a machine with internet access (e.g. the cluster login
    node) before training; compute nodes then only read ``map_dir``.
    """
    from f1tenth_gym_jax.envs.track.utils import find_track_dir

    map_dir = use_map_dir(map_dir)
    manifest = {"map_dir": str(map_dir), "maps": {}}
    seeds = {gen_name(s): (generate_duct_track, s) for s in TRAIN_GEN_SEEDS + HELDOUT_GEN_SEEDS}
    seeds.update({lab_name(s): (generate_lab_track, s)
                  for s in LAB_TRAIN_SEEDS + LAB_HELDOUT_SEEDS + LAB_UNSEEN_SEEDS})
    seeds.update({room_name(s): (generate_room_track, s)
                  for s in ROOM_TRAIN_SEEDS + ROOM_HELDOUT_SEEDS + ROOM_UNSEEN_SEEDS})
    seeds.update({LAST_RACE_MAPS[0]: (last_race_track, False), LAST_RACE_MAPS[1]: (last_race_track, True)})
    seeds.update({race_like_name(s): (generate_race_like_track, s)
                  for s in RACE_LIKE_TRAIN_SEEDS + RACE_LIKE_HELDOUT_SEEDS + RACE_LIKE_UNSEEN_SEEDS})
    seeds.update({trap_name(s): (generate_trap_track, s) for s in TRAP_TRAIN_SEEDS + TRAP_HELDOUT_SEEDS + TRAP_UNSEEN_SEEDS})
    seeds.update({n: (drawn_track, n) for n in DRAWN_TRAIN_MAPS + DRAWN_HELDOUT_MAPS})
    for name in maps:
        if name in seeds:
            gen, seed = seeds[name]
            track_dir = map_dir / name
            if not (track_dir / f"{name}.png").exists():
                g = gen(seed)
                if g is None:
                    raise RuntimeError(f"seed {seed} no longer yields a valid track")
                write_map(map_dir, name, g["free"], g["resolution"], g["origin"], g["centerline"])
            entry = {"kind": "generated", "seed": seed, "generator": gen.__name__}
        else:
            track_dir = find_track_dir(name)  # downloads if missing
            entry = {"kind": "racetrack"}
        stem = track_dir.name
        entry["files"] = {
            p.name: _sha256(p)
            for p in sorted(track_dir.iterdir())
            if p.suffix in (".png", ".pgm", ".yaml", ".csv") and p.name.startswith(stem)
        }
        manifest["maps"][name] = entry
    return manifest


def map_lists_config() -> dict:
    """The lists and generator settings, for the run config."""
    return {
        "train_maps": list(TRAIN_MAPS),
        "heldout_maps": list(HELDOUT_MAPS),
        "train_gen_seeds": list(TRAIN_GEN_SEEDS),
        "heldout_gen_seeds": list(HELDOUT_GEN_SEEDS),
        "lab_train_seeds": list(LAB_TRAIN_SEEDS),
        "lab_heldout_seeds": list(LAB_HELDOUT_SEEDS),
        "lab_unseen_seeds": list(LAB_UNSEEN_SEEDS),
        "lab_generator": {
            "length": list(LAB_LENGTH), "width": list(LAB_WIDTH), "min_width": LAB_MIN_WIDTH,
            "max_width": LAB_MAX_WIDTH, "wall": LAB_WALL, "min_radius": LAB_MIN_RADIUS,
            "clearance": LAB_CLEARANCE, "wander": LAB_WANDER, "bumps": list(LAB_BUMPS),
            "bump_amp": list(LAB_BUMP_AMP), "bump_sigma": list(LAB_BUMP_SIGMA), "squareness": list(LAB_SQUARENESS),
        },
        "generator": {
            "resolution": GEN_RESOLUTION,
            "wall": GEN_WALL,
            "width": list(GEN_WIDTH),
            "length": list(GEN_LENGTH),
            "min_radius": GEN_MIN_RADIUS,
            "clearance": GEN_CLEARANCE,
            "rad_scale": GEN_RAD_SCALE,
        },
    }


# ---------------------------------------------------------------------------
# Multi-map env: one env per map, all stepped inside one jitted function
# ---------------------------------------------------------------------------
class MultiMapEnv:
    """``num_envs`` split evenly over ``map_names``; flat [num_envs, ...] arrays.

    Each map gets its own F110Env (grid sizes differ), wrapped in
    Phase1Wrapper and vmapped over its share of the envs. The Python loop
    over maps unrolls inside jit.
    """

    def __init__(self, map_names, num_envs: int, randomize: bool = True, wrapper_kwargs=None, race=None,
                 accel_range=(), obstacle_maps=(), **param_overrides):
        """``race``: None for phase 1, else RaceWrapper kwargs (two-car env, phase 2).
        ``accel_range`` (lo, hi): each map gets its own acceleration / braking limit spread over the
        range (Param.a_max; one value per map, as the env compiles one per map). ``obstacle_maps``
        (race): these maps park an obstacle instead of drawing an opponent."""
        import jax  # noqa: F401  (keep module import light for the CLI)

        from f1tenth_gym_jax.envs import F110Env

        from .domain_rand import make_param
        from .wrappers import Phase1Wrapper

        self.map_names = tuple(map_names)
        if num_envs % len(self.map_names):
            raise ValueError(f"num_envs={num_envs} must split evenly over {len(self.map_names)} maps")
        self.num_envs = num_envs
        self.envs_per_map = num_envs // len(self.map_names)
        self.randomize = randomize
        self.race = race is not None
        if self.race:
            from .race import RaceWrapper

            self.wrappers = [
                RaceWrapper(F110Env(2, make_param(m, randomize=randomize, scan_other_agents=True,
                                                  **self._accel(i, accel_range), **param_overrides)),
                            randomize, **race, obstacles=m in set(obstacle_maps), **(wrapper_kwargs or {}))
                for i, m in enumerate(self.map_names)
            ]
        else:
            self.wrappers = [
                Phase1Wrapper(F110Env(1, make_param(m, randomize=randomize, **self._accel(i, accel_range),
                                                    **param_overrides)), randomize, **(wrapper_kwargs or {}))
                for i, m in enumerate(self.map_names)
            ]
        self.obs_dim = self.wrappers[0].obs_dim
        self.action_dim = self.wrappers[0].action_dim
        import jax.numpy as jnp

        self.map_ids = jnp.repeat(jnp.arange(len(self.map_names)), self.envs_per_map)

    @staticmethod
    def _accel(i, accel_range):
        """Map i's acceleration limit: spread over the range by a golden-ratio sequence."""
        if not accel_range:
            return {}
        lo, hi = accel_range
        return {"a_max": float(lo + (hi - lo) * ((i * 0.6180339887) % 1.0))}

    def reset(self, key):
        import jax
        import jax.numpy as jnp

        keys = jax.random.split(key, len(self.wrappers))
        obs, states = [], []
        for w, k in zip(self.wrappers, keys):
            o, s = jax.vmap(w.reset)(jax.random.split(k, self.envs_per_map))
            obs.append(o)
            states.append(s)
        return jnp.concatenate(obs), tuple(states)

    def step(self, key, states, actions, opp_params=None):
        """``opp_params``: race only, the snapshot-opponent pool (same for every env)."""
        import jax
        import jax.numpy as jnp

        keys = jax.random.split(key, len(self.wrappers))
        n = self.envs_per_map
        outs = []
        for i, (w, k, s) in enumerate(zip(self.wrappers, keys, states)):
            a = actions[i * n : (i + 1) * n]
            if self.race:
                outs.append(jax.vmap(w.step, in_axes=(0, 0, 0, None))(jax.random.split(k, n), s, a, opp_params))
            else:
                outs.append(jax.vmap(w.step)(jax.random.split(k, n), s, a))
        obs = jnp.concatenate([o[0] for o in outs])
        new_states = tuple(o[1] for o in outs)
        reward = jnp.concatenate([o[2] for o in outs])
        done = jnp.concatenate([o[3] for o in outs])
        info = {key_: jnp.concatenate([o[4][key_] for o in outs]) for key_ in outs[0][4]}
        info["map_id"] = self.map_ids
        return obs, new_states, reward, done, info


def main(argv=None):
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Prepare phase-1 maps.")
    parser.add_argument("command", choices=["prepare", "list"])
    parser.add_argument("--map-dir", default=str(DEFAULT_MAP_DIR))
    parser.add_argument("--lab", action="store_true", help="also the lab, room, last-race, race-like and trap tracks")
    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps(map_lists_config(), indent=2))
        return
    lab = (LAB_TRAIN_MAPS + LAB_HELDOUT_MAPS + LAB_UNSEEN_MAPS + ROOM_TRAIN_MAPS + ROOM_HELDOUT_MAPS
           + ROOM_UNSEEN_MAPS + LAST_RACE_MAPS + LAB_FT_TRAIN_MAPS + LAB_FT_HELDOUT_MAPS + LAB_EVAL_UNSEEN_MAPS)
    lab = tuple(dict.fromkeys(lab))
    maps = TRAIN_MAPS + HELDOUT_MAPS + (lab if args.lab else ())
    manifest = prepare_maps(args.map_dir, maps)
    out = pathlib.Path(args.map_dir) / "manifest.json"
    out.write_text(json.dumps({**map_lists_config(), **manifest}, indent=2))
    print(f"prepared {len(manifest['maps'])} maps in {args.map_dir}; manifest {out}")


if __name__ == "__main__":
    main()
