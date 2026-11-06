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
    w_left = width / 2 + _wall_drift(rng, s, length)
    w_right = width / 2 + _wall_drift(rng, s, length)
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

# Lab-size tracks (generate_lab_track): the first valid seeds from 2000 (train),
# 2500 (held-out: checkpoint selection) and 2900 (never-seen: final evaluation).
LAB_TRAIN_SEEDS = (2004, 2015, 2019, 2021, 2022, 2027, 2036, 2037, 2038, 2042, 2046, 2056,
                   2068, 2069, 2071, 2081, 2082, 2093, 2094, 2096, 2100, 2101, 2103, 2107)
LAB_HELDOUT_SEEDS = (2506, 2511, 2524, 2526, 2529, 2533)
LAB_UNSEEN_SEEDS = (2902, 2903, 2904, 2906, 2907, 2910)
LAB_TRAIN_MAPS = tuple(lab_name(s) for s in LAB_TRAIN_SEEDS)
LAB_HELDOUT_MAPS = tuple(lab_name(s) for s in LAB_HELDOUT_SEEDS)
LAB_UNSEEN_MAPS = tuple(lab_name(s) for s in LAB_UNSEEN_SEEDS)
# Lab fine-tune: lab tracks plus the duct tracks (kept so wider corridors are not forgotten)
LAB_FT_TRAIN_MAPS = LAB_TRAIN_MAPS + tuple(gen_name(s) for s in TRAIN_GEN_SEEDS)
LAB_FT_HELDOUT_MAPS = LAB_HELDOUT_MAPS + tuple(gen_name(s) for s in HELDOUT_GEN_SEEDS[:2])
assert not set(LAB_FT_TRAIN_MAPS) & set(LAB_FT_HELDOUT_MAPS + LAB_UNSEEN_MAPS)
assert not set(LAB_FT_HELDOUT_MAPS) & set(LAB_UNSEEN_MAPS)

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
                 **param_overrides):
        """``race``: None for phase 1, else RaceWrapper kwargs (two-car env, phase 2)."""
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
                RaceWrapper(F110Env(2, make_param(m, randomize=randomize, scan_other_agents=True, **param_overrides)),
                            randomize, **race, **(wrapper_kwargs or {}))
                for m in self.map_names
            ]
        else:
            self.wrappers = [
                Phase1Wrapper(F110Env(1, make_param(m, randomize=randomize, **param_overrides)), randomize,
                              **(wrapper_kwargs or {}))
                for m in self.map_names
            ]
        self.obs_dim = self.wrappers[0].obs_dim
        self.action_dim = self.wrappers[0].action_dim
        import jax.numpy as jnp

        self.map_ids = jnp.repeat(jnp.arange(len(self.map_names)), self.envs_per_map)

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
    parser.add_argument("--lab", action="store_true", help="also the lab-size tracks (train, held-out, never-seen)")
    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps(map_lists_config(), indent=2))
        return
    maps = TRAIN_MAPS + HELDOUT_MAPS + ((LAB_TRAIN_MAPS + LAB_HELDOUT_MAPS + LAB_UNSEEN_MAPS) if args.lab else ())
    manifest = prepare_maps(args.map_dir, maps)
    out = pathlib.Path(args.map_dir) / "manifest.json"
    out.write_text(json.dumps({**map_lists_config(), **manifest}, indent=2))
    print(f"prepared {len(manifest['maps'])} maps in {args.map_dir}; manifest {out}")


if __name__ == "__main__":
    main()
