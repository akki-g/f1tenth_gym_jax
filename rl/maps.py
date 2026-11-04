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
