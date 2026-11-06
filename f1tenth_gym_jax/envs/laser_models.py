"""2D LiDAR simulation by ray marching on a Euclidean distance transform.

Same algorithm as ``jax_pf.ray_marching.get_scan``, with two differences:

- Beam ``i`` is cast at exactly ``theta - fov/2 + i * fov / (num_beams - 1)``.
  ``get_scan`` looks angles up in a ``theta_dis``-bin table (truncating to the
  bin below) and its linspace end point spreads the beams over
  ``fov * num_beams / (num_beams - 1)`` instead of ``fov``.
- No ``chex.assert_max_traces``, so environments on maps with different grid
  sizes can coexist in one process.
"""

import chex
import jax
import jax.numpy as jnp


def beam_offsets(fov: float, num_beams: int) -> chex.Array:
    """Beam angles relative to the sensor heading, evenly over [-fov/2, fov/2]."""
    return jnp.linspace(-fov / 2.0, fov / 2.0, num_beams)


def _distance_at(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt):
    # world -> map frame -> (row, col), clipped to the grid
    x_trans = x - orig_x
    y_trans = y - orig_y
    x_rot = x_trans * orig_c + y_trans * orig_s
    y_rot = -x_trans * orig_s + y_trans * orig_c
    r = jnp.clip(y_rot / resolution, 0, height - 1).astype(int)
    c = jnp.clip(x_rot / resolution, 0, width - 1).astype(int)
    return dt[r, c]


def get_scan_exact(
    pose: chex.Array,
    offsets: chex.Array,
    eps: float,
    orig_x: float,
    orig_y: float,
    orig_c: float,
    orig_s: float,
    height: int,
    width: int,
    resolution: float,
    dt: chex.Array,
    max_range: float,
) -> chex.Array:
    """Ranges for one sensor pose ``(x, y, theta)`` at ``theta + offsets``."""
    dt = jnp.asarray(dt)
    angles = pose[2] + offsets
    cosines = jnp.cos(angles)
    sines = jnp.sin(angles)

    def dist(x, y):
        return _distance_at(
            x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt
        )

    def trace_ray(c, s):
        d0 = dist(pose[0], pose[1])

        def cond(v):
            return (v[0] > eps) & (v[1] <= max_range)

        def body(v):
            x = v[2] + v[0] * c
            y = v[3] + v[0] * s
            d = dist(x, y)
            return jnp.array([d, v[1] + d, x, y])

        out = jax.lax.while_loop(cond, body, jnp.array([d0, d0, pose[0], pose[1]]))
        return jnp.minimum(out[1], max_range)

    return jax.vmap(trace_ray)(cosines, sines)


def ray_rectangle_ranges(origin: chex.Array, angles: chex.Array, rects: chex.Array, max_range: float) -> chex.Array:
    """Distance along each ray to the nearest edge of a set of rectangles.

    Parameters
    ----------
    origin : chex.Array, shape (2,)
        Ray origin ``[x, y]`` (the LiDAR position).
    angles : chex.Array, shape (num_beams,)
        World-frame ray directions, rad.
    rects : chex.Array, shape (num_rects, 4, 2)
        Rectangle corners in winding order (e.g. ``get_vertices`` of other cars).
    max_range : float
        Returned where a ray hits no rectangle.

    Returns
    -------
    chex.Array, shape (num_beams,)
        Nearest hit distance per ray, capped at ``max_range``.
    """
    a = rects  # (R, 4, 2) segment starts
    e = jnp.roll(rects, -1, axis=1) - rects  # (R, 4, 2) segment vectors
    d = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)  # (B, 2)

    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    ap = a[None] - origin  # (1, R, 4, 2)
    dd = d[:, None, None, :]  # (B, 1, 1, 2)
    denom = cross(dd, e[None])  # (B, R, 4)
    safe = jnp.where(jnp.abs(denom) < 1e-12, 1.0, denom)
    t = cross(ap, e[None]) / safe  # distance along the ray
    u = cross(ap, dd) / safe  # position along the segment
    hit = (jnp.abs(denom) >= 1e-12) & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
    t = jnp.where(hit, t, max_range)
    return jnp.minimum(jnp.min(t, axis=(1, 2)), max_range)
