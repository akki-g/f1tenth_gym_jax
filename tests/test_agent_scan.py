import unittest

import jax.numpy as jnp
import numpy as np

from f1tenth_gym_jax.envs.laser_models import ray_rectangle_ranges


class TestRayRectangleRanges(unittest.TestCase):
    def test_hits_nearest_face_and_misses(self):
        # unit square from x=2 to x=3, y=-0.5..0.5
        sq = jnp.array([[[2.0, -0.5], [3.0, -0.5], [3.0, 0.5], [2.0, 0.5]]])
        angles = jnp.array([0.0, np.pi / 2, np.pi, np.arctan2(0.49, 2.0)])
        r = np.asarray(ray_rectangle_ranges(jnp.zeros(2), angles, sq, 10.0))
        self.assertAlmostEqual(r[0], 2.0, places=6)  # straight at the near face
        self.assertEqual(r[1], 10.0)  # sideways: miss
        self.assertEqual(r[2], 10.0)  # backwards: miss
        self.assertAlmostEqual(r[3], np.hypot(2.0, 0.49), places=5)  # just inside the corner

    def test_nearest_of_several_and_cap(self):
        rects = jnp.array([
            [[5.0, -1.0], [6.0, -1.0], [6.0, 1.0], [5.0, 1.0]],
            [[1.5, -1.0], [2.0, -1.0], [2.0, 1.0], [1.5, 1.0]],
        ])
        r = ray_rectangle_ranges(jnp.zeros(2), jnp.array([0.0]), rects, 1.0)
        self.assertAlmostEqual(float(r[0]), 1.0)  # capped at max_range (hit is at 1.5)
        r = ray_rectangle_ranges(jnp.zeros(2), jnp.array([0.0]), rects, 10.0)
        self.assertAlmostEqual(float(r[0]), 1.5, places=6)


if __name__ == "__main__":
    unittest.main()
