import logging

import jax
import pytest


class _CompileCounter(logging.Handler):
    """Counts XLA compilations reported by ``jax.log_compiles``."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages = []

    def emit(self, record):
        msg = record.getMessage()
        if msg.startswith("Compiling "):
            self.messages.append(msg)


@pytest.fixture
def compile_counter():
    counter = _CompileCounter()
    logger = logging.getLogger("jax")
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(counter)
    try:
        with jax.log_compiles(True):
            yield counter
    finally:
        logger.removeHandler(counter)
        logger.setLevel(old_level)


# --------------------------------------------------------------------------
# Synthetic test maps
# --------------------------------------------------------------------------
import numpy as np  # noqa: E402

from rl.maps import DEFAULT_MAP_DIR, use_map_dir, write_map  # noqa: E402

# FineBox: 6 m x 6 m room at 5 mm/px, walls 5 cm thick, inner face at x = +-2.95 m.
FINE_RES = 0.005
FINE_HALF = 3.0
FINE_WALL = 0.05


def _box_map(map_dir, name, half, res, wall, radius):
    n = int(round(2 * half / res))
    w = int(round(wall / res))
    free = np.zeros((n, n), dtype=bool)
    free[w:-w, w:-w] = True
    th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    cl = np.c_[radius * np.cos(th), radius * np.sin(th), np.full_like(th, 0.5), np.full_like(th, 0.5)]
    write_map(map_dir, name, free, res, (-half, -half), cl)


@pytest.fixture(scope="session")
def test_map_dir():
    """Synthetic test maps, written next to the training maps (gitignored).

    One map directory for every test: the env finds maps through a single
    environment variable, so per-test directories would fight over it.
    """
    map_dir = use_map_dir(DEFAULT_MAP_DIR)
    if not (map_dir / "FineBox" / "FineBox.png").exists():
        _box_map(map_dir, "FineBox", FINE_HALF, FINE_RES, FINE_WALL, radius=1.5)
    if not (map_dir / "OpenBox" / "OpenBox.png").exists():
        _box_map(map_dir, "OpenBox", 50.0, 0.05, 0.5, radius=30.0)
    return map_dir
