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
