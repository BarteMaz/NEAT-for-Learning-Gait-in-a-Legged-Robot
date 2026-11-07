"""Shared fixtures for the test suite.

Every test gets its own PyBullet server in DIRECT mode, so tests never share
simulation state and can run in any order.
"""

from collections.abc import Iterator

import pybullet as p
import pytest

from neat_walking_robot.environments.base import BaseEnvironmentData
from neat_walking_robot.environments.heightfield import RoughEnvironmentData


@pytest.fixture
def client() -> Iterator[int]:
    """A fresh DIRECT physics server, disconnected after the test."""
    client_id = p.connect(p.DIRECT)
    try:
        yield client_id
    finally:
        p.disconnect(client_id)


@pytest.fixture
def two_clients() -> Iterator[tuple[int, int]]:
    """Two independent DIRECT servers, used to catch a missing physicsClientId.

    Commands without physicsClientId go to server 0, so an environment built in
    the second server must leave the first one untouched.
    """
    first = p.connect(p.DIRECT)
    second = p.connect(p.DIRECT)
    try:
        yield first, second
    finally:
        p.disconnect(second)
        p.disconnect(first)


# Small but realistic dimensions keep the tests fast.
BASE_PARAMS = {
    "track_length": 6.0,
    "track_width": 2.0,
    "wall_height": 1.0,
    "wall_thickness": 0.15,  # deliberately different from ground_thickness
    "ground_thickness": 0.2,
    "friction": 0.8,
    "start_distance": 1.0,
    "end_distance": 1.0,
}

ROUGH_PARAMS = {
    **BASE_PARAMS,
    "lowest_ground_point": -0.15,
    "cell_size": 0.05,
    "amplitude": 0.1,
    "transition_distance": 0.5,
    "n_waves": 6,
    "wavelength_range": (0.5, 2.0),
}


@pytest.fixture
def make_base_params():
    """Factory for valid flat-track parameters; keyword overrides change single fields."""

    def factory(**overrides) -> BaseEnvironmentData:
        return BaseEnvironmentData(**{**BASE_PARAMS, **overrides})

    return factory


@pytest.fixture
def make_rough_params():
    """Factory for valid rough-terrain parameters; keyword overrides change single fields."""

    def factory(**overrides) -> RoughEnvironmentData:
        return RoughEnvironmentData(**{**ROUGH_PARAMS, **overrides})

    return factory
