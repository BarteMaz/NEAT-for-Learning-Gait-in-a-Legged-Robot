"""Ramp geometry and parameters specific to the slope environment (c).

Layout along x: flat start platform at z = 0, ramp rising at `incline`,
flat end platform (plateau) at the top. The platforms are
start_distance + buffer_space and end_distance + buffer_space long.
"""

import math

import numpy as np
import pybullet as p
import pytest

from neat_walking_robot.environments.slope import SlopeEnvironment

# Ray casts against a rotated box are accurate to about 1 mm (measured: up to 0.9 mm
# on a plain 10 deg ramp). A real construction error, e.g. ground_thickness not
# accounted for at a junction, is centimetres, so 2 mm still catches it.
RAY_TOL = 2e-3


def _ramp_span(params) -> float:
    """Horizontal length of the ramp."""
    return params.track_length - (
        params.start_distance + params.end_distance + 2 * params.buffer_space
    )


def _expected_rise(params) -> float:
    return math.tan(math.radians(params.incline)) * _ramp_span(params)


def _ray_surface(client: int, xs, y: float = 0.0) -> np.ndarray:
    """Height of the topmost surface the physics engine sees, by rays cast downwards."""
    hits = p.rayTestBatch(
        [[x, y, 10.0] for x in xs], [[x, y, -10.0] for x in xs], physicsClientId=client
    )
    return np.array([hit[3][2] for hit in hits])


@pytest.fixture
def env_and_client(make_slope_params, client):
    return SlopeEnvironment(make_slope_params(), client, seed=0), client


# --- Parameters ------------------------------------------------------------------


@pytest.mark.parametrize("incline", [-1.0, 31.0])
def test_incline_outside_allowed_range_is_rejected(make_slope_params, incline):
    with pytest.raises(ValueError):
        make_slope_params(incline=incline)


def test_slope_params_keep_base_validation(make_slope_params):
    with pytest.raises(ValueError):
        make_slope_params(track_width=0.0)


def test_walls_lower_than_the_plateau_are_rejected(make_slope_params):
    # at 30 deg the plateau is tan(30) * 3 = 1.73 m high; 1.0 m walls would end below it
    with pytest.raises(ValueError):
        make_slope_params(incline=30.0, wall_height=1.0)


def test_walls_taller_than_the_plateau_are_accepted(make_slope_params):
    params = make_slope_params(incline=30.0, wall_height=2.5)
    assert params.wall_height > _expected_rise(params)


# --- Geometry --------------------------------------------------------------------


def test_start_platform_is_level_at_zero(env_and_client, make_slope_params):
    env, _ = env_and_client
    params = make_slope_params()
    back = -params.track_length / 2
    for x in np.linspace(
        back + 0.05, back + params.start_distance + params.buffer_space, 6
    ):
        assert env.get_ground_level(x, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_plateau_height_matches_incline(env_and_client, make_slope_params):
    env, _ = env_and_client
    params = make_slope_params()
    front = params.track_length / 2
    for x in np.linspace(
        front - params.end_distance - params.buffer_space, front - 0.05, 6
    ):
        assert env.get_ground_level(x, 0.0) == pytest.approx(
            _expected_rise(params), abs=1e-9
        )


def test_ramp_angle_matches_incline(env_and_client, make_slope_params):
    # measured on the physical surface, not on the analytic formula
    _env, client = env_and_client
    params = make_slope_params()
    ramp_start = -params.track_length / 2 + params.start_distance + params.buffer_space
    xs = np.linspace(ramp_start + 0.3, ramp_start + _ramp_span(params) - 0.3, 20)
    z = _ray_surface(client, xs)
    measured = math.degrees(math.atan(np.polyfit(xs, z, 1)[0]))
    assert measured == pytest.approx(params.incline, abs=0.05)


@pytest.mark.parametrize("incline", [0.0, 10.0, 25.0])
def test_ground_level_matches_physical_surface(make_slope_params, client, incline):
    # get_ground_level must agree with the surface the robot will actually stand on
    params = make_slope_params(incline=incline)
    env = SlopeEnvironment(params, client, seed=0)
    half = params.track_length / 2
    xs = np.linspace(-half + 0.01, half - 0.01, 400)
    for y in (-0.9, 0.0, 0.9):
        analytic = np.array([env.get_ground_level(x, y) for x in xs])
        np.testing.assert_allclose(_ray_surface(client, xs, y), analytic, atol=RAY_TOL)


@pytest.mark.parametrize("incline", [10.0, 25.0])
def test_surface_has_no_step_at_the_junctions(make_slope_params, client, incline):
    # between neighbouring samples the surface may rise at most by the ramp gradient
    params = make_slope_params(incline=incline)
    SlopeEnvironment(params, client, seed=0)
    dx = 0.005
    xs = np.arange(-params.track_length / 2 + 0.01, params.track_length / 2 - 0.01, dx)
    jumps = np.abs(np.diff(_ray_surface(client, xs)))
    assert jumps.max() <= math.tan(math.radians(incline)) * dx + RAY_TOL


def test_ramp_spans_the_whole_track_width(env_and_client, make_slope_params):
    # the robot must not be able to walk around the ramp (project decision)
    env, client = env_and_client
    params = make_slope_params()
    x = 0.0  # middle of the ramp for the default parameters
    ys = np.linspace(-params.track_width / 2 + 0.01, params.track_width / 2 - 0.01, 9)
    hits = p.rayTestBatch(
        [[x, y, 10.0] for y in ys], [[x, y, -10.0] for y in ys], physicsClientId=client
    )
    expected = env.get_ground_level(x, 0.0)
    assert all(hit[3][2] == pytest.approx(expected, abs=RAY_TOL) for hit in hits)


def test_zero_incline_gives_a_flat_track(make_slope_params, client):
    params = make_slope_params(incline=0.0)
    env = SlopeEnvironment(params, client, seed=0)
    xs = np.linspace(
        -params.track_length / 2 + 0.01, params.track_length / 2 - 0.01, 50
    )
    assert np.abs(_ray_surface(client, xs)).max() < 1e-6
    assert all(env.get_ground_level(x, 0.0) == pytest.approx(0.0) for x in xs)


def test_plateau_is_below_the_top_of_the_walls(env_and_client, make_slope_params):
    env, client = env_and_client
    params = make_slope_params()
    _, wall_top = p.getAABB(env.wall_front, physicsClientId=client)
    assert wall_top[2] > _expected_rise(params)
