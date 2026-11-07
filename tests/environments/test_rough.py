"""Terrain generation and height queries specific to the rough environment (b)."""

import numpy as np
import pybullet as p
import pytest

from neat_walking_robot.environments.rough import RoughEnvironment


def _height_profile(env, xs, ys):
    return np.array([[env.get_ground_level(x, y) for y in ys] for x in xs])


@pytest.fixture
def env_and_client(make_rough_params, client):
    return RoughEnvironment(make_rough_params(), client, seed=0), client


def test_grid_size_follows_track_and_cell_size(env_and_client, make_rough_params):
    env, _ = env_and_client
    params = make_rough_params()
    assert env.rows == round(params.track_length / params.cell_size) + 1
    assert env.cols == round(params.track_width / params.cell_size) + 1


def test_terrain_heights_stay_within_amplitude(env_and_client, make_rough_params):
    env, client = env_and_client
    params = make_rough_params()
    lower, upper = p.getAABB(env.ground, physicsClientId=client)
    # small margin: Bullet may pad the AABB of a concave shape
    assert lower[2] >= -params.amplitude - 0.01
    assert upper[2] <= params.amplitude + 0.01


def test_terrain_is_not_flat(env_and_client):
    env, _ = env_and_client
    xs = np.linspace(0.0, 2.5, 15)
    heights = _height_profile(env, xs, [0.0])
    assert heights.max() - heights.min() > 0.02


def test_start_strip_is_flat(env_and_client, make_rough_params):
    # the robot must start on level ground, otherwise it is tilted before it moves
    env, _ = env_and_client
    params = make_rough_params()
    back = -params.track_length / 2
    xs = np.linspace(back + 0.1, back + params.start_distance, 8)
    ys = np.linspace(-params.track_width / 2 + 0.1, params.track_width / 2 - 0.1, 5)
    assert np.abs(_height_profile(env, xs, ys)).max() < 1e-3


def test_transition_from_start_strip_has_no_step(env_and_client, make_rough_params):
    # neighbouring samples across the transition never jump by more than the steepest allowed slope
    env, _ = env_and_client
    params = make_rough_params()
    back = -params.track_length / 2
    step = params.cell_size
    xs = np.arange(
        back + params.start_distance,
        back + params.start_distance + 1.5 + params.transition_distance,
        step,
    )
    profile = _height_profile(env, xs, [0.0])[:, 0]
    max_jump = np.abs(np.diff(profile)).max()
    assert max_jump <= np.tan(np.radians(env.max_slope_deg)) * step + 1e-6


def test_same_seed_gives_same_terrain(make_rough_params, two_clients):
    first, second = two_clients
    params = make_rough_params()
    xs, ys = np.linspace(-2.5, 2.5, 11), np.linspace(-0.9, 0.9, 5)
    a = _height_profile(RoughEnvironment(params, first, seed=3), xs, ys)
    b = _height_profile(RoughEnvironment(params, second, seed=3), xs, ys)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_give_different_terrain(make_rough_params, two_clients):
    first, second = two_clients
    params = make_rough_params()
    xs, ys = np.linspace(0.0, 2.5, 11), np.linspace(-0.9, 0.9, 5)
    a = _height_profile(RoughEnvironment(params, first, seed=3), xs, ys)
    b = _height_profile(RoughEnvironment(params, second, seed=4), xs, ys)
    assert not np.allclose(a, b)


def test_ground_level_ignores_a_body_above_the_point(env_and_client):
    # the robot will stand exactly where we ask for the ground height
    env, client = env_and_client
    x, y = 1.2, 0.3
    before = env.get_ground_level(x, y)
    shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.2, physicsClientId=client)
    p.createMultiBody(
        1.0, shape, basePosition=[x, y, before + 0.5], physicsClientId=client
    )
    assert env.get_ground_level(x, y) == pytest.approx(before, abs=1e-9)


def test_dropped_box_rests_at_reported_ground_level(env_and_client):
    # get_ground_level must agree with what the physics engine treats as the surface
    env, client = env_and_client
    x, y, half = 1.5, -0.2, 0.02
    ground = env.get_ground_level(x, y)
    shape = p.createCollisionShape(p.GEOM_SPHERE, radius=half, physicsClientId=client)
    ball = p.createMultiBody(
        0.1, shape, basePosition=[x, y, ground + 0.05], physicsClientId=client
    )
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    for _ in range(30):  # long enough to touch down, short enough not to roll away
        p.stepSimulation(physicsClientId=client)
    contacts = p.getContactPoints(ball, env.ground, physicsClientId=client)
    assert contacts, "the ball never touched the terrain"
    (bx, by, _), _ = p.getBasePositionAndOrientation(ball, physicsClientId=client)
    contact_z = min(c[6][2] for c in contacts)
    assert contact_z == pytest.approx(env.get_ground_level(bx, by), abs=5e-3)


def test_max_slope_is_reported(env_and_client):
    env, _ = env_and_client
    assert env.max_slope_deg is not None
    assert 0.0 < env.max_slope_deg < 90.0
