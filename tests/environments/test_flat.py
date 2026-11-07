"""Geometry specific to the flat environment (a)."""

import pybullet as p
import pytest

from neat_walking_robot.environments.flat import FlatEnvironment

TOL = 1e-6


@pytest.fixture
def env_and_client(make_base_params, client):
    return FlatEnvironment(make_base_params(), client, seed=0), client


def test_ground_top_surface_is_at_zero(env_and_client):
    env, client = env_and_client
    _, upper = p.getAABB(env.ground, physicsClientId=client)
    assert upper[2] == pytest.approx(0.0, abs=TOL)


def test_ground_thickness_matches_params(make_base_params, client):
    params = make_base_params(ground_thickness=0.35)
    env = FlatEnvironment(params, client, seed=0)
    lower, upper = p.getAABB(env.ground, physicsClientId=client)
    assert upper[2] - lower[2] == pytest.approx(0.35, abs=TOL)


def test_ground_covers_the_whole_track(env_and_client, make_base_params):
    env, client = env_and_client
    params = make_base_params()
    lower, upper = p.getAABB(env.ground, physicsClientId=client)
    assert (
        lower[0] <= -params.track_length / 2 + TOL
        and upper[0] >= params.track_length / 2 - TOL
    )
    assert (
        lower[1] <= -params.track_width / 2 + TOL
        and upper[1] >= params.track_width / 2 - TOL
    )


def test_dropped_box_rests_on_the_ground(env_and_client):
    # physical check, not only geometry: a box released above the track settles at z = half its height
    _env, client = env_and_client
    half = 0.1
    shape = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[half] * 3, physicsClientId=client
    )
    box = p.createMultiBody(
        1.0, shape, basePosition=[0.0, 0.0, 0.5], physicsClientId=client
    )
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    for _ in range(480):
        p.stepSimulation(physicsClientId=client)
    (_, _, z), _ = p.getBasePositionAndOrientation(box, physicsClientId=client)
    assert z == pytest.approx(half, abs=2e-3)  # small solver penetration is allowed
