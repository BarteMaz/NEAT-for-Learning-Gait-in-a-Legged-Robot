"""Contract shared by every environment: server isolation, enclosure, start pose.

Each test runs once per environment class, so a new environment (e.g. the ramp)
only needs to be added to ENVIRONMENTS to get the same checks.
"""

import pybullet as p
import pytest

from neat_walking_robot.environments.base import BaseEnvironment
from neat_walking_robot.environments.flat import FlatEnvironment
from neat_walking_robot.environments.heightfield import RoughEnvironment

TOL = 1e-6  # metres; geometry is computed exactly, only float rounding is allowed

ENVIRONMENTS = [
    pytest.param((FlatEnvironment, "make_base_params"), id="flat"),
    pytest.param((RoughEnvironment, "make_rough_params"), id="rough"),
]


@pytest.fixture(params=ENVIRONMENTS)
def build(request):
    """Returns (build_function, params) for one environment class.

    build_function(client, seed=0) creates the environment in the given server.
    """
    env_class, params_fixture = request.param
    params = request.getfixturevalue(params_fixture)()

    def build_function(client: int, seed: int = 0):
        return env_class(params, client, seed)

    return build_function, params


def test_base_environment_is_abstract(make_base_params, client):
    with pytest.raises(TypeError):
        BaseEnvironment(make_base_params(), client, 0)


# --- Server isolation (the physicsClientId bug class) -----------------------


def test_builds_only_in_its_own_server(build, two_clients):
    build_env, _ = build
    first, second = two_clients
    build_env(second)
    assert p.getNumBodies(physicsClientId=first) == 0
    assert p.getNumBodies(physicsClientId=second) > 0


def test_does_not_modify_bodies_in_another_server(build, two_clients):
    build_env, _ = build
    first, second = two_clients
    shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1, physicsClientId=first)
    other = p.createMultiBody(1.0, shape, physicsClientId=first)
    friction_before = p.getDynamicsInfo(other, -1, physicsClientId=first)[1]

    build_env(second)

    assert p.getNumBodies(physicsClientId=first) == 1
    assert p.getDynamicsInfo(other, -1, physicsClientId=first)[1] == friction_before


def test_ground_exists_after_construction(build, client):
    build_env, _ = build
    env = build_env(client)
    assert env.ground is not None and env.ground >= 0


def test_ground_friction_matches_params(build, two_clients):
    # built in the second server on purpose: a missing physicsClientId would leave the default friction
    build_env, params = build
    _, second = two_clients
    env = build_env(second)
    assert p.getDynamicsInfo(env.ground, -1, physicsClientId=second)[
        1
    ] == pytest.approx(params.friction)


def test_ground_level_query_uses_own_server(build, two_clients):
    build_env, _params = build
    _, second = two_clients
    env = build_env(second)
    start_x, start_y, _ = env.get_starting_pos()
    assert env.get_ground_level(start_x, start_y) == pytest.approx(0.0, abs=1e-3)


# --- Enclosure -----------------------------------------------------------------


def _aabb(body: int, client: int):
    lower, upper = p.getAABB(body, physicsClientId=client)
    return lower, upper


def test_side_walls_inner_faces_bound_the_track(build, client):
    build_env, params = build
    env = build_env(client)
    _left_lower, left_upper = _aabb(env.wall_left, client)
    right_lower, _right_upper = _aabb(env.wall_right, client)
    assert left_upper[1] == pytest.approx(-params.track_width / 2, abs=TOL)
    assert right_lower[1] == pytest.approx(params.track_width / 2, abs=TOL)


def test_end_walls_inner_faces_bound_the_track(build, client):
    build_env, params = build
    env = build_env(client)
    _back_lower, back_upper = _aabb(env.wall_back, client)
    front_lower, _front_upper = _aabb(env.wall_front, client)
    assert back_upper[0] == pytest.approx(-params.track_length / 2, abs=TOL)
    assert front_lower[0] == pytest.approx(params.track_length / 2, abs=TOL)


def test_walls_reach_below_the_lowest_ground_point(build, client):
    # otherwise a valley in the terrain opens a gap under the wall
    build_env, params = build
    env = build_env(client)
    lowest = params.lowest_ground_point
    for wall in (env.wall_left, env.wall_right, env.wall_front, env.wall_back):
        lower, _ = _aabb(wall, client)
        assert lower[2] <= lowest + TOL, (
            f"wall bottom at {lower[2]:.3f} m, lowest ground at {lowest:.3f} m"
        )


def test_walls_reach_the_declared_height(build, client):
    build_env, params = build
    env = build_env(client)
    for wall in (env.wall_left, env.wall_right, env.wall_front, env.wall_back):
        _, upper = _aabb(wall, client)
        assert upper[2] >= params.wall_height - TOL


def test_ground_lies_between_wall_bottom_and_top(build, client):
    build_env, _ = build
    env = build_env(client)
    _ground_lower, ground_upper = _aabb(env.ground, client)
    wall_lower, wall_upper = _aabb(env.wall_left, client)
    # the top surface of the ground must stay inside the vertical span of the walls
    assert wall_lower[2] <= ground_upper[2] <= wall_upper[2]


# --- Start pose and goal ------------------------------------------------------------


def test_start_position_matches_params(build, client):
    build_env, params = build
    env = build_env(client)
    x, y, z = env.get_starting_pos()
    assert x == pytest.approx(-params.track_length / 2 + params.start_distance)
    assert y == pytest.approx(0.0)
    assert z == pytest.approx(env.get_ground_level(x, y), abs=1e-6)


def test_start_is_inside_the_track_and_before_the_goal(build, client):
    build_env, params = build
    env = build_env(client)
    x, y, _ = env.get_starting_pos()
    goal_x = params.track_length / 2 - params.end_distance
    assert -params.track_length / 2 < x < goal_x
    assert -params.track_width / 2 < y < params.track_width / 2


def test_rebuilding_with_same_seed_gives_identical_world(build):
    # reproducibility: two independent servers, same params and seed -> same geometry
    build_env, _params = build
    probes = [(x, y) for x in (-2.0, -0.5, 0.7, 2.2) for y in (-0.8, 0.0, 0.6)]
    results = []
    for _ in range(2):
        client = p.connect(p.DIRECT)
        try:
            env = build_env(client, seed=7)
            results.append([env.get_ground_level(x, y) for x, y in probes])
        finally:
            p.disconnect(client)
    assert results[0] == results[1]
