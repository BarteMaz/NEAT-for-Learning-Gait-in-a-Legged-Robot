"""Tests for the shared episode loop (training/episode.py) with A1 on the flat track.

End reasons are produced on purpose: the test policy teleports the robot at a chosen
action index (tilted, next to a wall, past the goal), so every branch is checked
deterministically. Contract:
- every end reason is detected, including the back wall,
- duration is the simulated time actually run: (actions executed) * k * dt,
- the same policy gives identical metrics twice,
- all PyBullet calls go to the robot's own server.
"""

import math

import numpy as np
import pybullet as p
import pytest

from neat_walking_robot.environments import BaseEnvironmentData, FlatEnvironment
from neat_walking_robot.robot import A1_ROBOT_SPEC, Robot
from neat_walking_robot.simulation import SimulationData, prepare_simulation
from neat_walking_robot.training.episode import EndReason, EpisodeParams, run_episode

DT = 1 / 240
EPISODE = EpisodeParams(episode_time=10.0, frequency=3, k=4)
ACTION_TIME = EPISODE.k * DT
# D = 6 - 1 - 1 = 4 m, walls at y = +-1.5 (agreed 2026-11-15)
TRACK = BaseEnvironmentData(
    track_length=6.0, track_width=3.0, wall_height=1.0, wall_thickness=0.25,
    ground_thickness=0.2, friction=1.0, start_distance=1.0, end_distance=1.0,
)


def build(client_id: int):
    prepare_simulation(SimulationData(), client_id)
    env = FlatEnvironment(TRACK, client_id, seed=0)
    return env, Robot(A1_ROBOT_SPEC, env.get_starting_pos(), client_id)


@pytest.fixture
def world(client):
    return build(client)


def stand(obs, t):
    return np.zeros(12)


def baseline(obs, t, f=3.0, at=0.4, ac=0.3):
    """The open-loop trot chosen as baseline on 2026-11-15."""
    ph = 2 * math.pi * f * t
    a, b = at * math.sin(ph), ac * math.sin(ph + math.pi / 2)
    c, d = at * math.sin(ph + math.pi), ac * math.sin(ph + 1.5 * math.pi)
    return np.array([0, a, b, 0, c, d, 0, c, d, 0, a, b])


def teleport_at(robot, action_index: int, position_fn, orientation=(0, 0, 0)):
    """Policy that stands, and moves the robot once, right before action `action_index`."""
    state = {"n": 0}

    def policy(obs, t):
        if state["n"] == action_index:
            pos, _ = p.getBasePositionAndOrientation(robot.robot_id, physicsClientId=robot.client_id)
            p.resetBasePositionAndOrientation(
                robot.robot_id, position_fn(pos), p.getQuaternionFromEuler(orientation),
                physicsClientId=robot.client_id)
        state["n"] += 1
        return np.zeros(12)

    return policy


def test_standing_reaches_the_time_limit(world):
    env, robot = world
    m = run_episode(robot, env, stand, DT, EPISODE)
    assert m.end_reason == EndReason.TIME_LIMIT
    assert m.duration == pytest.approx(EPISODE.episode_time)
    assert abs(m.progress) < 0.05
    assert m.goal_distance == pytest.approx(4.0)


def test_baseline_reaches_the_goal(world):
    env, robot = world
    m = run_episode(robot, env, baseline, DT, EPISODE)
    assert m.end_reason == EndReason.GOAL
    assert m.duration < EPISODE.episode_time
    assert m.progress >= m.goal_distance - 0.05


def test_fall_is_detected(world):
    env, robot = world
    policy = teleport_at(robot, 30, lambda pos: pos, orientation=(1.2, 0, 0))
    m = run_episode(robot, env, policy, DT, EPISODE)
    assert m.end_reason == EndReason.FALL


@pytest.mark.parametrize("side", [-1, 1])
def test_side_wall_contact_is_detected(world, side):
    env, robot = world
    y_wall = side * (TRACK.track_width / 2 - 0.12)
    policy = teleport_at(robot, 30, lambda pos: (pos[0], y_wall, pos[2]))
    m = run_episode(robot, env, policy, DT, EPISODE)
    assert m.end_reason == EndReason.WALL


def test_back_wall_contact_is_detected(world):
    env, robot = world
    x_wall = -(TRACK.track_length / 2 - 0.15)
    policy = teleport_at(robot, 30, lambda pos: (x_wall, pos[1], pos[2]))
    m = run_episode(robot, env, policy, DT, EPISODE)
    assert m.end_reason == EndReason.WALL


def test_goal_is_detected_and_progress_is_measured(world):
    env, robot = world
    x_goal = TRACK.track_length / 2 - TRACK.end_distance
    policy = teleport_at(robot, 30, lambda pos: (x_goal + 0.05, pos[1], pos[2]))
    m = run_episode(robot, env, policy, DT, EPISODE)
    assert m.end_reason == EndReason.GOAL
    assert m.progress == pytest.approx(x_goal + 0.05 - robot.start_base_position[0], abs=0.02)


def test_duration_is_simulated_time_actually_run(world):
    """Teleport before action 30 ends the episode after it: 31 actions = 31 * k * dt."""
    env, robot = world
    x_goal = TRACK.track_length / 2 - TRACK.end_distance
    policy = teleport_at(robot, 30, lambda pos: (x_goal + 0.05, pos[1], pos[2]))
    m = run_episode(robot, env, policy, DT, EPISODE)
    assert m.duration == pytest.approx(31 * ACTION_TIME)


def test_episode_is_deterministic(world):
    env, robot = world
    first = run_episode(robot, env, baseline, DT, EPISODE)
    run_episode(robot, env, stand, DT, EPISODE)
    again = run_episode(robot, env, baseline, DT, EPISODE)
    assert first == again


def test_episode_runs_in_the_robot_server(two_clients):
    """Server 0 holds other bodies; stepping the wrong server would leave the robot frozen."""
    first, second = two_clients
    build(first)
    env, robot = build(second)
    m = run_episode(robot, env, baseline, DT, EPISODE)
    assert m.progress > 1.0
