"""Tests for the shared episode loop (training/episode.py) with A1 on the flat track.

End reasons and distances are produced on purpose: the test policy teleports the robot at chosen
action indices, so every branch is checked deterministically. Contract (agreed 2026-11-27):
- the policy is called once per action with the observation only (no time argument),
- max_distance = the LARGEST horizontal (x, y) distance of the base from its position after reset,
  in any direction; height changes do not count,
- toes_path_length = total path of the four toes in the world frame: the sum over toes and over
  actions of |toe position after the action - before it|, sampled once per action (k physics steps);
  the last action before an end condition counts too (2026-11-28),
- the episode ends on a fall or on contact with any of the four walls, otherwise at the time limit,
- duration is the simulated time actually run: (actions executed) * k * dt,
- the same policy gives identical metrics twice,
- all PyBullet calls go to the robot's own server.
"""

import dataclasses
import math

import numpy as np
import pybullet as p
import pytest

from neat_walking_robot.environments import BaseEnvironmentData, FlatEnvironment
from neat_walking_robot.robot import A1_ROBOT_SPEC, Robot
from neat_walking_robot.simulation import SimulationData, prepare_simulation
from neat_walking_robot.training.episode import EndReason, EpisodeParams, run_episode

DT = 1 / 240
EPISODE = EpisodeParams(episode_time=2.0, k=4)  # 120 actions; short, so the tests stay fast
ACTION_TIME = EPISODE.k * DT
N_ACTIONS = round(EPISODE.episode_time / ACTION_TIME)
OBS_LEN = 33
# Start at x = -2, walls: back at x = -3, front at x = +3, sides at y = +-1.5.
TRACK = BaseEnvironmentData(
    track_length=6.0, track_width=3.0, wall_height=1.0, wall_thickness=0.25,
    ground_thickness=0.2, friction=1.0, start_distance=1.0, end_distance=1.0,
)


def build(client_id: int, spec=A1_ROBOT_SPEC):
    prepare_simulation(SimulationData(), client_id)
    env = FlatEnvironment(TRACK, client_id, seed=0)
    return env, Robot(spec, env.get_starting_pos(), client_id)


@pytest.fixture
def world(client):
    return build(client)


def stand(obs):
    return np.zeros(12)


def make_wave():
    """Deterministic open-loop velocity pattern; the legs move, so the robot does not stand still."""
    state = {"n": 0}
    phases = np.linspace(0, math.pi, 12)

    def policy(obs):
        a = 0.6 * np.sin(2 * math.pi * state["n"] / 20 + phases)
        state["n"] += 1
        return a

    return policy


def teleports(robot, moves: dict):
    """Policy that stands and, right before action i, moves the robot as given by moves[i].

    moves[i] = (position_fn, euler): position_fn gets the start position after reset.
    """
    state = {"n": 0}

    def policy(obs):
        if state["n"] in moves:
            position_fn, euler = moves[state["n"]]
            p.resetBasePositionAndOrientation(
                robot.robot_id, position_fn(np.array(robot.start_base_position)),
                p.getQuaternionFromEuler(euler), physicsClientId=robot.client_id)
        state["n"] += 1
        return np.zeros(12)

    return policy


def shifted(dx: float, dy: float, dz: float = 0.0):
    return lambda start: (start[0] + dx, start[1] + dy, start[2] + dz)


LEVEL = (0, 0, 0)


# --- policy interface -------------------------------------------------------

def test_policy_gets_only_the_observation_once_per_action(world):
    env, robot = world
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return np.zeros(12)

    run_episode(robot, env, spy, DT, EPISODE)
    assert len(calls) == N_ACTIONS
    for args, kwargs in calls:
        assert kwargs == {} and len(args) == 1
        assert np.shape(args[0]) == (OBS_LEN,)


# --- progress ---------------------------------------------------------------

def test_standing_reaches_the_time_limit(world):
    env, robot = world
    m = run_episode(robot, env, stand, DT, EPISODE)
    assert m.end_reason == EndReason.TIME_LIMIT
    assert m.duration == pytest.approx(EPISODE.episode_time)
    assert m.episode_length == pytest.approx(EPISODE.episode_time)
    assert m.max_distance < 0.05


def test_max_distance_is_the_horizontal_distance_from_the_start(world):
    """Moved by (0.3, 0.4): max_distance = 0.5 m (Pythagoras), not 0.3 m along x."""
    env, robot = world
    m = run_episode(robot, env, teleports(robot, {30: (shifted(0.3, 0.4), LEVEL)}), DT, EPISODE)
    assert m.max_distance == pytest.approx(0.5, abs=0.03)


@pytest.mark.parametrize("dx, dy", [(0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)],
                         ids=["forward", "backward", "left", "right"])
def test_every_direction_counts_the_same(world, dx, dy):
    env, robot = world
    m = run_episode(robot, env, teleports(robot, {30: (shifted(dx, dy), LEVEL)}), DT, EPISODE)
    assert m.end_reason == EndReason.TIME_LIMIT
    assert m.max_distance == pytest.approx(0.5, abs=0.03)


def test_max_distance_is_the_largest_distance_not_the_final_one(world):
    """Out 0.5 m at action 30, back to the start at action 60: max_distance stays 0.5 m."""
    env, robot = world
    moves = {30: (shifted(0.5, 0), LEVEL), 60: (shifted(0, 0), LEVEL)}
    m = run_episode(robot, env, teleports(robot, moves), DT, EPISODE)
    assert m.max_distance == pytest.approx(0.5, abs=0.03)


def test_height_does_not_count_as_distance(world):
    """Lifted 0.3 m and dropped on the same spot: no horizontal distance."""
    env, robot = world
    m = run_episode(robot, env, teleports(robot, {30: (shifted(0, 0, 0.3), LEVEL)}), DT, EPISODE)
    assert m.max_distance < 0.05


# --- toes path --------------------------------------------------------------

def test_standing_toes_path_is_about_zero(world):
    """Planted feet barely move: the slow sag of a velocity-controlled robot is millimetres."""
    env, robot = world
    m = run_episode(robot, env, stand, DT, EPISODE)
    assert m.toes_path_length < 0.02


def test_toes_path_sums_the_four_toes_in_the_world_frame(world):
    """The whole robot moved by (0.3, 0.4) m: every toe goes 0.5 m, together 2.0 m."""
    env, robot = world
    m = run_episode(robot, env, teleports(robot, {30: (shifted(0.3, 0.4), LEVEL)}), DT, EPISODE)
    assert m.toes_path_length == pytest.approx(4 * 0.5, abs=0.05)


def test_toes_path_adds_up_back_and_forth(world):
    """Out 0.5 m and back: the path is 2 * 4 * 0.5 = 4 m, while max_distance stays 0.5 m."""
    env, robot = world
    moves = {30: (shifted(0.5, 0), LEVEL), 60: (shifted(0, 0), LEVEL)}
    m = run_episode(robot, env, teleports(robot, moves), DT, EPISODE)
    assert m.toes_path_length == pytest.approx(2 * 4 * 0.5, abs=0.1)


def test_last_action_before_the_end_counts(world):
    """Moved into the front wall before action 30: the episode ends after that action, and the
    jump of about 4.85 m per toe must already be in the path."""
    env, robot = world
    m = run_episode(robot, env, teleports(robot, {30: (wall_positions()["front"], LEVEL)}), DT, EPISODE)
    assert m.end_reason == EndReason.WALL
    jump = (TRACK.track_length / 2 - 0.15) - (-(TRACK.track_length / 2 - TRACK.start_distance))
    assert m.toes_path_length == pytest.approx(4 * jump, abs=0.1)


def test_toes_path_of_one_swinging_joint_matches_the_geometry(client):
    """Fixed trunk, legs in the air, only FR_hip turns at 0.5 * 21 rad/s for 4 actions (0.7 rad).

    The FR toe moves on a circle around the hip axis, so its path is r * |delta q|, where r is the
    distance of the toe from the axis. Per-action sampling measures chords of 0.175 rad arcs,
    0.13 % shorter than the arc, so 1 % tolerance.
    """
    spec = dataclasses.replace(A1_ROBOT_SPEC, use_fixed_base=True, start_height=1.0)
    env, robot = build(client, spec)
    hip = robot.moving_joints.index("FR_hip_joint")
    action = np.zeros(12)
    action[hip] = 0.5

    joint_id = robot.moving_joints_id["FR_hip_joint"]
    info = p.getJointInfo(robot.robot_id, joint_id, physicsClientId=client)
    episode = EpisodeParams(episode_time=4 * ACTION_TIME, k=EPISODE.k)
    m = run_episode(robot, env, lambda obs: action, DT, episode)

    # geometry after the episode: hip axis (world) through the joint, toe position, joint angle
    state = p.getLinkState(robot.robot_id, joint_id, computeForwardKinematics=True, physicsClientId=client)
    axis_point = np.array(state[4])  # link frame origin lies on the joint axis
    axis = np.array(p.getMatrixFromQuaternion(state[5])).reshape(3, 3) @ np.array(info[13])
    toe = np.array(robot.get_toes_world_position()["FR_toe"])
    r = np.linalg.norm(np.cross(toe - axis_point, axis))
    swept = abs(p.getJointState(robot.robot_id, joint_id, physicsClientId=client)[0]
                - A1_ROBOT_SPEC.rest_position["FR_hip_joint"])

    assert swept == pytest.approx(4 * ACTION_TIME * 0.5 * A1_ROBOT_SPEC.max_joint_velocity, rel=0.05)
    assert m.toes_path_length == pytest.approx(r * swept, rel=0.01)


# --- end reasons ------------------------------------------------------------

def test_fall_is_detected(world):
    env, robot = world
    m = run_episode(robot, env, teleports(robot, {30: (shifted(0, 0), (1.2, 0, 0))}), DT, EPISODE)
    assert m.end_reason == EndReason.FALL


def test_fall_uses_fall_threshold_not_the_observation_scale(client):
    """Roll 1.0 rad is past fall_threshold 0.8: the episode ends right after the tilt,
    even though roll_and_pitch_scale (2.0) is larger than the tilt."""
    spec = dataclasses.replace(A1_ROBOT_SPEC, fall_threshold=0.8, roll_and_pitch_scale=2.0)
    env, robot = build(client, spec)
    m = run_episode(robot, env, teleports(robot, {30: (shifted(0, 0), (1.0, 0, 0))}), DT, EPISODE)
    assert m.end_reason == EndReason.FALL
    assert m.duration <= 32 * ACTION_TIME + 1e-9


def test_no_fall_when_threshold_is_above_any_angle(client):
    """|roll| and |pitch| never exceed pi, so with fall_threshold = pi nothing counts as a fall,
    even with a small roll_and_pitch_scale (0.3) that a mixed-up field would use as threshold."""
    spec = dataclasses.replace(A1_ROBOT_SPEC, fall_threshold=math.pi, roll_and_pitch_scale=0.3)
    env, robot = build(client, spec)
    m = run_episode(robot, env, teleports(robot, {30: (shifted(0, 0), (1.0, 0, 0))}), DT, EPISODE)
    assert m.end_reason != EndReason.FALL


def wall_positions():
    """Base positions about 0.12-0.15 m from the inner face of each wall: the robot touches it."""
    half_l, half_w = TRACK.track_length / 2, TRACK.track_width / 2
    return {
        "left": lambda start: (start[0], -(half_w - 0.12), start[2]),
        "right": lambda start: (start[0], half_w - 0.12, start[2]),
        "back": lambda start: (-(half_l - 0.15), start[1], start[2]),
        "front": lambda start: (half_l - 0.15, start[1], start[2]),
    }


@pytest.mark.parametrize("wall", ["left", "right", "back", "front"])
def test_contact_with_any_wall_ends_the_episode(world, wall):
    """No wall is special any more: without a goal line the front wall ends the episode too.
    The episode stops right after the contact: moved before action 30, so 31 actions were run."""
    env, robot = world
    m = run_episode(robot, env, teleports(robot, {30: (wall_positions()[wall], LEVEL)}), DT, EPISODE)
    assert m.end_reason == EndReason.WALL
    assert m.duration == pytest.approx(31 * ACTION_TIME)


def test_wall_contact_stops_calling_the_policy(world):
    """After a wall contact no more actions are taken (a broken loop would keep the robot going)."""
    env, robot = world
    inner = teleports(robot, {30: (wall_positions()["front"], LEVEL)})
    calls = {"n": 0}

    def counting(obs):
        calls["n"] += 1
        return inner(obs)

    run_episode(robot, env, counting, DT, EPISODE)
    assert calls["n"] == 31


def test_duration_is_simulated_time_actually_run(world):
    """A tilt before action 30 ends the episode after it: 31 actions = 31 * k * dt."""
    env, robot = world
    m = run_episode(robot, env, teleports(robot, {30: (shifted(0, 0), (1.2, 0, 0))}), DT, EPISODE)
    assert m.duration == pytest.approx(31 * ACTION_TIME)


# --- determinism and isolation ----------------------------------------------

def test_episode_is_deterministic(world):
    env, robot = world
    first = run_episode(robot, env, make_wave(), DT, EPISODE)
    run_episode(robot, env, stand, DT, EPISODE)
    again = run_episode(robot, env, make_wave(), DT, EPISODE)
    assert first == again
    assert first.max_distance > 0.0
    assert first.toes_path_length > 0.0


def test_episode_runs_in_the_robot_server(two_clients):
    """Server 0 holds other bodies; the result must equal a run in a server of its own."""
    first, second = two_clients
    build(first)
    env, robot = build(second)
    shared = run_episode(robot, env, make_wave(), DT, EPISODE)

    alone = p.connect(p.DIRECT)
    try:
        env_alone, robot_alone = build(alone)
        reference = run_episode(robot_alone, env_alone, make_wave(), DT, EPISODE)
    finally:
        p.disconnect(alone)
    assert shared == reference
