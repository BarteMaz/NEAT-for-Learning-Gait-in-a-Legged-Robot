"""Tests for the Robot class and RobotSpec (neat_walking_robot.robot.robot), on A1.

Contract checked here (agreed 2026-11-12):
- the spec refers to joints and feet by name; unknown names give ValueError,
- reset() puts the robot in the rest pose and lets it settle on its feet,
- actions are in [-1, 1]; 0 means the rest pose, +-1 the ends of the working range,
- observations have a fixed length and order and are scaled to roughly [-1, 1],
- every PyBullet call goes to the robot's own server.

Observation layout (from robot.py, documented order):
  [0:12] joint angles, [12:24] joint velocities, [24:26] roll, pitch,
  [26:29] base angular velocity, [29:33] toe contacts, [33] yaw (added 2026-11-14).
"""

import dataclasses
import math

import numpy as np
import pybullet as p
import pytest

from neat_walking_robot.environments import BaseEnvironmentData, FlatEnvironment
from neat_walking_robot.robot.robot import (
    A1_ROBOT_SPEC,
    Robot,
    scale_to_angles,
    scale_to_neat_input,
)
from neat_walking_robot.simulation import SimulationData, prepare_simulation

N_JOINTS = 12
OBS_LEN = 34
JOINT_POS = slice(0, 12)
JOINT_VEL = slice(12, 24)
ROLL, PITCH = 24, 25
ANG_VEL = slice(26, 29)
CONTACTS = slice(29, 33)
YAW = 33

ANGLE_TOL = 0.05  # rad; position control under load leaves a small error


# Flat track large enough for A1; same shape as BASE_PARAMS in conftest.py.
FLAT_PARAMS = BaseEnvironmentData(
    track_length=6.0, track_width=2.0, wall_height=1.0, wall_thickness=0.15,
    ground_thickness=0.2, friction=0.8, start_distance=1.0, end_distance=1.0,
)


def make_robot(client_id: int, spec=A1_ROBOT_SPEC) -> Robot:
    prepare_simulation(SimulationData(), client_id)
    env = FlatEnvironment(FLAT_PARAMS, client_id, seed=0)
    return Robot(spec, env.get_starting_pos(), client_id)


def step(robot: Robot, n: int) -> None:
    for _ in range(n):
        p.stepSimulation(physicsClientId=robot.client_id)


def joint_angles(robot: Robot) -> dict[str, float]:
    return {name: state[0] for name, state in robot.get_joints_state().items()}


def base_state(robot: Robot):
    pos, orn = p.getBasePositionAndOrientation(robot.robot_id, physicsClientId=robot.client_id)
    return np.array(pos), np.array(p.getEulerFromQuaternion(orn))


@pytest.fixture
def robot(client) -> Robot:
    return make_robot(client)


# --- spec and loading -------------------------------------------------------

def test_finds_twelve_moving_joints_and_four_toes(robot):
    assert len(robot.moving_joints_id) == N_JOINTS
    assert set(robot.toes_links_id) == set(A1_ROBOT_SPEC.toes_links_names)


def test_unknown_toe_name_raises(client):
    """A typo in a foot name must fail loudly, not silently drop a foot."""
    spec = dataclasses.replace(A1_ROBOT_SPEC, toes_links_names=("FR_toe", "FL_toe", "RR_toe", "RL_foot"))
    with pytest.raises(ValueError):
        make_robot(client, spec)


def test_unknown_joint_in_rest_position_raises(client):
    """An extra or misspelled joint name in rest_position must fail loudly, not be ignored."""
    rest = {**A1_ROBOT_SPEC.rest_position, "FR_knee_joint": -1.5}
    spec = dataclasses.replace(A1_ROBOT_SPEC, rest_position=rest)
    with pytest.raises(ValueError):
        make_robot(client, spec)


def test_working_ranges_lie_inside_joint_limits(robot):
    for name, (lo, hi) in robot.moving_joints_limits.items():
        info = p.getJointInfo(robot.robot_id, robot.moving_joints_id[name], physicsClientId=robot.client_id)
        assert info[8] - 1e-9 <= lo < hi <= info[9] + 1e-9


# --- reset ------------------------------------------------------------------

def test_reset_puts_joints_in_rest_pose(robot):
    for name, angle in joint_angles(robot).items():
        assert angle == pytest.approx(A1_ROBOT_SPEC.rest_position[name], abs=ANGLE_TOL)


def test_robot_stands_after_reset(robot):
    """After settling all four toes touch the ground and the trunk is level."""
    pos, rpy = base_state(robot)
    assert robot.get_toes_contact_points() == dict.fromkeys(A1_ROBOT_SPEC.toes_links_names, 1)
    assert abs(rpy[0]) < 0.05 and abs(rpy[1]) < 0.05
    assert 0.2 < pos[2] < 0.4


def test_start_position_is_recorded_after_settling(robot):
    pos, _ = base_state(robot)
    assert np.allclose(robot.start_base_position, pos, atol=1e-12)


def test_reset_is_reproducible(robot):
    """Reset after an episode of random actions restores exactly the same state."""
    first_pos, first_rpy = base_state(robot)
    first_joints = robot.get_joints_state()

    rng = np.random.default_rng(0)
    for _ in range(60):
        robot.apply_action(rng.uniform(-1, 1, N_JOINTS))
        step(robot, 4)
    robot.reset()

    pos, rpy = base_state(robot)
    assert np.allclose(pos, first_pos, atol=1e-9)
    assert np.allclose(rpy, first_rpy, atol=1e-9)
    for name, state in robot.get_joints_state().items():
        assert np.allclose(state, first_joints[name], atol=1e-9)


def run_episode(robot: Robot, seed: int, n_actions: int = 60) -> np.ndarray:
    """Reset, then apply seeded random actions (4 physics steps each); return the final state."""
    robot.reset()
    rng = np.random.default_rng(seed)
    for _ in range(n_actions):
        robot.apply_action(rng.uniform(-1, 1, N_JOINTS))
        step(robot, 4)
    pos, rpy = base_state(robot)
    joints = np.array([s[0] for s in robot.get_joints_state().values()])
    return np.concatenate([pos, rpy, joints])


def test_episode_is_reproducible_regardless_of_previous_episode(robot):
    """The same actions after reset give bit-identical results, even if another episode ran in between.

    Without this, a genome's fitness would depend on which genome was evaluated before it.
    """
    first = run_episode(robot, seed=1)
    run_episode(robot, seed=2)
    again = run_episode(robot, seed=1)
    assert np.array_equal(first, again)


# --- actions ----------------------------------------------------------------

def test_scale_to_angles_maps_ends_and_middle():
    rng = np.array([-0.5, 1.5])
    assert scale_to_angles(-1.0, rng) == pytest.approx(-0.5)
    assert scale_to_angles(1.0, rng) == pytest.approx(1.5)
    assert scale_to_angles(0.0, rng) == pytest.approx(0.5)
    assert scale_to_angles(7.0, rng) == pytest.approx(1.5)  # clipped
    assert scale_to_angles(-7.0, rng) == pytest.approx(-0.5)


def test_scale_to_neat_input_inverts_scale_to_angles():
    rng = np.array([-2.0, -0.9])
    for a in (-1.0, -0.3, 0.0, 0.8, 1.0):
        assert scale_to_neat_input(scale_to_angles(a, rng), rng) == pytest.approx(a)


def test_apply_action_passes_effort_as_force_limit(robot, monkeypatch):
    """Without forces= PyBullet allows hundreds of N*m; every joint must be limited to its URDF effort.

    A1 rarely needs its full effort, so the limit is checked on the call itself.
    """
    calls = []
    real = p.setJointMotorControlArray

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(p, "setJointMotorControlArray", spy)
    robot.apply_action(np.zeros(N_JOINTS))
    assert len(calls) == 1
    forces = dict(zip(calls[0]["jointIndices"], calls[0]["forces"]))
    for name, joint_id in robot.moving_joints_id.items():
        assert forces[joint_id] == pytest.approx(robot.moving_joints_efforts[name])


def test_zero_action_holds_rest_pose(robot):
    """Action 0 is the rest pose: the robot keeps standing for 2 s."""
    for _ in range(120):
        robot.apply_action(np.zeros(N_JOINTS))
        step(robot, 4)
    for name, angle in joint_angles(robot).items():
        assert angle == pytest.approx(A1_ROBOT_SPEC.rest_position[name], abs=ANGLE_TOL)
    pos, rpy = base_state(robot)
    assert pos[2] > 0.2 and abs(rpy[0]) < 0.1 and abs(rpy[1]) < 0.1


@pytest.mark.parametrize("value", [-1.0, 1.0])
def test_extreme_actions_stay_in_working_range(client, value):
    """+-1 drives every joint to an end of its working range, never beyond."""
    robot = make_robot(client)
    p.resetBasePositionAndOrientation(  # hang in the air so legs move freely
        robot.robot_id, [0, 0, 1.0], [0, 0, 0, 1], physicsClientId=client)
    p.createConstraint(robot.robot_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 1.0],
                       physicsClientId=client)
    max_torque = dict.fromkeys(robot.moving_joints_id, 0.0)
    for _ in range(120):
        robot.apply_action(np.full(N_JOINTS, value))
        step(robot, 4)
        for name, joint_id in robot.moving_joints_id.items():
            torque = p.getJointState(robot.robot_id, joint_id, physicsClientId=client)[3]
            max_torque[name] = max(max_torque[name], abs(torque))
    for name, torque in max_torque.items():
        assert torque <= robot.moving_joints_efforts[name] + 1e-6, f"{name} exceeded effort"
    for name, angle in joint_angles(robot).items():
        lo, hi = robot.moving_joints_limits[name]
        target = lo if value < 0 else hi
        assert angle == pytest.approx(target, abs=ANGLE_TOL)


# --- observations -----------------------------------------------------------

def test_observation_has_fixed_length_and_is_finite(robot):
    obs = robot.get_observation()
    assert obs.shape == (OBS_LEN,)
    assert np.all(np.isfinite(obs))


def test_observation_at_rest(robot):
    """After reset: joints at the middle of their range, nothing moving, all toes down."""
    obs = robot.get_observation()
    assert np.allclose(obs[JOINT_POS], 0.0, atol=0.1)
    assert np.allclose(obs[JOINT_VEL], 0.0, atol=0.05)
    assert np.all(obs[CONTACTS] == 1)


def test_joint_velocities_are_scaled(robot):
    name = "FR_upper_joint"
    idx = list(robot.moving_joints_id).index(name)
    p.resetJointState(robot.robot_id, robot.moving_joints_id[name], A1_ROBOT_SPEC.rest_position[name],
                      targetVelocity=2.0, physicsClientId=robot.client_id)
    obs = robot.get_observation()
    assert obs[JOINT_VEL][idx] == pytest.approx(2.0 / A1_ROBOT_SPEC.joints_velocity_scale, abs=1e-9)


def test_roll_and_pitch_are_scaled_by_roll_and_pitch_scale(client):
    """Roll equal to roll_and_pitch_scale reads as 1.0.

    The observation scale and the fall threshold are separate fields (2026-11-21), so the
    spec gives them different values: dividing by fall_threshold would fail here.
    """
    spec = dataclasses.replace(A1_ROBOT_SPEC, roll_and_pitch_scale=0.5, fall_threshold=1.2)
    robot = make_robot(client, spec)
    pos, _ = base_state(robot)
    roll, pitch = 0.4, -0.2
    p.resetBasePositionAndOrientation(robot.robot_id, pos, p.getQuaternionFromEuler([roll, pitch, 0]),
                                      physicsClientId=robot.client_id)
    obs = robot.get_observation()
    assert obs[ROLL] == pytest.approx(roll / 0.5, abs=1e-6)
    assert obs[PITCH] == pytest.approx(pitch / 0.5, abs=1e-6)


def test_yaw_is_scaled_by_half_pi(robot):
    """Heading error of 90 degrees reads as 1.0; the robot starts facing +x, so yaw starts near 0."""
    assert abs(robot.get_observation()[YAW]) < 0.05
    pos, _ = base_state(robot)
    p.resetBasePositionAndOrientation(robot.robot_id, pos, p.getQuaternionFromEuler([0, 0, 0.6]),
                                      physicsClientId=robot.client_id)
    assert robot.get_observation()[YAW] == pytest.approx(0.6 / (math.pi / 2), abs=1e-6)


def test_base_angular_velocity_is_scaled(robot):
    p.resetBaseVelocity(robot.robot_id, [0, 0, 0], [0.5, -1.0, 2.5], physicsClientId=robot.client_id)
    obs = robot.get_observation()
    expected = np.array([0.5, -1.0, 2.5]) / A1_ROBOT_SPEC.base_velocity_scale
    assert np.allclose(obs[ANG_VEL], expected, atol=1e-6)


def test_contacts_drop_to_zero_in_the_air(robot):
    pos, orn = p.getBasePositionAndOrientation(robot.robot_id, physicsClientId=robot.client_id)
    p.resetBasePositionAndOrientation(robot.robot_id, [pos[0], pos[1], pos[2] + 0.5], orn,
                                      physicsClientId=robot.client_id)
    step(robot, 1)
    assert np.all(robot.get_observation()[CONTACTS] == 0)


# --- isolation between servers ----------------------------------------------

def test_robot_uses_its_own_server(two_clients):
    """A robot in the second server gives the same observation as alone and leaves server 0 untouched."""
    first, second = two_clients
    bodies_before = p.getNumBodies(physicsClientId=first)

    robot = make_robot(second)
    obs = robot.get_observation()
    assert p.getNumBodies(physicsClientId=first) == bodies_before

    alone = p.connect(p.DIRECT)
    try:
        reference = make_robot(alone).get_observation()
    finally:
        p.disconnect(alone)
    assert np.allclose(obs, reference, atol=1e-12)
