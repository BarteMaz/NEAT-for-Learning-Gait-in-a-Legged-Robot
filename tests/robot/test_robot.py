"""Tests for the Robot class and RobotSpec (neat_walking_robot.robot.robot), on A1.

Contract (velocity control, agreed 2026-11-27; replaces the position-control contract of 2026-11-12):
- the spec refers to joints and feet by name; unknown names give ValueError,
- reset() puts the robot in the rest pose and lets it settle on its feet,
- actions are in [-1, 1] (clipped outside) and mean joint velocities: a -> a * max_joint_velocity,
  sent with VELOCITY_CONTROL and max_joint_effort as the torque limit of every joint,
- max_joint_velocity = 21 rad/s and max_joint_effort = 33.5 N*m: the A1 motor spec given by Unitree
  (one source for both, agreed 2026-11-27; the URDF values 28.6 / 52.4 rad/s and 20 / 55 N*m are not used),
- reset() holds the rest pose with POSITION_CONTROL limited to max_joint_effort,
- joint position limits are the URDF limits (no software working range),
- use_fixed_base pins the trunk in place (diagnostic experiment with swinging legs),
- contact friction (agreed 2026-11-28): the toe links get toes_friction, every other part of the robot,
  the trunk (base, index -1) included, gets body_friction; Bullet multiplies them with the ground
  friction (physics/README.md). PyBullet's default damping 0.04 (a drag with no physical source) is
  switched off; in Bullet it is one value for the whole multibody, not per link,
- get_toes_world_position() gives the world position of every toe, by toe name (2026-11-28),
- observations have a fixed length and order and are scaled to roughly [-1, 1],
- every PyBullet call goes to the robot's own server.

Observation layout (33 values; phase sin/cos and yaw removed 2026-11-27):
  [0:12] joint angles (URDF limits -> [-1, 1]), [12:24] joint velocities / max_joint_velocity,
  [24:26] roll, pitch, [26:29] base angular velocity in the BODY frame (what the A1 IMU gyroscope measures,
  agreed 2026-11-27) / base_velocity_scale, [29:33] toe contacts.
The observation does not depend on the heading: no yaw, and no world-frame vectors.
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
    scale_to_neat_input,
    scale_to_velocity,
)
from neat_walking_robot.simulation import SimulationData, prepare_simulation

N_JOINTS = 12
K = 4  # physics steps per action, as in the training config
OBS_LEN = 33
JOINT_POS = slice(0, 12)
JOINT_VEL = slice(12, 24)
ROLL, PITCH = 24, 25
ANG_VEL = slice(26, 29)
CONTACTS = slice(29, 33)

V_MAX = A1_ROBOT_SPEC.max_joint_velocity
EFFORT = A1_ROBOT_SPEC.max_joint_effort
ANGLE_TOL = 0.05  # rad
# Bullet joint limits are soft: driven into a limit at full torque a joint overshoots it by ~0.07 rad.
LIMIT_TOL = 0.1  # rad

# Flat track large enough for A1; same shape as BASE_PARAMS in conftest.py.
FLAT_PARAMS = BaseEnvironmentData(
    track_length=6.0, track_width=2.0, wall_height=1.0, wall_thickness=0.15,
    ground_thickness=0.2, friction=0.8, start_distance=1.0, end_distance=1.0,
)
# Trunk pinned 1 m above the ground, so the legs swing freely.
HANGING_SPEC = dataclasses.replace(A1_ROBOT_SPEC, use_fixed_base=True, start_height=1.0)


def make_robot(client_id: int, spec=A1_ROBOT_SPEC) -> Robot:
    prepare_simulation(SimulationData(), client_id)
    env = FlatEnvironment(FLAT_PARAMS, client_id, seed=0)
    return Robot(spec, env.get_starting_pos(), client_id)


def step(robot: Robot, n: int) -> None:
    for _ in range(n):
        p.stepSimulation(physicsClientId=robot.client_id)


def act(robot: Robot, action, n_actions: int = 1) -> None:
    for _ in range(n_actions):
        robot.apply_action(action)
        step(robot, K)


def joint_angles(robot: Robot) -> dict[str, float]:
    return {name: state[0] for name, state in robot.get_joints_state().items()}


def joint_velocities(robot: Robot) -> dict[str, float]:
    return {name: state[1] for name, state in robot.get_joints_state().items()}


def base_state(robot: Robot):
    pos, orn = p.getBasePositionAndOrientation(robot.robot_id, physicsClientId=robot.client_id)
    return np.array(pos), np.array(p.getEulerFromQuaternion(orn))


def spy_on_motor_calls(monkeypatch) -> list[dict]:
    calls = []
    real = p.setJointMotorControlArray

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(p, "setJointMotorControlArray", spy)
    return calls


@pytest.fixture
def robot(client) -> Robot:
    return make_robot(client)


@pytest.fixture
def hanging(client) -> Robot:
    return make_robot(client, HANGING_SPEC)


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


def test_position_limits_are_the_urdf_limits(robot):
    for name, (lo, hi) in robot.moving_joints_position_limits.items():
        info = p.getJointInfo(robot.robot_id, robot.moving_joints_id[name], physicsClientId=robot.client_id)
        assert (lo, hi) == (info[8], info[9])
        assert lo < hi


def test_motor_limits_are_the_a1_motor_spec():
    """21 rad/s and 33.5 N*m from Unitree's A1 motor spec, not the URDF (28.6 / 52.4 rad/s, 20 / 55 N*m)."""
    assert A1_ROBOT_SPEC.max_joint_velocity == pytest.approx(21.0)
    assert A1_ROBOT_SPEC.max_joint_effort == pytest.approx(33.5)


def test_reset_limits_the_holding_torque_to_max_joint_effort(client, monkeypatch):
    """The rest pose is held by POSITION_CONTROL motors; every one of them gets force = max_joint_effort."""
    robot = make_robot(client)
    calls = []
    real = p.setJointMotorControl2

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(p, "setJointMotorControl2", spy)
    robot.reset()
    assert len(calls) == N_JOINTS
    assert {c["jointIndex"] for c in calls} == set(robot.moving_joints_id.values())
    for call in calls:
        assert call["controlMode"] == p.POSITION_CONTROL
        assert call["force"] == pytest.approx(EFFORT)


def test_velocity_limits_are_symmetric_max_joint_velocity(robot):
    for lo, hi in robot.moving_joints_velocity_limits.values():
        assert (lo, hi) == (-V_MAX, V_MAX)


def test_floating_base_is_the_default():
    assert A1_ROBOT_SPEC.use_fixed_base is False


def test_fixed_base_keeps_the_trunk_in_place(hanging):
    """Random leg motion must not move a pinned trunk, neither in position nor in orientation."""
    start_pos, start_rpy = base_state(hanging)
    rng = np.random.default_rng(0)
    for _ in range(120):
        act(hanging, rng.uniform(-1, 1, N_JOINTS))
    pos, rpy = base_state(hanging)
    assert np.allclose(pos, start_pos, atol=1e-9)
    assert np.allclose(rpy, start_rpy, atol=1e-9)
    assert pos[2] > 0.9


# --- contact friction and damping -------------------------------------------

def all_link_indices(robot: Robot) -> list[int]:
    return [-1, *range(p.getNumJoints(robot.robot_id, physicsClientId=robot.client_id))]


def lateral_friction(robot: Robot, link: int) -> float:
    return p.getDynamicsInfo(robot.robot_id, link, physicsClientId=robot.client_id)[1]


def test_toes_get_toes_friction(robot):
    for link in robot.toes_links_id.values():
        assert lateral_friction(robot, link) == pytest.approx(A1_ROBOT_SPEC.toes_friction)


def test_every_other_part_gets_body_friction_trunk_included(robot):
    """The trunk is the base (index -1), not one of the joints, so a loop over joints misses it.
    It is the part that slides on the ground when the robot falls or crawls."""
    toes = set(robot.toes_links_id.values())
    for link in all_link_indices(robot):
        if link not in toes:
            assert lateral_friction(robot, link) == pytest.approx(A1_ROBOT_SPEC.body_friction), f"link {link}"


def test_friction_comes_from_the_spec(client):
    spec = dataclasses.replace(A1_ROBOT_SPEC, toes_friction=0.37, body_friction=0.21)
    robot = make_robot(client, spec)
    toes = set(robot.toes_links_id.values())
    for link in all_link_indices(robot):
        expected = 0.37 if link in toes else 0.21
        assert lateral_friction(robot, link) == pytest.approx(expected), f"link {link}"


def test_damping_is_switched_off(robot):
    """With PyBullet's default damping 0.04 the robot falls 4.58 m in 1 s instead of g*t^2/2 (measured
    2026-11-28). Damping is a property of the whole multibody, so one changeDynamics call switches it off
    for every link. Checked by a free fall well above the ground: the rigid robot must fall like a stone."""
    pos, orn = p.getBasePositionAndOrientation(robot.robot_id, physicsClientId=robot.client_id)
    p.resetBasePositionAndOrientation(robot.robot_id, [pos[0], pos[1], 20.0], orn, physicsClientId=robot.client_id)
    p.resetBaseVelocity(robot.robot_id, [0, 0, 0], [0, 0, 0], physicsClientId=robot.client_id)
    steps = 240
    step(robot, steps)
    fallen = 20.0 - p.getBasePositionAndOrientation(robot.robot_id, physicsClientId=robot.client_id)[0][2]
    dt = SimulationData().dt
    # semi-implicit Euler: after n steps the drop is g * dt^2 * n(n+1)/2
    expected = 9.81 * dt ** 2 * steps * (steps + 1) / 2
    assert fallen == pytest.approx(expected, abs=1e-3)


def test_friction_is_set_in_the_robot_server(two_clients):
    """Without physicsClientId, changeDynamics goes to server 0. A robot there has the same body id,
    so it would silently get the friction while the robot in server 1 keeps the default 0.5."""
    first, second = two_clients
    spec = dataclasses.replace(A1_ROBOT_SPEC, toes_friction=0.37, body_friction=0.21)
    prepare_simulation(SimulationData(), first)
    FlatEnvironment(FLAT_PARAMS, first, seed=0)  # ground and 4 walls first, as in make_robot
    bystander = p.loadURDF(A1_ROBOT_SPEC.model_path, physicsClientId=first)
    robot = make_robot(second, spec)

    assert robot.robot_id == bystander  # same id in both servers: the dangerous case
    for link in all_link_indices(robot):
        assert p.getDynamicsInfo(bystander, link, physicsClientId=first)[1] == pytest.approx(0.5)
    for link in robot.toes_links_id.values():
        assert lateral_friction(robot, link) == pytest.approx(0.37)


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
    """Reset after an episode of random velocity commands restores exactly the same state."""
    first_pos, first_rpy = base_state(robot)
    first_joints = robot.get_joints_state()

    rng = np.random.default_rng(0)
    for _ in range(60):
        act(robot, rng.uniform(-1, 1, N_JOINTS))
    robot.reset()

    pos, rpy = base_state(robot)
    assert np.allclose(pos, first_pos, atol=1e-9)
    assert np.allclose(rpy, first_rpy, atol=1e-9)
    for name, state in robot.get_joints_state().items():
        assert np.allclose(state, first_joints[name], atol=1e-9)


def random_episode(robot: Robot, seed: int, n_actions: int = 60) -> np.ndarray:
    """Reset, then apply seeded random actions; return the final state."""
    robot.reset()
    rng = np.random.default_rng(seed)
    for _ in range(n_actions):
        act(robot, rng.uniform(-1, 1, N_JOINTS))
    pos, rpy = base_state(robot)
    joints = np.array([s[0] for s in robot.get_joints_state().values()])
    return np.concatenate([pos, rpy, joints])


def test_episode_is_reproducible_regardless_of_previous_episode(robot):
    """The same actions after reset give bit-identical results, even if another episode ran in between.

    Without this, a genome's fitness would depend on which genome was evaluated before it.
    """
    first = random_episode(robot, seed=1)
    random_episode(robot, seed=2)
    again = random_episode(robot, seed=1)
    assert np.array_equal(first, again)


# --- actions ----------------------------------------------------------------

def test_scale_to_velocity_maps_ends_middle_and_clips():
    rng = np.array([-V_MAX, V_MAX])
    assert scale_to_velocity(-1.0, rng) == pytest.approx(-V_MAX)
    assert scale_to_velocity(1.0, rng) == pytest.approx(V_MAX)
    assert scale_to_velocity(0.0, rng) == pytest.approx(0.0)
    assert scale_to_velocity(0.5, rng) == pytest.approx(0.5 * V_MAX)
    assert scale_to_velocity(7.0, rng) == pytest.approx(V_MAX)  # clipped
    assert scale_to_velocity(-7.0, rng) == pytest.approx(-V_MAX)


def test_apply_action_sends_velocity_targets_with_effort_limits(robot, monkeypatch):
    """One VELOCITY_CONTROL call: target = action * max_joint_velocity, force = max_joint_effort.

    Without forces= PyBullet allows hundreds of N*m, so the limit is checked on the call itself.
    """
    calls = spy_on_motor_calls(monkeypatch)
    action = np.linspace(-1.0, 1.0, N_JOINTS)
    robot.apply_action(action)

    assert len(calls) == 1
    call = calls[0]
    assert call["controlMode"] == p.VELOCITY_CONTROL
    targets = dict(zip(call["jointIndices"], call["targetVelocities"]))
    forces = dict(zip(call["jointIndices"], call["forces"]))
    for i, name in enumerate(robot.moving_joints):
        joint_id = robot.moving_joints_id[name]
        assert targets[joint_id] == pytest.approx(action[i] * V_MAX)
        assert forces[joint_id] == pytest.approx(EFFORT)


def test_apply_action_clips_to_max_joint_velocity(robot, monkeypatch):
    calls = spy_on_motor_calls(monkeypatch)
    robot.apply_action(np.array([5.0, -5.0] * 6))
    assert np.allclose(calls[0]["targetVelocities"], [V_MAX, -V_MAX] * 6)


def test_joints_reach_the_commanded_velocity_within_one_action(hanging):
    """The supervisor's question: is the command reached before the next observation?

    With free legs every joint runs at 0.5 * max_joint_velocity after k = 4 physics steps.
    """
    act(hanging, np.full(N_JOINTS, 0.5))
    for name, velocity in joint_velocities(hanging).items():
        assert velocity == pytest.approx(0.5 * V_MAX, abs=0.1), name


def test_zero_action_keeps_the_robot_standing_for_two_seconds(robot):
    """Action 0 means 'do not move', not 'go to the rest pose'.

    A velocity motor has no position target, so a standing robot sags slowly under its weight
    (measured 2026-11-27: thigh about -0.013 rad/s, base 1 cm per second). After 2 s it still stands.
    """
    act(robot, np.zeros(N_JOINTS), n_actions=120)
    pos, rpy = base_state(robot)
    assert robot.get_toes_contact_points() == dict.fromkeys(A1_ROBOT_SPEC.toes_links_names, 1)
    assert pos[2] > 0.25 and abs(rpy[0]) < 0.1 and abs(rpy[1]) < 0.1
    for name, angle in joint_angles(robot).items():
        assert angle == pytest.approx(A1_ROBOT_SPEC.rest_position[name], abs=0.1)


@pytest.mark.parametrize("value", [-1.0, 1.0])
def test_extreme_actions_stop_at_the_urdf_limits(hanging, value):
    """+-1 for 2 s drives every free joint to the end of its URDF range and keeps it there,
    without ever exceeding the effort."""
    max_torque = dict.fromkeys(hanging.moving_joints_id, 0.0)
    for _ in range(120):
        act(hanging, np.full(N_JOINTS, value))
        for name, joint_id in hanging.moving_joints_id.items():
            torque = p.getJointState(hanging.robot_id, joint_id, physicsClientId=hanging.client_id)[3]
            max_torque[name] = max(max_torque[name], abs(torque))
    for name, torque in max_torque.items():
        assert torque <= EFFORT + 1e-6, f"{name} exceeded max_joint_effort"
    for name, angle in joint_angles(hanging).items():
        lo, hi = hanging.moving_joints_position_limits[name]
        assert lo - LIMIT_TOL <= angle <= hi + LIMIT_TOL, name
        assert angle == pytest.approx(lo if value < 0 else hi, abs=LIMIT_TOL), name


# --- observations -----------------------------------------------------------

def test_scale_to_neat_input_maps_the_limits_to_minus_one_and_one():
    limits = np.array([-2.0, -0.9])
    assert scale_to_neat_input(-2.0, limits) == pytest.approx(-1.0)
    assert scale_to_neat_input(-0.9, limits) == pytest.approx(1.0)
    assert scale_to_neat_input(-1.45, limits) == pytest.approx(0.0)


def test_observation_has_fixed_length_and_is_finite(robot):
    obs = robot.get_observation()
    assert obs.shape == (OBS_LEN,)
    assert np.all(np.isfinite(obs))


def test_observation_at_rest(robot):
    """After reset: joint angles at the rest pose (scaled by URDF limits), nothing moving, all toes down."""
    obs = robot.get_observation()
    expected = [scale_to_neat_input(A1_ROBOT_SPEC.rest_position[name], robot.moving_joints_position_limits[name])
                for name in robot.moving_joints]
    assert np.allclose(obs[JOINT_POS], expected, atol=0.05)
    assert np.allclose(obs[JOINT_VEL], 0.0, atol=0.05)
    assert np.all(obs[CONTACTS] == 1)


@pytest.mark.parametrize("end, expected", [(0, -1.0), (1, 1.0)])
def test_joint_angles_are_scaled_by_the_urdf_limits(robot, end, expected):
    name = "FR_upper_joint"
    idx = robot.moving_joints.index(name)
    angle = robot.moving_joints_position_limits[name][end]
    p.resetJointState(robot.robot_id, robot.moving_joints_id[name], angle, physicsClientId=robot.client_id)
    assert robot.get_observation()[JOINT_POS][idx] == pytest.approx(expected, abs=1e-9)


def test_joint_velocities_are_scaled_by_max_joint_velocity(robot):
    name = "FR_upper_joint"
    idx = robot.moving_joints.index(name)
    p.resetJointState(robot.robot_id, robot.moving_joints_id[name], A1_ROBOT_SPEC.rest_position[name],
                      targetVelocity=10.5, physicsClientId=robot.client_id)
    assert robot.get_observation()[JOINT_VEL][idx] == pytest.approx(10.5 / V_MAX, abs=1e-9)


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


def test_observation_does_not_depend_on_heading(robot):
    """No yaw in the observation: turning a standing robot about the vertical axis changes nothing.

    Both poses are set with resetBasePositionAndOrientation (it also zeroes the base velocity);
    the turned one is the same orientation rotated about the world z axis, so roll and pitch stay.
    """
    pos, orn = p.getBasePositionAndOrientation(robot.robot_id, physicsClientId=robot.client_id)
    p.resetBasePositionAndOrientation(robot.robot_id, pos, orn, physicsClientId=robot.client_id)
    before = robot.get_observation()

    _, turned = p.multiplyTransforms([0, 0, 0], p.getQuaternionFromEuler([0, 0, 1.0]), [0, 0, 0], orn)
    p.resetBasePositionAndOrientation(robot.robot_id, pos, turned, physicsClientId=robot.client_id)
    assert np.allclose(robot.get_observation(), before, atol=1e-6)


def set_pose_and_body_angular_velocity(robot: Robot, euler, body_omega) -> None:
    """Puts the base in orientation `euler` and gives it angular velocity `body_omega` (body frame)."""
    pos, _ = base_state(robot)
    orn = p.getQuaternionFromEuler(euler)
    rotation = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)  # columns: body axes in world
    p.resetBasePositionAndOrientation(robot.robot_id, pos, orn, physicsClientId=robot.client_id)
    p.resetBaseVelocity(robot.robot_id, [0, 0, 0], (rotation @ np.asarray(body_omega)).tolist(),
                        physicsClientId=robot.client_id)


def test_base_angular_velocity_is_scaled(robot):
    """Level base facing +x: body frame = world frame, so only the scale is checked here."""
    set_pose_and_body_angular_velocity(robot, (0, 0, 0), [0.5, -1.0, 2.5])
    obs = robot.get_observation()
    expected = np.array([0.5, -1.0, 2.5]) / A1_ROBOT_SPEC.base_velocity_scale
    assert np.allclose(obs[ANG_VEL], expected, atol=1e-6)


@pytest.mark.parametrize("euler", [(0, 0, math.pi / 2), (0, 0, -2.0), (0.3, -0.2, 1.0)],
                         ids=["facing +y", "turned -2 rad", "tilted and turned"])
def test_base_angular_velocity_is_in_the_body_frame(robot, euler):
    """The same rotation of the trunk relative to itself reads the same whatever the heading.

    Pitching nose-down at 1 rad/s is +1 about the body y axis. Facing +y that is (-1, 0, 0) in the world
    frame; an observation in the world frame would show it as a roll rate.
    """
    body_omega = [0.5, -1.0, 2.5]
    set_pose_and_body_angular_velocity(robot, euler, body_omega)
    obs = robot.get_observation()
    assert np.allclose(obs[ANG_VEL], np.array(body_omega) / A1_ROBOT_SPEC.base_velocity_scale, atol=1e-6)


def test_contacts_drop_to_zero_in_the_air(robot):
    pos, orn = p.getBasePositionAndOrientation(robot.robot_id, physicsClientId=robot.client_id)
    p.resetBasePositionAndOrientation(robot.robot_id, [pos[0], pos[1], pos[2] + 0.5], orn,
                                      physicsClientId=robot.client_id)
    step(robot, 1)
    assert np.all(robot.get_observation()[CONTACTS] == 0)


# --- toe positions ----------------------------------------------------------

def test_toes_world_position_gives_every_toe_by_name(robot):
    positions = robot.get_toes_world_position()
    assert set(positions) == set(A1_ROBOT_SPEC.toes_links_names)
    for name, link in robot.toes_links_id.items():
        expected = p.getLinkState(robot.robot_id, link, computeForwardKinematics=True,
                                  physicsClientId=robot.client_id)[0]
        assert np.allclose(positions[name], expected, atol=1e-12)


def test_standing_toes_are_on_the_ground(robot):
    """Toe spheres have r = 0.02 m and the ground top is at z = 0."""
    for name, position in robot.get_toes_world_position().items():
        assert position[2] == pytest.approx(0.02, abs=0.005), name


def test_toes_world_position_follows_the_joints(hanging):
    """Turning only FR_hip moves only the FR toe (fixed trunk, current joint angles, no stale state)."""
    before = hanging.get_toes_world_position()
    p.resetJointState(hanging.robot_id, hanging.moving_joints_id["FR_hip_joint"], 0.5,
                      physicsClientId=hanging.client_id)
    after = hanging.get_toes_world_position()
    assert np.linalg.norm(np.subtract(after["FR_toe"], before["FR_toe"])) > 0.05
    for name in ("FL_toe", "RR_toe", "RL_toe"):
        assert np.allclose(after[name], before[name], atol=1e-9), name


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
