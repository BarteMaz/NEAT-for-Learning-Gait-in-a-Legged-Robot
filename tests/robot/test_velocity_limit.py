"""Tests for the optional joint velocity limit in Robot.apply_action (added 2026-11-23, stage 5).

Contract checked here:
- RobotSpec.apply_velocity_limit defaults to False, and a spec pickled before the field existed
  (first training run, 2026-11-21) reads as False, so old runs replay unchanged,
- without the limit apply_action keeps the old path (one setJointMotorControlArray call),
- with the limit every moving joint gets setJointMotorControl2 with its URDF velocity as maxVelocity,
  its URDF effort as force, the same target angle as before, and the robot's own physicsClientId,
- the limit really bounds the joint speed in every physics step,
- an unreachable limit gives exactly the same motion as no limit (only the speed cap differs).

Velocities are checked in every physics step, not every control step: a joint that reverses at every
control step is sampled right at its turning point, where its speed is close to zero.
"""

import dataclasses
import pickle

import numpy as np
import pybullet as p
import pytest

from neat_walking_robot.robot.robot import A1_ROBOT_SPEC, RobotSpec, scale_to_angles
from test_robot import N_JOINTS, make_robot, step

LIMITED_SPEC = dataclasses.replace(A1_ROBOT_SPEC, apply_velocity_limit=True)

LOW_CAP = 3.0        # rad/s, far below what the joints reach without a limit (about 15-23 rad/s)
CAP_TOLERANCE = 1.5  # contacts and gravity can push a joint a little over its motor's cap


def alternating_actions(n: int) -> list[np.ndarray]:
    """Full-range square wave on all joints, reversing at every control step (the 'vibration' pattern)."""
    return [np.full(N_JOINTS, 1.0 if i % 2 == 0 else -1.0) for i in range(n)]


def max_joint_speeds(robot, actions) -> np.ndarray:
    """Apply actions (4 physics steps each) and return max |velocity| of every moving joint over all steps."""
    ids = list(robot.moving_joints_id.values())
    speeds = np.zeros(len(ids))
    for action in actions:
        robot.apply_action(action)
        for _ in range(4):
            p.stepSimulation(physicsClientId=robot.client_id)
            states = p.getJointStates(robot.robot_id, ids, physicsClientId=robot.client_id)
            speeds = np.maximum(speeds, np.abs([s[1] for s in states]))
    return speeds


def final_state(robot) -> np.ndarray:
    ids = list(robot.moving_joints_id.values())
    joints = [s[0] for s in p.getJointStates(robot.robot_id, ids, physicsClientId=robot.client_id)]
    pos, orn = p.getBasePositionAndOrientation(robot.robot_id, physicsClientId=robot.client_id)
    return np.concatenate([joints, pos, orn])


def set_cap(robot, cap: float) -> None:
    for name in robot.moving_joints_velocity:
        robot.moving_joints_velocity[name] = cap


# --- spec --------------------------------------------------------------------

def test_velocity_limit_is_off_by_default():
    assert A1_ROBOT_SPEC.apply_velocity_limit is False


def test_spec_pickled_before_the_field_existed_reads_as_off():
    """training_config.pkl of the first run has a RobotSpec without apply_velocity_limit.

    Pickle restores the attribute dict without calling __init__, so the missing field must fall back to
    the class default (False). If the default were True, the first run would silently replay with the limit.
    """
    old_state = dict(A1_ROBOT_SPEC.__dict__)
    del old_state["apply_velocity_limit"]
    old_spec = RobotSpec.__new__(RobotSpec)
    object.__setattr__(old_spec, "__dict__", old_state)

    restored = pickle.loads(pickle.dumps(old_spec))
    assert "apply_velocity_limit" not in restored.__dict__
    assert restored.apply_velocity_limit is False
    assert dataclasses.replace(restored, apply_velocity_limit=True).apply_velocity_limit is True


def test_velocity_limits_are_read_from_urdf(client):
    robot = make_robot(client)
    for name, joint_id in robot.moving_joints_id.items():
        urdf_velocity = p.getJointInfo(robot.robot_id, joint_id, physicsClientId=client)[11]
        assert urdf_velocity > 0
        assert robot.moving_joints_velocity[name] == pytest.approx(urdf_velocity)


# --- which PyBullet call is used ----------------------------------------------

def spy_on(monkeypatch, name: str) -> list[dict]:
    calls = []
    real = getattr(p, name)

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(p, name, spy)
    return calls


def test_without_limit_one_array_call_and_no_per_joint_calls(client, monkeypatch):
    robot = make_robot(client)
    array_calls = spy_on(monkeypatch, "setJointMotorControlArray")
    single_calls = spy_on(monkeypatch, "setJointMotorControl2")

    robot.apply_action(np.full(N_JOINTS, 0.5))
    assert len(array_calls) == 1
    assert single_calls == []


def test_with_limit_every_joint_gets_urdf_velocity_effort_target_and_client(client, monkeypatch):
    robot = make_robot(client, LIMITED_SPEC)
    array_calls = spy_on(monkeypatch, "setJointMotorControlArray")
    single_calls = spy_on(monkeypatch, "setJointMotorControl2")

    action = np.linspace(-1.0, 1.0, N_JOINTS)
    robot.apply_action(action)

    assert array_calls == []
    by_joint = {call["jointIndex"]: call for call in single_calls}
    assert len(single_calls) == N_JOINTS
    assert set(by_joint) == set(robot.moving_joints_id.values())

    for i, name in enumerate(robot.moving_joints):
        call = by_joint[robot.moving_joints_id[name]]
        assert call["controlMode"] == p.POSITION_CONTROL
        assert call["maxVelocity"] == pytest.approx(robot.moving_joints_velocity[name])
        assert call["force"] == pytest.approx(robot.moving_joints_efforts[name])
        assert call["targetPosition"] == pytest.approx(
            scale_to_angles(action[i], robot.moving_joints_limits[name])
        )
        assert call.get("physicsClientId") == client


# --- physics -------------------------------------------------------------------

def test_low_limit_bounds_joint_speed_in_every_physics_step(client):
    """With a 3 rad/s cap the square wave stays near 3 rad/s; without it the same joints reach about 15 rad/s."""
    free = max_joint_speeds(make_robot(client), alternating_actions(30))
    assert free.min() > 3 * LOW_CAP  # sanity: the test motion is fast enough to hit the cap

    limited_robot = make_robot(client, LIMITED_SPEC)
    set_cap(limited_robot, LOW_CAP)
    limited = max_joint_speeds(limited_robot, alternating_actions(30))
    assert limited.max() <= CAP_TOLERANCE * LOW_CAP


def test_unreachable_limit_gives_exactly_the_same_motion_as_no_limit():
    """The limited branch must differ from the old one only by the speed cap (same targets, forces, joints)."""
    states = []
    for spec, cap in ((A1_ROBOT_SPEC, None), (LIMITED_SPEC, 1e3)):
        client_id = p.connect(p.DIRECT)
        try:
            robot = make_robot(client_id, spec)
            if cap is not None:
                set_cap(robot, cap)
            robot.reset()
            rng = np.random.default_rng(7)
            for _ in range(60):
                robot.apply_action(rng.uniform(-1, 1, N_JOINTS))
                step(robot, 4)
            states.append(final_state(robot))
        finally:
            p.disconnect(client_id)
    assert np.array_equal(states[0], states[1])


def test_limited_episode_is_reproducible(client):
    robot = make_robot(client, LIMITED_SPEC)

    def episode(seed: int) -> np.ndarray:
        robot.reset()
        rng = np.random.default_rng(seed)
        for _ in range(60):
            robot.apply_action(rng.uniform(-1, 1, N_JOINTS))
            step(robot, 4)
        return final_state(robot)

    first = episode(1)
    episode(2)
    assert np.array_equal(first, episode(1))


# --- isolation between servers ----------------------------------------------

def test_limited_robot_commands_only_its_own_server(two_clients):
    """Identical robots in both servers; only the second one is commanded. The first must not move.

    Both robots get the same body id, so a call without physicsClientId would drive the robot in server 0.
    """
    first, second = two_clients
    idle = make_robot(first, LIMITED_SPEC)
    active = make_robot(second, LIMITED_SPEC)
    assert idle.robot_id == active.robot_id

    idle_before = final_state(idle)
    active_before = final_state(active)
    for action in alternating_actions(20):
        active.apply_action(action)
        step(idle, 4)
        step(active, 4)

    assert np.abs(final_state(active) - active_before).max() > 0.1
    assert np.abs(final_state(idle)[:N_JOINTS] - idle_before[:N_JOINTS]).max() < 0.05
