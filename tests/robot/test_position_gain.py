"""Tests for RobotSpec.position_gain (added 2026-11-23, stage 5, block 1, step 5b).

Contract checked here:
- position_gain defaults to 0.1 (PyBullet's own default), also for a spec pickled before the field existed,
- both branches of apply_action pass the spec's gain to PyBullet (positionGains / positionGain),
- passing the default 0.1 explicitly gives bit-identical motion to omitting the argument,
  so runs made before the field existed replay unchanged,
- a lower gain makes the joints follow their targets more slowly.
"""

import dataclasses
import pickle

import numpy as np
import pybullet as p
import pytest

from neat_walking_robot.robot.robot import A1_ROBOT_SPEC, RobotSpec
from test_robot import N_JOINTS, make_robot, step

PYBULLET_DEFAULT_GAIN = 0.1


def spy_on(monkeypatch, name: str, drop: str | None = None) -> list[dict]:
    """Record kwargs of p.<name>; with drop, remove that kwarg before calling the real function."""
    calls = []
    real = getattr(p, name)

    def spy(*args, **kwargs):
        calls.append(dict(kwargs))
        if drop is not None:
            kwargs.pop(drop, None)
        return real(*args, **kwargs)

    monkeypatch.setattr(p, name, spy)
    return calls


def final_state(robot) -> np.ndarray:
    ids = list(robot.moving_joints_id.values())
    joints = [s[0] for s in p.getJointStates(robot.robot_id, ids, physicsClientId=robot.client_id)]
    pos, orn = p.getBasePositionAndOrientation(robot.robot_id, physicsClientId=robot.client_id)
    return np.concatenate([joints, pos, orn])


def random_episode(robot, seed: int = 3, n_actions: int = 60) -> np.ndarray:
    robot.reset()
    rng = np.random.default_rng(seed)
    for _ in range(n_actions):
        robot.apply_action(rng.uniform(-1, 1, N_JOINTS))
        step(robot, 4)
    return final_state(robot)


# --- spec --------------------------------------------------------------------

def test_position_gain_defaults_to_pybullet_default():
    assert A1_ROBOT_SPEC.position_gain == PYBULLET_DEFAULT_GAIN


def test_spec_pickled_before_the_field_existed_has_default_gain():
    old_state = dict(A1_ROBOT_SPEC.__dict__)
    del old_state["position_gain"]
    old_spec = RobotSpec.__new__(RobotSpec)
    object.__setattr__(old_spec, "__dict__", old_state)

    restored = pickle.loads(pickle.dumps(old_spec))
    assert "position_gain" not in restored.__dict__
    assert restored.position_gain == PYBULLET_DEFAULT_GAIN


# --- the gain reaches PyBullet -------------------------------------------------

@pytest.mark.parametrize("limited", [False, True], ids=["array", "per_joint"])
def test_apply_action_passes_spec_gain(client, monkeypatch, limited):
    spec = dataclasses.replace(A1_ROBOT_SPEC, position_gain=0.037, apply_velocity_limit=limited)
    robot = make_robot(client, spec)
    array_calls = spy_on(monkeypatch, "setJointMotorControlArray")
    single_calls = spy_on(monkeypatch, "setJointMotorControl2")

    robot.apply_action(np.zeros(N_JOINTS))

    if limited:
        assert len(single_calls) == N_JOINTS
        assert all(call["positionGain"] == pytest.approx(0.037) for call in single_calls)
    else:
        assert len(array_calls) == 1
        gains = array_calls[0]["positionGains"]
        assert len(gains) == N_JOINTS
        assert all(g == pytest.approx(0.037) for g in gains)


@pytest.mark.parametrize("limited", [False, True], ids=["array", "per_joint"])
def test_explicit_default_gain_is_bit_identical_to_omitting_it(monkeypatch, limited):
    """Old runs were made without positionGain(s); passing 0.1 must not change a single bit."""
    spec = dataclasses.replace(A1_ROBOT_SPEC, apply_velocity_limit=limited)
    states = []
    for strip in (False, True):
        client_id = p.connect(p.DIRECT)
        try:
            robot = make_robot(client_id, spec)
            with monkeypatch.context() as m:
                if strip:
                    spy_on(m, "setJointMotorControlArray", drop="positionGains")
                    spy_on(m, "setJointMotorControl2", drop="positionGain")
                states.append(random_episode(robot))
        finally:
            p.disconnect(client_id)
    assert np.array_equal(states[0], states[1])


# --- physics -------------------------------------------------------------------

def test_lower_gain_follows_targets_more_slowly():
    """Within one control step from rest, half the gain moves the joints clearly less towards their targets.

    Compared in total, not joint by joint: a loaded calf pressed against the ground can move either way.
    """
    moved = {}
    for gain in (0.1, 0.05):
        client_id = p.connect(p.DIRECT)
        try:
            robot = make_robot(client_id, dataclasses.replace(A1_ROBOT_SPEC, position_gain=gain))
            robot.reset()
            ids = list(robot.moving_joints_id.values())
            before = np.array([s[0] for s in p.getJointStates(robot.robot_id, ids, physicsClientId=client_id)])
            robot.apply_action(np.ones(N_JOINTS))
            step(robot, 4)
            after = np.array([s[0] for s in p.getJointStates(robot.robot_id, ids, physicsClientId=client_id)])
            moved[gain] = np.abs(after - before)
        finally:
            p.disconnect(client_id)
    assert moved[0.05].sum() < 0.8 * moved[0.1].sum()
