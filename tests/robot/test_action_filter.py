"""Tests for the EMA action filter in Robot (RobotSpec.action_filter_alpha, added 2026-11-24, stage 5, block 2).

Contract checked here:
- action_filter_alpha defaults to 1.0 (no filter), also for a spec pickled before the field existed,
  and must lie in (0, 1] (ValueError otherwise),
- with alpha = 1 the joint targets are exactly the unfiltered ones, bit for bit, so runs made before
  the filter existed replay unchanged,
- with alpha < 1 the action applied in step n is y[n] = alpha * u[n] + (1 - alpha) * y[n-1], y[-1] = 0
  (the rest pose), in both branches of apply_action (with and without the velocity limit),
- reset() puts the filter back to the rest pose, so an episode never depends on the previous one,
- apply_action does not modify the caller's action, and the filter state does not share memory with it.

Targets are read from the PyBullet call (spy), then compared with scale_to_angles of the expected
filtered action, so the tests do not depend on how the filter state is stored.
"""

import dataclasses
import pickle

import numpy as np
import pybullet as p
import pytest

from neat_walking_robot.robot.robot import A1_ROBOT_SPEC, RobotSpec, scale_to_angles
from test_robot import N_JOINTS, make_robot, step

ALPHA = 0.3


def filtered_spec(alpha: float, limited: bool = False) -> RobotSpec:
    return dataclasses.replace(A1_ROBOT_SPEC, action_filter_alpha=alpha, apply_velocity_limit=limited)


def record_targets(monkeypatch, robot) -> list[np.ndarray]:
    """After each apply_action, the list gets one array of joint targets in robot.moving_joints order."""
    targets: list[np.ndarray] = []
    order = [robot.moving_joints_id[name] for name in robot.moving_joints]
    pending: dict[int, float] = {}

    real_array = p.setJointMotorControlArray
    real_single = p.setJointMotorControl2

    def array_spy(*args, **kwargs):
        by_id = dict(zip(kwargs["jointIndices"], kwargs["targetPositions"]))
        targets.append(np.array([by_id[j] for j in order]))
        return real_array(*args, **kwargs)

    def single_spy(*args, **kwargs):
        pending[kwargs["jointIndex"]] = kwargs["targetPosition"]
        if len(pending) == len(order):
            targets.append(np.array([pending[j] for j in order]))
            pending.clear()
        return real_single(*args, **kwargs)

    monkeypatch.setattr(p, "setJointMotorControlArray", array_spy)
    monkeypatch.setattr(p, "setJointMotorControl2", single_spy)
    return targets


def angles(robot, action: np.ndarray) -> np.ndarray:
    return np.array([
        scale_to_angles(action[i], robot.moving_joints_limits[name]) for i, name in enumerate(robot.moving_joints)
    ])


def expected_filtered(actions: list[np.ndarray], alpha: float) -> list[np.ndarray]:
    y = np.zeros(N_JOINTS)
    out = []
    for u in actions:
        y = alpha * np.asarray(u, dtype=float) + (1 - alpha) * y
        out.append(y.copy())
    return out


def random_actions(seed: int, n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.uniform(-1, 1, N_JOINTS) for _ in range(n)]


# --- spec --------------------------------------------------------------------

def test_filter_is_off_by_default():
    assert A1_ROBOT_SPEC.action_filter_alpha == 1.0


def test_spec_pickled_before_the_field_existed_has_no_filter():
    old_state = dict(A1_ROBOT_SPEC.__dict__)
    del old_state["action_filter_alpha"]
    old_spec = RobotSpec.__new__(RobotSpec)
    object.__setattr__(old_spec, "__dict__", old_state)

    restored = pickle.loads(pickle.dumps(old_spec))
    assert "action_filter_alpha" not in restored.__dict__
    assert restored.action_filter_alpha == 1.0


@pytest.mark.parametrize("alpha", [0.0, -0.2, 1.0001, 2.0])
def test_alpha_outside_zero_one_is_rejected(alpha):
    with pytest.raises(ValueError):
        filtered_spec(alpha)


@pytest.mark.parametrize("alpha", [1e-3, 0.3, 1.0])
def test_alpha_inside_zero_one_is_accepted(alpha):
    assert filtered_spec(alpha).action_filter_alpha == alpha


# --- filter math -----------------------------------------------------------------

@pytest.mark.parametrize("limited", [False, True], ids=["array", "per_joint"])
def test_alpha_one_gives_exactly_the_unfiltered_targets(client, monkeypatch, limited):
    """Bit-identical, not approximately equal: old runs must replay unchanged.

    Note: prev + alpha * (u - prev) is not exactly u for alpha = 1 in floating point;
    alpha * u + (1 - alpha) * prev is.
    """
    robot = make_robot(client, filtered_spec(1.0, limited))
    targets = record_targets(monkeypatch, robot)
    actions = random_actions(seed=5, n=60)
    for action in actions:
        robot.apply_action(action.copy())
        step(robot, 4)

    for got, action in zip(targets, actions):
        assert np.array_equal(got, angles(robot, action))


def test_alpha_one_is_exact_for_a_pair_where_the_difference_form_rounds(client, monkeypatch):
    """0.3 + (-0.9 - 0.3) == -0.8999999999999999 in floating point, not -0.9.

    Random actions in [-1, 1] almost never hit such a pair, so it is checked explicitly.
    """
    robot = make_robot(client, filtered_spec(1.0))
    targets = record_targets(monkeypatch, robot)
    first, second = np.full(N_JOINTS, 0.3), np.full(N_JOINTS, -0.9)
    robot.apply_action(first.copy())
    robot.apply_action(second.copy())
    assert np.array_equal(targets[1], angles(robot, second))


@pytest.mark.parametrize("limited", [False, True], ids=["array", "per_joint"])
def test_filtered_targets_follow_the_ema_recursion(client, monkeypatch, limited):
    robot = make_robot(client, filtered_spec(ALPHA, limited))
    targets = record_targets(monkeypatch, robot)
    actions = random_actions(seed=6, n=40)
    for action in actions:
        robot.apply_action(action.copy())
        step(robot, 4)

    for got, filtered in zip(targets, expected_filtered(actions, ALPHA)):
        assert got == pytest.approx(angles(robot, filtered), abs=1e-12)


def test_step_response_starts_from_the_rest_pose(client, monkeypatch):
    """Input jumps from 0 to 1: applied actions 0.3, 0.51, 0.657, ... = 1 - 0.7^(n+1)."""
    robot = make_robot(client, filtered_spec(ALPHA))
    targets = record_targets(monkeypatch, robot)
    for _ in range(6):
        robot.apply_action(np.ones(N_JOINTS))

    for n, got in enumerate(targets):
        level = 1 - (1 - ALPHA) ** (n + 1)
        assert got == pytest.approx(angles(robot, np.full(N_JOINTS, level)), abs=1e-12)


def test_alternating_input_settles_at_alpha_over_two_minus_alpha(client, monkeypatch):
    """+1, -1, +1, ... (the 'vibration') settles at +-alpha / (2 - alpha) = +-0.176 for alpha = 0.3."""
    robot = make_robot(client, filtered_spec(ALPHA))
    targets = record_targets(monkeypatch, robot)
    for i in range(60):
        robot.apply_action(np.full(N_JOINTS, 1.0 if i % 2 == 0 else -1.0))

    amplitude = ALPHA / (2 - ALPHA)
    assert targets[-2] == pytest.approx(angles(robot, np.full(N_JOINTS, amplitude)), abs=1e-9)
    assert targets[-1] == pytest.approx(angles(robot, np.full(N_JOINTS, -amplitude)), abs=1e-9)


# --- state ---------------------------------------------------------------------

def test_reset_returns_the_filter_to_the_rest_pose(client, monkeypatch):
    robot = make_robot(client, filtered_spec(ALPHA))
    for action in random_actions(seed=7, n=20):
        robot.apply_action(action)
        step(robot, 4)
    robot.reset()

    targets = record_targets(monkeypatch, robot)
    robot.apply_action(np.ones(N_JOINTS))
    assert targets[0] == pytest.approx(angles(robot, np.full(N_JOINTS, ALPHA)), abs=1e-12)


def test_filtered_episode_does_not_depend_on_the_previous_one(client):
    """Same check as for the unfiltered robot: a genome's fitness must not depend on the genome before it."""
    robot = make_robot(client, filtered_spec(ALPHA))

    def episode(seed: int) -> np.ndarray:
        robot.reset()
        for action in random_actions(seed, n=60):
            robot.apply_action(action)
            step(robot, 4)
        ids = list(robot.moving_joints_id.values())
        joints = [s[0] for s in p.getJointStates(robot.robot_id, ids, physicsClientId=client)]
        pos, orn = p.getBasePositionAndOrientation(robot.robot_id, physicsClientId=client)
        return np.concatenate([joints, pos, orn])

    first = episode(1)
    episode(2)
    assert np.array_equal(first, episode(1))


@pytest.mark.parametrize("container", [np.array, list], ids=["ndarray", "list"])
def test_apply_action_does_not_modify_the_callers_action(client, container):
    """run_episode, diagnose.py and fitness_v2 read the raw network action after apply_action."""
    robot = make_robot(client, filtered_spec(ALPHA))
    action = container([0.9] * N_JOINTS)
    robot.apply_action(action)
    assert list(action) == [0.9] * N_JOINTS


def test_filter_state_does_not_share_memory_with_the_callers_action(client, monkeypatch):
    """A caller reusing its action buffer must not change the filter state behind the robot's back."""
    robot = make_robot(client, filtered_spec(ALPHA))
    targets = record_targets(monkeypatch, robot)

    buffer = np.full(N_JOINTS, 1.0)
    robot.apply_action(buffer)
    buffer[:] = -1.0            # caller overwrites its buffer in place
    robot.apply_action(buffer)

    expected = expected_filtered([np.full(N_JOINTS, 1.0), np.full(N_JOINTS, -1.0)], ALPHA)
    assert targets[1] == pytest.approx(angles(robot, expected[1]), abs=1e-12)
