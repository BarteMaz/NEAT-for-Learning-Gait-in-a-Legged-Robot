"""Tests for EpisodeMetrics.max_joint_action_change (run_episode) and the TrainingConfig validation of target_speed
(stage 5, 2026-11-24).

Contract for max_joint_action_change (changed 2026-11-25):
- for each joint j, the mean over executed action steps n of |a_j[n] - a_j[n-1]|, with a[-1] = 0 (the rest pose);
  the metric is the LARGEST of these 12 per-joint means,
- it is computed from the RAW policy output (before the robot's action filter),
- mean_yaw is the mean over executed action steps of |yaw| after the step (not the maximum),
- the policy is called exactly once per executed action step (no extra call before the loop:
  stateful policies, the recording wrapper in diagnose.py and the real-time replay rely on it).

Contract for TrainingConfig: fitness_v2 divides by (target_speed - D / T), so target_speed must be > D / T.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from neat_walking_robot.robot import A1_ROBOT_SPEC, Robot
import pickle

from neat_walking_robot.training.config import TrainingConfig, default_training_config
from neat_walking_robot.training.fitness import FitnessFunction
from neat_walking_robot.training.episode import EndReason, run_episode
from test_episode import DT, EPISODE, build, teleport_at

N_JOINTS = 12


def steps_run(m) -> int:
    return round(m.duration / (EPISODE.k * DT))


def constant(value: float):
    return lambda obs, t: np.full(N_JOINTS, value)


def alternating(a: float):
    """+a, -a, +a, ... on all joints."""
    state = {"n": 0}

    def policy(obs, t):
        sign = 1.0 if state["n"] % 2 == 0 else -1.0
        state["n"] += 1
        return np.full(N_JOINTS, sign * a)

    return policy


@pytest.fixture
def world(client):
    return build(client)


# --- value ----------------------------------------------------------------------

def test_standing_still_has_zero_action_change(world):
    env, robot = world
    m = run_episode(robot, env, constant(0.0), DT, EPISODE)
    assert m.max_joint_action_change == 0.0


def test_constant_action_counts_only_the_first_step_from_rest(world):
    """a[-1] = 0, then 0.2 forever: one change of 0.2 per joint, averaged over all steps."""
    env, robot = world
    m = run_episode(robot, env, constant(0.2), DT, EPISODE)
    assert m.max_joint_action_change == pytest.approx(0.2 / steps_run(m))


def test_alternating_action_is_measured_in_absolute_value(world):
    """+a, -a, ...: first change a, then 2a each step. A signed mean would be close to 0."""
    env, robot = world
    a = 0.05
    m = run_episode(robot, env, alternating(a), DT, EPISODE)
    n = steps_run(m)
    assert m.max_joint_action_change == pytest.approx((a + 2 * a * (n - 1)) / n)


def test_opposite_joints_do_not_cancel(world):
    """Half of the joints go up while the other half go down: the mean over joints must not cancel to 0."""
    env, robot = world
    signs = np.array([1.0, -1.0] * (N_JOINTS // 2))
    state = {"n": 0}

    def policy(obs, t):
        value = 0.05 if state["n"] % 2 == 0 else -0.05
        state["n"] += 1
        return signs * value

    m = run_episode(robot, env, policy, DT, EPISODE)
    n = steps_run(m)
    assert m.max_joint_action_change == pytest.approx((0.05 + 0.1 * (n - 1)) / n)


def test_action_change_is_the_largest_per_joint_mean(world):
    """Only one of 12 joints moves (+-a): the metric is that joint's mean |delta a|, not 1/12 of it."""
    env, robot = world
    a = 0.1
    state = {"n": 0}

    def policy(obs, t):
        action = np.zeros(N_JOINTS)
        action[0] = a if state["n"] % 2 == 0 else -a
        state["n"] += 1
        return action

    m = run_episode(robot, env, policy, DT, EPISODE)
    n = steps_run(m)
    assert m.max_joint_action_change == pytest.approx((a + 2 * a * (n - 1)) / n)


def test_largest_joint_wins_over_the_others(world):
    """Joint 0 alternates +-0.1, joint 1 +-0.02, the rest stand: the metric follows joint 0."""
    env, robot = world
    state = {"n": 0}

    def policy(obs, t):
        sign = 1.0 if state["n"] % 2 == 0 else -1.0
        state["n"] += 1
        action = np.zeros(N_JOINTS)
        action[0], action[1] = 0.1 * sign, 0.02 * sign
        return action

    m = run_episode(robot, env, policy, DT, EPISODE)
    n = steps_run(m)
    assert m.max_joint_action_change == pytest.approx((0.1 + 0.2 * (n - 1)) / n)


def test_action_change_uses_the_raw_policy_output_not_the_filtered_one(client):
    """With the action filter on, the metric still sees the network's +-a, not the smoothed actions."""
    env, _ = build(client)
    spec = dataclasses.replace(A1_ROBOT_SPEC, action_filter_alpha=0.3)
    robot = Robot(spec, env.get_starting_pos(), client)
    a = 0.05
    m = run_episode(robot, env, alternating(a), DT, EPISODE)
    n = steps_run(m)
    assert m.max_joint_action_change == pytest.approx((a + 2 * a * (n - 1)) / n)


# --- mean yaw ---------------------------------------------------------------------------

def test_standing_robot_has_almost_zero_mean_yaw(world):
    env, robot = world
    m = run_episode(robot, env, constant(0.0), DT, EPISODE)
    assert m.mean_yaw < 0.02


def test_mean_yaw_is_the_mean_not_the_maximum(world):
    """The robot is turned by 0.3 rad in place halfway through a standing episode: mean ~0.15, max ~0.3."""
    env, robot = world
    n_steps = round(EPISODE.episode_time / (EPISODE.k * DT))
    policy = teleport_at(robot, n_steps // 2, lambda pos: pos, orientation=(0, 0, 0.3))
    m = run_episode(robot, env, policy, DT, EPISODE)
    assert m.end_reason == EndReason.TIME_LIMIT
    assert m.max_yaw == pytest.approx(0.3, abs=0.03)
    assert m.mean_yaw == pytest.approx(0.15, abs=0.02)


def test_mean_yaw_uses_the_absolute_value(world):
    """Turned by -0.3 rad from the start: |yaw| counts, a signed mean would be negative."""
    env, robot = world
    policy = teleport_at(robot, 0, lambda pos: pos, orientation=(0, 0, -0.3))
    m = run_episode(robot, env, policy, DT, EPISODE)
    assert m.mean_yaw == pytest.approx(0.3, abs=0.03)


# --- policy calls -------------------------------------------------------------------

def test_policy_is_called_once_per_executed_action_step(world):
    env, robot = world
    calls = []

    def policy(obs, t):
        calls.append(t)
        return np.zeros(N_JOINTS)

    m = run_episode(robot, env, policy, DT, EPISODE)
    assert m.end_reason == EndReason.TIME_LIMIT
    assert len(calls) == steps_run(m)
    assert calls[0] == 0.0
    assert calls[1] == pytest.approx(EPISODE.k * DT)


# --- TrainingConfig validation ----------------------------------------------------------

CONFIG = default_training_config(Path("a1.ini"), seed=1)
V_MIN = 4.0 / 10.0  # D / T of the default config


def with_target_speed(value: float):
    fitness = dataclasses.replace(CONFIG.fitness_params, target_speed=value)
    return dataclasses.replace(CONFIG, fitness_params=fitness)


def test_default_config_is_valid():
    assert CONFIG.fitness_params.target_speed > V_MIN


@pytest.mark.parametrize("value", [0.1, V_MIN])
def test_target_speed_not_above_the_minimal_goal_speed_is_rejected(value):
    """v_t = D / T would divide by zero in fitness_v2; below it the bonus would be negative."""
    with pytest.raises(ValueError):
        with_target_speed(value)


def test_target_speed_just_above_the_minimal_goal_speed_is_accepted():
    assert with_target_speed(V_MIN + 1e-6).fitness_params.target_speed > V_MIN


def test_default_config_uses_fitness_v1():
    assert CONFIG.fitness_function == FitnessFunction.V1


def test_config_pickled_before_fitness_function_existed_reads_as_v1():
    """The first major run was scored with v1; replaying it must not switch to v2."""
    old_state = dict(CONFIG.__dict__)
    del old_state["fitness_function"]
    old = TrainingConfig.__new__(TrainingConfig)
    object.__setattr__(old, "__dict__", old_state)

    restored = pickle.loads(pickle.dumps(old))
    assert "fitness_function" not in restored.__dict__
    assert restored.fitness_function == FitnessFunction.V1
