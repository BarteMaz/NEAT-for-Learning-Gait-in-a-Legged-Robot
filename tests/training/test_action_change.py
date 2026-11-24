"""Tests for EpisodeMetrics.mean_action_change (run_episode) and the TrainingConfig validation of target_speed
(stage 5, 2026-11-24).

Contract for mean_action_change:
- m = mean over executed action steps n and joints j of |a_j[n] - a_j[n-1]|, with a[-1] = 0 (the rest pose),
- it is computed from the RAW policy output (before the robot's action filter),
- the policy is called exactly once per executed action step (no extra call before the loop:
  stateful policies, the recording wrapper in diagnose.py and the real-time replay rely on it).

Contract for TrainingConfig: fitness_v2 divides by (target_speed - D / T), so target_speed must be > D / T.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from neat_walking_robot.robot import A1_ROBOT_SPEC, Robot
from neat_walking_robot.training.config import default_training_config
from neat_walking_robot.training.episode import EndReason, run_episode
from test_episode import DT, EPISODE, build

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
    assert m.mean_action_change == 0.0


def test_constant_action_counts_only_the_first_step_from_rest(world):
    """a[-1] = 0, then 0.2 forever: one change of 0.2 per joint, averaged over all steps."""
    env, robot = world
    m = run_episode(robot, env, constant(0.2), DT, EPISODE)
    assert m.mean_action_change == pytest.approx(0.2 / steps_run(m))


def test_alternating_action_is_measured_in_absolute_value(world):
    """+a, -a, ...: first change a, then 2a each step. A signed mean would be close to 0."""
    env, robot = world
    a = 0.05
    m = run_episode(robot, env, alternating(a), DT, EPISODE)
    n = steps_run(m)
    assert m.mean_action_change == pytest.approx((a + 2 * a * (n - 1)) / n)


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
    assert m.mean_action_change == pytest.approx((0.05 + 0.1 * (n - 1)) / n)


def test_action_change_is_averaged_over_joints(world):
    """Only one of 12 joints moves (+-a): the metric is 1/12 of that joint's mean |delta a|, not its maximum."""
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
    assert m.mean_action_change == pytest.approx((a + 2 * a * (n - 1)) / n / N_JOINTS)


def test_action_change_uses_the_raw_policy_output_not_the_filtered_one(client):
    """With the action filter on, the metric still sees the network's +-a, not the smoothed actions."""
    env, _ = build(client)
    spec = dataclasses.replace(A1_ROBOT_SPEC, action_filter_alpha=0.3)
    robot = Robot(spec, env.get_starting_pos(), client)
    a = 0.05
    m = run_episode(robot, env, alternating(a), DT, EPISODE)
    n = steps_run(m)
    assert m.mean_action_change == pytest.approx((a + 2 * a * (n - 1)) / n)


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
