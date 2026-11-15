"""Tests for the preliminary fitness function (training/fitness.py).

Contract (agreed 2026-11-15, refined in stage 5):
- progress along x counts, clipped to [0, goal_distance],
- a time bonus bonus * (T - t_goal) / T only when the goal is reached,
- a constant penalty on a fall or a wall contact,
- no discontinuity at the goal: reaching it exactly at T scores like walking D without the goal,
- falling must never pay better than standing still.
The function is pure, so these tests need no simulator.
"""

import pytest

from neat_walking_robot.training.episode import EndReason, EpisodeMetrics
from neat_walking_robot.training.fitness import FitnessParams
from neat_walking_robot.training.fitness import fitness_v1 as fitness

T = 10.0
D = 4.0
PARAMS = FitnessParams(goal_bonus=2.0, penalty=1.0)


def metrics(progress: float, reason: EndReason, duration: float = T) -> EpisodeMetrics:
    return EpisodeMetrics(
        progress=progress, goal_distance=D, end_reason=reason,
        episode_length=T, duration=duration, max_y=0.0, max_yaw=0.0,
    )


def test_time_limit_scores_progress():
    assert fitness(metrics(2.5, EndReason.TIME_LIMIT), PARAMS) == pytest.approx(2.5)


def test_backwards_progress_counts_as_zero():
    assert fitness(metrics(-0.7, EndReason.TIME_LIMIT), PARAMS) == pytest.approx(0.0)


def test_goal_scores_distance_plus_time_bonus():
    """Goal after 6 s of 10 s: D + 2 * 4/10."""
    assert fitness(metrics(D + 0.02, EndReason.GOAL, duration=6.0), PARAMS) == pytest.approx(D + 0.8)


def test_goal_bonus_is_continuous_at_the_time_limit():
    at_limit = fitness(metrics(D, EndReason.GOAL, duration=T), PARAMS)
    not_quite = fitness(metrics(D, EndReason.TIME_LIMIT), PARAMS)
    assert at_limit == pytest.approx(not_quite)


@pytest.mark.parametrize("reason", [EndReason.FALL, EndReason.WALL])
def test_fall_and_wall_get_only_the_penalty(reason):
    """No time bonus for ending early by falling: 1.5 - 1.0, not 1.5 - 1.0 + 2 * 0.8."""
    assert fitness(metrics(1.5, reason, duration=2.0), PARAMS) == pytest.approx(0.5)


@pytest.mark.parametrize("fall_time", [0.1, 1.0, 5.0, 9.9])
def test_falling_never_beats_standing(fall_time):
    """An immediate fall without progress must score below standing still for the whole episode."""
    fall = fitness(metrics(0.0, EndReason.FALL, duration=fall_time), PARAMS)
    stand = fitness(metrics(0.0, EndReason.TIME_LIMIT), PARAMS)
    assert fall < stand


def test_earlier_goal_scores_higher():
    early = fitness(metrics(D, EndReason.GOAL, duration=5.0), PARAMS)
    late = fitness(metrics(D, EndReason.GOAL, duration=8.0), PARAMS)
    assert early > late
