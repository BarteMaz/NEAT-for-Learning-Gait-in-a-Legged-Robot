"""Tests for the fitness functions (training/fitness.py).

Contract (agreed 2026-11-27, first training stage: distance from the start only):
- `distance` = EpisodeMetrics.max_distance, the largest horizontal distance from the start; nothing else
  counts: a fall or a wall contact costs nothing (agreed with the supervisor; later stages may change it),
- `toes_total_path` = EpisodeMetrics.toes_path_length (diagnostic experiment with a fixed trunk,
  2026-11-28); nothing else counts either,
- FitnessFunction (enum with text values) and the FITNESS registry stay, so new functions are added by
  name; every FitnessFunction has a function in FITNESS,
- FitnessParams rejects a negative bonus or penalty.
The functions are pure, so these tests need no simulator.
"""

import pytest

from neat_walking_robot.training.episode import EndReason, EpisodeMetrics
from neat_walking_robot.training.fitness import FITNESS, FitnessFunction, FitnessParams, distance, toes_total_path

T = 10.0
PARAMS = FitnessParams(goal_bonus=2.0, penalty=1.0)


def metrics(max_distance: float, reason: EndReason, duration: float = T, toes_path_length: float = 0.0) -> EpisodeMetrics:
    return EpisodeMetrics(max_distance=max_distance, toes_path_length=toes_path_length, end_reason=reason,
                          episode_length=T, duration=duration)


@pytest.mark.parametrize("reason", list(EndReason))
def test_distance_is_the_max_distance_whatever_the_end_reason(reason):
    assert distance(metrics(1.7, reason, duration=3.0), PARAMS) == pytest.approx(1.7)


def test_falling_costs_nothing():
    """The same distance scores the same, fallen or not."""
    fallen = distance(metrics(1.2, EndReason.FALL, duration=2.0), PARAMS)
    walked = distance(metrics(1.2, EndReason.TIME_LIMIT), PARAMS)
    assert fallen == walked


def test_distance_ignores_bonus_and_penalty():
    other = FitnessParams(goal_bonus=50.0, penalty=50.0)
    for m in (metrics(1.7, EndReason.TIME_LIMIT), metrics(1.7, EndReason.FALL, duration=2.0),
              metrics(1.7, EndReason.WALL, duration=2.0)):
        assert distance(m, other) == distance(m, PARAMS)


def test_standing_still_scores_zero():
    assert distance(metrics(0.0, EndReason.TIME_LIMIT), PARAMS) == 0.0


def test_distance_ignores_the_toes_path():
    assert distance(metrics(1.7, EndReason.TIME_LIMIT, toes_path_length=40.0), PARAMS) == pytest.approx(1.7)


@pytest.mark.parametrize("reason", list(EndReason))
def test_toes_total_path_is_the_toes_path_whatever_the_end_reason(reason):
    m = metrics(0.3, reason, duration=3.0, toes_path_length=12.5)
    assert toes_total_path(m, PARAMS) == pytest.approx(12.5)


def test_toes_total_path_ignores_distance_and_params():
    other = FitnessParams(goal_bonus=50.0, penalty=50.0)
    m = metrics(0.0, EndReason.TIME_LIMIT, toes_path_length=12.5)
    assert toes_total_path(m, other) == toes_total_path(m, PARAMS) == pytest.approx(12.5)
    assert toes_total_path(metrics(3.0, EndReason.TIME_LIMIT, toes_path_length=12.5), PARAMS) == pytest.approx(12.5)


def test_registry_maps_every_fitness_function():
    assert FITNESS[FitnessFunction.DISTANCE] is distance
    assert FITNESS[FitnessFunction.TOES_PATH] is toes_total_path
    assert FitnessFunction("distance") is FitnessFunction.DISTANCE  # the values used by --fitness
    assert FitnessFunction("toes_path") is FitnessFunction.TOES_PATH
    assert set(FITNESS) == set(FitnessFunction)


@pytest.mark.parametrize("field", ["goal_bonus", "penalty"])
def test_negative_params_are_rejected(field):
    with pytest.raises(ValueError):
        FitnessParams(**{"goal_bonus": 2.0, "penalty": 1.0, field: -0.1})


def test_zero_bonus_and_penalty_are_allowed():
    FitnessParams(goal_bonus=0.0, penalty=0.0)
