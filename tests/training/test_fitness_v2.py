"""Tests for fitness_v2 (training/fitness.py, stage 5, block 4; agreed 2026-11-24).

Contract:
- progress along x counts, clipped to [0, goal_distance],
- at the goal: bonus = goal_bonus * clip((v - v_min) / (v_t - v_min), 0, 1), v = progress / t_goal,
  v_min = D / T (the slowest speed that still reaches the goal), so the bonus is 0 at t_goal = T
  (no jump at the goal line) and saturates at the target speed v_t,
- the positive part (progress + bonus) is multiplied by s = min(1, m0 / m), m = mean |delta action|;
  s = 1 for m <= m0, including m = 0,
- the fall / wall penalty is subtracted AFTER the multiplication, so rough actions never soften it,
- falling never beats standing still.
"""

import pytest

from neat_walking_robot.training.episode import EndReason, EpisodeMetrics
from neat_walking_robot.training.fitness import FITNESS, FitnessFunction, FitnessParams, fitness_v1, fitness_v2

T = 10.0
D = 4.0
V_MIN = D / T  # 0.4 m/s
PARAMS = FitnessParams(goal_bonus=2.0, penalty=1.0, target_speed=1.0, mean_action_change_threshold=0.2)
M0 = PARAMS.mean_action_change_threshold


def metrics(progress: float, reason: EndReason, duration: float = T, change: float = 0.0) -> EpisodeMetrics:
    return EpisodeMetrics(
        progress=progress, goal_distance=D, end_reason=reason,
        episode_length=T, duration=duration, max_y=0.0, max_yaw=0.0,
        mean_action_change=change,
    )


# --- progress and goal bonus ------------------------------------------------

def test_time_limit_scores_progress():
    assert fitness_v2(metrics(2.5, EndReason.TIME_LIMIT), PARAMS) == pytest.approx(2.5)


def test_backwards_progress_counts_as_zero():
    assert fitness_v2(metrics(-0.7, EndReason.TIME_LIMIT), PARAMS) == pytest.approx(0.0)


def test_goal_bonus_interpolates_between_minimal_and_target_speed():
    """Goal after 8 s: v = 0.5 m/s, bonus = 2 * (0.5 - 0.4) / (1.0 - 0.4) = 1/3."""
    assert fitness_v2(metrics(D, EndReason.GOAL, duration=8.0), PARAMS) == pytest.approx(D + 1 / 3)


def test_goal_bonus_saturates_at_the_target_speed():
    """Faster than v_t earns nothing more: 1 m/s and 2 m/s score the same."""
    at_target = fitness_v2(metrics(D, EndReason.GOAL, duration=D / 1.0), PARAMS)
    faster = fitness_v2(metrics(D, EndReason.GOAL, duration=D / 2.0), PARAMS)
    assert at_target == pytest.approx(D + PARAMS.goal_bonus)
    assert faster == pytest.approx(at_target)


def test_goal_bonus_is_continuous_at_the_time_limit():
    at_limit = fitness_v2(metrics(D, EndReason.GOAL, duration=T), PARAMS)
    not_quite = fitness_v2(metrics(D, EndReason.TIME_LIMIT), PARAMS)
    assert at_limit == pytest.approx(not_quite)


def test_faster_goal_never_scores_lower():
    durations = [10.0, 9.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0]
    scores = [fitness_v2(metrics(D, EndReason.GOAL, duration=t), PARAMS) for t in durations]
    assert all(a <= b + 1e-12 for a, b in zip(scores, scores[1:]))


# --- smoothness ---------------------------------------------------------------

@pytest.mark.parametrize("change", [0.0, 0.05, M0])
def test_smooth_actions_are_not_penalised(change):
    """m <= m0 gives factor 1; m = 0 (constant network output) must not divide by zero."""
    assert fitness_v2(metrics(2.5, EndReason.TIME_LIMIT, change=change), PARAMS) == pytest.approx(2.5)


def test_rough_actions_scale_the_positive_part():
    """m = 2 * m0 halves progress + bonus: goal at 1 m/s -> (4 + 2) / 2."""
    m = metrics(D, EndReason.GOAL, duration=D / 1.0, change=2 * M0)
    assert fitness_v2(m, PARAMS) == pytest.approx((D + PARAMS.goal_bonus) / 2)


def test_rougher_never_scores_higher():
    changes = [0.0, 0.2, 0.3, 0.5, 1.0, 2.0]
    scores = [fitness_v2(metrics(3.0, EndReason.TIME_LIMIT, change=c), PARAMS) for c in changes]
    assert all(a >= b - 1e-12 for a, b in zip(scores, scores[1:]))


# --- penalty -------------------------------------------------------------------

@pytest.mark.parametrize("reason", [EndReason.FALL, EndReason.WALL])
def test_penalty_is_subtracted_after_the_smoothness_factor(reason):
    """Progress 1.5 m, m = 2 * m0: 1.5 * 0.5 - 1.0 = -0.25 (not (1.5 - 1.0) * 0.5 = 0.25)."""
    m = metrics(1.5, reason, duration=2.0, change=2 * M0)
    assert fitness_v2(m, PARAMS) == pytest.approx(-0.25)


@pytest.mark.parametrize("progress", [0.0, 0.5, 2.0])
def test_vibrating_while_falling_never_pays(progress):
    """With the penalty inside the product a rough fall (s = 0.17) would lose only 0.17 instead of 1."""
    smooth = fitness_v2(metrics(progress, EndReason.FALL, duration=2.0, change=0.0), PARAMS)
    rough = fitness_v2(metrics(progress, EndReason.FALL, duration=2.0, change=1.2), PARAMS)
    assert rough <= smooth


def test_fall_gets_no_goal_bonus():
    assert fitness_v2(metrics(1.5, EndReason.FALL, duration=2.0), PARAMS) == pytest.approx(0.5)


@pytest.mark.parametrize("fall_time", [0.1, 1.0, 5.0, 9.9])
@pytest.mark.parametrize("change", [0.0, 1.5])
def test_falling_never_beats_standing(fall_time, change):
    fall = fitness_v2(metrics(0.0, EndReason.FALL, duration=fall_time, change=change), PARAMS)
    stand = fitness_v2(metrics(0.0, EndReason.TIME_LIMIT), PARAMS)
    assert fall < stand


# --- the case that motivated v2 ---------------------------------------------------

def test_baseline_beats_the_vibrating_winner_of_the_first_run():
    """Numbers from the block 1 diagnosis (2026-11-23).

    Baseline: goal after 7.383 s, progress 3.993 m, m = 0.053 -> 3.993 + 2 * (0.5408 - 0.4) / 0.6 = 4.462.
    First winner: goal after 2.0 s, progress 4.0 m, m = 1.17 -> (4 + 2) * 0.2 / 1.17 = 1.026.
    Under v1 the winner scored 5.6 and the baseline 4.52.
    """
    baseline = fitness_v2(metrics(3.993, EndReason.GOAL, duration=7.383, change=0.053), PARAMS)
    winner = fitness_v2(metrics(4.0, EndReason.GOAL, duration=2.0, change=1.17), PARAMS)
    assert baseline == pytest.approx(4.462, abs=1e-3)
    assert winner == pytest.approx(1.026, abs=1e-3)
    assert baseline > winner


# --- registry (2026-11-25) ------------------------------------------------------------

def test_every_fitness_function_has_an_implementation():
    """A new FitnessFunction member without a FITNESS entry would only fail inside a worker process."""
    assert set(FITNESS) == set(FitnessFunction)


def test_registry_maps_names_to_the_right_functions():
    assert FITNESS[FitnessFunction.V1] is fitness_v1
    assert FITNESS[FitnessFunction.V2] is fitness_v2
