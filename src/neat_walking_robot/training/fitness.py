from dataclasses import dataclass

import numpy as np

from neat_walking_robot.training.episode import EndReason, EpisodeMetrics


@dataclass(frozen=True, kw_only=True)
class FitnessParams:
    goal_bonus: float
    penalty: float
    def __post_init__(self):
        if self.goal_bonus < 0:
            raise ValueError("Goal bonus must be greater or equal 0")
        if self.penalty < 0:
            raise ValueError("Penalty must be greater or equal 0")


def fitness_v1(metrics: EpisodeMetrics, fitness_params: FitnessParams) -> float:
    gain = np.clip(metrics.progress, 0.0, metrics.goal_distance)
    bonus = 0.0
    if metrics.end_reason in [EndReason.FALL, EndReason.WALL]:
        bonus = -fitness_params.penalty
    elif metrics.end_reason == EndReason.GOAL:
        bonus = fitness_params.goal_bonus * (metrics.episode_length - metrics.duration) / metrics.episode_length

    return gain + bonus