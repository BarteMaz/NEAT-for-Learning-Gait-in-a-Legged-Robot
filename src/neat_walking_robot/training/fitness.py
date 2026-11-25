from dataclasses import dataclass
from enum import Enum

import numpy as np

from neat_walking_robot.training.episode import EndReason, EpisodeMetrics


@dataclass(frozen=True, kw_only=True)
class FitnessParams:
    goal_bonus: float
    penalty: float
    target_speed: float
    mean_action_change_threshold: float
    yaw_limit:float
    def __post_init__(self):
        if self.goal_bonus < 0:
            raise ValueError("Goal bonus must be greater or equal 0")
        if self.penalty < 0:
            raise ValueError("Penalty must be greater or equal 0")
        if self.target_speed <= 0:
            raise ValueError("Target speed must be greater than 0")
        if self.mean_action_change_threshold <= 0:
            raise ValueError("Mean action change threshold must be greater than 0")
        if self.yaw_limit <= 0:
            raise ValueError("Yaw limit must be greater than 0")

def fitness_v1(metrics: EpisodeMetrics, fitness_params: FitnessParams) -> float:
    gain = np.clip(metrics.progress, 0.0, metrics.goal_distance)
    bonus = 0.0
    if metrics.end_reason in [EndReason.FALL, EndReason.WALL]:
        bonus = -fitness_params.penalty
    elif metrics.end_reason == EndReason.GOAL:
        bonus = fitness_params.goal_bonus * (metrics.episode_length - metrics.duration) / metrics.episode_length

    return gain + bonus

def fitness_v2(metrics: EpisodeMetrics, fitness_params: FitnessParams) -> float:
    gain = np.clip(metrics.progress, 0.0, metrics.goal_distance)
    target_velocity = fitness_params.target_speed
    min_velocity = metrics.goal_distance / metrics.episode_length
    velocity = gain / metrics.duration

    mean_0 = fitness_params.mean_action_change_threshold
    mean = metrics.max_joint_action_change
    smoothness_multiplier = 1.0 if mean <= mean_0 else mean_0 / mean

    yaw_limit = fitness_params.yaw_limit
    mean_yaw = metrics.mean_yaw
    yaw_multiplier = max(0.0, 1 - mean_yaw / yaw_limit)

    bonus = 0.0
    penalty = 0.0
    if metrics.end_reason in [EndReason.FALL, EndReason.WALL]:
        penalty = fitness_params.penalty
    elif metrics.end_reason == EndReason.GOAL:
        bonus = fitness_params.goal_bonus * np.clip((velocity - min_velocity) / (target_velocity - min_velocity), 0, 1)

    return (gain + bonus) * smoothness_multiplier * yaw_multiplier - penalty


class FitnessFunction(Enum):
    V1 = "v1"
    V2 = "v2"

FITNESS = {
    FitnessFunction.V1: fitness_v1,
    FitnessFunction.V2: fitness_v2,
}

