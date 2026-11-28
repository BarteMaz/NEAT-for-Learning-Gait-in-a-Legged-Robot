from dataclasses import dataclass
from enum import Enum

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

def distance(metrics: EpisodeMetrics, fitness_params: FitnessParams) -> float:
    return metrics.max_distance

def toes_total_path(metrics: EpisodeMetrics, fitness_params: FitnessParams) -> float:
    return metrics.toes_path_length

class FitnessFunction(Enum):
    DISTANCE = "distance"
    TOES_PATH = "toes_path"

FITNESS = {
    FitnessFunction.DISTANCE: distance,
    FitnessFunction.TOES_PATH: toes_total_path,
}

