import json
import pickle
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from neat_walking_robot.environments import (
    BaseEnvironmentData,
    EnvironmentType,
)
from neat_walking_robot.robot import A1_ROBOT_SPEC, RobotSpec
from neat_walking_robot.simulation import SimulationData
from neat_walking_robot.training.episode import EpisodeParams
from neat_walking_robot.training.fitness import FitnessParams


@dataclass(frozen=True, kw_only=True)
class TrainingConfig:
    simulation_data: SimulationData
    env_type: EnvironmentType
    env_data: BaseEnvironmentData
    robot_spec: RobotSpec
    episode_params: EpisodeParams
    fitness_params: FitnessParams
    neat_config_path: Path
    seed: int


def default_training_config(neat_config_path: Path, seed: int) -> TrainingConfig:
    simulation_data = SimulationData()
    env_data = BaseEnvironmentData(
        track_length=6.0,
        track_width=3.0,
        wall_height=3.0,
        wall_thickness=0.25,
        ground_thickness=0.20,
        end_distance=1.0,
        start_distance=1.0,
        friction=1.0,
        lowest_ground_point=-0.20,
    )
    robot_data = A1_ROBOT_SPEC
    episode_params = EpisodeParams(
        episode_time=10.0,
        frequency=3.0,
        k=4
    )
    fit_params = FitnessParams(
        goal_bonus=2.0,
        penalty=1.0,
    )
    return TrainingConfig(
        simulation_data=simulation_data,
        env_type=EnvironmentType.FLAT,
        env_data=env_data,
        robot_spec=robot_data,
        episode_params=episode_params,
        fitness_params=fit_params,
        neat_config_path=neat_config_path,
        seed=seed
    )


def save_training_config_to_pickle(training_config: TrainingConfig, path: Path):
    file_path = path / "training_config.pkl"
    with open(file_path, 'wb') as f:
        pickle.dump(training_config, f)


def to_json(o):
    if isinstance(o, Enum):
        return o.name
    return str(o)


def save_training_config_to_json(training_config: TrainingConfig, path: Path):
    file_path = path / "training_config.json"
    data = {
        "simulation_data": asdict(training_config.simulation_data),
        "environment_type": training_config.env_type,
        "environment_data": asdict(training_config.env_data),
        "robot_specification": asdict(training_config.robot_spec),
        "episode_params": asdict(training_config.episode_params),
        "fitness_params": asdict(training_config.fitness_params),
        "neat_config_path": str(training_config.neat_config_path),
        "seed": training_config.seed,
    }
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(data, f, default=to_json, ensure_ascii=False, indent=2)


