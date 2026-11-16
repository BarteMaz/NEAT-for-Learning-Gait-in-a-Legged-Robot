from neat_walking_robot.environments import (
    EnvironmentType,
    BaseEnvironmentData,
    create_environment
)
from neat_walking_robot.robot import (
    Robot,
    A1_ROBOT_SPEC
)
from neat_walking_robot.simulation import (
    SimulationData,
    prepare_simulation
)
from neat_walking_robot.training import (
    TrainingConfig,
    FitnessParams,
    EpisodeParams,
    run_episode,
    fitness_v1,
)
from enum import Enum
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
import argparse
import pickle
import json
import shutil
import numpy as np
import pybullet as p
import neat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "conf.ini"
RESULTS_PATH = PROJECT_ROOT / "results"

def default_training_config() -> TrainingConfig:
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

    scripts_folder = Path(__file__).resolve().parent
    config_path = scripts_folder.parent / 'configs' / 'a1.ini'

    return TrainingConfig(
        simulation_data=simulation_data,
        env_type=EnvironmentType.FLAT,
        env_data=env_data,
        robot_spec=robot_data,
        episode_params=episode_params,
        fitness_params=fit_params,
        neat_config_path=config_path,
        seed=42
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
        json.dump(data, f, default=to_json, ensure_ascii=False ,indent=2)


def main(args):

    # training's folder
    now = datetime.now()
    date = now.strftime("%Y-%m-%d_%H-%M-%S")

    results_path = RESULTS_PATH / f"{args.name}_{args.seed}_{date}"
    results_path.mkdir(parents=True, exist_ok=True)

    # read and save config
    training_config = default_training_config()
    save_training_config_to_pickle(training_config, results_path)
    save_training_config_to_json(training_config, results_path)
    shutil.copy2(training_config.neat_config_path, results_path)

    # neat_config
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        training_config.neat_config_path
    )
    population = neat.Population(neat_config, seed=args.seed)


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--generations", type=int, default=100)
parser.add_argument("--name", type=str, default="a1_train")
args = parser.parse_args()

main(args)

