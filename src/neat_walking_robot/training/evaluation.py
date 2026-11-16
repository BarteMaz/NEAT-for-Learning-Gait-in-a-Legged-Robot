from dataclasses import dataclass
from typing import Callable
from pathlib import Path

import neat
import numpy as np
import pybullet as p

from neat_walking_robot.environments import (
    BaseEnvironmentData,
    EnvironmentType,
    create_environment,
)
from neat_walking_robot.robot import Robot, RobotSpec
from neat_walking_robot.simulation import SimulationData, prepare_simulation
from neat_walking_robot.training.episode import EpisodeParams, run_episode
from neat_walking_robot.training.fitness import FitnessParams, fitness_v1


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

_worker = {}

def init_worker(training_config: TrainingConfig):
    client = p.connect(p.DIRECT)
    prepare_simulation(training_config.simulation_data, client)
    env = create_environment(training_config.env_type, training_config.env_data, client, training_config.seed)
    robot = Robot(training_config.robot_spec, env.get_starting_pos(), client)
    _worker.update(client=client, env=env, robot=robot, config=training_config)

def make_policy(net, episode_params) -> Callable[[np.ndarray, float], list[float]]:
    return lambda obs, t : net.activate(list(obs) + [np.sin(np.pi * 2 * episode_params.frequency * t), np.cos(np.pi * 2 * episode_params.frequency * t)])

def eval_genome(genome, config):
    episode_params: EpisodeParams = _worker["config"].episode_params
    fitness_params: FitnessParams = _worker["config"].fitness_params
    dt = _worker["config"].simulation_data.dt

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    policy = make_policy(net, episode_params)
    metrics = run_episode(_worker["robot"], _worker["env"], policy, dt, episode_params)

    return fitness_v1(metrics, fitness_params)