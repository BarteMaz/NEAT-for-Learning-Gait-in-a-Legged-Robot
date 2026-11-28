from collections.abc import Callable

import neat
import numpy as np
import pybullet as p

from neat_walking_robot.environments import create_environment
from neat_walking_robot.robot import Robot
from neat_walking_robot.simulation import prepare_simulation
from neat_walking_robot.training.config import TrainingConfig
from neat_walking_robot.training.episode import EpisodeParams, run_episode
from neat_walking_robot.training.fitness import FitnessParams, FITNESS

_worker = {}

def init_worker(training_config: TrainingConfig):
    client = p.connect(p.DIRECT)
    prepare_simulation(training_config.simulation_data, client)
    env = create_environment(training_config.env_type, training_config.env_data, client, training_config.seed)
    robot = Robot(training_config.robot_spec, env.get_starting_pos(), client)
    _worker.update(client=client, env=env, robot=robot, config=training_config)

def make_policy(net) -> Callable[[np.ndarray], list[float]]:
    return lambda obs : net.activate(list(obs))

def eval_genome(genome, config):
    episode_params: EpisodeParams = _worker["config"].episode_params
    fitness_params: FitnessParams = _worker["config"].fitness_params
    dt = _worker["config"].simulation_data.dt

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    policy = make_policy(net)
    metrics = run_episode(_worker["robot"], _worker["env"], policy, dt, episode_params)

    return FITNESS[_worker["config"].fitness_function](metrics, fitness_params)

# for series evaluation
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)