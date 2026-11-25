"""Tests for the NEAT evaluation layer (training/evaluation.py) and the A1 config file.

Contract (agreed 2026-11-15):
- configs/a1.ini: 36 inputs (34 observations + sin and cos of the phase), 12 tanh outputs,
- the network sees obs followed by [sin(2 pi f t), cos(2 pi f t)],
- init_worker builds its own DIRECT server from the TrainingConfig,
- eval_genome is deterministic: the same genome gives the same fitness twice,
- a network with all weights and biases 0 outputs 0, i.e. the rest pose, so the robot stands.
"""

import math
from pathlib import Path

import neat
import numpy as np
import pybullet as p
import pytest

import neat_walking_robot
from neat_walking_robot.environments import BaseEnvironmentData, EnvironmentType
from neat_walking_robot.robot import A1_ROBOT_SPEC
from neat_walking_robot.simulation import SimulationData
from neat_walking_robot.training import evaluation
from neat_walking_robot.training.episode import EpisodeParams
from neat_walking_robot.training.fitness import FitnessParams

CONFIG_PATH =  "configs/a1.ini"

TRAINING = evaluation.TrainingConfig(
    simulation_data=SimulationData(num_solver_iterations=60),  # non-default on purpose
    env_type=EnvironmentType.FLAT,
    env_data=BaseEnvironmentData(
        track_length=6.0, track_width=3.0, wall_height=1.0, wall_thickness=0.25,
        ground_thickness=0.2, friction=1.0, start_distance=1.0, end_distance=1.0,
    ),
    robot_spec=A1_ROBOT_SPEC,
    episode_params=EpisodeParams(episode_time=2.0, frequency=3, k=4),  # short: tests stay fast
    # target_speed must exceed D / T = 4 m / 2 s (TrainingConfig validation, 2026-11-24)
    fitness_params=FitnessParams(goal_bonus=2.0, penalty=1.0, target_speed=3.0, mean_action_change_threshold=0.2,
                                 yaw_limit=math.pi / 2),
    neat_config_path=Path(CONFIG_PATH),
    seed=0,
)


@pytest.fixture
def neat_config() -> neat.Config:
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                       neat.DefaultSpeciesSet, neat.DefaultStagnation, str(CONFIG_PATH))


@pytest.fixture
def worker():
    """Runs init_worker in this process and disconnects its server afterwards."""
    evaluation._worker.clear()
    evaluation.init_worker(TRAINING)
    yield evaluation._worker
    p.disconnect(evaluation._worker["client"])
    evaluation._worker.clear()


def first_genome(neat_config):
    population = neat.Population(neat_config, seed=1)
    return next(iter(population.population.values()))


def zero_genome(neat_config):
    genome = first_genome(neat_config)
    for conn in genome.connections.values():
        conn.weight = 0.0
    for node in genome.nodes.values():
        node.bias = 0.0
    return genome


# --- config file --------------------------------------------------------------

def test_config_has_36_inputs_and_12_outputs(neat_config):
    assert neat_config.genome_config.num_inputs == 36
    assert neat_config.genome_config.num_outputs == 12


def test_outputs_use_tanh(neat_config):
    """tanh gives [-1, 1] with 0 = rest pose; sigmoid would give [0, 1] with 0.5 at zero input."""
    assert neat_config.genome_config.activation_default == "tanh"
    assert neat_config.genome_config.activation_options == ["tanh"]


# --- worker -------------------------------------------------------------------

def test_init_worker_builds_its_own_server(worker):
    client = worker["client"]
    assert p.isConnected(physicsClientId=client)
    assert worker["robot"].client_id == client
    assert worker["env"].client == client
    assert worker["config"] is TRAINING


def test_init_worker_applies_the_simulation_data(worker):
    params = p.getPhysicsEngineParameters(physicsClientId=worker["client"])
    assert params["fixedTimeStep"] == pytest.approx(TRAINING.simulation_data.dt)
    assert params["gravityAccelerationZ"] == pytest.approx(TRAINING.simulation_data.gravity[2])
    assert params["numSolverIterations"] == TRAINING.simulation_data.num_solver_iterations


# --- eval_genome --------------------------------------------------------------

def test_network_gets_observation_and_phase(worker, neat_config, monkeypatch):
    """Spy network: records every input vector and returns the rest pose."""
    seen = []

    class SpyNet:
        def activate(self, inputs):
            seen.append(list(inputs))
            return [0.0] * 12

    monkeypatch.setattr(neat.nn.FeedForwardNetwork, "create", staticmethod(lambda g, c: SpyNet()))
    evaluation.eval_genome(first_genome(neat_config), neat_config)

    f = TRAINING.episode_params.frequency
    step = TRAINING.episode_params.k * TRAINING.simulation_data.dt
    assert all(len(x) == 36 for x in seen)
    assert seen[0][34:] == pytest.approx([0.0, 1.0])
    assert seen[1][34:] == pytest.approx([math.sin(2 * math.pi * f * step), math.cos(2 * math.pi * f * step)])


def test_eval_genome_returns_a_finite_float(worker, neat_config):
    fitness = evaluation.eval_genome(first_genome(neat_config), neat_config)
    assert isinstance(fitness, float)
    assert np.isfinite(fitness)


def test_eval_genome_is_deterministic(worker, neat_config):
    genome = first_genome(neat_config)
    first = evaluation.eval_genome(genome, neat_config)
    evaluation.eval_genome(zero_genome(neat_config), neat_config)
    assert evaluation.eval_genome(genome, neat_config) == first


def test_zero_network_keeps_the_robot_standing(worker, neat_config):
    """All weights and biases 0: tanh(0) = 0 = rest pose, so no fall, no progress, fitness about 0."""
    fitness = evaluation.eval_genome(zero_genome(neat_config), neat_config)
    assert fitness == pytest.approx(0.0, abs=0.05)
