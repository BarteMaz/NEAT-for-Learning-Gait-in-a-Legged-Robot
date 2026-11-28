"""Tests for the NEAT evaluation layer (training/evaluation.py) and the A1 config file.

Contract (agreed 2026-11-27; the phase inputs sin/cos were removed):
- configs/a1.ini: 33 inputs (exactly the robot observation), 12 tanh outputs,
- make_policy(net) passes the observation to the network unchanged and returns its output,
- init_worker builds its own DIRECT server from the TrainingConfig,
- eval_genome is deterministic: the same genome gives the same fitness twice,
- a network with all weights and biases 0 outputs 0, i.e. zero joint velocities, so the robot
  stays where it is and scores about 0 (it sags slowly: about 1.5 cm in a 2 s episode),
- default_training_config uses FitnessFunction.DISTANCE (TrainingConfig itself has no default).
"""

from pathlib import Path

import neat
import numpy as np
import pybullet as p
import pytest

from neat_walking_robot.environments import BaseEnvironmentData, EnvironmentType
from neat_walking_robot.robot import A1_ROBOT_SPEC
from neat_walking_robot.simulation import SimulationData
from neat_walking_robot.training import evaluation
from neat_walking_robot.training.config import default_training_config
from neat_walking_robot.training.episode import EpisodeParams
from neat_walking_robot.training.fitness import FitnessFunction, FitnessParams

CONFIG_PATH = "configs/a1.ini"
OBS_LEN = 33

TRAINING = evaluation.TrainingConfig(
    simulation_data=SimulationData(num_solver_iterations=60),  # non-default on purpose
    env_type=EnvironmentType.FLAT,
    env_data=BaseEnvironmentData(
        track_length=6.0, track_width=3.0, wall_height=1.0, wall_thickness=0.25,
        ground_thickness=0.2, friction=1.0, start_distance=1.0, end_distance=1.0,
    ),
    robot_spec=A1_ROBOT_SPEC,
    episode_params=EpisodeParams(episode_time=2.0, k=4),  # short: tests stay fast
    fitness_params=FitnessParams(goal_bonus=2.0, penalty=1.0),
    fitness_function=FitnessFunction.DISTANCE,
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

def test_config_has_33_inputs_and_12_outputs(neat_config):
    assert neat_config.genome_config.num_inputs == OBS_LEN
    assert neat_config.genome_config.num_outputs == 12


def test_config_inputs_match_the_robot_observation(worker, neat_config):
    assert worker["robot"].get_observation().shape == (neat_config.genome_config.num_inputs,)


def test_outputs_use_tanh(neat_config):
    """tanh gives [-1, 1] with 0 = zero velocity; sigmoid would give [0, 1] with 0.5 at zero input."""
    assert neat_config.genome_config.activation_default == "tanh"
    assert neat_config.genome_config.activation_options == ["tanh"]


# --- make_policy --------------------------------------------------------------

def test_make_policy_passes_the_observation_unchanged():
    seen = []

    class FakeNet:
        def activate(self, inputs):
            seen.append(inputs)
            return [0.25] * 12

    obs = np.linspace(-1.0, 1.0, OBS_LEN)
    policy = evaluation.make_policy(FakeNet())
    assert policy(obs) == [0.25] * 12
    assert seen == [list(obs)]


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

def test_network_gets_exactly_the_robot_observation(worker, neat_config, monkeypatch):
    """Spy network and spy observation: every network input is the observation of that step."""
    robot = worker["robot"]
    observations, inputs = [], []
    real_observation = robot.get_observation

    def recording_observation():
        obs = real_observation()
        observations.append(obs)
        return obs

    class SpyNet:
        def activate(self, values):
            inputs.append(list(values))
            return [0.0] * 12

    monkeypatch.setattr(robot, "get_observation", recording_observation)
    monkeypatch.setattr(neat.nn.FeedForwardNetwork, "create", staticmethod(lambda g, c: SpyNet()))
    evaluation.eval_genome(first_genome(neat_config), neat_config)

    assert len(inputs) == len(observations) > 0
    for given, obs in zip(inputs, observations):
        assert given == list(obs)


def test_default_training_config_uses_the_distance_fitness():
    config = default_training_config(Path(CONFIG_PATH), seed=0)
    assert config.fitness_function == FitnessFunction.DISTANCE


def test_eval_genome_returns_a_finite_float(worker, neat_config):
    fitness = evaluation.eval_genome(first_genome(neat_config), neat_config)
    assert isinstance(fitness, float)
    assert np.isfinite(fitness)


def test_eval_genome_is_deterministic(worker, neat_config):
    genome = first_genome(neat_config)
    first = evaluation.eval_genome(genome, neat_config)
    evaluation.eval_genome(zero_genome(neat_config), neat_config)
    assert evaluation.eval_genome(genome, neat_config) == first


def test_zero_network_keeps_the_robot_in_place(worker, neat_config):
    """All weights and biases 0: tanh(0) = 0 = zero joint velocity, so no fall and about no distance."""
    fitness = evaluation.eval_genome(zero_genome(neat_config), neat_config)
    assert fitness == pytest.approx(0.0, abs=0.05)
