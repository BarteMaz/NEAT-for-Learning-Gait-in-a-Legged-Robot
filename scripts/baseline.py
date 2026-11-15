from dataclasses import asdict

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

import pybullet as p
import numpy as np
import json

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
    return TrainingConfig(
        simulation_data=simulation_data,
        env_type=EnvironmentType.FLAT,
        env_data=env_data,
        robot_spec=robot_data,
        episode_params=episode_params,
        fitness_params=fit_params,
        seed=42
    )

def sine_gait_action(t: float, f: float):
    a_thigh = 0.4
    a_calf = 0.3

    FR_RL_thigh = a_thigh * np.sin(2 * np.pi * f * t)
    FR_RL_calf = a_calf * np.sin(2 * np.pi * f * t + (np.pi / 2))

    FL_RR_thigh = a_thigh * np.sin(2 * np.pi * f * t + np.pi)
    FL_RR_calf = a_calf * np.sin(2 * np.pi * f * t + (np.pi / 2) + np.pi)

    return [0.0, FR_RL_thigh, FR_RL_calf, 0.0, FL_RR_thigh, FL_RR_calf, 0.0, FL_RR_thigh, FL_RR_calf, 0.0, FR_RL_thigh,
            FR_RL_calf]

def run_baseline():
    training_config = default_training_config()

    sd = training_config.simulation_data
    et = training_config.env_type
    ed = training_config.env_data
    rs = training_config.robot_spec
    ep = training_config.episode_params
    fp = training_config.fitness_params
    seed = training_config.seed

    client = p.connect(p.DIRECT)
    prepare_simulation(sd, client)
    env = create_environment(et, ed, client, seed)
    robot = Robot(rs, env.get_starting_pos(), client)
    policy = lambda obs,t : sine_gait_action(t, ep.frequency)
    results = run_episode(robot, env, policy, sd.dt, ep)
    fitness = fitness_v1(results, fp)
    v = results.progress / results.duration

    data = {
        "metrics" : asdict(results),
        "fitness" : fitness,
        "velocity" : v,
    }

    print(data)

run_baseline()