from neat_walking_robot.environments import create_environment
from neat_walking_robot.robot import Robot
from neat_walking_robot.simulation import prepare_simulation
from neat_walking_robot.training import (
    default_training_config,
    run_episode,
    fitness_v1,
)
from dataclasses import asdict
from enum import Enum
from pathlib import Path
import pybullet as p
import numpy as np
import json

def to_json(o):
    if isinstance(o, Enum):
        return o.name
    return str(o)

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
    training_config = default_training_config(Path(""), 42)

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
    with open("baseline_data.json", 'w', encoding="utf-8") as f:
        json.dump(data, f, default=to_json, ensure_ascii=False, indent=2)

run_baseline()