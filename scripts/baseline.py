import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pybullet as p
from neat_walking_robot.environments import create_environment
from neat_walking_robot.robot import Robot
from neat_walking_robot.simulation import prepare_simulation
from neat_walking_robot.training import (
    default_training_config,
    FITNESS,
    FitnessFunction,
    run_episode,
    to_json,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "a1.ini"
RESULTS_PATH = PROJECT_ROOT / "results"

def sine_gait_action(t: float, f: float):
    a_thigh = 0.4
    a_calf = 0.3

    fr_rl_thigh = a_thigh * np.sin(2 * np.pi * f * t)
    fr_rl_calf = a_calf * np.sin(2 * np.pi * f * t + (np.pi / 2))

    fl_rr_thigh = a_thigh * np.sin(2 * np.pi * f * t + np.pi)
    fl_rr_calf = a_calf * np.sin(2 * np.pi * f * t + (np.pi / 2) + np.pi)

    return [0.0, fr_rl_thigh, fr_rl_calf, 0.0, fl_rr_thigh, fl_rr_calf, 0.0, fl_rr_thigh, fl_rr_calf, 0.0, fr_rl_thigh,
            fr_rl_calf]

def run_baseline(args):
    training_config = default_training_config(CONFIG_PATH, 42)

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

    if args.fitness == "v1":
        fitness = FITNESS[FitnessFunction.V1](results, fp)
    else:
        fitness = FITNESS[FitnessFunction.V2](results, fp)

    v = results.progress / results.duration

    data = {
        "metrics" : asdict(results),
        "fitness" : fitness,
        "velocity" : v,
    }
    with open(RESULTS_PATH / "baseline_data.json", 'w', encoding="utf-8") as f:
        json.dump(data, f, default=to_json, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fitness", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()

    run_baseline(args)