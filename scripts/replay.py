import argparse
import time
from pathlib import Path

import neat
import pybullet as p
from neat_walking_robot.environments import BaseEnvironment, create_environment
from neat_walking_robot.robot import Robot
from neat_walking_robot.simulation import prepare_simulation
from neat_walking_robot.training import make_policy, run_episode, FITNESS, load_run_config


def realtime(policy, delay):
    def wrapped(obs, t):
        time.sleep(delay)
        return policy(obs, t)
    return wrapped


def change_walls_opacity(env: BaseEnvironment, opacity: float, client_id: int) -> None:
    walls_id = [env.wall_back, env.wall_left, env.wall_right, env.wall_front]
    for wall in walls_id:
        p.changeVisualShape(wall, -1, rgbaColor=[0.0, 1.0, 0.0, opacity], physicsClientId=client_id)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main(arguments):
    # read run's training configuration and best genome
    run_dir = PROJECT_ROOT / "results" / arguments.run_name
    training_config, winner, neat_config = load_run_config(run_dir)

    # create ANN from best genome
    net = neat.nn.FeedForwardNetwork.create(winner, neat_config)

    # create policy that runs realtime
    dt = training_config.simulation_data.dt
    policy = make_policy(net, training_config.episode_params)
    realtime_policy = realtime(policy, dt * training_config.episode_params.k)

    # recreate simulation
    client = p.connect(p.GUI)
    prepare_simulation(training_config.simulation_data, client)
    env = create_environment(training_config.env_type, training_config.env_data, client, training_config.seed)
    robot = Robot(training_config.robot_spec, env.get_starting_pos(), client)

    # visual adjustments
    change_walls_opacity(env, 0.1, client)
    p.resetDebugVisualizerCamera(5.5, 0, -89.9, [0, 0, 0], physicsClientId=client)

    # run episode
    metrics = run_episode(robot, env, realtime_policy, dt, training_config.episode_params)
    p.disconnect(physicsClientId=client)

    # results
    fitness = FITNESS[training_config.fitness_function](metrics, training_config.fitness_params)
    velocity = metrics.progress / metrics.duration

    print(f"================METRICS================")
    print(metrics)
    print(f"=======================================")
    print(f"Winner's fitness: {fitness}\nWinner's' velocity: {velocity}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    args = parser.parse_args()

    main(args)


