import argparse
import time
from pathlib import Path
from dataclasses import replace

import neat
import numpy as np
import pandas as pd
import pybullet as p
import time

from neat_walking_robot.environments import create_environment
from neat_walking_robot.robot import Robot, get_joint_info
from neat_walking_robot.simulation import prepare_simulation
from neat_walking_robot.training import load_run_config, make_policy, run_episode

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent

HIGH_FREQUENCY = 10.0      # Hz, threshold for the "high-frequency energy share"

# Observation layout (see Robot.get_observation)
VELOCITY_SLICE = slice(12, 24)
CONTACT_SLICE = slice(29, 33)


# ---------------------------------------------------------------- policies
def sine_gait_action(t: float, f: float) -> list[float]:
    a_thigh = 0.4
    a_calf = 0.3

    fr_rl_thigh = a_thigh * np.sin(2 * np.pi * f * t)
    fr_rl_calf = a_calf * np.sin(2 * np.pi * f * t + np.pi / 2)
    fl_rr_thigh = a_thigh * np.sin(2 * np.pi * f * t + np.pi)
    fl_rr_calf = a_calf * np.sin(2 * np.pi * f * t + np.pi / 2 + np.pi)

    return [0.0, fr_rl_thigh, fr_rl_calf,
            0.0, fl_rr_thigh, fl_rr_calf,
            0.0, fl_rr_thigh, fl_rr_calf,
            0.0, fr_rl_thigh, fr_rl_calf]


def recording(policy, log: list):
    """Wrap a policy(obs, t) so that every call appends (obs, action) to log."""
    def wrapper(obs, t):
        action = policy(obs, t)
        log.append((np.array(obs, copy=True), np.asarray(action, dtype=float)))
        return action
    return wrapper


# ---------------------------------------------------------------- metrics
def joint_metrics(action: np.ndarray, velocity: np.ndarray, max_velocity: float, fs: float) -> dict:
    """Metrics for one joint. action and velocity are 1-D arrays sampled at fs [Hz]."""
    n = len(action)

    centered = action - action.mean()
    spectrum = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    power = spectrum ** 2

    total = power[freqs > 0].sum()
    if total < 1e-12:
        # constant signal (e.g. baseline hips): no oscillation, frequency undefined
        peak, peak_amplitude, high_share = np.nan, 0.0, np.nan
    else:
        k = int(np.argmax(spectrum))
        peak = freqs[k]
        is_nyquist = (n % 2 == 0) and (k == len(freqs) - 1)
        peak_amplitude = spectrum[k] / n if is_nyquist else 2 * spectrum[k] / n
        high_share = power[freqs > HIGH_FREQUENCY].sum() / total

    return {
        "max_abs_velocity": np.abs(velocity).max(),
        "velocity_limit": max_velocity,
        "share_over_limit": np.mean(np.abs(velocity) > max_velocity),
        "mean_abs_action_change": np.abs(np.diff(action)).mean(),
        "peak_frequency_hz": peak,
        "peak_amplitude": peak_amplitude,
        "high_freq_energy_share": high_share,
    }


def contact_changes_per_second(contacts: np.ndarray, duration: float) -> np.ndarray:
    """contacts: (n, 4) array of 0/1. Returns number of touch-downs + lift-offs per second for each toe."""
    changes = np.abs(np.diff(contacts, axis=0)).sum(axis=0)
    return changes / duration


# ---------------------------------------------------------------- main
def main(arguments):
    run_dir = PROJECT_ROOT / "results" / arguments.run_name
    training_config, winner, neat_config = load_run_config(run_dir)

    if arguments.fine_dt:
        simulation_data = replace(training_config.simulation_data, dt=1/480)
        episode_params =  replace(training_config.episode_params, k=8)
    else:
        simulation_data = training_config.simulation_data
        episode_params = training_config.episode_params
    dt = simulation_data.dt
    fs = 1.0 / (episode_params.k * dt)

    log: list = []
    if arguments.baseline:
        tag = "baseline"
        policy = recording(lambda obs, t: sine_gait_action(t, 3.0), log)
    else:
        tag = "winner"
        net = neat.nn.FeedForwardNetwork.create(winner, neat_config)
        policy = recording(make_policy(net, episode_params), log)

    if arguments.velocity_limit:
        robot_spec = replace(training_config.robot_spec, apply_velocity_limit=True)
    else:
        robot_spec = training_config.robot_spec

    if arguments.fine_dt:
        tag += "_fine_dt"

    robot_spec = replace(robot_spec, position_gain=0.05)


    client = p.connect(p.DIRECT)
    try:
        prepare_simulation(simulation_data, client)
        env = create_environment(training_config.env_type, training_config.env_data, client, training_config.seed)
        robot = Robot(robot_spec, env.get_starting_pos(), client)

        joint_names = list(robot.moving_joints)
        velocity_limits = [
            get_joint_info(robot.robot_id, robot.moving_joints_id[name], client).velocity for name in joint_names
        ]
        toe_names = list(robot.toes_links_id.keys())
        velocity_scale = robot_spec.joints_velocity_scale

        metrics = run_episode(robot, env, policy, dt, episode_params)
        print(metrics)
    finally:
        p.disconnect(physicsClientId=client)

    observations = np.stack([obs for obs, _ in log])        # (n, 34)
    actions = np.stack([action for _, action in log])       # (n, 12)
    velocities = observations[:, VELOCITY_SLICE] * velocity_scale   # rad/s
    contacts = observations[:, CONTACT_SLICE]               # (n, 4)

    joints_df = pd.DataFrame(
        [joint_metrics(actions[:, i], velocities[:, i], velocity_limits[i], fs) for i in range(len(joint_names))],
        index=pd.Index(joint_names, name="joint"),
    )
    contacts_df = pd.DataFrame(
        {"changes_per_second": contact_changes_per_second(contacts, metrics.duration)},
        index=pd.Index(toe_names, name="toe"),
    )

    joints_path = OUTPUT_DIR / f"joints_{tag}_pg.csv"
    contacts_path = OUTPUT_DIR / f"contacts_{tag}_pg.csv"
    joints_df.to_csv(joints_path)
    contacts_df.to_csv(contacts_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--velocity_limit", action="store_true")
    parser.add_argument("--fine_dt", action="store_true")
    main(parser.parse_args())
