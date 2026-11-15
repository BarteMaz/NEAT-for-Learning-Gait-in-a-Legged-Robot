from dataclasses import dataclass
from enum import Enum

import numpy as np
import pybullet as p

from neat_walking_robot.environments import BaseEnvironment
from neat_walking_robot.robot import Robot


class EndReason(Enum):
    WALL = "wall contact detected"
    FALL = "fall detected"
    GOAL = "goal reached"
    TIME_LIMIT = "time limit reached"


@dataclass(frozen=True, kw_only=True)
class EpisodeParams:
    episode_time: float
    frequency: float
    k: int
    def __post_init__(self):
        if self.episode_time <= 0:
            raise ValueError("Episode time must be greater than 0")
        if self.frequency <= 0:
            raise ValueError("Frequency must be greater than 0")
        if self.k <= 0:
            raise ValueError("k must be greater than 0")


@dataclass(frozen=True, kw_only=True)
class EpisodeMetrics:
    progress: float
    goal_distance: float
    end_reason: EndReason
    episode_length: float
    duration: float
    max_y: float
    max_yaw: float


def run_episode(robot: Robot, env: BaseEnvironment, policy, dt: float, params: EpisodeParams) -> EpisodeMetrics:
    # prep variables
    action_number = round(params.episode_time / (params.k * dt))
    client_id = robot.client_id
    x_goal = env.params.track_length / 2 - env.params.end_distance

    progress = 0.0
    max_y = 0.0
    max_yaw = 0.0
    episode_duration = 0.0
    reason = None

    robot.reset()
    x_0 = robot.start_base_position[0]

    for n in range(action_number):
        t = n * params.k * dt
        observation = robot.get_observation()
        action = policy(observation, t)
        robot.apply_action(action)

        for _ in range(params.k):
            p.stepSimulation(physicsClientId=client_id)

        # check end conditions
        info = p.getBasePositionAndOrientation(robot.robot_id, physicsClientId=client_id)
        current_x, current_y = info[0][0:2]
        roll, pitch, yaw = p.getEulerFromQuaternion(info[1])

        progress = current_x - x_0

        max_y = max(max_y, np.abs(current_y))

        max_yaw = max(max_yaw, np.abs(yaw))

        if len(p.getContactPoints(bodyA=env.wall_right, bodyB=robot.robot_id, physicsClientId=client_id)):
            reason = EndReason.WALL
            episode_duration = (n + 1) * params.k * dt
            break
        if len(p.getContactPoints(bodyA=env.wall_left, bodyB=robot.robot_id, physicsClientId=client_id)) > 0:
            reason = EndReason.WALL
            episode_duration = (n + 1) * params.k * dt
            break
        if len(p.getContactPoints(bodyA=env.wall_back, bodyB=robot.robot_id, physicsClientId=client_id)):
            reason = EndReason.WALL
            episode_duration = (n + 1) * params.k * dt
            break



        if np.abs(roll) > robot.fall_threshold or np.abs(pitch) > robot.fall_threshold:
            reason = EndReason.FALL
            episode_duration = (n + 1) * params.k * dt
            break
        if current_x >= x_goal:
            reason = EndReason.GOAL
            episode_duration = (n + 1) * params.k * dt
            break

    if reason is None:
        reason = EndReason.TIME_LIMIT
        episode_duration = params.episode_time

    measurements = EpisodeMetrics(
        progress=progress,
        goal_distance=env.params.track_length - env.params.end_distance - env.params.start_distance,
        end_reason=reason,
        episode_length=params.episode_time,
        duration=episode_duration,
        max_y=max_y,
        max_yaw=max_yaw
    )

    return measurements
