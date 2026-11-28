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
    k: int
    def __post_init__(self):
        if self.episode_time <= 0:
            raise ValueError("Episode time must be greater than 0")
        if self.k <= 0:
            raise ValueError("k must be greater than 0")


@dataclass(frozen=True, kw_only=True)
class EpisodeMetrics:
    max_distance: float
    toes_path_length: float
    end_reason: EndReason
    episode_length: float
    duration: float



def run_episode(robot: Robot, env: BaseEnvironment, policy, dt: float, params: EpisodeParams) -> EpisodeMetrics:
    # prep variables
    action_number = round(params.episode_time / (params.k * dt))
    client_id = robot.client_id
    flag_wall_contact = False

    max_distance = 0.0
    toes_path_length = 0.0
    episode_duration = 0.0
    reason = None


    robot.reset()
    start_x, start_y = robot.start_base_position[0:2]
    toes_position = robot.get_toes_world_position()
    walls_id = [env.wall_back, env.wall_left, env.wall_right, env.wall_front]

    for n in range(action_number):
        observation = robot.get_observation()
        action = policy(observation)
        robot.apply_action(action)

        # step simulation
        for _ in range(params.k):
            p.stepSimulation(physicsClientId=client_id)

        new_toes_position = robot.get_toes_world_position()
        for link_name, pos in new_toes_position.items():
            toes_path_length += np.linalg.norm(np.array(pos) - np.array(toes_position[link_name]))
        toes_position = new_toes_position

        info = p.getBasePositionAndOrientation(robot.robot_id, physicsClientId=client_id)
        current_x, current_y = info[0][0:2]
        roll, pitch = p.getEulerFromQuaternion(info[1])[0:2]

        # calculate distance from starting point
        current_distance = np.sqrt((current_x - start_x) ** 2 + (current_y - start_y) ** 2)
        if max_distance < current_distance:
            max_distance = current_distance

        # check end conditions
        for wall in walls_id:
            if len(p.getContactPoints(bodyA=wall, bodyB=robot.robot_id, physicsClientId=client_id)):
                reason = EndReason.WALL
                episode_duration = (n + 1) * params.k * dt
                flag_wall_contact = True
                break

        if flag_wall_contact:
            break

        if np.abs(roll) > robot.fall_threshold or np.abs(pitch) > robot.fall_threshold:
            reason = EndReason.FALL
            episode_duration = (n + 1) * params.k * dt
            break


    if reason is None:
        reason = EndReason.TIME_LIMIT
        episode_duration = params.episode_time

    measurements = EpisodeMetrics(
        max_distance=max_distance,
        toes_path_length=float(toes_path_length),
        end_reason=reason,
        episode_length=params.episode_time,
        duration=episode_duration,
    )

    return measurements
