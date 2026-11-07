from neat_walking_robot.environments.base import BaseEnvironment, BaseEnvironmentData
from dataclasses import dataclass
import pybullet as p
import numpy as np


@dataclass(frozen=True, kw_only=True)
class SlopeEnvironmentData(BaseEnvironmentData):
    incline: float

    def __post_init__(self):
        super().__post_init__()
        if self.incline < 0 or self.incline > 30:
            raise ValueError(f"Incline ({self.incline}) must be between 0 and 30 degree")

        max_ground_height = np.tan(np.deg2rad(self.incline)) * (self.track_length - (
                    self.start_distance + self.end_distance + self.buffer_space * 2))
        if self.wall_height <= max_ground_height:
            raise ValueError(
                f"For SlopeEnvironment wall height ({self.wall_height}) must be larger then max ground height ({max_ground_height}) indicated by incline ({self.incline}) ")


class SlopeEnvironment(BaseEnvironment[SlopeEnvironmentData]):
    def __init__(self, params: SlopeEnvironmentData, client: int, seed: int):
        super().__init__(params, client, seed)
        self.alpha = np.deg2rad(self.params.incline)
        # slope
        self.slope_bottom = self.params.track_length - (
                    self.params.start_distance + self.params.end_distance + self.params.buffer_space * 2)
        self.slope_length = self.slope_bottom / np.cos(self.alpha)
        self.slope_z = (self.slope_length / 2 + np.tan(self.alpha) * self.params.ground_thickness / 2) * np.sin(
            self.alpha) - self.params.ground_thickness / 2 / np.cos(self.alpha)
        self.slope_x = -self.params.track_length / 2 + self.params.start_distance + self.params.buffer_space + (
                    self.slope_length / 2 + np.tan(self.alpha) * self.params.ground_thickness / 2) * np.cos(self.alpha)
        # start platform
        self.start_platform_length = self.params.start_distance + self.params.buffer_space
        self.start_platform_x = (-self.params.track_length + self.start_platform_length) / 2
        # end platform
        self.end_platform_length = self.params.end_distance + self.params.buffer_space
        self.end_platform_x = (self.params.track_length - self.end_platform_length) / 2
        self.end_platform_z = np.tan(self.alpha) * self.slope_bottom - self.params.ground_thickness / 2

        self.ground = self.build_ground()

    def build_ground(self):
        ground_collision = p.createCollisionShapeArray(
            shapeTypes=[p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX],
            halfExtents=[
                [self.start_platform_length / 2, self.params.track_width / 2, self.params.ground_thickness / 2],
                [self.slope_length / 2, self.params.track_width / 2, self.params.ground_thickness / 2],
                [self.end_platform_length / 2, self.params.track_width / 2, self.params.ground_thickness / 2]
            ],
            collisionFramePositions=[
                [self.start_platform_x, 0, -self.params.ground_thickness / 2],
                [self.slope_x, 0, self.slope_z],
                [self.end_platform_x, 0, self.end_platform_z]
            ],
            collisionFrameOrientations=[
                [0.0, 0.0, 0.0, 1.0],
                p.getQuaternionFromEuler([0, -self.alpha, 0]),
                [0.0, 0.0, 0.0, 1.0]
            ],
            physicsClientId=self.client
        )
        ground = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=ground_collision,
            basePosition=[0,0,0],
            physicsClientId=self.client
        )
        p.changeVisualShape(ground, -1, rgbaColor=[0.6, 0.7, 0.6, 1], physicsClientId=self.client)
        p.changeDynamics(ground, -1, lateralFriction=self.params.friction, physicsClientId=self.client)

        return ground

    def get_ground_level(self, x: float, y: float):
        ground_level = -1.0
        if x < -self.params.track_length / 2 or x > self.params.track_length / 2:
            pass
        elif y < -self.params.track_width / 2 or y > self.params.track_width / 2:
            pass
        elif -self.params.track_length / 2 <= x <= (-self.params.track_length / 2 + self.start_platform_length):
            ground_level = 0.0
        elif (-self.params.track_length / 2 + self.start_platform_length) < x < (
                self.params.track_length / 2 - self.end_platform_length):
            distance = x - (-self.params.track_length / 2 + self.start_platform_length)
            ground_level = np.tan(self.alpha) * distance
        elif (self.params.track_length / 2 - self.end_platform_length) <= x <= self.params.track_length / 2:
            ground_level = np.tan(self.alpha) * self.slope_bottom

        return ground_level
