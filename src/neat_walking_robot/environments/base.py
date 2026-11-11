from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import pybullet as p


@dataclass(frozen=True, kw_only=True)
class BaseEnvironmentData:
    track_length: float
    track_width: float
    wall_height: float
    wall_thickness: float
    ground_thickness: float
    end_distance: float
    start_distance: float
    friction: float
    lowest_ground_point: float = 0.0
    buffer_space: float = 0.5

    def __post_init__(self):
        # Check for negatives
        if self.track_length <= 0:
            raise ValueError("Track length must be greater than 0")
        if self.track_width <= 0:
            raise ValueError("Track width must be greater than 0")
        if self.wall_height <= 0:
            raise ValueError("Wall height must be greater than 0")
        if self.wall_thickness <= 0:
            raise ValueError("Wall thickness must be greater than 0")
        if self.friction <= 0:
            raise ValueError("Friction must be greater than 0")
        if self.ground_thickness <= 0:
            raise ValueError("Ground height must be greater than 0")
        # Check for contradictions
        if self.end_distance + self.start_distance >= self.track_length:
            raise ValueError("The sum of start and end distance must be less than the track length")
        if self.start_distance < 0.5 or self.end_distance < 0.5:
            raise ValueError("The start and end distance must be more or equal to 0.5")


ParamsT = TypeVar("ParamsT", bound=BaseEnvironmentData)


class BaseEnvironment(ABC, Generic[ParamsT]):
    def __init__(self, params: ParamsT, client: int, seed: int):
        self.seed = seed
        self.params = params
        self.client = client
        self.wall_left, self.wall_right = self.build_side_walls()
        self.wall_front, self.wall_back = self.build_front_and_back_wall()
        self.line = None

    def build_static_box(self, dimensions, position, color):
        collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=dimensions,
            physicsClientId=self.client
        )
        visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=dimensions,
            rgbaColor=color,
            physicsClientId=self.client
        )
        object_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position,
            physicsClientId=self.client
        )
        return object_id

    def build_side_walls(self):
        wall_left = self.build_static_box(
            [self.params.track_length / 2, self.params.wall_thickness / 2,
             (self.params.wall_height - self.params.lowest_ground_point) / 2],
            [0, -(self.params.track_width + self.params.wall_thickness) / 2,
             (self.params.wall_height + self.params.lowest_ground_point) / 2],
            [0, 0, 1, 1]
        )
        wall_right = self.build_static_box(
            [self.params.track_length / 2, self.params.wall_thickness / 2,
             (self.params.wall_height - self.params.lowest_ground_point) / 2],
            [0, (self.params.track_width + self.params.wall_thickness) / 2,
             (self.params.wall_height + self.params.lowest_ground_point) / 2],
            [0, 0, 1, 1]
        )
        return wall_left, wall_right

    def build_front_and_back_wall(self):
        wall_front = self.build_static_box(
            [self.params.wall_thickness / 2, self.params.track_width / 2,
             (self.params.wall_height - self.params.lowest_ground_point) / 2],
            [(self.params.track_length + self.params.wall_thickness) / 2, 0,
             (self.params.wall_height + self.params.lowest_ground_point) / 2],
            [0, 0, 1, 1]
        )
        wall_back = self.build_static_box(
            [self.params.wall_thickness / 2, self.params.track_width / 2,
             (self.params.wall_height - self.params.lowest_ground_point) / 2],
            [-(self.params.track_length + self.params.wall_thickness) / 2, 0,
             (self.params.wall_height + self.params.lowest_ground_point) / 2],
            [0, 0, 1, 1]
        )

        return wall_front, wall_back

    def show_goal_line(self):
        self.line = p.addUserDebugLine(
            lineFromXYZ=[self.params.track_length / 2 - self.params.end_distance, -self.params.track_width / 2,
                         self.get_ground_level(self.params.track_length / 2 - self.params.end_distance,
                                               -self.params.track_width / 2) + 1],
            lineToXYZ=[self.params.track_length / 2 - self.params.end_distance, self.params.track_width / 2,
                       self.get_ground_level(self.params.track_length / 2 - self.params.end_distance,
                                             -self.params.track_width / 2) + 1],
            lineColorRGB=[0, 1, 0],
            physicsClientId=self.client
        )

    def get_starting_pos(self):
        return [-(self.params.track_length / 2 - self.params.start_distance), 0.0,
                self.get_ground_level(-(self.params.track_length / 2 - self.params.start_distance), 0.0)]

    @abstractmethod
    def build_ground(self) -> int: ...

    @abstractmethod
    def get_ground_level(self, x: float, y: float) -> float: ...
