import pybullet as p

from neat_walking_robot.environments.base import BaseEnvironment, BaseEnvironmentData


class FlatEnvironment(BaseEnvironment[BaseEnvironmentData]):
    def __init__(self, params: BaseEnvironmentData, client: int, seed: int):
        super().__init__(params, client, seed)
        self.ground = self.build_ground()


    def build_ground(self):
        plane = self.build_static_box(
            [self.params.track_length / 2, self.params.track_width / 2, self.params.ground_thickness / 2],
            [0, 0, -self.params.ground_thickness / 2],
            [1, 0, 0, 0.5]
        )
        p.changeDynamics(plane, -1, lateralFriction=self.params.friction, physicsClientId=self.client)
        return plane

    def get_ground_level(self, x: float, y: float):
        return 0.0


