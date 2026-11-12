from neat_walking_robot.environments import FlatEnvironment, BaseEnvironmentData
from neat_walking_robot.simulation import SimulationData, connect_and_configure_simulation
from neat_walking_robot.robot.robot import Robot, A1_ROBOT_SPEC

import pybullet as p
import time

sim_data = SimulationData()
client = connect_and_configure_simulation(sim_data, client_type=p.GUI)

env_data = BaseEnvironmentData(
    track_length= 20.0,
    track_width= 8.0,
    wall_height= 3.0,
    wall_thickness= 0.5,
    ground_thickness= 0.20,
    end_distance= 1.0,
    start_distance= 1.0,
    friction= 1.0,
    lowest_ground_point= -2.0,
)

flat_env = FlatEnvironment(env_data, client, 0)
p.changeVisualShape(flat_env.wall_left, -1, rgbaColor=[0.0, 0.0, 1.0, 0.2], physicsClientId=client)
p.changeVisualShape(flat_env.wall_right, -1, rgbaColor=[0.0, 0.0, 1.0, 0.2], physicsClientId=client)
p.changeVisualShape(flat_env.wall_front, -1, rgbaColor=[0.0, 0.0, 1.0, 0.2], physicsClientId=client)
p.changeVisualShape(flat_env.wall_back, -1, rgbaColor=[0.0, 0.0, 1.0, 0.2], physicsClientId=client)

a1_robot = Robot(A1_ROBOT_SPEC, flat_env, client)


while p.isConnected(physicsClientId=client):
    p.stepSimulation(physicsClientId=client)
    time.sleep(1/240)