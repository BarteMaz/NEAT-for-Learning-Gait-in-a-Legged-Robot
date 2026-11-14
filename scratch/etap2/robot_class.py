from neat_walking_robot.environments import FlatEnvironment, BaseEnvironmentData
from neat_walking_robot.simulation import SimulationData, connect_and_configure_simulation
from neat_walking_robot.robot.robot import Robot, A1_ROBOT_SPEC, scale_to_neat_input

import pybullet as p
import time
import numpy as np

sim_data = SimulationData()
client = connect_and_configure_simulation(sim_data, client_type=p.DIRECT)

env_data = BaseEnvironmentData(
    track_length= 8.0,
    track_width= 3.0,
    wall_height= 3.0,
    wall_thickness= 0.25,
    ground_thickness= 0.20,
    end_distance= 1.0,
    start_distance= 1.0,
    friction= 1.0,
    lowest_ground_point= 0.0,
)

flat_env = FlatEnvironment(env_data, client, 0)
# p.changeVisualShape(flat_env.wall_left, -1, rgbaColor=[0.0, 0.0, 1.0, 0.2], physicsClientId=client)
# p.changeVisualShape(flat_env.wall_right, -1, rgbaColor=[0.0, 0.0, 1.0, 0.2], physicsClientId=client)
# p.changeVisualShape(flat_env.wall_front, -1, rgbaColor=[0.0, 0.0, 1.0, 0.2], physicsClientId=client)
# p.changeVisualShape(flat_env.wall_back, -1, rgbaColor=[0.0, 0.0, 1.0, 0.2], physicsClientId=client)

a1_robot = Robot(A1_ROBOT_SPEC, flat_env.get_starting_pos(), client)

value = [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5]

# qKey = ord('q')
# rKey = ord('r')
#
# while p.isConnected(physicsClientId=client):
#     p.stepSimulation(physicsClientId=client)
#     keys = p.getKeyboardEvents()
#     if qKey in keys and keys[qKey] & p.KEY_WAS_TRIGGERED:
#         for i in range(180):
#             a1_robot.apply_action(value)
#             p.stepSimulation(physicsClientId=client)
#             b_pos, b_or = p.getBasePositionAndOrientation(a1_robot.robot_id)
#             X, Y, Z = p.getEulerFromQuaternion(b_or)
#             print(
#                 f"Base position: {b_pos[0]}, {b_pos[1]}, {b_pos[2]}; Base orientation: {X}, {Y}, {Z}")
#             time.sleep(1 / 240)
#             value[2] += -0.01
#             value[8] += -0.01
#         joints_state2 = a1_robot.get_joints_state()
#     elif rKey in keys and keys[rKey] & p.KEY_WAS_TRIGGERED:
#         a1_robot.reset()
#     time.sleep(1/240)

for i in range(120):
    p.stepSimulation(physicsClientId=client)

a1_robot.apply_action(value)
for i in range(360):
    p.stepSimulation(physicsClientId=client)

for i in a1_robot.get_observation():
    print(f"{i:.3f}", end=" ")
