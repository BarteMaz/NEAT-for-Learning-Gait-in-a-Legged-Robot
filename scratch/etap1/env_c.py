from neat_walking_robot.environments.slope import SlopeEnvironment, SlopeEnvironmentData

import pybullet as p
import pybullet_data
import time
# World's parameters:
client_type = p.GUI
dt = 1/240
g = 9.81

env_data = SlopeEnvironmentData(
    track_length= 6,
    track_width= 3.0,
    wall_height= 3.0,
    wall_thickness= 0.5,
    ground_thickness= 0.20,
    end_distance= 1.0,
    start_distance= 1.0,
    friction= 1.0,
    lowest_ground_point= 0.0,
    buffer_space=0.5,
    incline=20
)

client = p.connect(client_type)

p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
p.setGravity(0, 0, -g, physicsClientId=client)
p.setTimeStep(dt, physicsClientId=client)



env1 = SlopeEnvironment(env_data, client, 0)
p.changeVisualShape(env1.wall_left, -1, rgbaColor=[0.0, 1.0, 0.0, 0.1], physicsClientId=client)
p.changeVisualShape(env1.wall_right, -1, rgbaColor=[0.0, 1.0, 0.0, 0.1], physicsClientId=client)
p.changeVisualShape(env1.wall_front, -1, rgbaColor=[0.0, 1.0, 0.0, 0.1], physicsClientId=client)
p.changeVisualShape(env1.wall_back, -1, rgbaColor=[0.0, 1.0, 0.0, 0.1], physicsClientId=client)

if client_type == p.DIRECT:
    for _ in range(100000):
        p.stepSimulation(physicsClientId=client)
    p.disconnect(client)
else:
    while p.isConnected(physicsClientId=client):
        p.stepSimulation(physicsClientId=client)
        time.sleep(dt)



