from neat_walking_robot.environments.base import BaseEnvironmentData
from neat_walking_robot.environments.flat import FlatEnvironment

import pybullet as p
import pybullet_data
import time
# World's parameters:
client_type = p.GUI
dt = 1/240
g = 9.81

env_data = BaseEnvironmentData(
    track_length= 6.0,
    track_width= 3.0,
    wall_height= 2.0,
    wall_thickness= 0.20,
    ground_thickness= 0.20,
    end_distance= 1.0,
    start_distance= 1.0,
    friction= 1.0,
    lowest_ground_point= -2.0,
)

client = p.connect(client_type)

p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
p.setGravity(0, 0, -g, physicsClientId=client)
p.setTimeStep(dt, physicsClientId=client)


env1 = FlatEnvironment(env_data, client, 0)


if client_type == p.DIRECT:
    for _ in range(100000):
        p.stepSimulation(physicsClientId=client)
    p.disconnect(client)
else:
    while p.isConnected(physicsClientId=client):
        p.stepSimulation(physicsClientId=client)
        time.sleep(dt)

