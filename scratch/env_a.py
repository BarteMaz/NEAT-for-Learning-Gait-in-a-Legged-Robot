from neat_walking_robot.environments.base import BaseEnvironmentData
from neat_walking_robot.environments.flat import FlatEnvironment

import pybullet as p
import pybullet_data
# World's parameters:
client_type = p.DIRECT
dt = 1/240
g = 9.81

env_data = BaseEnvironmentData(
    track_length= 20.0,
    track_width= 8.0,
    wall_height= 3.0,
    wall_thickness= 0.5,
    ground_height= 0.5,
    friction=1.0,
    goal_pos= 9.0
)

# robot's bounding box size
robot_length = 1
robot_width = 1
robot_height = 1
# ------------------

client = p.connect(client_type)
client1 = p.connect(client_type)

p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
p.setGravity(0, 0, -g, physicsClientId=client)
p.setTimeStep(dt, physicsClientId=client)

p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client1)
p.setGravity(0, 0, -g, physicsClientId=client1)
p.setTimeStep(dt, physicsClientId=client1)

env1 = FlatEnvironment(env_data, client1, 0)


print(p.getNumBodies(client))
print(p.getNumBodies(client1))

# if client_type == p.DIRECT:
#     for _ in range(100000):
#         p.stepSimulation(physicsClientId=client)
#     p.disconnect(client)
# else:
#     while p.isConnected(physicsClientId=client):
#         p.stepSimulation(physicsClientId=client)
#         time.sleep(dt)

p.disconnect(client)
p.disconnect(client1)