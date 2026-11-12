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

wallf_min, wallf_max = p.getAABB(env1.wall_front)

print(f"wallf_min_z: {wallf_min[2]}, wallf_max_z: {wallf_max[2]}")


# if client_type == p.DIRECT:
#     for _ in range(100000):
#         p.stepSimulation(physicsClientId=client)
#     p.disconnect(client)
# else:
#     while p.isConnected(physicsClientId=client):
#         p.stepSimulation(physicsClientId=client)
#         time.sleep(dt)

p.disconnect(client)
