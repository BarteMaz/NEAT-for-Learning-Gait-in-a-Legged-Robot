from neat_walking_robot.environments.rough import RoughEnvironment, RoughEnvironmentData

import pybullet as p
import pybullet_data
import time
# World's parameters:
client_type = p.GUI
dt = 1/240
g = 9.81

env_data = RoughEnvironmentData(
    track_length= 10,
    track_width= 3.0,
    wall_height= 2,
    wall_thickness= 0.5,
    ground_thickness= 0.20,
    end_distance= 1.0,
    start_distance= 1.0,
    friction= 1.0,
    lowest_ground_point= -1,
    buffer_space=0.5,
    cell_size= 0.05,
    amplitude=0.3,
    transition_distance=2.0,
    n_waves=6,
    wavelength_range=(1.0, 3.0),
)

client = p.connect(client_type)

p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
p.setGravity(0, 0, -g, physicsClientId=client)
p.setTimeStep(dt, physicsClientId=client)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=client)

env1 = RoughEnvironment(env_data, client, 0)
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



