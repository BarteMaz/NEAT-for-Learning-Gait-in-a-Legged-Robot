import pybullet as p
import time
from neat_walking_robot.simulation import SimulationData, connect_and_configure_simulation
from neat_walking_robot.robot.joints import (
    joint_table
)


sim_data = SimulationData()
client = connect_and_configure_simulation(sim_data, client_type=p.DIRECT)
a1 = p.loadURDF("a1/a1.urdf", flags=p.URDF_USE_INERTIA_FROM_FILE, basePosition=[0,0,0.3119956548683467], physicsClientId=client)

# while p.isConnected(physicsClientId=client):
#     p.stepSimulation(physicsClientId=client)
#     time.sleep(1/240)


p.disconnect(client)


