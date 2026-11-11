import pybullet as p
import time
from neat_walking_robot.simulation import SimulationData, connect_and_configure_simulation


sim_data = SimulationData()
client = connect_and_configure_simulation(sim_data, client_type=p.GUI)

a1 = p.loadURDF("a1/a1.urdf", basePosition=[0,0,0.5], flags=p.URDF_USE_INERTIA_FROM_FILE, useFixedBase=True, physicsClientId=client)

numJoints = p.getNumJoints(a1, physicsClientId=client)
mode = p.POSITION_CONTROL

hip_joints = [1, 6, 11, 16]
hip_effort = 22.0

upper_joints = [3, 8, 13, 18]
upper_effort = 55.0

lower_joints = [4, 9, 14, 19]
lower_effort = 55.0

for hip in hip_joints:
    p.resetJointState(a1, hip, targetValue=0.0, physicsClientId=client)
    p.setJointMotorControl2(a1, hip, controlMode=mode, force=hip_effort, physicsClientId=client)

for upper in upper_joints:
    p.resetJointState(a1, upper, targetValue=0.8, physicsClientId=client)
    p.setJointMotorControl2(a1, upper, controlMode=mode, force=upper_effort, physicsClientId=client)

for lower in lower_joints:
    p.resetJointState(a1, lower, targetValue=-1.5, physicsClientId=client)
    p.setJointMotorControl2(a1, lower, controlMode=mode, force=lower_effort, physicsClientId=client)

while p.isConnected(physicsClientId=client):
    p.stepSimulation(physicsClientId=client)
    time.sleep(1/240)


# p.disconnect(client)


