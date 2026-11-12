import pybullet as p
import time
import numpy as np
import matplotlib.pyplot as plt
from neat_walking_robot.simulation import SimulationData, connect_and_configure_simulation

sim_data = SimulationData()
client = connect_and_configure_simulation(sim_data, client_type=p.DIRECT)

a1 = p.loadURDF("a1/a1.urdf", basePosition=[0,0,0.5], flags=p.URDF_USE_INERTIA_FROM_FILE, useFixedBase=True, physicsClientId=client)

def reset_joints(robot_id: int, position_gain: float, client_id: int):
    mode = p.POSITION_CONTROL
    hip_joints = [1, 6, 11, 16]
    upper_joints = [3, 8, 13, 18]
    lower_joints = [4, 9, 14, 19]

    for hip in hip_joints:
        p.resetJointState(robot_id, hip, targetValue=0.0, physicsClientId=client_id)

    for upper in upper_joints:
        p.resetJointState(robot_id, upper, targetValue=0.8, physicsClientId=client_id)

    for lower in lower_joints:
        p.resetJointState(robot_id, lower, targetValue=-1.5, physicsClientId=client_id)

pos_gains = [0.02, 0.1, 0.5]
for pos_gain in pos_gains:
    reset_joints(a1, pos_gain, client_id=client)
    for i in range(120):
        p.stepSimulation(physicsClientId=client)

    p.setJointMotorControl2(
        bodyUniqueId=a1,
        jointIndex=3,
        positionGain=pos_gain,
        force=55.0,
        controlMode=p.POSITION_CONTROL,
        targetPosition =(0.8 + 0.5),
        physicsClientId=client
    )
    applied_torque = []
    positions = []
    for i in range(240):
        p.stepSimulation(physicsClientId=client)
        state = p.getJointState(a1, 3, physicsClientId=client)
        applied_torque.append(state[3])
        positions.append(state[0])
    x = np.arange(0, 240, 1)

    fig, ax = plt.subplots()
    ax.plot(x, positions)
    ax.set(xlabel='step', ylabel='position (rad)', title=f"position_gain={pos_gain}")
    fig.savefig(f"position_{pos_gain}.png")

    fig, ax = plt.subplots()
    ax.plot(x, applied_torque)
    ax.set(xlabel='step', ylabel='applied torque (Nm)', title=f"position_gain={pos_gain}")
    fig.savefig(f"torque_{pos_gain}.png")

# while p.isConnected(physicsClientId=client):
#     p.stepSimulation(physicsClientId=client)
#     time.sleep(1/240)

p.disconnect(client)


