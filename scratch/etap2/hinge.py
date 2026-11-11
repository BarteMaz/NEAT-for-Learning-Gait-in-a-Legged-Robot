import pybullet as p
import pybullet_data
import numpy as np
import time
from neat_walking_robot.simulation import *

sim_data = SimulationData()
client = connect_and_configure_simulation(sim_data, client_type=p.GUI)


hinge = p.loadURDF("hinge.urdf",basePosition=[0, 0, 0] ,useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=client)
p.setJointMotorControl2(hinge, 0, p.VELOCITY_CONTROL, force=0,physicsClientId=client)
p.resetJointState(hinge, 0, targetValue=np.pi/4, physicsClientId=client)

p.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=0,
    cameraPitch=0,
    cameraTargetPosition=[0, 0, 0.75],
    physicsClientId=client,
)

def draw_lines(pos, ori, L, ids=None):
    pos = np.array(pos)
    ori = np.array(ori).reshape(3,3)
    if ids is None:
        line_x = p.addUserDebugLine(pos, pos + L * ori[:,0], [1, 0, 0], physicsClientId=client)
        line_y = p.addUserDebugLine(pos, pos + L * ori[:,1], [0, 1, 0], physicsClientId=client)
        line_z = p.addUserDebugLine(pos, pos + L * ori[:,2], [0, 0, 1], physicsClientId=client)
    else:
        line_x = p.addUserDebugLine(pos, pos + L * ori[:, 0], [1, 0, 0], replaceItemUniqueId=ids[0], physicsClientId=client)
        line_y = p.addUserDebugLine(pos, pos + L * ori[:, 1], [0, 1, 0], replaceItemUniqueId=ids[1], physicsClientId=client)
        line_z = p.addUserDebugLine(pos, pos + L * ori[:, 2], [0, 0, 1], replaceItemUniqueId=ids[2], physicsClientId=client)
    return line_x, line_y, line_z


state = p.getLinkState(hinge,0,physicsClientId=client)
pos1, or1, pos2, or2 = state[0], state[1], state[4], state[5]

L = 2.0
# dla ramki środka masy:
or1 = p.getMatrixFromQuaternion(or1)
mass_ids = draw_lines(pos1, or1, L)

# dla ramki linku:
or2 = p.getMatrixFromQuaternion(or2)
link_ids = draw_lines(pos2, or2, L)


count = 0
while p.isConnected(physicsClientId=client):
    p.stepSimulation(physicsClientId=client)

    if count == 3:
        state = p.getLinkState(hinge, 0, physicsClientId=client)
        # dla ramki środka masy:
        mass_ids = draw_lines(state[0], p.getMatrixFromQuaternion(state[1]), L, mass_ids)
        # dla ramki linku:
        link_ids = draw_lines(state[4], p.getMatrixFromQuaternion(state[5]), L, link_ids)

    count = (count + 1) % 4
    time.sleep(1/240)


# p.disconnect(client)


