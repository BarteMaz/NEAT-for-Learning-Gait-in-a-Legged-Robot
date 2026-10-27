import pybullet as p
import pybullet_data
import numpy as np
import time

# World's parameters:
client_type = p.GUI
dt = 1/240

deg = 0
alpha = np.deg2rad(deg)

g = 9.81
# ------------------

client = p.connect(client_type)

p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
p.setGravity(0, 0, -g, physicsClientId=client)
p.setTimeStep(dt, physicsClientId=client)

# PLANE
plane_collision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[2,2,0.05], physicsClientId=client)
plane_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[2,2,0.05], rgbaColor=[255,0,0,0.5], physicsClientId=client)
plane = p.createMultiBody(
    baseMass=0.0,
    baseCollisionShapeIndex=plane_collision,
    baseVisualShapeIndex=plane_visual,
    basePosition=[0,0,0],
    baseOrientation=p.getQuaternionFromEuler([0,alpha,0]),
    physicsClientId=client
)
p.changeDynamics(plane, -1, lateralFriction=1.0 ,physicsClientId=client)

# CUBE
cube_collision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[0.1,0.1,0.1], physicsClientId=client)
cube_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.1,0.1,0.1], rgbaColor=[0,0,1,1], physicsClientId=client)
cube = p.createMultiBody(
    baseMass=1.0,
    baseCollisionShapeIndex=cube_collision,
    baseVisualShapeIndex=cube_visual,
    basePosition=[0,0,(0.15 / np.cos(alpha)) + 0.01],
    baseOrientation=p.getQuaternionFromEuler([0,alpha,0]),
    physicsClientId=client
)
p.changeDynamics(cube, -1, lateralFriction=0.5 ,physicsClientId=client)

# VISUAL sphere
# sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.1, rgbaColor=[0,1,0,1], physicsClientId=client)
# sphere = p.createMultiBody(
#     baseMass=1.0,
#     baseVisualShapeIndex=sphere_visual,
#     basePosition=[0,0,2],
#     baseOrientation=p.getQuaternionFromEuler([0,0,0]),
#     physicsClientId=client
# )


if client_type == p.DIRECT:
    for _ in range(100000):
        p.stepSimulation(physicsClientId=client)
    p.disconnect(client)
else:
    while p.isConnected(physicsClientId=client):
        p.stepSimulation(physicsClientId=client)
        time.sleep(dt)


