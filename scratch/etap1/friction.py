import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt

g = 9.81
dt = 1/240

distances = []
degrees = np.arange(0, 60.5, 0.5)

for deg in degrees:
    client = p.connect(p.DIRECT)

    alpha = np.deg2rad(deg)
    gx = -np.sin(alpha)*g
    gz = -np.cos(alpha)*g

    p.setTimeStep(dt, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -g, physicsClientId=client)

    plane = p.loadURDF("plane.urdf", physicsClientId=client)
    cube = p.loadURDF("cube_small.urdf", [0, 0, 0.025], physicsClientId=client)

    p.changeDynamics(cube, -1, lateralFriction=0.5 ,physicsClientId=client)

    for _ in range(round(3 / dt)):
        p.stepSimulation(physicsClientId=client)

    pos1, orn1 = p.getBasePositionAndOrientation(cube, physicsClientId=client)

    p.setGravity(gx, 0, gz, physicsClientId=client)
    for _ in range(round(1 / dt)):
        p.stepSimulation(physicsClientId=client)


    pos2, orn2 = p.getBasePositionAndOrientation(cube, physicsClientId=client)

    distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
    distances.append(distance)


fig, ax = plt.subplots()
ax.plot(degrees, distances)
ax.set(xlabel='degrees', ylabel='distance', yscale='log')
plt.show()

