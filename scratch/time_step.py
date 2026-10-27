import pybullet as p
import pybullet_data

client = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
p.setGravity(0, 0, -9.81, physicsClientId=client)

plane = p.loadURDF("plane.urdf", physicsClientId=client)
ball = p.loadURDF("sphere_small.urdf", [0, 0, 1.0], physicsClientId=client)

p.changeDynamics(ball, -1,linearDamping=0.0, angularDamping=0.0, physicsClientId=client)
p.setTimeStep(1/1000, physicsClientId=client)
flag = 0
for _ in range(100000):
    p.stepSimulation(physicsClientId=client)
    if len(p.getContactPoints(ball, plane, physicsClientId=client)) > 0 and flag == 0:
        print(_)
        flag = 1

print(p.getPhysicsEngineParameters(physicsClientId=client))
pos, orn = p.getBasePositionAndOrientation(ball, physicsClientId=client)
print(pos)
p.disconnect(client)