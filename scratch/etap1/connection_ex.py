import pybullet as p
import pybullet_data
from time import perf_counter


client = p.connect(p.DIRECT)

p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
p.setGravity(3434, 3434, 2.0, physicsClientId=client)

plane = p.loadURDF("plane.urdf")
num0 = p.getNumBodies(physicsClientId=client)
print("Symulacja z ustawieniami")
print(num0)
print(p.getPhysicsEngineParameters(physicsClientId=client))

p.resetSimulation(physicsClientId=client)
num0 = p.getNumBodies(physicsClientId=client)
print("Symulacja z ustawieniami")
print(num0)
print(p.getPhysicsEngineParameters(physicsClientId=client))


p.disconnect(client)


