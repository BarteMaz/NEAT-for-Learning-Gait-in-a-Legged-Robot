import pybullet as p
import pybullet_data
from time import perf_counter


client = p.connect(p.DIRECT)
client1 = p.connect(p.DIRECT)

p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
p.setGravity(0, 0, -9.81, physicsClientId=client)

p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client1)
p.setGravity(0, 0, -9.81, physicsClientId=client1)

plane = p.loadURDF("plane.urdf")

num0 = p.getNumBodies(physicsClientId=client)
num1 = p.getNumBodies(physicsClientId=client1)

print(num0, num1)
print(f"Serwer0: {num0}; Serwer1: {num1}")
p.disconnect(client)
p.disconnect(client1)

