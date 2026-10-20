import pybullet as p
import time 
import pybullet_data
from exceptions.exceptions import URDFLoadError
from pathlib import Path

# PATHS
PROJECT_ROOT = Path(__file__).resolve().parent
URDF_PATH = PROJECT_ROOT / "robots" / "test_robot.urdf"

physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version

connection_info = p.getConnectionInfo()
print(connection_info)

p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,2]
startOrientation = p.getQuaternionFromEuler([0,0,0])

boxId = p.loadURDF(str(URDF_PATH), startPos, startOrientation)

if boxId < 0:
    print("Failed to load URDF file.")
    raise URDFLoadError("Failed to load URDF file.", value="robots/robot_2dof.urdf")



#set the center of mass frame (loadURDF sets base link frame)
# startPos.resetBasePositionAndOrientation(boxId, startPos, startOrientation)

for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)

cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()


