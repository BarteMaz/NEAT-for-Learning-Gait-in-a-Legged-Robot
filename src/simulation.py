from pathlib import Path

import pybullet as p
import pybullet_data


# ==========================================
# Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent
URDF_PATH = PROJECT_ROOT / "robots" / "test_robot.urdf"


# ==========================================
# Start PyBullet
# ==========================================

physics_client = p.connect(p.GUI)

if physics_client < 0:
    raise RuntimeError("Nie udało się połączyć z PyBullet.")


# ==========================================
# PyBullet configuration
# ==========================================

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.81)

# Ground plane
plane_id = p.loadURDF("plane.urdf")


# ==========================================
# Load our robot
# ==========================================

robot_id = p.loadURDF(
    str(URDF_PATH),
    basePosition=[0, 0, 0.1],
    useFixedBase=True,
)


print(f"Robot ID: {robot_id}")
print(f"URDF: {URDF_PATH}")
print(f"Liczba jointów: {p.getNumJoints(robot_id)}")


# ==========================================
# Simulation loop
# ==========================================

while p.isConnected():

    p.stepSimulation()

    # PyBullet GUI działa w czasie rzeczywistym.