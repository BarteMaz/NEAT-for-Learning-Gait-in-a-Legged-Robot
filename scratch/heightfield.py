import numpy as np
import time
import pybullet as p
import pybullet_data

ROWS, COLS, CELL, SZ = 256, 256, 0.05, 1.0


def make_terrain(rows, cols, seed=0, difficulty=1.0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, rows)[:, None]
    y = np.linspace(0, 1, cols)[None, :]

    # 1) łagodne pagórki
    h = 0.15 * difficulty * np.sin(2 * np.pi * 2 * x) * np.cos(2 * np.pi * 3 * y)

    # 2) drobne „kamienie”: zgrubny szum powiększony blokami 4x4
    coarse = rng.uniform(0, 0.04 * difficulty, size=(rows // 4, cols // 4))
    h += np.repeat(np.repeat(coarse, 4, axis=0), 4, axis=1)

    # 4) płaski obszar startowy w środku (promień ~1 m)
    ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    r = np.hypot((ii - rows / 2) * CELL, (jj - cols / 2) * CELL)
    h[r < 1.0] = 0.0
    return h


def build_terrain(h, replace=None):
    kwargs = dict(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=[CELL, CELL, SZ],
        heightfieldData=h.flatten(order="F").tolist(),
        numHeightfieldRows=ROWS,
        numHeightfieldColumns=COLS,
        heightfieldTextureScaling=(ROWS - 1) / 2,
        flags=p.GEOM_CONCAVE_INTERNAL_EDGE,
    )
    if replace is not None:
        kwargs["replaceHeightfieldIndex"] = replace
    return p.createCollisionShape(**kwargs)


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # szybsze ładowanie

h = make_terrain(ROWS, COLS)
shape = build_terrain(h)
terrain = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape)
p.resetBasePositionAndOrientation(
    terrain, [0, 0, (h.min() + h.max()) / 2 * SZ], [0, 0, 0, 1])
p.changeDynamics(terrain, -1, lateralFriction=1.0)
p.changeVisualShape(terrain, -1, rgbaColor=[0.6, 0.7, 0.6, 1])

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


# Wysokość terenu w dowolnym punkcie: raycast, odporny na wszelkie offsety
def ground_z(x, y):
    hit = p.rayTest([x, y, 10], [x, y, -10])[0]
    return hit[3][2] if hit[0] == terrain else 0.0


robot = p.loadURDF("r2d2.urdf", [3, 0, ground_z(0, 0) + 0.5])

while p.isConnected():
    time.sleep(1/240)
    p.stepSimulation()
