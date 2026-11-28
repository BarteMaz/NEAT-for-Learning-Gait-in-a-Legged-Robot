"""Incline-plane experiment: calibrating the ground friction of the A1 simulation.

A block (and a "toe sled" with the A1 toe collision spheres) gets the same contact
dynamics as the A1 toes and slides down an incline. From its motion the experiment reads
the friction coefficient that Bullet really applies, checks how Bullet combines the
coefficients of the two bodies, and fits the ground friction so that the effective
coefficient equals the one of rubber on dry concrete.

Description, sources and conclusions: physics/README.md
Run from the repository root:  uv run python physics/friction_incline.py
Results:                       physics/friction_results.json
"""

import json
import math
from dataclasses import replace
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pybullet as p

from neat_walking_robot.simulation import SimulationData, prepare_simulation

RESULTS_PATH = Path(__file__).with_name("friction_results.json")

A1_MODEL = "a1/a1.urdf"
TOE_LINKS = ("FR_toe", "FL_toe", "RR_toe", "RL_toe")

# Rubber on dry concrete (see README.md for the sources). Bullet has a single Coulomb
# coefficient, so one value has to stand for both static and kinetic friction.
TARGET_MU = 0.8
TARGET_MU_RANGE = (0.6, 1.0)
CURRENT_GROUND_FRICTION = 1.0  # BaseEnvironmentData.friction in default_training_config()

MEASURE_ANGLE_DEG = 60.0  # tan(60 deg) = 1.73, so every tested mu_eff below that slides
SETTLE_TIME = 0.1         # s, skipped before fitting the acceleration
MEASURE_TIME = 1.0        # s of sliding used for the fit
STICK_TIME = 2.0          # s, run length when checking whether the body sticks
SLIDE_SPEED = 0.01        # m/s, faster than this after STICK_TIME counts as sliding

PLANE_HALF = (6.0, 1.0, 0.05)  # incline half extents [m], long axis = downhill
BOX_HALF = (0.1, 0.1, 0.02)    # flat block, so it slides instead of tipping over
SLED_HALF_SPAN = 0.1           # toe spheres at (+-0.1, +-0.1) in the sled frame
BODY_MASS = 1.0                # kg; Coulomb friction should not depend on it (checked below)
# PyBullet gives every body linearDamping = angularDamping = 0.04 by default: a drag that grows with
# speed (Bullet multibody: F = -m * c * (1 + |v|) * v). On a sliding body it looks like extra friction,
# so the friction measurements switch it off; its effect is measured separately in robustness.
PYBULLET_DEFAULT_DAMPING = 0.04
MEASURE_DAMPING = 0.0

COMBINATION_PAIRS = [(0.5, 0.5), (0.5, 1.0), (0.5, 1.6), (1.0, 0.6), (0.8, 1.2), (1.0, 1.0), (0.3, 2.0)]
COMBINATION_MODELS = {
    "product": lambda a, b: a * b,
    "min": min,
    "max": max,
    "arithmetic_mean": lambda a, b: (a + b) / 2,
    "geometric_mean": lambda a, b: math.sqrt(a * b),
}


def new_client(sim: SimulationData) -> int:
    client = p.connect(p.DIRECT)
    prepare_simulation(sim, client)
    return client


def read_a1_contact_dynamics(sim: SimulationData) -> dict:
    """Contact parameters of every A1 link plus the default friction of a box built like the arena ground."""
    client = new_client(sim)
    try:
        robot = p.loadURDF(A1_MODEL, flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=client)
        links = {}
        toe_radius = None
        for index in range(-1, p.getNumJoints(robot, physicsClientId=client)):
            if index == -1:
                name = p.getBodyInfo(robot, physicsClientId=client)[0].decode()
            else:
                name = p.getJointInfo(robot, index, physicsClientId=client)[12].decode()
            info = p.getDynamicsInfo(robot, index, physicsClientId=client)
            links[name] = {
                "lateral_friction": info[1],
                "restitution": info[5],
                "rolling_friction": info[6],
                "spinning_friction": info[7],
                "contact_damping": info[8],
                "contact_stiffness": info[9],
            }
            if name == TOE_LINKS[0]:
                shape = p.getCollisionShapeData(robot, index, physicsClientId=client)[0]
                toe_radius = shape[3][0]

        # same call as BaseEnvironment.build_static_box, before changeDynamics sets the friction
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1.0, 1.0, 0.1], physicsClientId=client)
        ground = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=collision, physicsClientId=client)
        default_box_friction = p.getDynamicsInfo(ground, -1, physicsClientId=client)[1]
    finally:
        p.disconnect(client)

    toes = {name: links[name] for name in TOE_LINKS}
    first = toes[TOE_LINKS[0]]
    if any(toe != first for toe in toes.values()):
        raise RuntimeError(f"A1 toes have different contact dynamics: {toes}")
    return {
        "toe_dynamics": first,
        "toe_collision_radius": toe_radius,
        "lateral_friction_of_all_links": sorted({v["lateral_friction"] for v in links.values()}),
        "links": links,
        "default_box_lateral_friction": default_box_friction,
    }


def incline_frame(angle_deg: float, heading_deg: float):
    """Orientation of the incline. Its local x axis points downhill, local z is the surface normal."""
    orientation = p.getQuaternionFromEuler([0.0, math.radians(angle_deg), math.radians(heading_deg)])
    rotation = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    return orientation, rotation


def build_incline(client: int, orientation, rotation: np.ndarray, friction: float) -> int:
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=PLANE_HALF, physicsClientId=client)
    center = rotation @ np.array([0.0, 0.0, -PLANE_HALF[2]])  # top surface goes through the origin
    plane = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        basePosition=center.tolist(),
        baseOrientation=orientation,
        physicsClientId=client,
    )
    p.changeDynamics(plane, -1, lateralFriction=friction, physicsClientId=client)
    return plane


def build_slider(client: int, shape: str, orientation, rotation: np.ndarray, dynamics: dict,
                 toe_radius: float, mass: float, damping: float) -> int:
    if shape == "box":
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=BOX_HALF, physicsClientId=client)
        lift = BOX_HALF[2]
    elif shape == "toe_sled":
        offsets = [[sx * SLED_HALF_SPAN, sy * SLED_HALF_SPAN, 0.0] for sx in (-1, 1) for sy in (-1, 1)]
        collision = p.createCollisionShapeArray(
            shapeTypes=[p.GEOM_SPHERE] * 4,
            radii=[toe_radius] * 4,
            collisionFramePositions=offsets,
            physicsClientId=client,
        )
        lift = toe_radius
    else:
        raise ValueError(f"Unknown slider shape: {shape}")

    start = rotation @ np.array([-PLANE_HALF[0] + 0.3, 0.0, lift + 1e-4])
    body = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        basePosition=start.tolist(),
        baseOrientation=orientation,
        physicsClientId=client,
    )
    p.changeDynamics(
        body, -1,
        lateralFriction=dynamics["lateral_friction"],
        rollingFriction=dynamics["rolling_friction"],
        spinningFriction=dynamics["spinning_friction"],
        restitution=dynamics["restitution"],
        linearDamping=damping,
        angularDamping=damping,
        physicsClientId=client,
    )
    return body


def run_slide(sim: SimulationData, angle_deg: float, ground_friction: float, slider: dict,
              duration: float, heading_deg: float = 0.0, mass: float = BODY_MASS,
              damping: float = MEASURE_DAMPING) -> dict:
    """Releases the slider at rest and records its displacement along the downhill direction."""
    client = new_client(sim)
    try:
        orientation, rotation = incline_frame(angle_deg, heading_deg)
        build_incline(client, orientation, rotation, ground_friction)
        body = build_slider(client, slider["shape"], orientation, rotation, slider["dynamics"],
                            slider["toe_radius"], mass, damping)
        downhill, normal = rotation[:, 0], rotation[:, 2]

        steps = round(duration / sim.dt)
        time = np.arange(1, steps + 1) * sim.dt
        along = np.empty(steps)
        start = np.array(p.getBasePositionAndOrientation(body, physicsClientId=client)[0])
        max_normal_offset = 0.0
        for i in range(steps):
            p.stepSimulation(physicsClientId=client)
            position = np.array(p.getBasePositionAndOrientation(body, physicsClientId=client)[0])
            along[i] = (position - start) @ downhill
            max_normal_offset = max(max_normal_offset, abs((position - start) @ normal))

        speed = np.array(p.getBaseVelocity(body, physicsClientId=client)[0]) @ downhill
        body_orientation = p.getBasePositionAndOrientation(body, physicsClientId=client)[1]
        body_normal = np.array(p.getMatrixFromQuaternion(body_orientation)).reshape(3, 3)[:, 2]
        tilt_deg = math.degrees(math.acos(float(np.clip(body_normal @ normal, -1.0, 1.0))))
    finally:
        p.disconnect(client)

    return {
        "time": time,
        "along": along,
        "final_speed": float(speed),
        "max_normal_offset": max_normal_offset,
        "tilt_deg": tilt_deg,
    }


def measure_mu(sim: SimulationData, ground_friction: float, slider: dict,
               angle_deg: float = MEASURE_ANGLE_DEG, heading_deg: float = 0.0, mass: float = BODY_MASS,
               damping: float = MEASURE_DAMPING) -> dict:
    """Effective friction from the sliding acceleration: a = g (sin(theta) - mu cos(theta))."""
    run = run_slide(sim, angle_deg, ground_friction, slider, MEASURE_TIME, heading_deg, mass, damping)
    mask = run["time"] >= SETTLE_TIME
    coefficients = np.polyfit(run["time"][mask], run["along"][mask], 2)
    residual = run["along"][mask] - np.polyval(coefficients, run["time"][mask])
    acceleration = 2.0 * coefficients[0]

    g = abs(sim.gravity[2])
    theta = math.radians(angle_deg)
    mu = (g * math.sin(theta) - acceleration) / (g * math.cos(theta))
    return {
        "mu_measured": mu,
        "acceleration": acceleration,
        "slid": bool(acceleration > 0.05),
        "fit_residual_rms": float(np.sqrt(np.mean(residual ** 2))),
        "max_normal_offset": run["max_normal_offset"],
        "final_tilt_deg": run["tilt_deg"],
    }


def slides(sim: SimulationData, angle_deg: float, ground_friction: float, slider: dict, heading_deg: float) -> bool:
    return run_slide(sim, angle_deg, ground_friction, slider, STICK_TIME, heading_deg)["final_speed"] > SLIDE_SPEED


def critical_angle(sim: SimulationData, ground_friction: float, slider: dict, heading_deg: float = 0.0,
                   low: float = 1.0, high: float = 80.0, tolerance: float = 0.01) -> dict:
    """Smallest incline angle at which the slider starts to slide (bisection)."""
    while high - low > tolerance:
        middle = (low + high) / 2
        if slides(sim, middle, ground_friction, slider, heading_deg):
            high = middle
        else:
            low = middle
    angle = (low + high) / 2
    creep = run_slide(sim, angle - 1.0, ground_friction, slider, STICK_TIME, heading_deg)
    return {
        "critical_angle_deg": angle,
        "mu_from_angle": math.tan(math.radians(angle)),
        "creep_1deg_below_m": float(creep["along"][-1]),
    }


def fit_ground_friction(sim: SimulationData, target_mu: float, slider: dict,
                        low: float = 0.05, high: float = 5.0, tolerance: float = 1e-4) -> float:
    """Ground lateralFriction for which the measured mu_eff equals target_mu (bisection, mu_eff grows with it)."""
    while high - low > tolerance:
        middle = (low + high) / 2
        if measure_mu(sim, middle, slider)["mu_measured"] < target_mu:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def package_version() -> str:
    for name in ("pybullet-arm64", "pybullet"):
        try:
            return f"{name} {metadata.version(name)}"
        except metadata.PackageNotFoundError:
            continue
    return "unknown"


def main() -> None:
    sim = SimulationData()
    a1 = read_a1_contact_dynamics(sim)
    toe = a1["toe_dynamics"]
    mu_foot = toe["lateral_friction"]

    def slider(shape: str, lateral_friction: float = mu_foot) -> dict:
        return {"shape": shape, "dynamics": {**toe, "lateral_friction": lateral_friction},
                "toe_radius": a1["toe_collision_radius"]}

    box, sled = slider("box"), slider("toe_sled")
    print(f"A1 toe lateral friction: {mu_foot}, default box friction: {a1['default_box_lateral_friction']}")

    # 1. How does Bullet combine the coefficients of the two bodies?
    combination = []
    for mu_block, mu_ground in COMBINATION_PAIRS:
        measured = measure_mu(sim, mu_ground, slider("box", mu_block))["mu_measured"]
        combination.append({"mu_block": mu_block, "mu_ground": mu_ground, "mu_measured": measured,
                            **{name: f(mu_block, mu_ground) for name, f in COMBINATION_MODELS.items()}})
    model_errors = {
        name: float(np.sqrt(np.mean([(row["mu_measured"] - row[name]) ** 2 for row in combination])))
        for name in COMBINATION_MODELS
    }
    best_model = min(model_errors, key=model_errors.get)
    print(f"combination rule: {best_model} (rms errors: {model_errors})")

    # 2. Current project setting: A1 toes on the arena ground (friction=1.0).
    current = {
        "ground_friction": CURRENT_GROUND_FRICTION,
        "expected_mu_eff": COMBINATION_MODELS[best_model](mu_foot, CURRENT_GROUND_FRICTION),
        "box_sliding": measure_mu(sim, CURRENT_GROUND_FRICTION, box),
        "toe_sled_sliding": measure_mu(sim, CURRENT_GROUND_FRICTION, sled),
        "box_critical_angle": critical_angle(sim, CURRENT_GROUND_FRICTION, box),
        "toe_sled_critical_angle": critical_angle(sim, CURRENT_GROUND_FRICTION, sled),
    }
    print(f"current mu_eff: {current['box_sliding']['mu_measured']:.4f}")

    # 3. Robustness of the measurement: time step, mass, sliding direction.
    robustness = {
        "time_step": [
            {"dt": dt, **measure_mu(replace(sim, dt=dt), CURRENT_GROUND_FRICTION, box)}
            for dt in (1 / 240, 1 / 960)
        ],
        "mass": [
            {"mass": mass, **measure_mu(sim, CURRENT_GROUND_FRICTION, box, mass=mass)}
            for mass in (0.25, 1.0, 3.0)
        ],
        "heading_sliding": [
            {"heading_deg": heading, **measure_mu(sim, CURRENT_GROUND_FRICTION, box, heading_deg=heading)}
            for heading in (0.0, 30.0, 45.0)
        ],
        "heading_sticking": [
            {"heading_deg": heading, **critical_angle(sim, CURRENT_GROUND_FRICTION, box, heading_deg=heading)}
            for heading in (0.0, 15.0, 30.0, 45.0, 60.0, 90.0)
        ],
        "pybullet_default_damping": [
            {"angle_deg": angle, "damping": damping,
             **measure_mu(sim, CURRENT_GROUND_FRICTION, box, angle_deg=angle, damping=damping)}
            for angle in (35.0, 60.0, 75.0)
            for damping in (MEASURE_DAMPING, PYBULLET_DEFAULT_DAMPING)
        ],
    }

    # 4. Calibration: ground friction that gives rubber-on-dry-concrete mu_eff with the A1 toes.
    calibration = []
    for target in (TARGET_MU_RANGE[0], TARGET_MU, TARGET_MU_RANGE[1]):
        ground = fit_ground_friction(sim, target, box)
        calibration.append({
            "target_mu": target,
            "fitted_ground_friction": ground,
            "analytic_ground_friction": target / mu_foot,
            "box_sliding": measure_mu(sim, ground, box),
            "toe_sled_sliding": measure_mu(sim, ground, sled),
            "box_critical_angle": critical_angle(sim, ground, box),
            "expected_critical_angle_deg": math.degrees(math.atan(target)),
        })
        print(f"target mu {target}: ground friction {ground:.4f}")

    recommended = next(row for row in calibration if row["target_mu"] == TARGET_MU)
    results = {
        "meta": {
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "pybullet": package_version(),
            "simulation": {"dt": sim.dt, "gravity": sim.gravity, "num_solver_iterations": sim.num_solver_iterations},
            "measure_angle_deg": MEASURE_ANGLE_DEG,
            "settle_time_s": SETTLE_TIME,
            "measure_time_s": MEASURE_TIME,
            "stick_time_s": STICK_TIME,
            "slide_speed_threshold_m_s": SLIDE_SPEED,
            "body_mass_kg": BODY_MASS,
            "slider_linear_and_angular_damping": MEASURE_DAMPING,
            "box_half_extents_m": BOX_HALF,
            "toe_sled_half_span_m": SLED_HALF_SPAN,
        },
        "a1_contact_dynamics": a1,
        "combination_rule": {"pairs": combination, "rms_error_per_model": model_errors, "best_model": best_model},
        "current_setting": current,
        "robustness": robustness,
        "calibration": calibration,
        "recommendation": {
            "target_mu_rubber_dry_concrete": TARGET_MU,
            "target_mu_range": TARGET_MU_RANGE,
            "foot_lateral_friction": mu_foot,
            "ground_friction": round(recommended["fitted_ground_friction"], 3),
            "ground_friction_range": [round(row["fitted_ground_friction"], 3) for row in (calibration[0], calibration[-1])],
        },
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
