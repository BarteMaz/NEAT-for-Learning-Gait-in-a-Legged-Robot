from neat_walking_robot.simulation import SimulationData, prepare_simulation
from neat_walking_robot.environments import (
    BaseEnvironment,
    BaseEnvironmentData,
    RoughEnvironmentData,
    SlopeEnvironmentData,
    create_environment,
    EnvironmentType
)
from neat_walking_robot.robot import Robot, A1_ROBOT_SPEC

import argparse
import pybullet as p
import time

FLAT_ENV_DATA = BaseEnvironmentData(
    track_length=6.0,
    track_width=3.0,
    wall_height=3.0,
    wall_thickness=0.25,
    ground_thickness=0.20,
    end_distance=1.0,
    start_distance=1.0,
    friction=1.0,
    lowest_ground_point=-0.20,
)

ROUGH_ENV_DATA = RoughEnvironmentData(
    track_length=6.0,
    track_width=3.0,
    wall_height=3.0,
    wall_thickness=0.25,
    ground_thickness=0.20,
    end_distance= 1.0,
    start_distance= 1.0,
    friction= 1.0,
    lowest_ground_point= -0.3,
    buffer_space=0.5,
    cell_size= 0.05,
    amplitude=0.3,
    transition_distance=2.0,
    n_waves=6,
    wavelength_range=(1.0, 3.0),
)

SLOPE_ENV_DATA = SlopeEnvironmentData(
    track_length= 6,
    track_width= 3.0,
    wall_height= 3.0,
    wall_thickness= 0.25,
    ground_thickness= 0.20,
    end_distance= 1.0,
    start_distance= 1.0,
    friction= 1.0,
    lowest_ground_point= 0.0,
    buffer_space=0.5,
    incline=20
)

# name, distance, yaw, pitch
CAMERA_PRESETS = [
    ("side", [0.0, 0.0, 1.0], 4.0, 0.0, -5.0),
    ("top", [0.0, 0.0, 1.0], 4.0, 0.0, -89.9),
    ("oblique", [0.0, 0.0, 0.6], 6.0, -80.0, -20.0),
]

def apply_camera(preset, client):
    _, target, distance, yaw, pitch = preset
    p.resetDebugVisualizerCamera(
        cameraDistance=distance,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=target,
        physicsClientId=client,
    )

def was_pressed(events, key):
    return bool(events.get(key, 0) & p.KEY_WAS_TRIGGERED)

def change_walls_opacity(env: BaseEnvironment, opacity: float, client_id: int) -> None:
    walls_id = [env.wall_back, env.wall_left, env.wall_right, env.wall_front]
    for wall in walls_id:
        p.changeVisualShape(wall, -1, rgbaColor=[0.0, 1.0, 0.0, opacity], physicsClientId=client_id)

def main(args):
    client = p.connect(p.GUI)
    prepare_simulation(SimulationData(), client)
    dt = SimulationData().dt

    if args.rough_env:
        env = create_environment(EnvironmentType.ROUGH, ROUGH_ENV_DATA, client, args.seed)
        texture_id = p.loadTexture("checker_blue.png", physicsClientId=client)
        p.changeVisualShape(env.ground, -1, textureUniqueId=texture_id, rgbaColor=[1.0, 1.0, 1.0, 1.0], physicsClientId=client)
    elif args.slope_env:
        env = create_environment(EnvironmentType.SLOPE, SLOPE_ENV_DATA, client, args.seed)
    else:
        env = create_environment(EnvironmentType.FLAT, FLAT_ENV_DATA, client, args.seed)

    change_walls_opacity(env, 0.1, client)

    camera_index = 0
    apply_camera(CAMERA_PRESETS[camera_index], client)
    robot_id = None

    while p.isConnected(physicsClientId=client):
        events = p.getKeyboardEvents(physicsClientId=client)

        if was_pressed(events, p.B3G_RIGHT_ARROW):
            camera_index = (camera_index + 1) % len(CAMERA_PRESETS)
            apply_camera(CAMERA_PRESETS[camera_index], client)
        if was_pressed(events, p.B3G_LEFT_ARROW):
            camera_index = (camera_index - 1) % len(CAMERA_PRESETS)
            apply_camera(CAMERA_PRESETS[camera_index], client)

        if was_pressed(events, ord("r")):
            if robot_id is not None:  # second press = respawn
                p.removeBody(robot_id, physicsClientId=client)
                robot_id = None
            else:
                a1 = Robot(A1_ROBOT_SPEC, env.get_starting_pos(), client)
                robot_id = a1.robot_id

        p.stepSimulation(physicsClientId=client)
        time.sleep(dt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flat_env", action="store_true")
    parser.add_argument("--rough_env", action="store_true")
    parser.add_argument("--slope_env", action="store_true")
    args = parser.parse_args()

    main(args)


