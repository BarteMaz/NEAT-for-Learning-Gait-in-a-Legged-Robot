from neat_walking_robot.environments import FlatEnvironment, BaseEnvironmentData
from neat_walking_robot.simulation import SimulationData, connect_and_configure_simulation
from neat_walking_robot.robot.robot import Robot, A1_ROBOT_SPEC

import pybullet as p
import time
import numpy as np
import pandas as pd

sim_data = SimulationData()
client = connect_and_configure_simulation(sim_data, client_type=p.GUI)

env_data = BaseEnvironmentData(
    track_length=20,
    track_width=8,
    wall_height=3.0,
    wall_thickness=0.25,
    ground_thickness=0.20,
    end_distance=1.0,
    start_distance=1.0,
    friction=1.0,
    lowest_ground_point=0.0,
)

flat_env = FlatEnvironment(env_data, client, 0)

a1 = Robot(A1_ROBOT_SPEC, flat_env.get_starting_pos(), client)

T = 20
k = 4
dt = sim_data.dt
n_actions = int(T / (k * dt))
n_warmup = int(round(1.0 / (k * dt)))


def sine_gait_action(t, params):
    f = params['f']
    a_thigh = params['at']
    a_calf = params['ac']

    FR_RL_thigh = a_thigh * np.sin(2 * np.pi * f * t)
    FR_RL_calf = a_calf * np.sin(2 * np.pi * f * t + (np.pi / 2))

    FL_RR_thigh = a_thigh * np.sin(2 * np.pi * f * t + np.pi)
    FL_RR_calf = a_calf * np.sin(2 * np.pi * f * t + (np.pi / 2) + np.pi)

    return [0.0, FR_RL_thigh, FR_RL_calf, 0.0, FL_RR_thigh, FL_RR_calf, 0.0, FL_RR_thigh, FL_RR_calf, 0.0, FR_RL_thigh,
            FR_RL_calf]


# results = {
#     "frequencies": [],
#     "thigh_amp": [],
#     "calf_amp": [],
#     "fall_time": [],
#     "wall_contact": [],
#     "v": [],
#     "max_yaw": [],
#     "x_end": [],
#     "y_max": []
# }
# frequencies = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]
# thigh_amps = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
# calves_amps = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
#
# for f in frequencies:
#     for at in thigh_amps:
#         for ac in calves_amps:
#             params = {
#                 "f": f,
#                 "at": at,
#                 "ac": ac
#             }
#
#             fall_time = None
#             v = None
#             x_start = None
#             max_yaw = 0.0
#             y_max = None
#             x_end = None
#             wall_contact = False
#             a1.reset()
#             for n in range(n_actions):
#                 t = n * k * sim_data.dt
#                 action = sine_gait_action(t, params)
#                 a1.apply_action(action)
#                 for _ in range(k):
#                     p.stepSimulation(physicsClientId=client)
#
#                 if len(p.getContactPoints(bodyA=flat_env.wall_right, linkIndexA=-1, physicsClientId=client)) > 0:
#                     wall_contact = True
#                 if len(p.getContactPoints(bodyA=flat_env.wall_left, linkIndexA=-1, physicsClientId=client)) > 0:
#                     wall_contact = True
#
#                 if n == n_warmup:
#                     x_start = p.getBasePositionAndOrientation(a1.robot_id, physicsClientId=client)[0][0]
#                 info = p.getBasePositionAndOrientation(a1.robot_id, physicsClientId=client)
#                 roll, pitch, yaw = p.getEulerFromQuaternion(info[1])
#
#                 if max_yaw < np.abs(yaw):
#                     max_yaw = np.abs(yaw)
#                 if y_max is None or np.abs(y_max) < np.abs(info[0][1]) :
#                     y_max = info[0][1]
#
#                 if np.abs(roll) > a1.fall_threshold or np.abs(pitch) > a1.fall_threshold:
#                     fall_time = t
#                     break
#
#             if fall_time is None:
#                 x_end, y_end = p.getBasePositionAndOrientation(a1.robot_id, physicsClientId=client)[0][0:2]
#                 v = (x_end - x_start) / ((n_actions - n_warmup - 1) * k * dt)
#             else:
#                 x_end, y_end = p.getBasePositionAndOrientation(a1.robot_id, physicsClientId=client)[0][0:2]
#
#             results["frequencies"].append(f)
#             results["thigh_amp"].append(at)
#             results["calf_amp"].append(ac)
#             results["fall_time"].append(fall_time)
#             results["wall_contact"].append(wall_contact)
#             results["v"].append(v)
#             results["max_yaw"].append(max_yaw)
#             results["x_end"].append(x_end)
#             results["y_max"].append(y_max)
#
# p.disconnect(physicsClientId=client)
#
# df = pd.DataFrame(results)
# df= df[df["fall_time"].isna()]
# df = df[df["wall_contact"] == False]
# df.to_csv("sine_gait.csv", index=False)


for i in range(360):
    p.stepSimulation(physicsClientId=client)

params = {
    "f": 3.0,
    "at": 0.4,
    "ac": 0.3
}
n = 0
while p.isConnected(physicsClientId=client):
    t = n * k * sim_data.dt
    n = n + 1
    action = sine_gait_action(t, params)
    a1.apply_action(action)
    for _ in range(k):
        p.stepSimulation(physicsClientId=client)
        time.sleep(dt)
