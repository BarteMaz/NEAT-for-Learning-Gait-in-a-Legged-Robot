from neat_walking_robot.training import default_training_config, TrainingConfig
from neat_walking_robot.simulation import prepare_simulation
from neat_walking_robot.environments import create_environment
import time
import pybullet as p
from pathlib import Path

training_config = default_training_config(Path(""), 42)

client = p.connect(p.GUI)
prepare_simulation(training_config.simulation_data, client)
env = create_environment(training_config.env_type, training_config.env_data, client, training_config.seed)

walls_id = [env.wall_back, env.wall_left, env.wall_right, env.wall_front]
for wall in walls_id:
    p.changeVisualShape(wall, -1, rgbaColor=[0.0, 1.0, 0.0, 0.3], physicsClientId=client)

p.configureDebugVisualizer(flag=p.COV_ENABLE_SHADOWS, enable=0,physicsClientId=client)
p.resetDebugVisualizerCamera(5.5, 0, -89.9, [0, 0, 0])

while p.isConnected(physicsClientId=client):
    p.stepSimulation(physicsClientId=client)
    time.sleep(training_config.simulation_data.dt)

