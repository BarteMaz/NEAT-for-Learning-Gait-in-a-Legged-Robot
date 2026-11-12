import pybullet as p
import numpy as np

from neat_walking_robot.robot.joints import (
    JointType,
    get_joints_info
)
from neat_walking_robot.environments import BaseEnvironment

from dataclasses import dataclass

@dataclass(frozen=True, kw_only=True)
class RobotSpec:
    model_path: str
    toes_links_names: tuple[str, str, str, str]
    rest_position: dict[str, float]
    start_height: float

A1_ROBOT_SPEC = RobotSpec(
    model_path="a1/a1.urdf",
    toes_links_names=("FR_toe_fixed", "FL_toe_fixed", "FR_toe_fixed", "FR_toe_fixed"),
    rest_position={
        "FR_hip_joint" : 0.0,
        "FL_hip_joint" : 0.0,
        "RR_hip_joint" : 0.0,
        "RL_hip_joint" : 0.0,
        "FR_upper_joint" : 0.8,
        "FL_upper_joint" : 0.8,
        "RR_upper_joint" : 0.8,
        "RL_upper_joint" : 0.8,
        "FR_lower_joint" : -1.5,
        "FL_lower_joint": -1.5,
        "RR_lower_joint": -1.5,
        "RL_lower_joint": -1.5,
    },
    start_height=0.32
)

def scale_to_angles(value: float, angular_range: np.ndarray):
    value = np.clip(value, -1.0, 1.0)
    return angular_range[0] + (value + 1.0) * 0.5 * (angular_range[1] - angular_range[0])

# def scale_to_neat_input(value: float):


class Robot:
    def __init__(self, spec: RobotSpec, env: BaseEnvironment, client_id: int):
        self.client_id = client_id
        self.env = env
        self.model_path = spec.model_path
        self.toes_links_names = spec.toes_links_names
        self.rest_position = spec.rest_position
        self.start_height = spec.start_height

        start_pos = env.get_starting_pos()
        self.robot_id = p.loadURDF(
            self.model_path,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            basePosition=[start_pos[0], start_pos[1], start_pos[2] + self.start_height],
            physicsClientId=self.client_id
        )
        self.reset_base_position = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)[0]
        self.moving_joints_limits = {}
        self.moving_joints_efforts = {}
        self.moving_joints_id = {}
        self.toes_links_id = {}

        joints_info = get_joints_info(self.robot_id, self.client_id)
        for joint_info in joints_info.values():
            if joint_info.joint_name in self.toes_links_names:
                self.toes_links_id[joint_info.joint_name] = joint_info.joint_index

            if joint_info.joint_type == JointType.REVOLUTE:
                joint_rest_position = self.rest_position[joint_info.joint_name]
                lower_limit = joint_info.joint_lower_limit
                upper_limit = joint_info.joint_upper_limit
                new_range = np.clip([joint_rest_position - 0.525, joint_rest_position + 0.525], lower_limit,
                                    upper_limit)

                self.moving_joints_limits[joint_info.joint_name] = new_range
                self.moving_joints_efforts[joint_info.joint_name] = joint_info.effort
                self.moving_joints_id[joint_info.joint_name] = joint_info.joint_index

        self.start_base_position = None
        self.start_base_orientation = None
        self.reset()


    def reset(self):
        # reset robot's base
        p.resetBasePositionAndOrientation(self.robot_id, self.reset_base_position, [0,0,0,1], physicsClientId=self.client_id)
        p.resetBaseVelocity(self.robot_id, linearVelocity=[0.0, 0.0, 0.0], angularVelocity=[0.0, 0.0, 0.0], physicsClientId=self.client_id)

        # set joints in rest position
        for joint_name, joint_id in self.moving_joints_id.items():
            p.resetJointState(
                self.robot_id,
                joint_id,
                targetValue=self.rest_position[joint_name],
                targetVelocity=0.0,
                physicsClientId=self.client_id
            )
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_id,
                force=self.moving_joints_efforts[joint_name],
                controlMode=p.POSITION_CONTROL,
                targetPosition=self.rest_position[joint_name],
                physicsClientId=self.client_id
            )

        # stabilization
        for i in range(120):
            p.stepSimulation(physicsClientId=self.client_id)

        base_info = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        self.start_base_position = base_info[0]
        self.start_base_orientation = base_info[1]

    def apply_action(self, values: list[float]) -> None:
        joints_id = []
        joints_target_angles = []
        joints_forces = []

        for i, (joint_name, joint_range) in enumerate(self.moving_joints_limits.items()):
            joints_id.append(self.moving_joints_id[joint_name])
            joints_target_angles.append(scale_to_angles(values[i], joint_range))
            joints_forces.append(self.moving_joints_efforts[joint_name])

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=joints_id,
            controlMode=p.POSITION_CONTROL,
            targetPositions=joints_target_angles,
            forces=joints_forces,
            physicsClientId=self.client_id
        )

    # def get_observation(self) -> np.ndarray:



