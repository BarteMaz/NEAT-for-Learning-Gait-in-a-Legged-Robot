from dataclasses import dataclass

import numpy as np
import pybullet as p
from scipy.constants import value

from neat_walking_robot.robot.joints import JointType, get_joints_info


@dataclass(frozen=True, kw_only=True)
class RobotSpec:
    model_path: str
    toes_links_names: tuple[str, str, str, str]
    rest_position: dict[str, float]
    angular_range: float
    stabilization_time: int
    start_height: float
    joints_velocity_scale: float
    base_velocity_scale: float
    fall_threshold: float
    roll_and_pitch_scale: float
    apply_velocity_limit: bool = False
    position_gain: float = 0.1
    action_filter_alpha: float = 1.0
    def __post_init__(self):
        if not (0 < self.action_filter_alpha <= 1.0):
            raise ValueError(f'action_filter_alpha must be in (0, 1]')


A1_ROBOT_SPEC = RobotSpec(
    model_path="a1/a1.urdf",
    toes_links_names=("FR_toe", "FL_toe", "RR_toe", "RL_toe"),
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
    start_height=0.32,
    angular_range=0.525,
    stabilization_time=120,
    joints_velocity_scale=10.0,
    base_velocity_scale=5.0,
    fall_threshold=0.8,
    roll_and_pitch_scale=0.8,
)

def scale_to_angles(value: float, angular_range: np.ndarray):
    value = np.clip(value, -1.0, 1.0)
    return angular_range[0] + (value + 1.0) * 0.5 * (angular_range[1] - angular_range[0])

def scale_to_neat_input(angle: float,angular_range: np.ndarray):
    return 2 * (angle - angular_range[0]) / (angular_range[1] - angular_range[0]) - 1.0

class Robot:
    def __init__(self, spec: RobotSpec, start_pos: list[float], client_id: int):
        self.client_id = client_id

        # specification
        self.model_path = spec.model_path
        self.toes_links_names = spec.toes_links_names
        self.rest_position = spec.rest_position
        self.start_height = spec.start_height
        self.stabilization_time = spec.stabilization_time
        self.joints_velocity_scale = spec.joints_velocity_scale
        self.base_velocity_scale = spec.base_velocity_scale
        self.fall_threshold = spec.fall_threshold
        self.roll_and_pitch_scale = spec.roll_and_pitch_scale
        self.apply_velocity_limit = spec.apply_velocity_limit
        self.position_gain = spec.position_gain
        self.action_filter_alpha = spec.action_filter_alpha

        # loading
        self.robot_id = p.loadURDF(
            self.model_path,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            basePosition=[start_pos[0], start_pos[1], start_pos[2] + self.start_height],
            physicsClientId=self.client_id
        )
        self.reset_base_position = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)[0]
        self.moving_joints_limits = {}
        self.moving_joints_efforts = {}
        self.moving_joints_velocity = {}
        self.moving_joints_id = {}
        self.toes_links_id = {}

        # check for joints and links names compatibility
        joints_info = get_joints_info(self.robot_id, self.client_id)
        available_joints_names = {joint.joint_name for joint in joints_info.values()}
        available_links_names = {joint.link_name for joint in joints_info.values()}
        if not set(self.toes_links_names).issubset(available_links_names):
            raise ValueError(f"Toes links names in robot specification are not valid! Available links names: {available_links_names}")
        if not set(self.rest_position.keys()).issubset(available_joints_names):
            raise ValueError(f"Joints names in robot specification are not valid! Available joints names: {available_joints_names}")

        # read and save joints info
        for joint_info in joints_info.values():
            if joint_info.link_name in self.toes_links_names:
                self.toes_links_id[joint_info.link_name] = joint_info.joint_index

            if joint_info.joint_type == JointType.REVOLUTE:
                joint_rest_position = self.rest_position[joint_info.joint_name]
                lower_limit = joint_info.joint_lower_limit
                upper_limit = joint_info.joint_upper_limit
                new_range = np.clip([joint_rest_position - spec.angular_range, joint_rest_position + spec.angular_range], lower_limit,
                                    upper_limit)

                self.moving_joints_limits[joint_info.joint_name] = new_range
                self.moving_joints_efforts[joint_info.joint_name] = joint_info.effort
                self.moving_joints_velocity[joint_info.joint_name] = joint_info.velocity
                self.moving_joints_id[joint_info.joint_name] = joint_info.joint_index

        self.moving_joints = tuple(self.moving_joints_id.keys())

        # initialize by reset()
        self.start_base_position = None
        self.start_base_orientation = None
        self.filter_state = None
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
                positionGain=self.position_gain,
                physicsClientId=self.client_id
            )

        # stabilization
        for i in range(self.stabilization_time):
            p.stepSimulation(physicsClientId=self.client_id)

        base_info = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        self.start_base_position = base_info[0]
        self.start_base_orientation = base_info[1]
        self.filter_state = np.zeros(len(self.moving_joints))

    def ema_filter(self, values: list[float]) -> list[float]:
        for i, prev in enumerate(self.filter_state):
            values[i] = prev + self.action_filter_alpha * (values[i] - prev)

        self.filter_state = values
        return values


    def apply_action(self, values: list[float]) -> None:
        # filter action
        values = self.ema_filter(values)

        joints_id = []
        joints_target_angles = []
        joints_forces = []
        joints_velocities = []

        for i, joint_name in enumerate(self.moving_joints):
            joint_range = self.moving_joints_limits[joint_name]
            joints_id.append(self.moving_joints_id[joint_name])
            joints_target_angles.append(scale_to_angles(values[i], joint_range))
            joints_forces.append(self.moving_joints_efforts[joint_name])
            joints_velocities.append(self.moving_joints_velocity[joint_name])

        if not self.apply_velocity_limit:
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=joints_id,
                controlMode=p.POSITION_CONTROL,
                targetPositions=joints_target_angles,
                forces=joints_forces,
                positionGains=[self.position_gain for _ in range(len(self.moving_joints))],
                physicsClientId=self.client_id
            )
        else:
            for i in range(len(self.moving_joints)):
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joints_id[i],
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joints_target_angles[i],
                    force=joints_forces[i],
                    positionGain=self.position_gain,
                    maxVelocity=joints_velocities[i],
                    physicsClientId=self.client_id
                )

    def get_toes_contact_points(self):
        toes_contact_points = {}
        for link_name, link_id in self.toes_links_id.items():
            if len(p.getContactPoints(bodyA=self.robot_id, linkIndexA=link_id, physicsClientId=self.client_id)) > 0:
                toes_contact_points[link_name] = 1
            else:
                toes_contact_points[link_name] = 0
        return toes_contact_points

    def get_joints_state(self):
        joints_state = {}
        joint_states = p.getJointStates(self.robot_id, self.moving_joints_id.values(), physicsClientId=self.client_id)
        for i, joint_name in enumerate(self.moving_joints):
            joints_state[joint_name] = joint_states[i][0:2]

        return joints_state

    def get_observation(self) -> np.ndarray:
        base_ang_velocity = list(p.getBaseVelocity(self.robot_id, physicsClientId=self.client_id)[1])
        base_ang_velocity_scaled = [x / self.base_velocity_scale for x in base_ang_velocity]

        orientation = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)[1])
        roll_and_pitch_scaled = [x / self.roll_and_pitch_scale for x in orientation[0:2]]
        yaw_scaled = [orientation[2] / (np.pi / 2)]

        toes_contacts = self.get_toes_contact_points()
        toes_contacts_list = [x for x in toes_contacts.values()]

        joints_state = self.get_joints_state()
        joints_position_list = [scale_to_neat_input(joint[0], self.moving_joints_limits[name]) for name, joint in joints_state.items()]
        joints_velocity_list = [joint[1] / self.joints_velocity_scale for joint in joints_state.values()]

        observation_vector = joints_position_list + joints_velocity_list + roll_and_pitch_scaled + base_ang_velocity_scaled + toes_contacts_list + yaw_scaled

        return np.array(observation_vector)



