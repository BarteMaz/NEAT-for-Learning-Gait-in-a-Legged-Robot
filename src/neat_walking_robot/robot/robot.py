from dataclasses import dataclass

import numpy as np
import pybullet as p


from neat_walking_robot.robot.joints import JointType, get_joints_info


@dataclass(frozen=True, kw_only=True)
class RobotSpec:
    model_path: str
    toes_links_names: tuple[str, str, str, str]
    rest_position: dict[str, float]
    stabilization_time: int
    start_height: float
    use_fixed_base: bool
    toes_friction: float
    body_friction:float
    max_joint_velocity: float
    max_joint_effort: float
    base_velocity_scale: float
    fall_threshold: float
    roll_and_pitch_scale: float



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
    stabilization_time=120,
    base_velocity_scale=5.0,
    use_fixed_base=False,
    toes_friction=0.8,
    body_friction=0.4,
    fall_threshold=0.8,
    roll_and_pitch_scale=0.8,
    max_joint_velocity=21.0,
    max_joint_effort=33.5,
)

def scale_to_velocity(value: float, velocity_range: np.ndarray):
    value = np.clip(value, -1.0, 1.0)
    return velocity_range[0] + (value + 1.0) * 0.5 * (velocity_range[1] - velocity_range[0])

def scale_to_neat_input(angle: float, scaled_range: np.ndarray):
    return 2 * (angle - scaled_range[0]) / (scaled_range[1] - scaled_range[0]) - 1.0

class Robot:
    def __init__(self, spec: RobotSpec, start_pos: list[float], client_id: int):
        self.client_id = client_id

        # specification
        self.model_path = spec.model_path
        self.toes_links_names = spec.toes_links_names
        # load params
        self.rest_position = spec.rest_position
        self.start_height = spec.start_height
        self.stabilization_time = spec.stabilization_time
        # flag for testing
        self.use_fixed_base = spec.use_fixed_base
        # observation scales
        self.base_velocity_scale = spec.base_velocity_scale
        self.fall_threshold = spec.fall_threshold
        self.roll_and_pitch_scale = spec.roll_and_pitch_scale
        # velocity and effort limits
        self.max_joint_velocity = spec.max_joint_velocity
        self.max_joint_effort = spec.max_joint_effort
        # friction
        self.toes_friction = spec.toes_friction
        self.body_friction = spec.body_friction

        # loading
        self.robot_id = p.loadURDF(
            self.model_path,
            useFixedBase=self.use_fixed_base,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            basePosition=[start_pos[0], start_pos[1], start_pos[2] + self.start_height],
            physicsClientId=self.client_id
        )
        p.changeDynamics(
            bodyUniqueId=self.robot_id,
            linkIndex=-1,
            lateralFriction=self.body_friction,
            angularDamping=0.0,
            linearDamping=0.0,
            physicsClientId=self.client_id
        )
        self.reset_base_position = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)[0]

        self.moving_joints_velocity_limits = {}
        self.moving_joints_position_limits = {}
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
                self.set_link_friction(joint_info.joint_index, self.toes_friction)
            else:
                self.set_link_friction(joint_info.joint_index, self.body_friction)

            if joint_info.joint_type == JointType.REVOLUTE:
                self.moving_joints_position_limits[joint_info.joint_name] = [joint_info.joint_lower_limit, joint_info.joint_upper_limit]
                self.moving_joints_velocity_limits[joint_info.joint_name] = [-self.max_joint_velocity, self.max_joint_velocity]
                self.moving_joints_id[joint_info.joint_name] = joint_info.joint_index

        self.moving_joints = tuple(self.moving_joints_id.keys())

        # initialize by reset()
        self.start_base_position = None
        self.start_base_orientation = None
        self.reset()


    def set_link_friction(self, links_id: int, friction: float) -> None:
        p.changeDynamics(
            bodyUniqueId=self.robot_id,
            linkIndex=links_id,
            lateralFriction=friction,
            physicsClientId=self.client_id
        )

    def reset(self) -> None:
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
                force=self.max_joint_effort,
                controlMode=p.POSITION_CONTROL,
                targetPosition=self.rest_position[joint_name],
                positionGain=0.1,
                physicsClientId=self.client_id
            )

        # stabilization
        for i in range(self.stabilization_time):
            p.stepSimulation(physicsClientId=self.client_id)

        base_info = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        self.start_base_position = base_info[0]
        self.start_base_orientation = base_info[1]
        

    def apply_action(self, values: list[float]) -> None:
        
        joints_id = []
        joints_target_velocity = []
        joints_forces = []

        for i, joint_name in enumerate(self.moving_joints):
            joint_velocity_range = self.moving_joints_velocity_limits[joint_name]
            joints_id.append(self.moving_joints_id[joint_name])
            joints_target_velocity.append(scale_to_velocity(values[i], joint_velocity_range))
            joints_forces.append(self.max_joint_effort)

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=joints_id,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=joints_target_velocity,
            forces=joints_forces,
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

    def get_toes_world_position(self):
        toes_world_position = {}
        for link_name, link_id in self.toes_links_id.items():
            toes_world_position[link_name] = p.getLinkState(
                self.robot_id,
                linkIndex=link_id,
                computeForwardKinematics=True,
                physicsClientId=self.client_id
            )[0]
        return toes_world_position

    def get_joints_state(self):
        joints_state = {}
        joint_states = p.getJointStates(self.robot_id, self.moving_joints_id.values(), physicsClientId=self.client_id)
        for i, joint_name in enumerate(self.moving_joints):
            joints_state[joint_name] = joint_states[i][0:2]

        return joints_state

    def get_observation(self) -> np.ndarray:
        quaterion = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)[1]
        orientation = p.getEulerFromQuaternion(quaterion)
        rotate = np.array(p.getMatrixFromQuaternion(quaterion)).reshape(3, 3)

        ang_velocity_vec = np.array(p.getBaseVelocity(self.robot_id, physicsClientId=self.client_id)[1])
        base_ang_velocity = rotate.T @ ang_velocity_vec
        base_ang_velocity_scaled = [x / self.base_velocity_scale for x in base_ang_velocity]

        roll_and_pitch_scaled = [x / self.roll_and_pitch_scale for x in orientation[0:2]]

        toes_contacts = self.get_toes_contact_points()
        toes_contacts_list = [x for x in toes_contacts.values()]

        joints_state = self.get_joints_state()
        joints_position_list = [scale_to_neat_input(joint[0], self.moving_joints_position_limits[name]) for name, joint in joints_state.items()]
        joints_velocity_list = [joint[1] / self.max_joint_velocity for joint in joints_state.values()]

        observation_vector = joints_position_list + joints_velocity_list + roll_and_pitch_scaled + base_ang_velocity_scaled + toes_contacts_list

        return np.array(observation_vector)



