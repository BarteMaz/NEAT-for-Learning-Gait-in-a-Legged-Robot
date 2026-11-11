import pybullet as p
import numpy as np

from dataclasses import dataclass
from enum import IntEnum

class JointType(IntEnum):
    FIXED = p.JOINT_FIXED
    PRISMATIC = p.JOINT_PRISMATIC
    REVOLUTE = p.JOINT_REVOLUTE
    SPHERICAL = p.JOINT_SPHERICAL
    PLANAR = p.JOINT_PLANAR

class ShapeType(IntEnum):
    SPHERE = p.GEOM_SPHERE
    BOX = p.GEOM_BOX
    CYLINDER = p.GEOM_CYLINDER
    MESH = p.GEOM_MESH
    CAPSULE = p.GEOM_CAPSULE

@dataclass(frozen=True, kw_only=True)
class JointInfo:
    joint_index: int
    joint_name: str
    joint_type: JointType
    joint_axis: tuple[float, float, float]
    q_index: int
    joint_lower_limit: float
    joint_upper_limit: float
    effort: float
    velocity: float
    parent_index: int
    parent_name: str
    link_name: str
    link_collision_shape: tuple[ShapeType, ...]
    link_mass: float
    link_friction: float

    def __str__(self):
        if self.joint_lower_limit > self.joint_upper_limit:
            limits = "joint angular limits: None\n"
        else:
            limits = f"joint angular limits: ({np.rad2deg(self.joint_lower_limit):.1f} deg, {np.rad2deg(self.joint_upper_limit):.1f} deg)\n"

        shapes = [shape.name for shape in self.link_collision_shape]

        return (f"joint index: {self.joint_index}\n"
                f"joint name: {self.joint_name}\n"
                f"joint type: {self.joint_type.name}\n"
                f"joint axis: {self.joint_axis}\n"
                f"q index: {self.q_index}\n"
                f"{limits}"
                f"joint effort: {self.effort}\n"
                f"joint velocity: {self.velocity}\n"
                f"parent index: {self.parent_index}\n"
                f"parent name: {self.parent_name}\n"
                f"link name: {self.link_name}\n"
                f"link collision shape: {shapes}\n"
                f"link mass: {self.link_mass}\n"
                f"link friction: {self.link_friction}")

def get_joint_info(robot_id, joint_id, client_id) -> JointInfo:
    base_name = p.getBodyInfo(robot_id, physicsClientId=client_id)[0].decode()
    joint_info = p.getJointInfo(robot_id, joint_id, physicsClientId=client_id)

    joint_index = joint_info[0]
    joint_name = joint_info[1].decode()
    joint_type = joint_info[2]
    q_index = joint_info[3]
    joint_axis = joint_info[13]
    joint_lower_limit = joint_info[8]
    joint_upper_limit = joint_info[9]

    effort = joint_info[10]
    velocity = joint_info[11]

    parent_index = joint_info[16]
    if parent_index == -1:
        parent_name = base_name
    else:
        parent_name = p.getJointInfo(robot_id, parent_index, physicsClientId=client_id)[12].decode()

    link_dynamics = p.getDynamicsInfo(robot_id, joint_id, physicsClientId=client_id)
    link_name = joint_info[12].decode()

    collision_data = p.getCollisionShapeData(robot_id, joint_id, physicsClientId=client_id)

    collision_shapes = []
    for collision in collision_data:
        collision_shapes.append(ShapeType(collision[2]))

    link_shape = tuple(collision_shapes)
    link_mass = link_dynamics[0]
    link_friction = link_dynamics[1]

    return JointInfo(
        joint_index=joint_index,
        joint_name=joint_name,
        joint_type=JointType(joint_type),
        joint_axis=joint_axis,
        q_index=q_index,
        joint_lower_limit=joint_lower_limit,
        joint_upper_limit=joint_upper_limit,
        effort=effort,
        velocity=velocity,
        parent_index=parent_index,
        parent_name=parent_name,
        link_name=link_name,
        link_collision_shape=link_shape,
        link_mass=link_mass,
        link_friction=link_friction,
    )


def get_joints_info(robot_id, client_id) -> dict[int, JointInfo]:
    num_joints = p.getNumJoints(robot_id, physicsClientId=client_id)
    joints_info = {}
    for i in range(num_joints):
        joints_info[i] = get_joint_info(robot_id, i, client_id)
    return joints_info

def total_mass(robot_id, client_id):
    joints_info = get_joints_info(robot_id, client_id)
    mass = 0
    for v in joints_info.values():
        mass += v.link_mass
    mass += p.getDynamicsInfo(robot_id, -1, physicsClientId=client_id)[0]
    return mass

def joint_table(robot_id, client_id):
    joints_info = get_joints_info(robot_id, client_id)
    for v in joints_info.values():
        print(v)
        print("--------------------------------------")





