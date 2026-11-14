"""Tests for the joint inspection helpers in neat_walking_robot.robot.joints.

The A1 model from pybullet_data is used as a real robot with known structure:
21 joints (12 revolute, 9 fixed), link names decoded to str, spherical toes.
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pybullet as p
import pybullet_data
import pytest

from neat_walking_robot.robot.joints import (
    JointType,
    ShapeType,
    get_joint_info,
    get_joints_info,
    total_mass,
)

A1_URDF = "a1/a1.urdf"


def load_a1(client_id: int, fixed_base: bool = False) -> int:
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    return p.loadURDF(
        A1_URDF,
        useFixedBase=fixed_base,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
        physicsClientId=client_id,
    )


def a1_mass_from_urdf() -> float:
    """Independent oracle: sum of all <mass value> entries in the A1 URDF file."""
    root = ET.parse(Path(pybullet_data.getDataPath()) / A1_URDF).getroot()
    return sum(float(m.get("value")) for m in root.iter("mass"))


# A two-link URDF whose child link has no <collision>, written to a temp file.
NO_COLLISION_URDF = """<?xml version="1.0"?>
<robot name="no_collision">
  <link name="base">
    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
    <inertial><mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
  </link>
  <joint name="hinge" type="revolute">
    <parent link="base"/><child link="arm"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <link name="arm">
    <inertial><mass value="0.5"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
  </link>
</robot>
"""


@pytest.fixture
def no_collision_urdf(tmp_path: Path) -> str:
    path = tmp_path / "no_collision.urdf"
    path.write_text(NO_COLLISION_URDF)
    return str(path)


def test_a1_joint_counts(client):
    """A1 has 21 joints: 12 revolute (3 per leg) and 9 fixed (imu, shoulders, toes)."""
    infos = get_joints_info(load_a1(client), client)
    types = [info.joint_type for info in infos.values()]
    assert len(infos) == 21
    assert types.count(JointType.REVOLUTE) == 12
    assert types.count(JointType.FIXED) == 9


def test_names_are_decoded_strings(client):
    """Names must be str, not bytes: b'FR_toe' == 'FR_toe' is silently False."""
    for info in get_joints_info(load_a1(client), client).values():
        assert isinstance(info.joint_name, str)
        assert isinstance(info.link_name, str)
        assert isinstance(info.parent_name, str)


def test_parent_name_is_link_name_not_joint_name(client):
    """The parent of FR_upper_joint is the link FR_hip (index 1), not the joint FR_hip_joint."""
    infos = {info.joint_name: info for info in get_joints_info(load_a1(client), client).values()}
    assert infos["FR_upper_joint"].parent_index == 1
    assert infos["FR_upper_joint"].parent_name == "FR_hip"
    assert infos["FR_hip_joint"].parent_index == -1
    assert infos["FR_hip_joint"].parent_name == "trunk"


def test_revolute_limits_match_urdf(client):
    """Limits stay in radians; the thigh range of A1 is -60 deg to 240 deg."""
    infos = {info.joint_name: info for info in get_joints_info(load_a1(client), client).values()}
    upper = infos["FR_upper_joint"]
    assert upper.joint_lower_limit == pytest.approx(math.radians(-60), abs=1e-6)
    assert upper.joint_upper_limit == pytest.approx(math.radians(240), abs=1e-6)
    assert upper.effort == pytest.approx(55.0)
    assert infos["FR_hip_joint"].effort == pytest.approx(20.0)


def test_fixed_joints_report_no_limits(client):
    """PyBullet marks 'no limits' with upper < lower, e.g. (0.0, -1.0)."""
    for info in get_joints_info(load_a1(client), client).values():
        if info.joint_type == JointType.FIXED:
            assert info.joint_upper_limit < info.joint_lower_limit


def test_toes_are_spheres(client):
    infos = {info.link_name: info for info in get_joints_info(load_a1(client), client).values()}
    for toe in ("FR_toe", "FL_toe", "RR_toe", "RL_toe"):
        assert infos[toe].link_collision_shape == (ShapeType.SPHERE,)


def test_link_without_collision_gives_empty_tuple(client, no_collision_urdf):
    """A missing <collision> (e.g. the '<collison>' typo) must show up as (), not crash."""
    robot = p.loadURDF(no_collision_urdf, physicsClientId=client)
    info = get_joint_info(robot, 0, client)
    assert info.link_collision_shape == ()


def test_total_mass_matches_urdf_for_free_base(client):
    """Sum of base and all links equals the sum of masses written in the URDF (about 12.46 kg)."""
    robot = load_a1(client)
    assert total_mass(robot, client) == pytest.approx(a1_mass_from_urdf(), rel=1e-6)


def test_fixed_base_reports_zero_base_mass(client):
    """Documents a PyBullet trap: with useFixedBase the base becomes static and its mass reads 0."""
    robot = load_a1(client, fixed_base=True)
    assert p.getDynamicsInfo(robot, -1, physicsClientId=client)[0] == 0.0
    assert total_mass(robot, client) < a1_mass_from_urdf() - 4.0


def test_functions_use_the_given_client(two_clients, no_collision_urdf):
    """Both servers hold a body with id 0; a call without physicsClientId reads the wrong one."""
    first, second = two_clients
    small = p.loadURDF(no_collision_urdf, physicsClientId=first)
    a1 = load_a1(second)
    assert small == a1 == 0

    infos = get_joints_info(a1, second)
    assert len(infos) == 21
    assert infos[3].parent_name == "FR_hip"
    assert total_mass(a1, second) == pytest.approx(a1_mass_from_urdf(), rel=1e-6)
