## What is an URDF File? (General Structure)

An URDF (Unified Robitics Description Format) file is na `.xml` file used to describe the visual, physics, and connections of a robot with joints and links. Below is the general sctructure of an URDF file

```xml
<xml version="1.0">
<robot name="robot_name">
    <link name="link_name">
        <visual>
        </visual>

        <collision>
        </collision>

        <inertial>
        </inertial>
    </link>

    <joint name="join_name"...>
        <axis .../>
        <parent link .../>
        <child link .../>
        <origin .../>
    </joint>
</robot>
```

## Links in URDF

A link is used to describe a segment of a robot, where you can define the `visual`, `collision` and `inertial` properties. The `geometry` and `origin` properties are used in `visual` and `collision` properties to describe the shapes and location of the link.

### General Structure of a Link

```xml
<link name="link_name">
    <visual> <!-- OPTIONAL -->
        <geometry> <!-- REQUIRED -->
            <!-- Geometry Specific Code -->
        </geometry>
        <origin rpy="r p y" xyz="x y z"/> <!-- OPTIONAL -->
        <material name="color_name"/> <!-- OPTIONAL -->
    </visual>

    <collision> <!-- OPTIONAL -->
        <geometry> <!-- REQUIRED -->
            <!-- Geometry Specific Code -->
        </geometry>
        <origin rpy="r p y" xyz="x y z"/> <!-- OPTIONAL -->
    </collsion>

    <inertial>
        <mass value="mass_in_kg"/>
        <inertia
            ixx="ixx" ixy="ixy" ixz="ixz"
            iyy="iyy" iyz="iyz"
            izz="izz"
        />
    </inertial>
</link>
```

### Visual

Visual is used to describe the geometry of the link. Usually this can be an accurate representation of the robot or just a simplified version depending on the goal. For accurate models, one can obtain the model from a 3D modeling software like Solidworks and export the `.stl` files for the geometry

### Types of Geometries used in URDF File

Units for the geometries are in SI units, so meters for all lengths. For the mesh, the files types can be `.stl` or `.dea` files

```xml
<geometry>
    <!-- Geometry Specific Code -->
</geometry>
```

| **Geometry** | **Geometry Specific Code**                   |
| ------------ | -------------------------------------------- |
| box          | `<box size="z y z"/>`                        |
| sphere       | `<sphere radius="r"/>`                       |
| cylinder     | `<cylinder radius="r" lenght="l"/>`          |
| mesh         | `<mesh filename="file_path" scale="x y z"/>` |

### Origin - rpy and xyz

The origin is used to define the rotation [radians] using roll (x), pitch (y), yaw (z) and the traslation [meters] using x, y, z relative to the link's coordinate frame. The order of rotation for rpy is ZYX, so first rotate about X axis, then Y axis and finally Z axis. _The rotation operation is applied first and then the translation._

For `visual` and `collision`, `origin` describes how the geometry is transformed relative to the link's frame.

For `interval`, `origin` describes the location of the center of mass relative to the link's frame.

```xml
<origin rpy="r p y" xyz="x y z"/>
```

Axis Colors:

| **Axis** | **Color** |
| -------- | --------- |
| X        | Red       |
| Y        | Green     |
| Z        | Blue      |

### Material

The material is used to define the colors and the transparency of the color by using `rgba`, normalized between 0 and 1.

```xml
<material name="color_name">
    <color rgba="r g b a"/>
</material>
```

Some common colors:

| `color_name` | `r g b a` |
| ------------ | --------- |
| red          | 1 0 0 1   |
| green        | 0 1 0 1   |
| blue         | 0 0 1 1   |
| yellow       | 1 1 0 1   |
| cyan         | 0 1 1 1   |

### Collision

The Collision property can be used to describe a boudning shape around the link for collision detection. Usually, peopel use a simplified shape for this for faster calculation when detecting for collisions between other collision boudnig shapes.

```xml
<collision> <!-- OPTIONAL -->
        <geometry> <!-- REQUIRED -->
            <!-- Geometry Specific Code -->
        </geometry>
        <origin rpy="r p y" xyz="x y z"/> <!-- OPTIONAL -->
    </collsion>
```

### Inertial

The `inertial` property is used to descibe the inertia [$kg \cdot m^2$] about the link's center of mass and the mass [kg] of the link.

```xml
<inertial>
        <mass value="mass_in_kg"/>
        <inertia
            ixx="ixx" ixy="ixy" ixz="ixz"
            iyy="iyy" iyz="iyz"
            izz="izz"
        />
    </inertial>
```

## Joints in URDF Files

Joints are used to desribe the connections between two links, a `parent link` and a `child link`.

### General Structure of a Joint

```xml
<joint name="joint_name" type="joint_type">
    <!-- OPTIONAL -->
        <parent link="parent_link_name"/>
        <child link="parent_link_name"/>
    <!-- REQUIRED -->
        <origin rpy="r p y" xyz="x y z"/>
        <axis rpy="r p y" xyz="x y z"/>
        <limit lower="lower" upper="upper" effort="max_effort" velocity="max_vel"/>
        <dynamics damping="damp_coef" friction="fric_coef"/>
</joint>
```

### Types of Joints

Options for `joint_type`

- `fixed` - no motions between links
- `continuous` - rotations about axis, not limit
- `revolute` - rotation about acis, with limits (in radians)
- `prismatic` - translation about axis
- `floating` 6 DOF (3 translations, 3 rotations)
- `planar` motions on plane (2 translations, 1 rotation)

### Parent and Child Link

The `parent link` is the link before the joint and the `child link` is the link after the joint.

```xml
<parent link="parent_link_name"/>
<child link="parent_link_name"/>
```

### Origin and Axis

The `origin` describes the locations of the child frame relative to the parent frame. The `axis` describes the axis of rotation for the joint.

```xml
<origin rpy="r p y" xyz="x y z"/>
<axis rpy="r p y" xyz="x y z"/>
```

### Dynamics

The dynamics include `damping` expressed in [N s/m] and `friction` (static friction) expressed in [N]

```xml
<dynamics damping="damp_coef" friction="fric_coef"/>
```

### Limit

Limits are for `revolute` and `prismatic` joints only. The `lower` and `upper` field are in radins or meters respectively, the `effort` is max effort in [Nm], and `velocity` is max velocity [rad/s].

```xml
<limit lower="lower" upper="upper" effort="max_effort" velocity="max_vel"/>
```
