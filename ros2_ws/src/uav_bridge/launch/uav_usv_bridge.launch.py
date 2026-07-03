"""Launch UAV-USV heterogeneous cross-domain bridge.

Starts the UAV-side perception, task assignment, visual servoing,
and BEV mapping nodes. USV-side nodes run in the swarm launch.

Architecture:
  UAV (this launch):
    - Overhead YOLO perception (downward camera)
    - Hungarian task assigner (global optimization)
    - Visual servoing controller (per-USV guidance)
    - BEV map builder (global orthomosaic)
    - Star topology hub (central communication)

  USVs (launched separately via swarm_3boat.launch.py):
    - DDS discovery → connected to UAV via star topology
    - Boids/ORCA local planning
    - Role assignment
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Generate launch description for the UAV-USV bridge."""
    uav_altitude = LaunchConfiguration("uav_altitude")
    num_usvs = LaunchConfiguration("num_usvs")

    declare_altitude = DeclareLaunchArgument(
        "uav_altitude",
        default_value="25.0",
        description="UAV flight altitude above water surface (m)",
    )
    declare_num_usvs = DeclareLaunchArgument(
        "num_usvs",
        default_value="3",
        description="Number of USVs in the swarm",
    )

    # UAV overhead perception (downward YOLO)
    overhead_perception = Node(
        package="uav_bridge",
        executable="overhead_perception",
        name="overhead_perception",
        output="screen",
        parameters=[{
            "uav_altitude": uav_altitude,
            "camera_fov_h": 1.2,
            "camera_fov_v": 0.9,
            "image_width": 1920,
            "image_height": 1080,
        }],
    )

    # Hungarian task assigner (global optimization)
    hungarian_assigner = Node(
        package="uav_bridge",
        executable="hungarian_assigner",
        name="hungarian_assigner",
        output="screen",
        parameters=[{
            "max_cost": 50.0,  # meters — ignore tasks further than this
        }],
    )

    # BEV map builder (global orthomosaic)
    bev_mapper = Node(
        package="uav_bridge",
        executable="bev_mapper",
        name="bev_mapper",
        output="screen",
        parameters=[{
            "world_width_m": 100.0,
            "world_height_m": 100.0,
            "resolution": 0.1,
        }],
    )

    # Star topology hub (central communication)
    star_topology = Node(
        package="uav_bridge",
        executable="star_topology",
        name="star_topology",
        output="screen",
        parameters=[{
            "own_id": "uav_0",
            "heartbeat_interval": 0.5,
            "heartbeat_timeout": 3.0,
        }],
    )

    # Visual servoing controllers (one per USV)
    visual_servoing_nodes = []
    for i in range(3):  # Default: 3 USVs
        visual_servoing_nodes.append(
            Node(
                package="uav_bridge",
                executable="visual_servoing",
                name=f"visual_servoing_usv_{i}",
                output="screen",
                parameters=[{
                    "usv_id": f"usv_{i}",
                    "kp_angle": 0.005,
                    "kd_angle": 0.001,
                    "kp_distance": 0.01,
                    "max_correction_angle": 0.3,
                    "dead_zone_px": 5.0,
                }],
            )
        )

    return LaunchDescription(
        [
            declare_altitude,
            declare_num_usvs,
            overhead_perception,
            hungarian_assigner,
            bev_mapper,
            star_topology,
        ]
        + visual_servoing_nodes
    )
