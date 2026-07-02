"""Launch a 3-boat decentralized USV swarm simulation.

Each boat runs in its own ROS 2 namespace with:
  - DDS peer discovery (automatic, no roscore)
  - Boids flocking + ORCA collision avoidance
  - Distributed SLAM map fusion
  - Leader-follower or consensus formation (configurable)

Boat roles:
  - usv_0: Scout (camera + Jetson)
  - usv_1: Worker (navigation only)
  - usv_2: Worker (navigation only)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Generate launch description for 3-boat swarm simulation."""
    formation_mode = LaunchConfiguration("formation_mode")

    declare_formation = DeclareLaunchArgument(
        "formation_mode",
        default_value="boids",
        description="Swarm formation mode: 'boids', 'leader_follower', or 'consensus'",
    )

    # Base simulation (Gazebo + 3 boats) from usv_bringup
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("usv_bringup"), "launch", "multi_boat_sim.launch.py"]
            )
        ),
    )

    # Per-boat swarm stack
    boat_ids = ["usv_0", "usv_1", "usv_2"]
    boat_groups = []

    for i, boat_id in enumerate(boat_ids):
        group = GroupAction([
            PushRosNamespace(boat_id),

            # DDS discovery monitor
            Node(
                package="usv_swarm",
                executable="dds_discovery",
                name="dds_discovery",
                output="screen",
            ),

            # Boids flocking controller
            Node(
                package="usv_swarm",
                executable="boids_controller",
                name="boids_controller",
                output="screen",
                parameters=[{
                    "cohesion_weight": 1.0,
                    "alignment_weight": 0.8,
                    "separation_weight": 1.5,
                    "perception_radius": 15.0,
                    "separation_radius": 3.0,
                }],
            ),

            # ORCA collision avoidance
            Node(
                package="usv_swarm",
                executable="orca_avoidance",
                name="orca_avoidance",
                output="screen",
                parameters=[{
                    "time_horizon": 5.0,
                    "radius": 0.5,
                    "max_speed": 1.5,
                }],
            ),

            # Distributed SLAM fusion
            Node(
                package="usv_swarm",
                executable="distributed_slam",
                name="distributed_slam",
                output="screen",
            ),

            # Role assignment
            Node(
                package="usv_swarm",
                executable="role_assignment",
                name="role_assignment",
                output="screen",
                parameters=[{
                    "initial_role": "scout" if i == 0 else "worker",
                }],
            ),

            # Consensus protocol (if enabled)
            Node(
                package="usv_swarm",
                executable="consensus",
                name="consensus",
                output="screen",
                condition=lambda ctx: ctx.launch_configurations.get(
                    "formation_mode"
                ) == "consensus",
            ),

            # Leader-Follower (if enabled)
            Node(
                package="usv_swarm",
                executable="leader_follower",
                name="leader_follower",
                output="screen",
                parameters=[{
                    "is_leader": i == 0,
                    "formation_distance": 3.0,
                    "formation_angle": 0.0,
                }],
                condition=lambda ctx: ctx.launch_configurations.get(
                    "formation_mode"
                ) == "leader_follower",
            ),

            # Mesh network manager
            Node(
                package="usv_swarm",
                executable="mesh_network",
                name="mesh_network",
                output="screen",
            ),
        ])
        boat_groups.append(group)

    return LaunchDescription(
        [declare_formation, sim] + boat_groups
    )
