"""Launch multi-USV swarm simulation with Gazebo."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Generate launch description for 3-boat swarm simulation."""
    world_file = LaunchConfiguration("world_file")
    num_boats = LaunchConfiguration("num_boats")

    declare_world = DeclareLaunchArgument(
        "world_file",
        default_value=PathJoinSubstitution(
            [FindPackageShare("usv_gazebo"), "worlds", "water_surface.world"]
        ),
        description="Gazebo world file path",
    )

    declare_num = DeclareLaunchArgument(
        "num_boats", default_value="3", description="Number of USVs in the swarm"
    )

    # Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("gazebo_ros"), "launch", "gazebo.launch.py"]
            )
        ),
        launch_arguments={"world": world_file}.items(),
    )

    # Spawn 3 USVs at different positions
    spawn_positions = [
        ("usv_0", "0.0", "0.0", "0.0"),
        ("usv_1", "3.0", "0.0", "0.0"),
        ("usv_2", "0.0", "3.0", "0.0"),
    ]

    spawn_nodes = []
    for name, x, y, yaw in spawn_positions:
        spawn_nodes.append(Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            arguments=[
                "-entity", name,
                "-file", PathJoinSubstitution(
                    [FindPackageShare("usv_description"), "urdf", "usv_hull.urdf"]
                ),
                "-x", x, "-y", y, "-z", "0.5",
                "-R", "0.0", "-P", "0.0", "-Y", yaw,
            ],
            output="screen",
        ))

    # Robot state publisher (one per boat)
    robot_state_pubs = []
    for i in range(3):
        robot_state_pubs.append(Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            namespace=f"usv_{i}",
            parameters=[{
                "robot_description": PathJoinSubstitution(
                    [FindPackageShare("usv_description"), "urdf", "usv_hull.urdf"]
                ),
            }],
        ))

    return LaunchDescription(
        [declare_world, declare_num, gazebo]
        + spawn_nodes
        + robot_state_pubs
    )
