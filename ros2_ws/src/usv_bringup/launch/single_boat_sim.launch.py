"""Launch single USV simulation with Gazebo and RViz."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Generate launch description for single-boat simulation."""
    world_file = LaunchConfiguration("world_file")
    use_rviz = LaunchConfiguration("use_rviz")

    declare_world = DeclareLaunchArgument(
        "world_file",
        default_value=PathJoinSubstitution(
            [FindPackageShare("usv_gazebo"), "worlds", "water_surface.world"]
        ),
        description="Gazebo world file path",
    )

    declare_rviz = DeclareLaunchArgument(
        "use_rviz", default_value="true", description="Launch RViz2"
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

    # Spawn USV
    spawn_usv = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity", "usv_0",
            "-file", PathJoinSubstitution(
                [FindPackageShare("usv_description"), "urdf", "usv_hull.urdf"]
            ),
            "-x", "0.0", "-y", "0.0", "-z", "0.5",
            "-R", "0.0", "-P", "0.0", "-Y", "0.0",
        ],
        output="screen",
    )

    # Robot state publisher
    robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{
            "robot_description": PathJoinSubstitution(
                [FindPackageShare("usv_description"), "urdf", "usv_hull.urdf"]
            ),
        }],
    )

    # RViz2
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        condition=lambda: use_rviz == "true",
        arguments=["-d", PathJoinSubstitution(
            [FindPackageShare("usv_bringup"), "config", "usv.rviz"]
        )],
    )

    return LaunchDescription([
        declare_world,
        declare_rviz,
        gazebo,
        spawn_usv,
        robot_state_pub,
        rviz,
    ])
