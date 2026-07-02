"""Launch Cartographer 2D SLAM for USV."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Generate launch description for Cartographer 2D SLAM."""
    config_dir = PathJoinSubstitution(
        [FindPackageShare("usv_slam"), "config"]
    )
    config_file = LaunchConfiguration("config_file")

    declare_config = DeclareLaunchArgument(
        "config_file",
        default_value=PathJoinSubstitution([config_dir, "cartographer_2d.lua"]),
        description="Cartographer 2D configuration file",
    )

    cartographer_node = Node(
        package="cartographer_ros",
        executable="cartographer_node",
        name="cartographer_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
        arguments=[
            "-configuration_directory", config_dir,
            "-configuration_basename", "cartographer_2d.lua",
        ],
    )

    occupancy_grid_node = Node(
        package="cartographer_ros",
        executable="occupancy_grid_node",
        name="occupancy_grid_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
        arguments=["-resolution", "0.05"],
    )

    return LaunchDescription([
        declare_config,
        cartographer_node,
        occupancy_grid_node,
    ])
