#!/bin/bash
# Docker entrypoint — launches the appropriate ROS 2 nodes for this USV
# Usage: docker_entrypoint.sh <usv_id>

set -e

USV_ID="${1:-usv_0}"
USV_ROLE="${USV_ROLE:-worker}"

source /opt/ros/humble/setup.sh
source /ros2_ws/install/setup.sh

echo "Starting ${USV_ID} (role: ${USV_ROLE})"

# Common nodes for all USVs
ros2 launch usv_bringup single_boat_sim.launch.py &

# Role-specific nodes
if [ "$USV_ROLE" = "scout" ]; then
    echo "Launching scout perception stack..."
    ros2 run usv_perception yolo_detector &
    ros2 run usv_calibration imu_dejitter &
fi

# Swarm coordination
ros2 run usv_swarm dds_discovery &
ros2 run usv_swarm boids_controller &
ros2 run usv_swarm orca_avoidance &
ros2 run usv_swarm role_assignment &

# Keep container alive
wait
