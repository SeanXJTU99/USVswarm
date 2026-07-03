# USV Swarm — Docker image based on ROS 2 Humble Desktop
# Target: one container per USV + one for Gazebo server
FROM osrf/ros:humble-desktop

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # ROS 2 navigation and SLAM
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-cartographer-ros \
    ros-humble-slam-toolbox \
    # Gazebo + VRX
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros2-control \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    # ROS 2 tools
    ros-humble-rviz2 \
    ros-humble-ros2bag \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-tf2-tools \
    # Python
    python3-pip \
    python3-numpy \
    python3-scipy \
    python3-opencv \
    # Networking
    iperf3 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Create workspace
RUN mkdir -p /ros2_ws/src
WORKDIR /ros2_ws

# Copy source code
COPY ros2_ws/src/ /ros2_ws/src/

# Build
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install --parallel-workers 2

# Entrypoint
COPY docker_entrypoint.sh /docker_entrypoint.sh
RUN chmod +x /docker_entrypoint.sh

ENTRYPOINT ["/docker_entrypoint.sh"]
CMD ["usv_0"]
