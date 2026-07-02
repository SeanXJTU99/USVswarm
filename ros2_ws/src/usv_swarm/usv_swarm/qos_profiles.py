"""QoS profiles for multi-USV swarm communication over lossy water-surface links.

The water surface is a harsh RF environment:
  - Multipath interference causes frequent packet loss
  - Antenna height is only 30-50 cm above water
  - 5.8 GHz mesh has ~20 Mbps shared bandwidth

QoS strategy splits traffic into two tiers:

  Tier 1 — Best Effort (sensor/pose data):
    - GPS coordinates, IMU attitude, YOLO detections
    - High frequency (~20 Hz), loss-tolerant
    - If a pose update drops, the next one arrives in 50 ms
    - Saves bandwidth by not retransmitting stale data

  Tier 2 — Reliable (control commands):
    - Waypoint assignments, role changes, emergency stop
    - Low frequency, must-not-drop
    - TCP-like reliability with DDS reliability QoS

Reference: ROS 2 QoS policies map to DDS reliability, durability, and
deadline QoS attributes.
"""

from rclpy.qos import (
    QoSPresetProfiles,
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
    LivelinessPolicy,
)


def best_effort_pose_qos() -> QoSProfile:
    """Best Effort QoS for high-frequency pose/sensor data.

    Use for: /usv_<id>/pose, /usv_<id>/odom, /usv_<id>/detections

    Returns:
        QoSProfile with BEST_EFFORT reliability, VOLATILE durability.
    """
    profile = QoSProfile(
        depth=5,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        liveliness=LivelinessPolicy.AUTOMATIC,
    )
    return profile


def reliable_command_qos() -> QoSProfile:
    """Reliable QoS for low-frequency control messages.

    Use for: /usv_<id>/waypoint, /usv_<id>/role, /swarm/emergency_stop

    Returns:
        QoSProfile with RELIABLE reliability, TRANSIENT_LOCAL durability.
    """
    profile = QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        liveliness=LivelinessPolicy.AUTOMATIC,
        lifespan=rclpy.duration.Duration(seconds=2.0),  # stale after 2s
    )
    return profile


def swarm_discovery_qos() -> QoSProfile:
    """QoS for swarm heartbeat/discovery beacons.

    Periodic, best-effort heartbeats to keep the swarm topology fresh
    without flooding the shared 5.8 GHz channel.

    Returns:
        QoSProfile for heartbeat topics.
    """
    profile = QoSProfile(
        depth=3,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
    )
    return profile
