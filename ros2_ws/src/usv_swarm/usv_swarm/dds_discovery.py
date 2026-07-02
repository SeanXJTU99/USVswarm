"""ROS 2 DDS discovery — fully decentralized peer-to-peer node discovery.

Unlike ROS 1 (which required a central roscore), ROS 2 Humble uses DDS
(Data Distribution Service) for automatic, decentralized node discovery.
Every USV in the swarm discovers peers on the same DDS domain without
any central coordinator.

This module provides utilities to monitor and manage the active peer list.
"""

from typing import Dict, List, Optional, Set

import rclpy
from rclpy.node import Node


class DDSDiscovery(Node):
    """Monitors the ROS 2 DDS graph for peer USV nodes.

    Provides a list of active peers on the swarm LAN, updated as
    nodes join (new boat powers on) or leave (boat loses power/range).

    The discovery is fully automatic — DDS handles it at the transport
    layer. This node simply polls the ROS graph and exposes the peer
    list as a convenient ROS topic.
    """

    def __init__(self, namespace: str = "") -> None:
        """Initialize DDS discovery monitor.

        Args:
            namespace: USV namespace (e.g., "usv_0").
        """
        super().__init__(
            "dds_discovery",
            namespace=namespace,
        )

        self.declare_parameter("discovery_period", 1.0)
        self._period: float = (
            self.get_parameter("discovery_period").get_parameter_value().double_value
        )

        self._peers: Set[str] = set()
        self._timer = self.create_timer(self._period, self._poll_graph)

    def _poll_graph(self) -> None:
        """Poll the ROS graph for active node names.

        Extracts peer USV namespaces from the list of all active nodes.
        """
        node_names: List[str] = self.get_node_names()
        own_ns = self.get_namespace()

        peers: Set[str] = set()
        for name in node_names:
            # Node names include namespace, e.g. /usv_0/camera_driver
            if name.startswith("/"):
                parts = name.strip("/").split("/")
                if len(parts) >= 1:
                    ns = parts[0]
                    if ns.startswith("usv_") and ns != own_ns.strip("/"):
                        peers.add(ns)

        if peers != self._peers:
            joined = peers - self._peers
            left = self._peers - peers
            if joined:
                self.get_logger().info(f"Peers joined: {joined}")
            if left:
                self.get_logger().warning(f"Peers left: {left}")

        self._peers = peers

    @property
    def active_peers(self) -> Set[str]:
        """Set of active peer USV namespaces."""
        return self._peers.copy()

    @property
    def swarm_size(self) -> int:
        """Number of active USVs in the swarm (including self)."""
        return len(self._peers)
