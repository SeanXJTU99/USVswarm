"""Star-topology communication network: UAV as central hub for USV fleet.

Replaces the water-surface mesh network with a cleaner star topology:
  - UAV is the central node (hub)
  - Each USV connects directly to the UAV (spoke)
  - No USV-to-USV direct communication needed

This eliminates the multipath interference problem because:
  1. UAV antenna is elevated 20-30 m above water → clear LOS to all USVs
  2. Radio waves travel vertically → surface reflection does not interfere
     with the direct path (unlike horizontal water-surface propagation)
  3. Centralized routing is simpler and more predictable than mesh routing

Trade-off: single point of failure (the UAV). Mitigated by:
  - Fallback to mesh mode if UAV is unavailable
  - UAV battery monitoring with automatic return-to-mothership
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import time


@dataclass
class StarNode:
    """A node (USV) in the star topology."""

    node_id: str
    ip_address: str
    role: str                      # "scout" or "worker"
    last_heartbeat: float = 0.0
    rssi_dbm: float = -100.0
    battery_pct: float = 100.0
    position: Tuple[float, float] = (0.0, 0.0)
    is_connected: bool = False


class StarTopology:
    """Star-topology network manager (runs on UAV).

    The UAV maintains the list of connected USVs, routes messages
    between them, and monitors link quality.

    On each USV, a lightweight client registers with the UAV hub
    and sends periodic heartbeats.
    """

    HEARTBEAT_INTERVAL: float = 0.5    # seconds
    HEARTBEAT_TIMEOUT: float = 3.0     # seconds
    MAX_USVS: int = 10                  # Maximum supported spokes

    def __init__(self, own_id: str = "uav_0") -> None:
        """Initialize star topology hub (runs on UAV).

        Args:
            own_id: UAV identifier.
        """
        self.own_id: str = own_id
        self._nodes: Dict[str, StarNode] = {}

    def register_node(
        self,
        node_id: str,
        ip_address: str,
        role: str = "worker",
    ) -> None:
        """Register a new USV spoke or update an existing one.

        Args:
            node_id: USV identifier ("usv_0", etc.).
            ip_address: USV IP address on the wireless network.
            role: Operational role ("scout" or "worker").
        """
        if node_id == self.own_id:
            return

        if node_id not in self._nodes:
            self._nodes[node_id] = StarNode(
                node_id=node_id,
                ip_address=ip_address,
                role=role,
            )

        node = self._nodes[node_id]
        node.ip_address = ip_address
        node.role = role
        node.last_heartbeat = time.time()
        node.is_connected = True

    def update_heartbeat(
        self,
        node_id: str,
        rssi: float = -100.0,
        battery_pct: float = 100.0,
        position: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Update a USV's status on heartbeat reception.

        Args:
            node_id: USV identifier.
            rssi: Received signal strength (dBm).
            battery_pct: Battery level percentage.
            position: USV position (x, y) in world frame.
        """
        node = self._nodes.get(node_id)
        if node is None:
            return

        node.rssi_dbm = rssi
        node.battery_pct = battery_pct
        node.position = position
        node.last_heartbeat = time.time()
        node.is_connected = True

    def check_timeouts(self) -> List[str]:
        """Check for spoke USVs that timed out.

        Returns:
            List of USV IDs that just disconnected.
        """
        now = time.time()
        disconnected: List[str] = []

        for node_id, node in self._nodes.items():
            if node.is_connected and (now - node.last_heartbeat) > self.HEARTBEAT_TIMEOUT:
                node.is_connected = False
                disconnected.append(node_id)

        return disconnected

    def remove_node(self, node_id: str) -> None:
        """Explicitly remove a USV from the topology.

        Args:
            node_id: USV to deregister.
        """
        self._nodes.pop(node_id, None)

    def get_route(self, target_id: str) -> Optional[str]:
        """Get the IP address to reach a target USV.

        In star topology, ALL routes go through the UAV:
          - Sender → UAV → Receiver

        So this returns the target's IP (UAV relays transparently).

        Args:
            target_id: Destination USV identifier.

        Returns:
            Target IP address, or None if unreachable.
        """
        node = self._nodes.get(target_id)
        if node and node.is_connected:
            return node.ip_address
        return None

    def broadcast_to_all(self, exclude: Optional[List[str]] = None) -> List[str]:
        """Get IP addresses of all connected USVs for broadcast.

        Args:
            exclude: List of node IDs to exclude from broadcast.

        Returns:
            List of IP addresses.
        """
        exclude_set = set(exclude or [])
        return [
            node.ip_address
            for node in self._nodes.values()
            if node.is_connected and node.node_id not in exclude_set
        ]

    @property
    def connected_nodes(self) -> List[StarNode]:
        """List of currently connected USV spokes."""
        return [n for n in self._nodes.values() if n.is_connected]

    @property
    def node_count(self) -> int:
        """Number of connected USVs."""
        return len(self.connected_nodes)
