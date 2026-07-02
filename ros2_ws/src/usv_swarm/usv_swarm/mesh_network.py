"""5.8 GHz Mesh network manager for USV swarm communication.

Manages the ad-hoc wireless mesh topology across the swarm.
Each USV acts as a relay node — if boat A cannot reach boat C directly
due to waves blocking line-of-sight, boat B automatically routes
A's packets to C through the mesh.

Uses a lightweight distance-vector routing protocol running at the
application layer (not kernel IP routing) for maximum portability
across different embedded Linux boards (Jetson, Raspberry Pi, etc.).

Topology is dynamic: boats join/leave the mesh as they power on/off
or move out of radio range (~30-50 m on water with elevated antennas).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import time


@dataclass
class MeshPeer:
    """Information about a peer in the mesh network."""

    node_id: str
    ip_address: str
    last_heartbeat: float = 0.0
    rssi_dbm: float = -100.0    # Signal strength indicator
    hop_count: int = 1           # Hops to reach this peer
    next_hop: Optional[str] = None  # Which peer to route through
    is_connected: bool = False


class MeshNetwork:
    """Application-layer mesh network manager.

    Maintains a routing table for all peers in the swarm and
    provides next-hop lookup for multi-hop message forwarding.

    Uses periodic heartbeat broadcasts to detect topology changes.
    If a peer misses N consecutive heartbeats, it is marked as
    disconnected and routes through it are invalidated.
    """

    HEARTBEAT_INTERVAL: float = 0.5   # seconds
    HEARTBEAT_TIMEOUT: float = 3.0    # seconds before peer marked offline
    MAX_HOPS: int = 4                  # Maximum relay hops

    def __init__(self, own_id: str, own_ip: str) -> None:
        """Initialize mesh network manager.

        Args:
            own_id: This USV's identifier ("usv_0").
            own_ip: This USV's IP address on the mesh interface.
        """
        self.own_id: str = own_id
        self.own_ip: str = own_ip

        # Routing table: peer_id → MeshPeer
        self._peers: Dict[str, MeshPeer] = {}

    def update_peer(
        self,
        peer_id: str,
        ip_address: str,
        rssi: float = -100.0,
        hop_count: int = 1,
        next_hop: Optional[str] = None,
    ) -> None:
        """Add or update a peer in the routing table.

        Args:
            peer_id: Peer USV identifier.
            ip_address: Peer's IP address.
            rssi: Received signal strength (dBm).
            hop_count: Number of hops to peer (1 = direct).
            next_hop: ID of intermediate relay peer, if multi-hop.
        """
        if peer_id == self.own_id:
            return

        if peer_id not in self._peers:
            self._peers[peer_id] = MeshPeer(
                node_id=peer_id,
                ip_address=ip_address,
            )

        peer = self._peers[peer_id]
        peer.ip_address = ip_address
        peer.rssi_dbm = rssi
        peer.hop_count = min(hop_count, self.MAX_HOPS)
        peer.next_hop = next_hop
        peer.last_heartbeat = time.time()
        peer.is_connected = True

    def remove_peer(self, peer_id: str) -> None:
        """Remove a peer from the routing table.

        Args:
            peer_id: Peer to remove.
        """
        self._peers.pop(peer_id, None)
        # Invalidate routes that used this peer as next hop
        for peer in self._peers.values():
            if peer.next_hop == peer_id:
                peer.next_hop = None
                peer.hop_count = self.MAX_HOPS + 1

    def check_timeouts(self) -> List[str]:
        """Check for timed-out peers and mark them disconnected.

        Should be called periodically (~1 Hz).

        Returns:
            List of peer IDs that just timed out.
        """
        now = time.time()
        timed_out: List[str] = []
        for peer_id, peer in self._peers.items():
            if peer.is_connected and (now - peer.last_heartbeat) > self.HEARTBEAT_TIMEOUT:
                peer.is_connected = False
                timed_out.append(peer_id)
        return timed_out

    def get_route(self, target_id: str) -> Optional[str]:
        """Get the next-hop IP address to reach a target peer.

        If the peer is directly reachable, returns its IP.
        If multi-hop, returns the next-hop peer's IP.

        Args:
            target_id: Destination peer identifier.

        Returns:
            IP address to forward the packet to, or None if unreachable.
        """
        peer = self._peers.get(target_id)
        if peer is None or not peer.is_connected:
            return None

        if peer.hop_count == 1:
            return peer.ip_address

        if peer.next_hop:
            next_peer = self._peers.get(peer.next_hop)
            if next_peer and next_peer.is_connected:
                return next_peer.ip_address

        return None

    @property
    def active_peers(self) -> List[MeshPeer]:
        """List of currently connected peers."""
        return [p for p in self._peers.values() if p.is_connected]

    @property
    def peer_count(self) -> int:
        """Number of connected peers."""
        return len(self.active_peers)

    def broadcast_addresses(self) -> List[str]:
        """Get IP addresses of all directly connected peers.

        Returns:
            List of IP addresses (direct 1-hop peers only).
        """
        return [
            p.ip_address
            for p in self._peers.values()
            if p.is_connected and p.hop_count == 1
        ]
