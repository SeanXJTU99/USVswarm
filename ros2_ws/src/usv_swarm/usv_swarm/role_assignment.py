"""Role assignment for heterogeneous USV swarm.

In a cost-optimized swarm (3-5 boats), not every vessel needs expensive
sensors. This module implements role-based task allocation:

  - Scout (感知船): Carries camera + Jetson for YOLO perception.
    Broadcasts obstacle detections to the swarm.
  - Worker (作业船): Carries only GPS + IMU + thruster control.
    Receives shared perception data, focuses on navigation.

Roles can be reassigned dynamically based on:
  - Battery level (low battery → relinquish scout role)
  - Sensor health (camera failure → downgrade to worker)
  - Swarm composition (ensure at least 1 scout in the swarm)

Role assignment uses a distributed election protocol — no central
arbiter needed.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import time


class Role(Enum):
    """USV operational role."""

    SCOUT = "scout"      # Perception + broadcast
    WORKER = "worker"    # Navigation only
    RELAY = "relay"      # Communication relay (no sensors, no work)
    IDLE = "idle"        # Not assigned


@dataclass
class USVStatus:
    """Status report from a single USV."""

    usv_id: str
    role: Role
    battery_pct: float          # 0.0 - 100.0
    camera_healthy: bool
    gps_healthy: bool
    cpu_load_pct: float         # 0.0 - 100.0
    timestamp: float = field(default_factory=time.time)


class RoleAssigner:
    """Distributed role assignment coordinator.

    Each USV runs one instance. All instances share status via the
    mesh network and independently converge to the same role
    assignment (deterministic algorithm with identical inputs).
    """

    # Minimum number of scouts desired in the swarm
    MIN_SCOUTS: int = 1

    # Battery threshold below which a scout should step down
    LOW_BATTERY_THRESHOLD: float = 20.0

    def __init__(self, own_id: str) -> None:
        """Initialize role assigner.

        Args:
            own_id: This USV's identifier.
        """
        self.own_id: str = own_id
        self._swarm_status: Dict[str, USVStatus] = {}

    def update_status(self, status: USVStatus) -> None:
        """Update the known status of a USV in the swarm.

        Args:
            status: Status report from any USV (including self).
        """
        self._swarm_status[status.usv_id] = status

        # Remove stale entries (older than 10 seconds)
        now = time.time()
        stale = [
            uid for uid, s in self._swarm_status.items()
            if now - s.timestamp > 10.0
        ]
        for uid in stale:
            del self._swarm_status[uid]

    def assign_role(self, own_status: USVStatus) -> Role:
        """Determine the optimal role for this USV.

        Algorithm (deterministic, runs identically on all USVs):
          1. If own camera is unhealthy → WORKER or RELAY
          2. If own battery is low → step down from SCOUT to WORKER
          3. Count scout-capable USVs. If fewer than MIN_SCOUTS,
             elect scouts by highest battery among capable.
          4. Remaining scout-capable USVs become WORKER or RELAY.

        Args:
            own_status: This USV's current status.

        Returns:
            Assigned role for this USV.
        """
        self.update_status(own_status)

        # Cannot be scout without a working camera
        if not own_status.camera_healthy:
            if own_status.battery_pct > self.LOW_BATTERY_THRESHOLD:
                return Role.WORKER
            return Role.IDLE

        # Scout-capable USVs: healthy camera, decent battery
        capable: List[USVStatus] = []
        for status in self._swarm_status.values():
            if status.camera_healthy and status.battery_pct > self.LOW_BATTERY_THRESHOLD:
                capable.append(status)

        # Sort by battery (descending) — higher battery gets scout priority
        capable.sort(key=lambda s: s.battery_pct, reverse=True)

        # Select top N as scouts
        num_scouts = max(self.MIN_SCOUTS, 1)
        scout_ids: Set[str] = {s.usv_id for s in capable[:num_scouts]}

        if self.own_id in scout_ids:
            return Role.SCOUT
        elif own_status.battery_pct > self.LOW_BATTERY_THRESHOLD:
            return Role.WORKER
        else:
            return Role.IDLE

    def get_scouts(self) -> List[str]:
        """Get the list of current scout USV IDs.

        Returns:
            Sorted list of scout identifiers.
        """
        return sorted(
            s.usv_id for s in self._swarm_status.values()
            if s.role == Role.SCOUT
        )

    def get_workers(self) -> List[str]:
        """Get the list of current worker USV IDs.

        Returns:
            Sorted list of worker identifiers.
        """
        return sorted(
            s.usv_id for s in self._swarm_status.values()
            if s.role == Role.WORKER
        )
