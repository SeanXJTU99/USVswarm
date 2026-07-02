"""2D occupancy grid costmap builder for USV navigation.

Converts YOLO detection results (world-frame obstacle positions) and
IPM free-space segmentation into ROS nav2 costmap layers.
"""

from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Point, Pose

from .ipm_transform import IPMTransformer


class CostmapBuilder(Node):
    """Builds and maintains a 2D costmap from perception outputs.

    Integrates:
      - Obstacle points from YOLO detections (projected to world frame via IPM).
      - Free space from water surface segmentation (if available).
      - Inflated cost values (Gaussian decay around obstacles).

    Publishes nav_msgs/OccupancyGrid for use by nav2 planners (A*, DWA).
    """

    # Costmap cell values (following nav2 convention)
    FREE = 0
    UNKNOWN = 128
    LETHAL = 254
    INFLATED = 200
    INSCRIBED = 253

    def __init__(self) -> None:
        """Initialize costmap builder node."""
        super().__init__("costmap_builder")

        # Parameters
        self.declare_parameter("resolution", 0.05)        # m/cell
        self.declare_parameter("width", 200)               # cells
        self.declare_parameter("height", 200)              # cells
        self.declare_parameter("inflation_radius", 0.5)    # m
        self.declare_parameter("robot_radius", 0.25)       # m

        self.resolution: float = (
            self.get_parameter("resolution").get_parameter_value().double_value
        )
        width_cells: int = (
            self.get_parameter("width").get_parameter_value().integer_value
        )
        height_cells: int = (
            self.get_parameter("height").get_parameter_value().integer_value
        )
        self.inflation_radius: float = (
            self.get_parameter("inflation_radius").get_parameter_value().double_value
        )
        self.robot_radius: float = (
            self.get_parameter("robot_radius").get_parameter_value().double_value
        )

        # Costmap grid (origin at center)
        self._grid: np.ndarray = np.full(
            (height_cells, width_cells), self.UNKNOWN, dtype=np.int8
        )
        self._origin_x: float = -width_cells * self.resolution / 2.0
        self._origin_y: float = -height_cells * self.resolution / 2.0

        # Inflated cells count (for precomputation)
        cells_per_inflation: int = int(self.inflation_radius / self.resolution)
        self._inflation_kernel: np.ndarray = self._make_disk_kernel(cells_per_inflation)

        # Publisher
        self.costmap_pub = self.create_publisher(
            OccupancyGrid, "/usv/costmap", 10
        )

        self.get_logger().info("CostmapBuilder initialized")

    def _make_disk_kernel(self, radius_cells: int) -> np.ndarray:
        """Create a circular kernel for costmap inflation.

        Args:
            radius_cells: Inflation radius in number of cells.

        Returns:
            Boolean kernel array where True = within inflation radius.
        """
        size = 2 * radius_cells + 1
        ys, xs = np.ogrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
        dist = np.sqrt(xs**2 + ys**2)
        return dist <= radius_cells

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert world coordinates to grid cell indices.

        Args:
            wx: World X coordinate (m).
            wy: World Y coordinate (m).

        Returns:
            (row, col) grid indices.
        """
        col = int((wx - self._origin_x) / self.resolution)
        row = int((wy - self._origin_y) / self.resolution)
        return row, col

    def add_obstacle(self, wx: float, wy: float, radius_m: float = 0.3) -> None:
        """Mark a grid cell as occupied (lethal) with inflation.

        Args:
            wx: World X of obstacle center.
            wy: World Y of obstacle center.
            radius_m: Obstacle physical radius (m).
        """
        row, col = self.world_to_grid(wx, wy)
        rows, cols = self._grid.shape

        if not (0 <= row < rows and 0 <= col < cols):
            return

        # Mark lethal cell
        self._grid[row, col] = self.LETHAL

        # Inflate around obstacle
        kr = self._inflation_kernel.shape[0] // 2
        r_start = max(0, row - kr)
        r_end = min(rows, row + kr + 1)
        c_start = max(0, col - kr)
        c_end = min(cols, col + kr + 1)

        k_r_start = r_start - (row - kr)
        k_r_end = k_r_start + (r_end - r_start)
        k_c_start = c_start - (col - kr)
        k_c_end = k_c_start + (c_end - c_start)

        region = self._grid[r_start:r_end, c_start:c_end]
        kernel_region = self._inflation_kernel[k_r_start:k_r_end, k_c_start:k_c_end]
        mask = (region != self.LETHAL) & kernel_region
        region[mask] = self.INFLATED

    def add_free_space(self, wx: float, wy: float) -> None:
        """Mark a cell as free (observed water surface).

        Args:
            wx: World X coordinate.
            wy: World Y coordinate.
        """
        row, col = self.world_to_grid(wx, wy)
        rows, cols = self._grid.shape
        if 0 <= row < rows and 0 <= col < cols:
            self._grid[row, col] = self.FREE

    def update_from_detections(
        self, obstacles: List[Tuple[float, float, float]]
    ) -> None:
        """Batch-update costmap from a list of detected obstacles.

        Args:
            obstacles: List of (world_x, world_y, radius_m) tuples.
        """
        for wx, wy, radius in obstacles:
            self.add_obstacle(wx, wy, radius)

    def publish(self) -> None:
        """Publish the current costmap as an OccupancyGrid message."""
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.info = MapMetaData()
        msg.info.resolution = self.resolution
        msg.info.width = self._grid.shape[1]
        msg.info.height = self._grid.shape[0]
        msg.info.origin.position.x = self._origin_x
        msg.info.origin.position.y = self._origin_y
        msg.info.origin.orientation.w = 1.0

        msg.data = self._grid.ravel().tolist()
        self.costmap_pub.publish(msg)

    def reset(self) -> None:
        """Reset costmap to all unknown (e.g., on map switch)."""
        self._grid.fill(self.UNKNOWN)


def main(args: Optional[List[str]] = None) -> None:
    """Entry point for the costmap builder node."""
    rclpy.init(args=args)
    node = CostmapBuilder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
