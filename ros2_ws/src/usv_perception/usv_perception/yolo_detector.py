"""YOLOv5-based water surface object detector with CBAM attention.

ROS 2 node that:
  - Subscribes to camera images
  - Runs YOLOv5 inference (with optional CBAM module for glare suppression)
  - Publishes bounding boxes and segmentation masks (if available)
  - Broadcasts obstacle world coordinates via GPS fusion
"""

from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from geometry_msgs.msg import Pose2D

from .cbam import CBAM


class WaterSurfaceDetector(Node):
    """YOLOv5-based detector for water surface objects.

    Detects: vessels, buoys, swimmers, floating debris, reefs, fishing nets.
    Uses CBAM attention to suppress water glare and wave reflections.
    """

    # Target classes for water surface operations
    WATER_CLASSES = {
        0: "vessel",
        1: "buoy",
        2: "swimmer",
        3: "floating_debris",
        4: "reef",
        5: "navigation_marker",
        6: "fishing_net",
    }

    def __init__(self) -> None:
        """Initialize the detector node with model and CBAM module."""
        super().__init__("water_surface_detector")

        # Parameters
        self.declare_parameter("model_path", "")
        self.declare_parameter("conf_threshold", 0.45)
        self.declare_parameter("nms_iou_threshold", 0.45)
        self.declare_parameter("img_size", 640)
        self.declare_parameter("use_cbam", True)
        self.declare_parameter("device", "cuda:0")

        self.model_path: str = (
            self.get_parameter("model_path").get_parameter_value().string_value
        )
        self.conf_threshold: float = (
            self.get_parameter("conf_threshold").get_parameter_value().double_value
        )
        self.nms_iou: float = (
            self.get_parameter("nms_iou_threshold").get_parameter_value().double_value
        )
        self.img_size: int = (
            self.get_parameter("img_size").get_parameter_value().integer_value
        )
        self.use_cbam: bool = (
            self.get_parameter("use_cbam").get_parameter_value().bool_value
        )

        # CBAM attention module (inserted after backbone, before neck)
        if self.use_cbam:
            self.cbam = CBAM(channels=256, reduction=16)

        # Subscribers & publishers
        self.image_sub = self.create_subscription(
            Image, "/usv/camera/image_raw", self._image_callback, 10
        )
        self.detection_pub = self.create_publisher(
            Detection2DArray, "/usv/perception/detections", 10
        )

        self.get_logger().info("WaterSurfaceDetector initialized")

    def _image_callback(self, msg: Image) -> None:
        """Process incoming camera frame.

        Args:
            msg: Raw camera image from the USV camera sensor.
        """
        # Convert ROS image to numpy array
        img_np: np.ndarray = self._ros_image_to_numpy(msg)

        # Run YOLO inference (uses model loaded via ultralytics or ONNX)
        detections: List[dict] = self._run_inference(img_np)

        # Publish detections
        self._publish_detections(detections, msg.header.stamp)

    def _ros_image_to_numpy(self, msg: Image) -> np.ndarray:
        """Convert ROS Image message to numpy array.

        Args:
            msg: ROS sensor_msgs/Image.

        Returns:
            BGR numpy array (H, W, 3).
        """
        dtype_map = {"8": np.uint8, "16": np.uint16}
        np_dtype = dtype_map.get(str(msg.encoding[-1]), np.uint8)

        img = np.frombuffer(msg.data, dtype=np_dtype).reshape(
            msg.height, msg.width, -1
        )
        if img.shape[2] == 1:
            img = img[:, :, 0]
        return img

    def _run_inference(self, img: np.ndarray) -> List[dict]:
        """Run YOLO inference with optional CBAM feature enhancement.

        This is a skeleton — actual inference delegates to ultralytics YOLO
        or an ONNX runtime backend. The CBAM module is applied as a feature
        refinement step between backbone and detection head.

        Args:
            img: Input image (H, W, 3) in BGR format.

        Returns:
            List of detection dicts with keys: class_id, confidence, bbox, mask.
        """
        # Placeholder: in production, load model via ultralytics.YOLO(model_path)
        # and run model(img, conf=self.conf_threshold, iou=self.nms_iou)
        # If self.use_cbam, inject CBAM into the model's feature pyramid.
        detections: List[dict] = []
        # --- production implementation would go here ---
        return detections

    def _publish_detections(
        self, detections: List[dict], stamp: rclpy.time.Time
    ) -> None:
        """Convert detection dicts to ROS Detection2DArray and publish.

        Args:
            detections: Raw detection results from inference.
            stamp: Timestamp from the source image.
        """
        msg = Detection2DArray()
        msg.header.stamp = stamp
        msg.header.frame_id = "usv_camera_link"

        for det in detections:
            d = Detection2D()
            d.results[0].id = det.get("class_id", -1)

            bbox = det.get("bbox", [0, 0, 0, 0])
            d.bbox = BoundingBox2D()
            d.bbox.center.x = float(bbox[0])
            d.bbox.center.y = float(bbox[1])
            d.bbox.size_x = float(bbox[2])
            d.bbox.size_y = float(bbox[3])

            msg.detections.append(d)

        self.detection_pub.publish(msg)


def main(args: Optional[List[str]] = None) -> None:
    """Entry point for the YOLO water surface detector node."""
    rclpy.init(args=args)
    node = WaterSurfaceDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
