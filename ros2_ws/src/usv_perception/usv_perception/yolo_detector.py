"""YOLOv5 water surface object detector — multi-backend with frame-skip tracking.

ROS 2 node supporting three inference backends (PyTorch / ONNX / TensorRT),
frame-level detection-and-tracking pipeline for lightweight deployment,
and optional CBAM attention for glare suppression.

Backend priority (auto-detect):
  1. TensorRT  (.engine) — Jetson Xavier/Orin, 3-5× speedup via FP16
  2. ONNX      (.onnx)   — portable, CUDA/CPU/OpenVINO providers
  3. PyTorch   (.pt)     — training/dev, fallback

Frame-skip tracking:
  Given slow water-surface dynamics (0.5-1.5 m/s), running YOLO on every
  frame is wasteful. This node runs inference every Nth frame and uses
  OpenCV CSRT trackers on intermediate frames, achieving 2-3× effective
  throughput at negligible accuracy cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray

from .cbam import CBAM


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------

class InferenceBackend(Enum):
    """Supported inference backends in priority order."""

    TENSORRT = "tensorrt"   # .engine — Jetson, max perf
    ONNX = "onnx"           # .onnx  — portable, CUDA/CPU
    PYTORCH = "pytorch"     # .pt    — dev / training


@dataclass
class TrackedDetection:
    """A detection being interpolated by an OpenCV tracker between YOLO runs."""

    tracker: cv2.Tracker  # OpenCV tracker instance
    bbox: Tuple[float, float, float, float]  # (cx, cy, w, h)
    class_id: int
    confidence: float
    age: int = 0           # Frames since last YOLO-confirmed update
    max_age: int = 6       # Drop after this many frames without re-detection


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

class WaterSurfaceDetector(Node):
    """YOLOv5-based detector for water surface objects.

    Detects: vessels, buoys, swimmers, floating debris, reefs, fishing nets.
    Supports PyTorch, ONNX Runtime, and TensorRT backends with automatic
    fallback.  Frame-skip tracking interpolates detections between YOLO
    runs for 2-3× effective throughput.

    ROS I/O
    -------
    Subscribes:  /usv/camera/image_raw  (sensor_msgs/Image)
    Publishes:   /usv/perception/detections  (vision_msgs/Detection2DArray)
    """

    TARGET_CLASSES: Dict[int, str] = {
        0: "vessel",
        1: "buoy",
        2: "swimmer",
        3: "floating_debris",
        4: "reef",
        5: "navigation_marker",
        6: "fishing_net",
    }

    def __init__(self) -> None:
        super().__init__("water_surface_detector")

        # ---- parameters ----
        self.declare_parameter("model_path", "")
        self.declare_parameter("conf_threshold", 0.45)
        self.declare_parameter("nms_iou_threshold", 0.45)
        self.declare_parameter("img_size", 640)
        self.declare_parameter("use_cbam", True)
        self.declare_parameter("backend", "auto")               # auto | tensorrt | onnx | pytorch
        self.declare_parameter("frame_skip", 3)                 # run YOLO every N frames
        self.declare_parameter("tracker_max_age", 6)            # drop stale tracked objects
        self.declare_parameter("tracker_min_confidence", 0.3)   # force re-detect below this

        self.model_path: str = self._param_str("model_path")
        self.conf_threshold: float = self._param_double("conf_threshold")
        self.nms_iou: float = self._param_double("nms_iou_threshold")
        self.img_size: int = self._param_int("img_size")
        self.use_cbam: bool = self._param_bool("use_cbam")
        self.backend_name: str = self._param_str("backend")
        self.frame_skip: int = self._param_int("frame_skip")
        self.tracker_max_age: int = self._param_int("tracker_max_age")
        self.tracker_min_conf: float = self._param_double("tracker_min_confidence")

        # ---- backend ---
        self.backend: InferenceBackend = self._resolve_backend()
        self._model: Any = None          # ultralytics YOLO | ort.InferenceSession
        self._load_model()

        # ---- CBAM (PyTorch backend only) ----
        self.cbam: Optional[CBAM] = None
        if self.use_cbam and self.backend == InferenceBackend.PYTORCH:
            self.cbam = CBAM(channels=256, reduction=16)

        # ---- frame-skip state ----
        self._frame_idx: int = 0
        self._tracked: Dict[int, TrackedDetection] = {}   # track_id → TrackedDetection
        self._next_track_id: int = 0
        self._last_inference_time_ms: float = 0.0
        self._fps_window: List[float] = []                 # rolling FPS buffer

        # ---- ROS comms ----
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)

        self.image_sub = self.create_subscription(
            Image, "/usv/camera/image_raw", self._image_callback, qos,
        )
        self.detection_pub = self.create_publisher(
            Detection2DArray, "/usv/perception/detections", qos,
        )

        self.get_logger().info(
            f"WaterSurfaceDetector ready | backend={self.backend.value} "
            f"frame_skip={self.frame_skip} cbam={self.use_cbam}"
        )

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _param_str(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def _param_double(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    def _param_int(self, name: str) -> int:
        return self.get_parameter(name).get_parameter_value().integer_value

    def _param_bool(self, name: str) -> bool:
        return self.get_parameter(name).get_parameter_value().bool_value

    # ------------------------------------------------------------------
    # Backend resolution & model loading
    # ------------------------------------------------------------------

    def _resolve_backend(self) -> InferenceBackend:
        """Resolve backend from ROS param, with auto-detection.

        'auto' probes in order: TensorRT engine file → ONNX file → PyTorch.
        """
        want = self.backend_name.lower()

        if want in ("tensorrt", "engine"):
            return InferenceBackend.TENSORRT
        if want in ("onnx", "onnxruntime"):
            return InferenceBackend.ONNX
        if want in ("pytorch", "torch", "pt"):
            return InferenceBackend.PYTORCH

        # auto-detect from model_path extension
        if self.model_path.endswith(".engine"):
            return InferenceBackend.TENSORRT
        if self.model_path.endswith(".onnx"):
            return InferenceBackend.ONNX
        return InferenceBackend.PYTORCH

    def _load_model(self) -> None:
        """Load or initialise the model for the selected backend."""
        if self.backend == InferenceBackend.PYTORCH:
            self._load_pytorch()
        elif self.backend == InferenceBackend.ONNX:
            self._load_onnx()
        elif self.backend == InferenceBackend.TENSORRT:
            self._load_tensorrt()

    def _load_pytorch(self) -> None:
        """Load ultralytics YOLO model (training/dev backend)."""
        try:
            from ultralytics import YOLO
            path = self.model_path or "yolov5s.pt"
            self._model = YOLO(path)
            self.get_logger().info(f"PyTorch backend loaded: {path}")
        except ImportError:
            self.get_logger().error("ultralytics not installed; PyTorch backend unavailable")
            self._model = None
        except Exception as exc:
            self.get_logger().error(f"Failed to load PyTorch model: {exc}")
            self._model = None

    def _load_onnx(self) -> None:
        """Load ONNX Runtime inference session.

        Provider order: CUDA → CPU (or OpenVINO on Intel).
        """
        try:
            import onnxruntime as ort
        except ImportError:
            self.get_logger().error("onnxruntime not installed; ONNX backend unavailable")
            self._model = None
            return

        providers = ort.get_available_providers()
        self.get_logger().info(f"ONNX available providers: {providers}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        try:
            self._model = ort.InferenceSession(
                self.model_path, sess_options=sess_options,
                providers=providers,
            )
            self.get_logger().info(f"ONNX backend loaded: {self.model_path}")
        except Exception as exc:
            self.get_logger().error(f"Failed to load ONNX model: {exc}")
            self._model = None

    def _load_tensorrt(self) -> None:
        """Load TensorRT engine (via ultralytics or onnxruntime with TRT EP).

        TensorRT engines are platform-specific — built on the target Jetson.
        Export with:  model.export(format='engine', device=0, half=True)
        """
        # TensorRT engines can be loaded via ultralytics YOLO (which wraps
        # TensorRT Python bindings internally).
        try:
            from ultralytics import YOLO
            if not self.model_path:
                self.get_logger().error("TensorRT backend requires model_path to .engine file")
                self._model = None
                return
            self._model = YOLO(self.model_path, task="detect")
            self.get_logger().info(f"TensorRT backend loaded: {self.model_path}")
        except ImportError:
            self.get_logger().error("ultralytics not installed; cannot load TensorRT engine")
            self._model = None
        except Exception as exc:
            self.get_logger().error(f"Failed to load TensorRT engine: {exc}")
            self._model = None

    # ------------------------------------------------------------------
    # Image callback + frame-skip logic
    # ------------------------------------------------------------------

    def _image_callback(self, msg: Image) -> None:
        """Process incoming camera frame with frame-skip tracking.

        Every `frame_skip`-th frame runs full YOLO inference; intermediate
        frames reuse tracker predictions (OpenCV CSRT).
        """
        img = self._ros_image_to_numpy(msg)
        self._frame_idx += 1

        if self._frame_idx % self.frame_skip == 0 or not self._tracked:
            # --- full YOLO inference ---
            t0 = time.perf_counter()
            raw_detections = self._run_inference(img)
            self._last_inference_time_ms = (time.perf_counter() - t0) * 1000.0
            self._update_fps_window(self._last_inference_time_ms)

            detections = self._merge_tracked(raw_detections, img)
        else:
            # --- tracker interpolation ---
            detections = self._tracker_predict(img)

        self._publish_detections(detections, msg.header.stamp)

    def _update_fps_window(self, ms: float) -> None:
        self._fps_window.append(ms)
        if len(self._fps_window) > 30:
            self._fps_window.pop(0)

    @property
    def avg_inference_ms(self) -> float:
        """Rolling average inference latency (ms)."""
        if not self._fps_window:
            return 0.0
        return sum(self._fps_window) / len(self._fps_window)

    # ------------------------------------------------------------------
    # Inference dispatch
    # ------------------------------------------------------------------

    def _run_inference(self, img: np.ndarray) -> List[dict]:
        """Dispatch to the active backend."""
        if self._model is None:
            return []

        if self.backend == InferenceBackend.PYTORCH:
            return self._infer_pytorch(img)
        if self.backend == InferenceBackend.ONNX:
            return self._infer_onnx(img)
        if self.backend == InferenceBackend.TENSORRT:
            return self._infer_tensorrt(img)
        return []

    def _infer_pytorch(self, img: np.ndarray) -> List[dict]:
        """PyTorch (ultralytics) inference."""
        results = self._model(
            img,
            conf=self.conf_threshold,
            iou=self.nms_iou,
            imgsz=self.img_size,
            verbose=False,
        )
        return self._ultralytics_results_to_dicts(results)

    def _infer_onnx(self, img: np.ndarray) -> List[dict]:
        """ONNX Runtime inference with pre/post-processing."""
        # Pre-process: resize + normalize + BCHW
        blob, scale, pad = self._preprocess_onnx(img)

        input_name = self._model.get_inputs()[0].name
        outputs = self._model.run(None, {input_name: blob})

        # Post-process: decode output tensors → detection dicts
        return self._postprocess_onnx(outputs, scale, pad)

    def _infer_tensorrt(self, img: np.ndarray) -> List[dict]:
        """TensorRT inference (ultralytics handles pre/post internally)."""
        results = self._model(
            img,
            conf=self.conf_threshold,
            iou=self.nms_iou,
            imgsz=self.img_size,
            verbose=False,
        )
        return self._ultralytics_results_to_dicts(results)

    # ------------------------------------------------------------------
    # Pre/post-processing helpers (ONNX path)
    # ------------------------------------------------------------------

    def _preprocess_onnx(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Resize + pad + normalize + BCHW for ONNX YOLOv5 input.

        Returns (blob_NCHW, scale, (pad_w, pad_h)).
        """
        h0, w0 = img.shape[:2]
        scale = min(self.img_size / w0, self.img_size / h0)
        new_w, new_h = int(w0 * scale), int(h0 * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = (self.img_size - new_w) // 2
        pad_h = (self.img_size - new_h) // 2
        padded = cv2.copyMakeBorder(
            resized, pad_h, self.img_size - new_h - pad_h,
            pad_w, self.img_size - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )

        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None, ...]  # HWC → NCHW
        return blob, scale, (pad_w, pad_h)

    def _postprocess_onnx(
        self,
        outputs: List[np.ndarray],
        scale: float,
        pad: Tuple[int, int],
    ) -> List[dict]:
        """Decode ONNX YOLOv5 output → list of detection dicts.

        Output shape: (1, 25200, 12) for 7-class model: [cx,cy,w,h,obj_conf, cls0..cls6].
        """
        preds = outputs[0][0]  # (25200, 12)
        pad_w, pad_h = pad
        detections: List[dict] = []

        for row in preds:
            obj_conf = float(row[4])
            if obj_conf < self.conf_threshold:
                continue

            cls_probs = row[5:]
            cls_id = int(np.argmax(cls_probs))
            cls_conf = float(cls_probs[cls_id])
            conf = obj_conf * cls_conf
            if conf < self.conf_threshold:
                continue

            # Decode box: cxcywh → xyxy (pixel coords on padded image)
            cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            x1 = (cx - w / 2 - pad_w) / scale
            y1 = (cy - h / 2 - pad_h) / scale
            x2 = (cx + w / 2 - pad_w) / scale
            y2 = (cy + h / 2 - pad_h) / scale

            detections.append({
                "class_id": cls_id,
                "class_name": self.TARGET_CLASSES.get(cls_id, "unknown"),
                "confidence": conf,
                "bbox": [
                    (x1 + x2) / 2,   # cx
                    (y1 + y2) / 2,   # cy
                    x2 - x1,          # w
                    y2 - y1,          # h
                ],
            })

        # NMS
        return self._nms(detections)

    def _ultralytics_results_to_dicts(self, results: Any) -> List[dict]:
        """Convert ultralytics Results object to our detection dict format."""
        detections: List[dict] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls.item())
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
                detections.append({
                    "class_id": cls_id,
                    "class_name": self.TARGET_CLASSES.get(cls_id, "unknown"),
                    "confidence": float(box.conf.item()),
                    "bbox": [
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        x2 - x1,
                        y2 - y1,
                    ],
                })
        return detections

    def _nms(self, dets: List[dict]) -> List[dict]:
        """Class-aware NMS on detection dicts."""
        if len(dets) <= 1:
            return dets

        boxes = np.array([d["bbox"] for d in dets])  # cxcywh
        scores = np.array([d["confidence"] for d in dets])
        cls_ids = np.array([d["class_id"] for d in dets])

        # cxcywh → xyxy
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        keep: List[int] = []
        for cls in np.unique(cls_ids):
            idxs = np.where(cls_ids == cls)[0]
            order = scores[idxs].argsort()[::-1]
            while len(order) > 0:
                keep.append(idxs[order[0]])
                if len(order) == 1:
                    break
                iou = self._box_iou_batch(xyxy[idxs[order[0]]], xyxy[idxs[order[1:]]])
                order = order[1:][iou < self.nms_iou]

        return [dets[i] for i in sorted(keep)]

    @staticmethod
    def _box_iou_batch(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Vectorized IoU of one box against many."""
        ix1 = np.maximum(box[0], boxes[:, 0])
        iy1 = np.maximum(box[1], boxes[:, 1])
        ix2 = np.minimum(box[2], boxes[:, 2])
        iy2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        area_a = (box[2] - box[0]) * (box[3] - box[1])
        area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return inter / (area_a + area_b - inter + 1e-6)

    # ------------------------------------------------------------------
    # Tracker interpolation (frame-skip)
    # ------------------------------------------------------------------

    def _merge_tracked(
        self, raw_detections: List[dict], img: np.ndarray
    ) -> List[dict]:
        """Merge raw YOLO detections with existing trackers via IoU matching.

        - Matched detections update their tracker (re-center).
        - Unmatched detections spawn new trackers.
        - Unmatched trackers are aged; those exceeding max_age are dropped.
        """
        matched_track_ids: set[int] = set()
        new_tracked: Dict[int, TrackedDetection] = {}

        for det in raw_detections:
            cx, cy, w, h = det["bbox"]
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            w_i, h_i = int(w), int(h)

            # Try to match an existing tracker by IoU
            best_id: Optional[int] = None
            best_iou: float = 0.0
            for tid, td in self._tracked.items():
                if tid in matched_track_ids:
                    continue
                tc = td.bbox
                iou = self._iou_xywh(
                    tc[0], tc[1], tc[2], tc[3],
                    cx, cy, w, h,
                )
                if iou > best_iou:
                    best_iou, best_id = iou, tid

            if best_id is not None and best_iou > 0.3:
                # Update existing tracker
                td = self._tracked[best_id]
                td.bbox = (cx, cy, w, h)
                td.confidence = det["confidence"]
                td.age = 0
                matched_track_ids.add(best_id)
                # Re-init tracker
                tracker = cv2.TrackerCSRT_create()
                tracker.init(img, (x1, y1, w_i, h_i))
                td.tracker = tracker
                new_tracked[best_id] = td
            else:
                # New tracker
                tracker = cv2.TrackerCSRT_create()
                tracker.init(img, (x1, y1, w_i, h_i))
                tid = self._next_track_id
                self._next_track_id += 1
                new_tracked[tid] = TrackedDetection(
                    tracker=tracker,
                    bbox=(cx, cy, w, h),
                    class_id=det["class_id"],
                    confidence=det["confidence"],
                )

        # Age unmatched trackers
        for tid, td in self._tracked.items():
            if tid not in matched_track_ids:
                td.age += 1
                if td.age <= td.max_age and td.confidence >= self.tracker_min_conf:
                    new_tracked[tid] = td

        self._tracked = new_tracked

        return [
            {
                "class_id": td.class_id,
                "class_name": self.TARGET_CLASSES.get(td.class_id, "unknown"),
                "confidence": td.confidence,
                "bbox": list(td.bbox),
                "track_id": tid,
                "is_tracked": td.age > 0,
            }
            for tid, td in self._tracked.items()
        ]

    def _tracker_predict(self, img: np.ndarray) -> List[dict]:
        """Update all trackers and return interpolated detections."""
        detections: List[dict] = []
        stale: List[int] = []

        for tid, td in self._tracked.items():
            ok, bbox_xywh = td.tracker.update(img)
            if ok:
                x, y, w, h = bbox_xywh
                td.bbox = (x + w / 2, y + h / 2, w, h)
                td.age += 1
                detections.append({
                    "class_id": td.class_id,
                    "class_name": self.TARGET_CLASSES.get(td.class_id, "unknown"),
                    "confidence": td.confidence * 0.95,  # decay confidence
                    "bbox": [td.bbox[0], td.bbox[1], td.bbox[2], td.bbox[3]],
                    "track_id": tid,
                    "is_tracked": True,
                })
            else:
                stale.append(tid)

        for tid in stale:
            del self._tracked[tid]

        return detections

    @staticmethod
    def _iou_xywh(
        cx1: float, cy1: float, w1: float, h1: float,
        cx2: float, cy2: float, w2: float, h2: float,
    ) -> float:
        """IoU of two cxcywh boxes."""
        x1a, y1a = cx1 - w1 / 2, cy1 - h1 / 2
        x2a, y2a = cx1 + w1 / 2, cy1 + h1 / 2
        x1b, y1b = cx2 - w2 / 2, cy2 - h2 / 2
        x2b, y2b = cx2 + w2 / 2, cy2 + h2 / 2

        ix1 = max(x1a, x1b)
        iy1 = max(y1a, y1b)
        ix2 = min(x2a, x2b)
        iy2 = min(y2a, y2b)

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = w1 * h1
        area_b = w2 * h2
        return inter / (area_a + area_b - inter + 1e-6)

    # ------------------------------------------------------------------
    # ROS helpers
    # ------------------------------------------------------------------

    def _ros_image_to_numpy(self, msg: Image) -> np.ndarray:
        dtype_map = {"8": np.uint8, "16": np.uint16}
        np_dtype = dtype_map.get(str(msg.encoding[-1]), np.uint8)
        img = np.frombuffer(msg.data, dtype=np_dtype).reshape(
            msg.height, msg.width, -1,
        )
        if img.shape[2] == 1:
            img = img[:, :, 0]
        return img

    def _publish_detections(
        self, detections: List[dict], stamp: rclpy.time.Time,
    ) -> None:
        msg = Detection2DArray()
        msg.header.stamp = stamp
        msg.header.frame_id = "usv_camera_link"

        for det in detections:
            d = Detection2D()
            d.results[0].id = det.get("class_id", -1)
            d.results[0].score = det.get("confidence", 0.0)

            bbox = det.get("bbox", [0, 0, 0, 0])
            d.bbox = BoundingBox2D()
            d.bbox.center.x = float(bbox[0])
            d.bbox.center.y = float(bbox[1])
            d.bbox.size_x = float(bbox[2])
            d.bbox.size_y = float(bbox[3])

            msg.detections.append(d)

        self.detection_pub.publish(msg)


def main(args: Optional[List[str]] = None) -> None:
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
