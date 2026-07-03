# Mock Sensor Data

This directory contains synthetic sensor logs for unit testing and
algorithm validation. All data is **fabricated** — no real sensor
recordings are included.

## Files

| File | Description |
|------|-------------|
| `sensor_log.csv` | Synthetic GPS + IMU + camera log (11 samples, 0.05s interval) |

## Format

### `sensor_log.csv`

| Column | Unit | Description |
|--------|------|-------------|
| `timestamp_s` | s | Elapsed time since start |
| `gps_lat` | deg | GPS latitude (synthetic, NOT a real location) |
| `gps_lon` | deg | GPS longitude (synthetic, NOT a real location) |
| `gps_heading_rad` | rad | Heading from GPS track |
| `imu_accel_x/y/z_mps2` | m/s² | Accelerometer readings |
| `imu_gyro_x/y/z_radps` | rad/s | Gyroscope readings |
| `camera_frame_id` | — | Sequential frame counter |
| `yolo_detections_count` | — | Number of objects YOLO detected in this frame |

## Usage

These files are consumed by the test suite:

```bash
pytest tests/ -v
```

## Regeneration

To regenerate mock data with different parameters, modify and run the
simulation, then export the ROS bag to CSV:

```bash
ros2 bag record -o mock_data /usv_0/gps /usv_0/imu /usv_0/camera
ros2 run rosbag2_to_csv rosbag2_to_csv mock_data
```
