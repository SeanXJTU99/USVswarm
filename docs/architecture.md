# Architecture — USV Swarm System Evolution

## Three-Tier Evolution Panorama

```
┌─────────────────────────────────────────────────────────────────┐
│                    2020: Single-USV Closed Loop                  │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐ │
│  │ YOLOv5   │   │ Carto-   │   │ A* + DWA │   │ LOS +        │ │
│  │ + CBAM   │──▶│ grapher  │──▶│ Planning │──▶│ Cascaded PID │ │
│  │ (Camera) │   │ (Lidar)  │   │ (nav2)   │   │ (Thrusters)  │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────┘ │
│       │              │               │               │          │
│       ▼              ▼               ▼               ▼          │
│  IPM → Costmap   Occupancy Grid   Global/Local    Dual Prop     │
│  (BEV proj.)     (0.05m res.)     Costmaps        Differential  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ ROS 2 Humble + DDS + 5.8GHz Mesh
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 2022: Decentralized Swarm Intelligence           │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ DDS Peer     │  │ QoS Profiles │  │ Mesh Network         │  │
│  │ Discovery    │  │ (Best Effort │  │ (Multi-hop Routing)  │  │
│  │ (No roscore) │  │  / Reliable) │  │                      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Boids        │  │ ORCA         │  │ Distributed SLAM     │  │
│  │ Cohesion     │  │ Reciprocal   │  │ Log-odds Fusion      │  │
│  │ Alignment    │  │ Avoidance    │  │ Submap Exchange      │  │
│  │ Separation   │  │ 50/50 Split  │  │                      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Leader-      │  │ Consensus    │  │ Role Assignment      │  │
│  │ Follower     │  │ Protocol     │  │ Scout vs Worker      │  │
│  │ (L-α Control)│  │ (Avg. agree) │  │ (Battery-aware)      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ UAV Airborne Compute Center
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               2022+: UAV-USV Cross-Domain Coordination           │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Overhead     │  │ Hungarian    │  │ Visual Servoing      │  │
│  │ Perception   │  │ Assigner     │  │ (IBVS, UAV-guided)   │  │
│  │ (Nadir YOLO) │  │ (Optimal)    │  │                      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │ Star Topology│  │ BEV Mapper   │                             │
│  │ (Hub-Spoke)  │  │ (Orthomosaic)│                             │
│  │ UAV Central  │  │ Global Map   │                             │
│  └──────────────┘  └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### Why ROS 2 DDS instead of ROS 1?
ROS 1 requires a central `roscore`. If the master node fails (boat sinks, power loss), the entire swarm collapses. ROS 2 Humble uses DDS peer-to-peer discovery — any boat can join or leave the swarm without affecting others.

### Why Boids + ORCA instead of centralized planning?
Centralized planning (one boat plans for all) is fragile and communication-intensive. Boids is fully decentralized — each USV computes its own force vectors using only locally observed neighbor states. ORCA adds collision-free guarantees with mathematically proven safety.

### Why UAV star topology instead of pure mesh?
The water-surface mesh suffers from multipath interference (radio waves reflecting off the water). UAV-mounted antenna at 20-30 m altitude provides clean line-of-sight to all USVs. The star topology is simpler than mesh routing and more reliable in this geometry.

### Why cascaded PID instead of MPC?
Model Predictive Control requires an accurate hydrodynamic model and significant computation (quadratic programming). For 50-80 cm storage-box vessels with limited onboard compute (Raspberry Pi / Jetson Nano), a well-tuned cascaded PID with anti-windup achieves comparable tracking performance at a fraction of the computational cost.

## Data Flow

```
Camera (30 Hz) ──▶ YOLOv5+CBAM ──▶ IPM (IMU-stabilized) ──▶ Costmap
                                                                 │
Lidar (10 Hz)  ──▶ Cartographer 2D SLAM ──▶ Occupancy Grid ─────┤
                                                                 │
                                                                 ▼
                                                          nav2 Planner
                                                          (A* + DWA)
                                                                 │
                                                                 ▼
                                                          LOS Guidance
                                                         (desired heading)
                                                                 │
                                                                 ▼
                                                      Cascaded PID
                                                 (throttle, diff_torque)
                                                                 │
                                                                 ▼
                                                      Differential Drive
                                                    (left, right thrust)
```

## Swarm Communication Flow

```
USV_0 (Scout)                    USV_1 (Worker)               USV_2 (Worker)
  │                                 │                            │
  │ YOLO detects obstacle           │                            │
  │ at GPS (lat, lon)               │                            │
  │                                 │                            │
  ├─ DDS publish ──────────────────┤                            │
  │  /swarm/obstacles              │                            │
  │  QoS: Best Effort              │                            │
  │                                 │                            │
  │                                 ├── receives obstacle ──────┤
  │                                 │   adds to local costmap   │
  │                                 │                            │
  │                                 ├── Boids force update ─────┤
  │                                 │   (separation from        │
  │                                 │    virtual obstacle)      │
  │                                 │                            │
  ├── ORCA check ──────────────────┤                            │
  │   (no collision with           │                            │
  │    USV_1 trajectory)           │                            │
```

## UAV Cross-Domain Flow

```
UAV (20-30m altitude)
  │
  ├── Downward camera (1920×1080, 30 Hz)
  │   └── YOLO detects all objects on water surface
  │       └── Direct BEV projection (no IPM needed)
  │
  ├── Hungarian Assigner
  │   └── Cost matrix: USV_i → Task_j distance
  │       └── Optimal assignment broadcast to USVs
  │
  ├── Visual Servoing (per USV)
  │   └── pixel_error → steering_correction
  │       └── Sent via star topology (Reliable QoS)
  │
  ├── BEV Mapper
  │   └── Accumulate frames → global orthomosaic
  │       └── Broadcast to all USVs as global_costmap
  │
  └── Star Topology Hub
      └── Heartbeat monitoring (3s timeout)
          └── Automatic fallback to mesh if UAV disconnects
```
