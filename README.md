# Spot MoCap Teleop

Teleoperation of the Boston Dynamics Spot robot arm via OptiTrack motion capture glove.

## Overview

This system captures hand movements from an operator wearing a glove with OptiTrack markers, processes and filters those movements, enforces safety limits, and sends commands to the Spot robot's arm end-effector in real-time. All data is recorded via rosbag2 for analysis.

## Architecture

<img width="4043" height="5116" alt="spot_mocap_teleop_diagram" src="https://github.com/user-attachments/assets/367591c7-c346-4037-a232-b1ead8812198" />


## Data Flow Pipeline

```
OptiTrack Motive
     │
     │  UDP Multicast (NatNet protocol)
     ▼
mocap4ros2 Driver ──────▶ /mocap4r2_optitrack/rigid_bodies
     │
     ▼
MocapGloveProcessor ───▶ /teleop/raw_glove_pose   (100 Hz)
  • Extracts glove rigid body     /teleop/glove_velocity
  • Applies calibration offset    /teleop/glove_tracking
  • Scales to robot workspace
     │
     ▼
MotionFilter ───────────▶ /teleop/filtered_pose    (100 Hz)
  • Butterworth LPF (fc=5Hz)
  • Velocity rate limiting (0.5 m/s max)
  • Acceleration limiting (2.0 m/s² max)
  • Dead zone (5mm)
  • SLERP orientation smoothing
     │
     ▼
WorkspaceLimiter ───────▶ /teleop/safe_pose         (100 Hz)
  • Clamp to arm workspace         /teleop/in_workspace
  • Self-collision avoidance       /teleop/workspace_marker
  • Soft boundary zones
     │
     ▼
SpotArmCommander ───────▶ /robot_command (Action)   (10 Hz)
  • ArmCartesianCommand            /teleop/arm_command_status
  • Body frame reference
  • Commanded at fixed 10Hz rate
     │
     ▼
spot_ros2 Driver ───────▶ Spot SDK ──────▶ Spot Arm
```

## Prerequisites

- Docker + Docker Compose
- OptiTrack camera system with Motive software
- Boston Dynamics Spot with arm
- Host machine on same network as OptiTrack and Spot

## Quick Start

### 1. Configure

Edit `config/teleop_params.yaml`:
- Set `glove_rigid_body_name` to match your Motive rigid body name
- Adjust `position_scale` based on your workspace ratio
- Tune filter parameters if needed

Edit `config/spot_config.yaml`:
- Set Spot IP, username, password

Set environment variables:
```bash
export SPOT_IP=192.168.80.3
export SPOT_USERNAME=admin
export SPOT_PASSWORD=your_password
export DISPLAY=$DISPLAY
```

### 2. Build

```bash
cd docker
docker-compose build
```

### 3. Run

```bash
# Allow X11 forwarding
xhost +local:docker

# Start all services
docker-compose up -d

# Or start individually:
docker-compose up -d mocap4ros2_optitrack
docker-compose up -d spot_driver
docker-compose up -d spot_teleop
```

### 4. Operate

```bash
# Access the teleop container
docker-compose exec spot_teleop bash

# Step 1: Initialize robot (claim, power on, stand, deploy arm)
ros2 service call /teleop/init_system std_srvs/srv/Trigger

# Step 2: Position glove in starting pose, then start teleop
ros2 service call /teleop/start_teleop std_srvs/srv/Trigger

# Step 3: Move your hand - the Spot arm follows!

# Step 4: Stop teleop
ros2 service call /teleop/stop_teleop std_srvs/srv/Trigger

# Emergency stop (any time)
ros2 service call /teleop/emergency_stop std_srvs/srv/Trigger
```

### 5. Retrieve Data

```bash
# Rosbags are saved to the rosbag_data volume
docker cp spot_teleop:/root/rosbag_data ./rosbag_data

# Play back
ros2 bag play rosbag_data/spot_teleop_YYYYMMDD_HHMMSS/
```

## ROS2 Topics

| Topic | Type | Description |
|---|---|---|
| `/mocap4r2_optitrack/rigid_bodies` | `mocap4r2_msgs/RigidBodies` | Raw OptiTrack data |
| `/teleop/raw_glove_pose` | `PoseStamped` | Calibrated, scaled glove pose |
| `/teleop/glove_velocity` | `TwistStamped` | Glove velocity |
| `/teleop/glove_tracking` | `Bool` | Glove tracking status |
| `/teleop/filtered_pose` | `PoseStamped` | Low-pass filtered pose |
| `/teleop/safe_pose` | `PoseStamped` | Workspace-limited pose |
| `/teleop/in_workspace` | `Bool` | Inside workspace bounds |
| `/teleop/safety_status` | `String` | Safety monitor status |
| `/teleop/e_stop` | `Bool` | Emergency stop flag |
| `/teleop/system_state` | `String` | Pipeline state machine |
| `/teleop/recording_status` | `String` | Rosbag recording status |
| `/teleop/arm_command_status` | `String` | Arm command feedback |

## ROS2 Services

| Service | Type | Description |
|---|---|---|
| `/teleop/init_system` | `Trigger` | Initialize robot |
| `/teleop/start_teleop` | `Trigger` | Start full pipeline |
| `/teleop/stop_teleop` | `Trigger` | Stop and stow |
| `/teleop/emergency_stop` | `Trigger` | E-stop |
| `/teleop/calibrate_glove` | `Trigger` | Set glove origin |
| `/teleop/enable_arm` | `Trigger` | Enable arm commands |
| `/teleop/disable_arm` | `Trigger` | Disable arm commands |
| `/teleop/stow_arm` | `Trigger` | Stow arm |
| `/teleop/unstow_arm` | `Trigger` | Deploy arm |
| `/teleop/start_recording` | `Trigger` | Start rosbag |
| `/teleop/stop_recording` | `Trigger` | Stop rosbag |
| `/teleop/reset_safety` | `Trigger` | Clear safety violations |

## Safety Features

- **Heartbeat monitor**: Stops arm if mocap data is lost for >0.5s
- **Velocity watchdog**: Limits end-effector speed to 0.5 m/s
- **Acceleration limiter**: Max 2.0 m/s² to prevent jerky motions
- **Force/torque limits**: E-stop at 50N force or 20Nm torque
- **Workspace boundaries**: Hard limits matching Spot arm reach
- **Self-collision avoidance**: Minimum distance from robot body
- **Low-pass filter**: 2nd order Butterworth at 5Hz removes noise
- **Dead zone**: 5mm threshold ignores micro-movements

## License

Apache-2.0
