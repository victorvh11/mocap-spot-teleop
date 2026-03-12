# =============================================================================
# Dockerfile - Spot MoCap Teleop
# Extends MOCAP4ROS2 OptiTrack with spot_ros2 and teleop pipeline
# =============================================================================

# Stage 1: Base ROS2 Humble with MOCAP4ROS2
FROM osrf/ros:humble-desktop AS base

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    curl wget git vim \
    build-essential \
    python3-pip \
    python3-colcon-common-extensions \
    python3-vcstool \
    python3-rosdep \
    clang-format \
    python3-numpy \
    python3-scipy \
    python3-transforms3d \
    ros-humble-rosbag2 \
    ros-humble-rosbag2-transport \
    ros-humble-rosbag2-storage-default-plugins \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-joint-state-publisher-gui \
    ros-humble-xacro \
    ros-humble-tf-transformations \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip3 install --no-cache-dir \
    bosdyn-client==4.1.0 \
    bosdyn-mission==4.1.0 \
    bosdyn-api==4.1.0 \
    bosdyn-core==4.1.0 \
    bosdyn-choreography-client==4.1.0

# =============================================================================
# Stage 2: Build MOCAP4ROS2 OptiTrack
# =============================================================================
WORKDIR /root/mocap4r2_ws

# Clone MOCAP4ROS2
RUN mkdir -p src && cd src && \
    git clone https://github.com/MOCAP4ROS2-Project/mocap4ros2_optitrack.git

# Import dependencies
RUN cd src && \
    vcs import < mocap4ros2_optitrack/dependency_repos.repos && \
    if [ -d mocap4r2 ]; then cd mocap4r2 && git checkout humble-devel; fi

# Install ROS deps + compatibility patch
RUN rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y \
        --skip-keys "ament_cmake_clang_format"

RUN find /root/mocap4r2_ws/src -name "*.hpp" -o -name "*.cpp" | \
    xargs grep -rl "tf2/buffer_core.h" 2>/dev/null | \
    xargs -r sed -i 's|#include "tf2/buffer_core.h"|#include "tf2_ros/buffer.h"|g'

# Build MOCAP4ROS2
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install"

# =============================================================================
# Stage 3: Build spot_ros2
# =============================================================================
WORKDIR /root/spot_ws

RUN mkdir -p src && cd src && \
    git clone https://github.com/bdaiinstitute/spot_ros2.git && \
    cd spot_ros2 && \
    git submodule init && \
    git submodule update

# Install spot_ros2 dependencies
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    cd src/spot_ros2 && \
    if [ -f install_spot_ros2.sh ]; then chmod +x install_spot_ros2.sh && ./install_spot_ros2.sh; fi"

RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -y" || true

# Build spot_ros2
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    source /root/mocap4r2_ws/install/setup.bash && \
    colcon build --symlink-install" || \
    echo "WARNING: spot_ros2 build had issues, some packages may not be available"

# =============================================================================
# Stage 4: Build teleop package
# =============================================================================
WORKDIR /root/teleop_ws

# Copy teleop package
COPY spot_mocap_teleop/ src/spot_mocap_teleop/

# Build teleop
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    source /root/mocap4r2_ws/install/setup.bash && \
    source /root/spot_ws/install/setup.bash 2>/dev/null || true && \
    colcon build --symlink-install"

# =============================================================================
# Environment setup
# =============================================================================
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /root/mocap4r2_ws/install/setup.bash" >> /root/.bashrc && \
    echo "source /root/spot_ws/install/setup.bash 2>/dev/null || true" >> /root/.bashrc && \
    echo "source /root/teleop_ws/install/setup.bash" >> /root/.bashrc && \
    echo "" >> /root/.bashrc && \
    echo "echo '========================================='" >> /root/.bashrc && \
    echo "echo ' Spot MoCap Teleop Container Ready'" >> /root/.bashrc && \
    echo "echo ' ROS2 Humble + MOCAP4ROS2 + spot_ros2'" >> /root/.bashrc && \
    echo "echo '========================================='" >> /root/.bashrc && \
    echo "echo ''" >> /root/.bashrc && \
    echo "echo 'Quick commands:'" >> /root/.bashrc && \
    echo "echo '  ros2 launch spot_mocap_teleop spot_mocap_teleop.launch.py'" >> /root/.bashrc && \
    echo "echo '  ros2 service call /teleop/init_system std_srvs/srv/Trigger'" >> /root/.bashrc && \
    echo "echo '  ros2 service call /teleop/start_teleop std_srvs/srv/Trigger'" >> /root/.bashrc && \
    echo "echo '  ros2 service call /teleop/stop_teleop std_srvs/srv/Trigger'" >> /root/.bashrc && \
    echo "echo ''" >> /root/.bashrc

# Create rosbag output directory
RUN mkdir -p /root/rosbag_data

# Expose OptiTrack multicast ports
EXPOSE 1510/udp 1511/udp

WORKDIR /root/teleop_ws

ENTRYPOINT ["/bin/bash", "-c", \
    "source /opt/ros/humble/setup.bash && \
     source /root/mocap4r2_ws/install/setup.bash && \
     source /root/spot_ws/install/setup.bash 2>/dev/null || true && \
     source /root/teleop_ws/install/setup.bash && \
     exec \"$@\"", "--"]

CMD ["/bin/bash"]
