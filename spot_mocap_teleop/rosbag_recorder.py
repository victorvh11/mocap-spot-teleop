#!/usr/bin/env python3
"""
rosbag_recorder.py
==================
Manages rosbag2 recording of all relevant teleoperation and robot
state topics. Supports start/stop via service calls.

Services:
    /teleop/start_recording  (std_srvs/srv/Trigger)
    /teleop/stop_recording   (std_srvs/srv/Trigger)

Publishes:
    /teleop/recording_status (std_msgs/msg/String)
"""

import rclpy
from rclpy.node import Node
import subprocess
import signal
import os
from datetime import datetime

from std_msgs.msg import String
from std_srvs.srv import Trigger


class RosbagRecorder(Node):
    """Manages rosbag2 recording via subprocess."""

    def __init__(self):
        super().__init__('rosbag_recorder')

        # Parameters
        self.declare_parameter('bag_output_dir', '/root/rosbag_data')
        self.declare_parameter('bag_name_prefix', 'spot_teleop')
        self.declare_parameter('auto_start', False)
        self.declare_parameter('max_bag_duration_sec', 600)
        self.declare_parameter('compression', 'zstd')
        self.declare_parameter('record_topics', [
            '/mocap4r2_optitrack/rigid_bodies',
            '/mocap4r2_optitrack/markers',
            '/teleop/raw_glove_pose',
            '/teleop/filtered_pose',
            '/teleop/safe_pose',
            '/spot/status/joint_states',
            '/spot/status/end_effector_force',
            '/spot/status/robot_state',
            '/joint_states',
            '/tf',
            '/tf_static',
            '/teleop/safety_status',
            '/teleop/diagnostics',
            '/teleop/arm_command_status',
            '/teleop/glove_velocity',
        ])

        self.output_dir = self.get_parameter('bag_output_dir').value
        self.prefix = self.get_parameter('bag_name_prefix').value
        self.auto_start = self.get_parameter('auto_start').value
        self.max_duration = self.get_parameter('max_bag_duration_sec').value
        self.compression = self.get_parameter('compression').value
        self.topics = self.get_parameter('record_topics').value

        # State
        self.recording_process = None
        self.is_recording = False
        self.current_bag_path = None

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Services
        self.srv_start = self.create_service(
            Trigger, '/teleop/start_recording', self.start_recording_cb
        )
        self.srv_stop = self.create_service(
            Trigger, '/teleop/stop_recording', self.stop_recording_cb
        )

        # Publisher
        self.pub_status = self.create_publisher(
            String, '/teleop/recording_status', 10
        )

        # Status timer
        self.create_timer(2.0, self.publish_status)

        # Auto-start if configured
        if self.auto_start:
            self.get_logger().info('Auto-starting recording...')
            self._start_recording()

        self.get_logger().info(
            f'RosbagRecorder ready. Output: {self.output_dir}, '
            f'Topics: {len(self.topics)}'
        )

    def _start_recording(self) -> tuple:
        """Start rosbag2 recording subprocess."""
        if self.is_recording:
            return False, 'Already recording'

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        bag_name = f'{self.prefix}_{timestamp}'
        self.current_bag_path = os.path.join(self.output_dir, bag_name)

        cmd = [
            'ros2', 'bag', 'record',
            '-o', self.current_bag_path,
            '--storage', 'sqlite3',
        ]

        # Add compression
        if self.compression != 'none':
            cmd.extend(['--compression-mode', 'message'])
            cmd.extend(['--compression-format', self.compression])

        # Add max duration
        if self.max_duration > 0:
            cmd.extend(['-d', str(self.max_duration)])

        # Add topics
        cmd.extend(self.topics)

        self.get_logger().info(f'Starting rosbag2: {bag_name}')
        self.get_logger().debug(f'Command: {" ".join(cmd)}')

        try:
            self.recording_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )
            self.is_recording = True
            return True, f'Recording to {self.current_bag_path}'

        except Exception as e:
            self.get_logger().error(f'Failed to start recording: {e}')
            return False, str(e)

    def _stop_recording(self) -> tuple:
        """Stop rosbag2 recording subprocess."""
        if not self.is_recording or self.recording_process is None:
            return False, 'Not currently recording'

        try:
            # Send SIGINT to the process group (graceful shutdown)
            os.killpg(os.getpgid(self.recording_process.pid), signal.SIGINT)
            self.recording_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # Force kill
            os.killpg(os.getpgid(self.recording_process.pid), signal.SIGKILL)
            self.recording_process.wait()
        except Exception as e:
            self.get_logger().warn(f'Error stopping recording: {e}')

        bag_path = self.current_bag_path
        self.recording_process = None
        self.is_recording = False
        self.current_bag_path = None

        self.get_logger().info(f'Recording stopped. Saved to {bag_path}')
        return True, f'Saved to {bag_path}'

    def start_recording_cb(self, request, response):
        """Service handler to start recording."""
        success, message = self._start_recording()
        response.success = success
        response.message = message
        return response

    def stop_recording_cb(self, request, response):
        """Service handler to stop recording."""
        success, message = self._stop_recording()
        response.success = success
        response.message = message
        return response

    def publish_status(self):
        """Periodically publish recording status."""
        status = String()
        if self.is_recording:
            status.data = f'RECORDING: {self.current_bag_path}'
        else:
            status.data = 'IDLE'
        self.pub_status.publish(status)

    def destroy_node(self):
        """Clean up: stop recording before shutdown."""
        if self.is_recording:
            self._stop_recording()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RosbagRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
