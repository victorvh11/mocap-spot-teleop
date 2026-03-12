#!/usr/bin/env python3
"""
motion_filter.py
================
Applies low-pass filtering and rate limiting to glove pose data.
Removes high-frequency jitter/noise from the motion capture system
and enforces velocity/acceleration constraints for safe robot control.

Subscribes:
    /teleop/raw_glove_pose   (geometry_msgs/msg/PoseStamped)

Publishes:
    /teleop/filtered_pose    (geometry_msgs/msg/PoseStamped)
    /teleop/filter_diagnostics (std_msgs/msg/String)
"""

import rclpy
from rclpy.node import Node
import numpy as np
from scipy.signal import butter, lfilter_zi, lfilter
from scipy.spatial.transform import Rotation, Slerp

from geometry_msgs.msg import PoseStamped, TwistStamped


class ButterworthFilter:
    """Real-time Butterworth low-pass filter for streaming data."""

    def __init__(self, order: int, cutoff_hz: float, sample_rate_hz: float, n_channels: int = 3):
        nyquist = sample_rate_hz / 2.0
        normalized_cutoff = min(cutoff_hz / nyquist, 0.99)
        self.b, self.a = butter(order, normalized_cutoff, btype='low')
        # One filter state per channel
        self.zi = [lfilter_zi(self.b, self.a) for _ in range(n_channels)]
        self.initialized = False

    def filter(self, data: np.ndarray) -> np.ndarray:
        """Filter a single sample (shape: (n_channels,))."""
        if not self.initialized:
            for i in range(len(self.zi)):
                self.zi[i] = self.zi[i] * data[i]
            self.initialized = True

        out = np.zeros_like(data)
        for i in range(len(data)):
            y, self.zi[i] = lfilter(self.b, self.a, [data[i]], zi=self.zi[i])
            out[i] = y[0]
        return out


class ExponentialFilter:
    """Exponential Moving Average filter."""

    def __init__(self, alpha: float, n_channels: int = 3):
        self.alpha = alpha
        self.state = None

    def filter(self, data: np.ndarray) -> np.ndarray:
        if self.state is None:
            self.state = data.copy()
            return data.copy()
        self.state = self.alpha * data + (1.0 - self.alpha) * self.state
        return self.state.copy()


class MotionFilter(Node):
    """Low-pass filter and rate limiter for mocap pose data."""

    def __init__(self):
        super().__init__('motion_filter')

        # Parameters
        self.declare_parameter('filter_type', 'butterworth')
        self.declare_parameter('filter_order', 2)
        self.declare_parameter('cutoff_frequency_hz', 5.0)
        self.declare_parameter('sampling_rate_hz', 100.0)
        self.declare_parameter('ema_alpha', 0.15)
        self.declare_parameter('max_linear_velocity', 0.5)
        self.declare_parameter('max_angular_velocity', 1.0)
        self.declare_parameter('max_linear_acceleration', 2.0)
        self.declare_parameter('max_angular_acceleration', 4.0)
        self.declare_parameter('position_deadzone', 0.005)
        self.declare_parameter('orientation_deadzone', 0.02)

        filter_type = self.get_parameter('filter_type').value
        order = self.get_parameter('filter_order').value
        cutoff = self.get_parameter('cutoff_frequency_hz').value
        fs = self.get_parameter('sampling_rate_hz').value
        alpha = self.get_parameter('ema_alpha').value

        self.max_lin_vel = self.get_parameter('max_linear_velocity').value
        self.max_ang_vel = self.get_parameter('max_angular_velocity').value
        self.max_lin_acc = self.get_parameter('max_linear_acceleration').value
        self.max_ang_acc = self.get_parameter('max_angular_acceleration').value
        self.pos_deadzone = self.get_parameter('position_deadzone').value
        self.ori_deadzone = self.get_parameter('orientation_deadzone').value

        # Initialize filter
        if filter_type == 'butterworth':
            self.pos_filter = ButterworthFilter(order, cutoff, fs, 3)
            self.get_logger().info(
                f'Butterworth filter: order={order}, cutoff={cutoff}Hz, fs={fs}Hz'
            )
        else:
            self.pos_filter = ExponentialFilter(alpha, 3)
            self.get_logger().info(f'EMA filter: alpha={alpha}')

        # State for rate limiting
        self.prev_pos = None
        self.prev_vel = np.zeros(3)
        self.prev_quat = None
        self.prev_time = None
        self.prev_output_pos = None

        # Subscribers / Publishers
        self.sub_raw = self.create_subscription(
            PoseStamped, '/teleop/raw_glove_pose', self.raw_pose_cb, 10
        )
        self.pub_filtered = self.create_publisher(
            PoseStamped, '/teleop/filtered_pose', 10
        )

        self.get_logger().info('MotionFilter node started.')

    def raw_pose_cb(self, msg: PoseStamped):
        """Filter incoming raw pose."""
        now = self.get_clock().now()
        pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        quat = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ])

        # ---- Low-pass filter position ----
        filtered_pos = self.pos_filter.filter(pos)

        # ---- Dead zone ----
        if self.prev_output_pos is not None:
            delta = np.linalg.norm(filtered_pos - self.prev_output_pos)
            if delta < self.pos_deadzone:
                filtered_pos = self.prev_output_pos.copy()

        # ---- Rate limiting (velocity + acceleration) ----
        if self.prev_time is not None:
            dt = (now - self.prev_time).nanoseconds * 1e-9
            if dt > 0.0005:
                velocity = (filtered_pos - self.prev_pos) / dt
                speed = np.linalg.norm(velocity)

                # Velocity clamp
                if speed > self.max_lin_vel:
                    velocity = velocity * (self.max_lin_vel / speed)
                    filtered_pos = self.prev_pos + velocity * dt

                # Acceleration clamp
                accel = (velocity - self.prev_vel) / dt
                accel_mag = np.linalg.norm(accel)
                if accel_mag > self.max_lin_acc:
                    accel = accel * (self.max_lin_acc / accel_mag)
                    velocity = self.prev_vel + accel * dt
                    # Re-check velocity after acceleration clamp
                    speed = np.linalg.norm(velocity)
                    if speed > self.max_lin_vel:
                        velocity = velocity * (self.max_lin_vel / speed)
                    filtered_pos = self.prev_pos + velocity * dt

                self.prev_vel = velocity
        else:
            self.prev_vel = np.zeros(3)

        # ---- Orientation filtering (SLERP-based smoothing) ----
        filtered_quat = self._filter_orientation(quat)

        # ---- Publish ----
        out = PoseStamped()
        out.header.stamp = now.to_msg()
        out.header.frame_id = msg.header.frame_id
        out.pose.position.x = float(filtered_pos[0])
        out.pose.position.y = float(filtered_pos[1])
        out.pose.position.z = float(filtered_pos[2])
        out.pose.orientation.x = float(filtered_quat[0])
        out.pose.orientation.y = float(filtered_quat[1])
        out.pose.orientation.z = float(filtered_quat[2])
        out.pose.orientation.w = float(filtered_quat[3])

        self.pub_filtered.publish(out)

        # Update state
        self.prev_pos = filtered_pos.copy()
        self.prev_output_pos = filtered_pos.copy()
        self.prev_time = now

    def _filter_orientation(self, quat: np.ndarray) -> np.ndarray:
        """Smooth orientation using SLERP interpolation."""
        # Ensure valid quaternion
        norm = np.linalg.norm(quat)
        if norm < 1e-6:
            return np.array([0.0, 0.0, 0.0, 1.0])
        quat = quat / norm

        if self.prev_quat is None:
            self.prev_quat = quat.copy()
            return quat

        # Check for quaternion flip (double cover)
        if np.dot(quat, self.prev_quat) < 0:
            quat = -quat

        # SLERP with smoothing factor
        alpha = 0.3  # Higher = more responsive, lower = smoother
        try:
            rots = Rotation.from_quat(np.array([self.prev_quat, quat]))
            slerp = Slerp([0, 1], rots)
            result = slerp(alpha)
            filtered = result.as_quat()
        except Exception:
            filtered = quat

        self.prev_quat = filtered.copy()
        return filtered


def main(args=None):
    rclpy.init(args=args)
    node = MotionFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
