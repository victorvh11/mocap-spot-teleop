#!/usr/bin/env python3
"""
workspace_limiter.py
====================
Enforces workspace boundaries for the Spot arm, ensuring the
commanded pose stays within safe operational limits.

Subscribes:
    /teleop/filtered_pose  (geometry_msgs/msg/PoseStamped)

Publishes:
    /teleop/safe_pose      (geometry_msgs/msg/PoseStamped)
    /teleop/workspace_status (std_msgs/msg/String)
"""

import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Bool
from visualization_msgs.msg import Marker


class WorkspaceLimiter(Node):
    """Clamps commanded poses to the Spot arm's safe workspace."""

    def __init__(self):
        super().__init__('workspace_limiter')

        # Workspace bounds
        self.declare_parameter('workspace_bounds.x_min', 0.2)
        self.declare_parameter('workspace_bounds.x_max', 0.85)
        self.declare_parameter('workspace_bounds.y_min', -0.50)
        self.declare_parameter('workspace_bounds.y_max', 0.50)
        self.declare_parameter('workspace_bounds.z_min', -0.25)
        self.declare_parameter('workspace_bounds.z_max', 0.45)
        self.declare_parameter('soft_margin', 0.10)
        self.declare_parameter('boundary_mode', 'clamp')
        self.declare_parameter('max_pitch', 1.57)
        self.declare_parameter('max_roll', 1.57)
        self.declare_parameter('max_yaw', 3.14)
        self.declare_parameter('enable_self_collision_check', True)
        self.declare_parameter('min_distance_to_body', 0.15)

        self.bounds = {
            'x_min': self.get_parameter('workspace_bounds.x_min').value,
            'x_max': self.get_parameter('workspace_bounds.x_max').value,
            'y_min': self.get_parameter('workspace_bounds.y_min').value,
            'y_max': self.get_parameter('workspace_bounds.y_max').value,
            'z_min': self.get_parameter('workspace_bounds.z_min').value,
            'z_max': self.get_parameter('workspace_bounds.z_max').value,
        }
        self.soft_margin = self.get_parameter('soft_margin').value
        self.boundary_mode = self.get_parameter('boundary_mode').value
        self.min_body_dist = self.get_parameter('min_distance_to_body').value
        self.check_collision = self.get_parameter('enable_self_collision_check').value

        # Subscribers / Publishers
        self.sub_filtered = self.create_subscription(
            PoseStamped, '/teleop/filtered_pose', self.filtered_cb, 10
        )
        self.pub_safe = self.create_publisher(PoseStamped, '/teleop/safe_pose', 10)
        self.pub_status = self.create_publisher(String, '/teleop/workspace_status', 10)
        self.pub_in_bounds = self.create_publisher(Bool, '/teleop/in_workspace', 10)
        self.pub_ws_marker = self.create_publisher(
            Marker, '/teleop/workspace_marker', 10
        )

        # Publish workspace visualization periodically
        self.create_timer(1.0, self._publish_workspace_marker)

        self.get_logger().info('WorkspaceLimiter node started.')

    def filtered_cb(self, msg: PoseStamped):
        """Apply workspace limits to filtered pose."""
        pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])

        original_pos = pos.copy()
        is_clamped = False
        status_details = []

        if self.boundary_mode == 'clamp':
            pos, clamped_axes = self._clamp_position(pos)
            is_clamped = len(clamped_axes) > 0
            if is_clamped:
                status_details.append(f'clamped: {clamped_axes}')

        elif self.boundary_mode == 'soft_limit':
            pos = self._soft_limit_position(pos)
            delta = np.linalg.norm(pos - original_pos)
            if delta > 0.001:
                is_clamped = True
                status_details.append('soft_limited')

        elif self.boundary_mode == 'reject':
            if not self._is_in_bounds(pos):
                # Reject: don't publish anything
                status_msg = String()
                status_msg.data = 'REJECTED: out of workspace'
                self.pub_status.publish(status_msg)

                in_bounds = Bool()
                in_bounds.data = False
                self.pub_in_bounds.publish(in_bounds)
                return

        # Self-collision check (simple: distance from body origin)
        if self.check_collision:
            dist_to_body = np.linalg.norm(pos)
            if dist_to_body < self.min_body_dist:
                direction = pos / max(dist_to_body, 1e-6)
                pos = direction * self.min_body_dist
                status_details.append('collision_avoidance')
                is_clamped = True

        # Build output
        out = PoseStamped()
        out.header = msg.header
        out.pose.position.x = float(pos[0])
        out.pose.position.y = float(pos[1])
        out.pose.position.z = float(pos[2])
        out.pose.orientation = msg.pose.orientation

        self.pub_safe.publish(out)

        # Status
        in_bounds = Bool()
        in_bounds.data = not is_clamped
        self.pub_in_bounds.publish(in_bounds)

        if status_details:
            status_msg = String()
            status_msg.data = f'LIMITED: {", ".join(status_details)}'
            self.pub_status.publish(status_msg)

    def _clamp_position(self, pos: np.ndarray):
        """Hard clamp to workspace boundaries."""
        clamped = []
        result = pos.copy()

        axes = ['x', 'y', 'z']
        for i, axis in enumerate(axes):
            lo = self.bounds[f'{axis}_min']
            hi = self.bounds[f'{axis}_max']
            if result[i] < lo:
                result[i] = lo
                clamped.append(f'{axis}_min')
            elif result[i] > hi:
                result[i] = hi
                clamped.append(f'{axis}_max')

        return result, clamped

    def _soft_limit_position(self, pos: np.ndarray):
        """Gradually reduce commanded position near boundaries."""
        result = pos.copy()
        axes = ['x', 'y', 'z']

        for i, axis in enumerate(axes):
            lo = self.bounds[f'{axis}_min']
            hi = self.bounds[f'{axis}_max']
            margin = self.soft_margin

            if result[i] < lo + margin:
                # Smooth transition in soft zone
                t = max(0.0, (result[i] - lo) / margin)
                result[i] = lo + t * t * margin  # Quadratic easing
            elif result[i] > hi - margin:
                t = max(0.0, (hi - result[i]) / margin)
                result[i] = hi - t * t * margin

            # Hard clamp as final safety
            result[i] = np.clip(result[i], lo, hi)

        return result

    def _is_in_bounds(self, pos: np.ndarray) -> bool:
        """Check if position is within workspace."""
        axes = ['x', 'y', 'z']
        for i, axis in enumerate(axes):
            if pos[i] < self.bounds[f'{axis}_min'] or pos[i] > self.bounds[f'{axis}_max']:
                return False
        return True

    def _publish_workspace_marker(self):
        """Publish a cube marker showing the workspace boundaries in RViz."""
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'body'
        marker.ns = 'workspace'
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Center of workspace
        marker.pose.position.x = (self.bounds['x_min'] + self.bounds['x_max']) / 2.0
        marker.pose.position.y = (self.bounds['y_min'] + self.bounds['y_max']) / 2.0
        marker.pose.position.z = (self.bounds['z_min'] + self.bounds['z_max']) / 2.0
        marker.pose.orientation.w = 1.0

        # Size
        marker.scale.x = self.bounds['x_max'] - self.bounds['x_min']
        marker.scale.y = self.bounds['y_max'] - self.bounds['y_min']
        marker.scale.z = self.bounds['z_max'] - self.bounds['z_min']

        # Semi-transparent green
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.15

        self.pub_ws_marker.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = WorkspaceLimiter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
