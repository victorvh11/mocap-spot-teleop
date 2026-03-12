#!/usr/bin/env python3
"""
safety_monitor.py
=================
Monitors system health and enforces safety constraints.
Triggers emergency stop if any safety condition is violated.

Monitors:
    - Mocap tracking heartbeat
    - End-effector force/torque limits
    - Velocity watchdog
    - Workspace boundary violations

Publishes:
    /teleop/safety_status  (std_msgs/msg/String)
    /teleop/e_stop         (std_msgs/msg/Bool)
"""

import rclpy
from rclpy.node import Node
import numpy as np
import time

from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger


class SafetyMonitor(Node):
    """Monitors teleoperation safety and triggers emergency stops."""

    def __init__(self):
        super().__init__('safety_monitor')

        # Parameters
        self.declare_parameter('mocap_timeout', 0.5)
        self.declare_parameter('command_timeout', 1.0)
        self.declare_parameter('max_force_threshold', 50.0)
        self.declare_parameter('max_torque_threshold', 20.0)
        self.declare_parameter('velocity_watchdog_limit', 1.0)
        self.declare_parameter('enable_force_limit', True)
        self.declare_parameter('enable_velocity_watchdog', True)
        self.declare_parameter('enable_heartbeat_monitor', True)
        self.declare_parameter('enable_workspace_monitor', True)
        self.declare_parameter('recovery_action', 'stow')

        self.mocap_timeout = self.get_parameter('mocap_timeout').value
        self.cmd_timeout = self.get_parameter('command_timeout').value
        self.max_force = self.get_parameter('max_force_threshold').value
        self.max_torque = self.get_parameter('max_torque_threshold').value
        self.max_vel = self.get_parameter('velocity_watchdog_limit').value
        self.recovery_action = self.get_parameter('recovery_action').value

        self.enable_force = self.get_parameter('enable_force_limit').value
        self.enable_vel = self.get_parameter('enable_velocity_watchdog').value
        self.enable_heartbeat = self.get_parameter('enable_heartbeat_monitor').value
        self.enable_workspace = self.get_parameter('enable_workspace_monitor').value

        # State tracking
        self.last_mocap_time = None
        self.last_command_time = None
        self.is_safe = True
        self.safety_violations = []
        self.e_stop_active = False

        # Subscribers
        self.sub_tracking = self.create_subscription(
            Bool, '/teleop/glove_tracking', self.tracking_cb, 10
        )
        self.sub_velocity = self.create_subscription(
            TwistStamped, '/teleop/glove_velocity', self.velocity_cb, 10
        )
        self.sub_safe_pose = self.create_subscription(
            PoseStamped, '/teleop/safe_pose', self.safe_pose_cb, 10
        )
        self.sub_in_workspace = self.create_subscription(
            Bool, '/teleop/in_workspace', self.workspace_cb, 10
        )
        # Spot end-effector force (if available)
        self.sub_wrench = self.create_subscription(
            WrenchStamped, '/spot/status/end_effector_force', self.wrench_cb, 10
        )

        # Publishers
        self.pub_safety = self.create_publisher(String, '/teleop/safety_status', 10)
        self.pub_estop = self.create_publisher(Bool, '/teleop/e_stop', 10)

        # Service clients for recovery
        self.stow_client = self.create_client(Trigger, '/teleop/stow_arm')
        self.disable_client = self.create_client(Trigger, '/teleop/disable_arm')

        # E-stop reset service
        self.srv_reset = self.create_service(
            Trigger, '/teleop/reset_safety', self.reset_safety_cb
        )

        # Monitor timer at 20Hz
        self.create_timer(0.05, self.monitor_tick)

        self.get_logger().info('SafetyMonitor started.')

    def tracking_cb(self, msg: Bool):
        """Update mocap tracking timestamp."""
        if msg.data:
            self.last_mocap_time = self.get_clock().now()

    def velocity_cb(self, msg: TwistStamped):
        """Check velocity limits."""
        if not self.enable_vel:
            return
        vel = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z,
        ])
        speed = np.linalg.norm(vel)
        if speed > self.max_vel:
            self._trigger_violation(
                f'VELOCITY_EXCEEDED: {speed:.2f} m/s > {self.max_vel:.2f} m/s'
            )

    def safe_pose_cb(self, msg: PoseStamped):
        """Track command heartbeat."""
        self.last_command_time = self.get_clock().now()

    def workspace_cb(self, msg: Bool):
        """Track workspace boundary status."""
        if self.enable_workspace and not msg.data:
            self._trigger_violation('WORKSPACE_BOUNDARY_VIOLATION')

    def wrench_cb(self, msg: WrenchStamped):
        """Check force/torque limits on end-effector."""
        if not self.enable_force:
            return
        force = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
        ])
        torque = np.array([
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z,
        ])

        force_mag = np.linalg.norm(force)
        torque_mag = np.linalg.norm(torque)

        if force_mag > self.max_force:
            self._trigger_violation(
                f'FORCE_EXCEEDED: {force_mag:.1f}N > {self.max_force:.1f}N'
            )
        if torque_mag > self.max_torque:
            self._trigger_violation(
                f'TORQUE_EXCEEDED: {torque_mag:.1f}Nm > {self.max_torque:.1f}Nm'
            )

    def monitor_tick(self):
        """Periodic safety check."""
        violations = []
        now = self.get_clock().now()

        # Check mocap heartbeat
        if self.enable_heartbeat and self.last_mocap_time is not None:
            dt = (now - self.last_mocap_time).nanoseconds * 1e-9
            if dt > self.mocap_timeout:
                violations.append(
                    f'MOCAP_TIMEOUT: {dt:.2f}s > {self.mocap_timeout:.2f}s'
                )

        # Publish current status
        status = String()
        if self.e_stop_active:
            status.data = f'E_STOP_ACTIVE | Violations: {self.safety_violations}'
        elif violations:
            status.data = f'WARNING: {violations}'
            for v in violations:
                self._trigger_violation(v)
        else:
            status.data = 'OK'

        self.pub_safety.publish(status)

        # Publish e-stop status
        estop_msg = Bool()
        estop_msg.data = self.e_stop_active
        self.pub_estop.publish(estop_msg)

    def _trigger_violation(self, reason: str):
        """Handle a safety violation."""
        if self.e_stop_active:
            return  # Already in e-stop

        self.get_logger().error(f'SAFETY VIOLATION: {reason}')
        self.safety_violations.append(reason)
        self.e_stop_active = True
        self.is_safe = False

        # Disable arm teleop
        if self.disable_client.service_is_ready():
            self.disable_client.call_async(Trigger.Request())

        # Execute recovery
        if self.recovery_action == 'stow':
            if self.stow_client.service_is_ready():
                self.stow_client.call_async(Trigger.Request())
                self.get_logger().warn('Recovery: stowing arm')

    def reset_safety_cb(self, request, response):
        """Reset safety monitor after violations are addressed."""
        self.e_stop_active = False
        self.is_safe = True
        self.safety_violations = []
        response.success = True
        response.message = 'Safety monitor reset. E-stop cleared.'
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = SafetyMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
