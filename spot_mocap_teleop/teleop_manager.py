#!/usr/bin/env python3
"""
teleop_manager.py
=================
High-level orchestrator for the Spot mocap teleoperation system.
Manages the startup sequence, calibration workflow, and provides
a unified interface for controlling the teleop pipeline.

Services:
    /teleop/init_system       - Initialize robot (claim, power on, stand, unstow arm)
    /teleop/start_teleop      - Start full teleop pipeline
    /teleop/stop_teleop       - Stop teleop and stow arm
    /teleop/emergency_stop    - Immediate stop of all operations

Publishes:
    /teleop/system_state      (std_msgs/msg/String)
    /teleop/diagnostics       (std_msgs/msg/String)
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
import time

from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger


class TeleopState:
    """Possible teleop system states."""
    IDLE = 'IDLE'
    INITIALIZING = 'INITIALIZING'
    CALIBRATING = 'CALIBRATING'
    READY = 'READY'
    TELEOPERATING = 'TELEOPERATING'
    STOPPED = 'STOPPED'
    ERROR = 'ERROR'
    E_STOP = 'E_STOP'


class TeleopManager(Node):
    """Orchestrates the Spot mocap teleoperation system."""

    def __init__(self):
        super().__init__('teleop_manager')

        self.cb_group = ReentrantCallbackGroup()
        self.state = TeleopState.IDLE

        self.declare_parameter('spot_name', '')
        self.spot_name = self.get_parameter('spot_name').value
        prefix = f'/{self.spot_name}' if self.spot_name else ''

        # ---- Service clients to Spot driver ----
        self.clients = {
            'claim': self.create_client(
                Trigger, f'{prefix}/claim', callback_group=self.cb_group),
            'power_on': self.create_client(
                Trigger, f'{prefix}/power_on', callback_group=self.cb_group),
            'stand': self.create_client(
                Trigger, f'{prefix}/stand', callback_group=self.cb_group),
            'sit': self.create_client(
                Trigger, f'{prefix}/sit', callback_group=self.cb_group),
            'arm_unstow': self.create_client(
                Trigger, f'{prefix}/arm_unstow', callback_group=self.cb_group),
            'arm_stow': self.create_client(
                Trigger, f'{prefix}/arm_stow', callback_group=self.cb_group),
            'power_off': self.create_client(
                Trigger, f'{prefix}/power_off', callback_group=self.cb_group),
        }

        # ---- Service clients to teleop nodes ----
        self.teleop_clients = {
            'calibrate_glove': self.create_client(
                Trigger, '/teleop/calibrate_glove', callback_group=self.cb_group),
            'enable_arm': self.create_client(
                Trigger, '/teleop/enable_arm', callback_group=self.cb_group),
            'disable_arm': self.create_client(
                Trigger, '/teleop/disable_arm', callback_group=self.cb_group),
            'stow_arm': self.create_client(
                Trigger, '/teleop/stow_arm', callback_group=self.cb_group),
            'start_recording': self.create_client(
                Trigger, '/teleop/start_recording', callback_group=self.cb_group),
            'stop_recording': self.create_client(
                Trigger, '/teleop/stop_recording', callback_group=self.cb_group),
            'reset_safety': self.create_client(
                Trigger, '/teleop/reset_safety', callback_group=self.cb_group),
        }

        # Subscribers
        self.sub_estop = self.create_subscription(
            Bool, '/teleop/e_stop', self.estop_cb, 10
        )
        self.sub_tracking = self.create_subscription(
            Bool, '/teleop/glove_tracking', self.tracking_cb, 10
        )

        # Services (orchestration commands)
        self.srv_init = self.create_service(
            Trigger, '/teleop/init_system', self.init_system_cb,
            callback_group=self.cb_group
        )
        self.srv_start = self.create_service(
            Trigger, '/teleop/start_teleop', self.start_teleop_cb,
            callback_group=self.cb_group
        )
        self.srv_stop = self.create_service(
            Trigger, '/teleop/stop_teleop', self.stop_teleop_cb,
            callback_group=self.cb_group
        )
        self.srv_estop = self.create_service(
            Trigger, '/teleop/emergency_stop', self.emergency_stop_cb,
            callback_group=self.cb_group
        )

        # Publishers
        self.pub_state = self.create_publisher(String, '/teleop/system_state', 10)
        self.pub_diag = self.create_publisher(String, '/teleop/diagnostics', 10)

        # State publish timer
        self.create_timer(0.5, self.publish_state)

        # Tracking state
        self.glove_tracked = False

        self.get_logger().info('TeleopManager started. State: IDLE')

    def estop_cb(self, msg: Bool):
        if msg.data and self.state == TeleopState.TELEOPERATING:
            self.state = TeleopState.E_STOP
            self.get_logger().error('E-STOP triggered by safety monitor!')

    def tracking_cb(self, msg: Bool):
        self.glove_tracked = msg.data

    def _call_service_sync(self, client, timeout=5.0) -> tuple:
        """Call a Trigger service and wait for result."""
        if not client.service_is_ready():
            if not client.wait_for_service(timeout_sec=timeout):
                return False, f'Service {client.srv_name} not available'

        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)

        if future.result() is not None:
            result = future.result()
            return result.success, result.message
        return False, 'Service call timed out'

    def init_system_cb(self, request, response):
        """Initialize the robot: claim -> power on -> stand -> unstow arm."""
        self.state = TeleopState.INITIALIZING
        self._publish_diag('Initializing robot system...')

        steps = [
            ('claim', 'Claiming robot lease...'),
            ('power_on', 'Powering on...'),
            ('stand', 'Standing up...'),
            ('arm_unstow', 'Deploying arm...'),
        ]

        for service_name, step_msg in steps:
            self._publish_diag(step_msg)
            self.get_logger().info(step_msg)

            if service_name in self.clients:
                success, msg = self._call_service_sync(
                    self.clients[service_name], timeout=30.0
                )
                if not success:
                    self.state = TeleopState.ERROR
                    response.success = False
                    response.message = f'Failed at {service_name}: {msg}'
                    self.get_logger().error(response.message)
                    return response
            time.sleep(1.0)  # Wait between steps

        self.state = TeleopState.READY
        response.success = True
        response.message = 'Robot initialized. Ready for calibration.'
        self._publish_diag(response.message)
        self.get_logger().info(response.message)
        return response

    def start_teleop_cb(self, request, response):
        """Start teleoperation: calibrate glove -> enable arm -> start recording."""
        if self.state not in [TeleopState.READY, TeleopState.STOPPED]:
            response.success = False
            response.message = f'Cannot start teleop in state: {self.state}'
            return response

        self.state = TeleopState.CALIBRATING
        self._publish_diag('Starting teleop pipeline...')

        # 1. Calibrate glove
        if not self.glove_tracked:
            response.success = False
            response.message = (
                'Glove not tracked! Ensure OptiTrack can see the glove markers.'
            )
            self.state = TeleopState.READY
            return response

        self._publish_diag('Calibrating glove position...')
        success, msg = self._call_service_sync(
            self.teleop_clients['calibrate_glove']
        )
        if not success:
            self.get_logger().warn(f'Calibration warning: {msg}')

        # 2. Reset safety
        self._call_service_sync(self.teleop_clients['reset_safety'])

        # 3. Enable arm control
        self._publish_diag('Enabling arm control...')
        success, msg = self._call_service_sync(
            self.teleop_clients['enable_arm']
        )

        # 4. Start recording
        self._publish_diag('Starting rosbag recording...')
        self._call_service_sync(self.teleop_clients['start_recording'])

        self.state = TeleopState.TELEOPERATING
        response.success = True
        response.message = 'Teleop ACTIVE. Recording started.'
        self._publish_diag(response.message)
        self.get_logger().info(response.message)
        return response

    def stop_teleop_cb(self, request, response):
        """Stop teleoperation gracefully."""
        self._publish_diag('Stopping teleop...')

        # 1. Disable arm
        self._call_service_sync(self.teleop_clients['disable_arm'])

        # 2. Stow arm
        self._call_service_sync(self.teleop_clients['stow_arm'])

        # 3. Stop recording
        self._call_service_sync(self.teleop_clients['stop_recording'])

        self.state = TeleopState.STOPPED
        response.success = True
        response.message = 'Teleop stopped. Recording saved.'
        self._publish_diag(response.message)
        self.get_logger().info(response.message)
        return response

    def emergency_stop_cb(self, request, response):
        """Emergency stop: immediately disable everything."""
        self.get_logger().error('EMERGENCY STOP TRIGGERED!')
        self.state = TeleopState.E_STOP

        # Disable arm immediately
        self._call_service_sync(self.teleop_clients['disable_arm'])

        # Stow arm
        self._call_service_sync(self.teleop_clients['stow_arm'])

        response.success = True
        response.message = 'EMERGENCY STOP executed. Arm disabled and stowing.'
        return response

    def publish_state(self):
        """Periodically publish system state."""
        msg = String()
        msg.data = self.state
        self.pub_state.publish(msg)

    def _publish_diag(self, text: str):
        """Publish a diagnostic message."""
        msg = String()
        msg.data = f'[{self.state}] {text}'
        self.pub_diag.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopManager()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
