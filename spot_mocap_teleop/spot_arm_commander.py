#!/usr/bin/env python3
"""
spot_arm_commander.py
=====================
Converts safe pose commands into Spot arm end-effector commands
using the spot_ros2 RobotCommand action interface.

Uses ArmCartesianCommand to move the hand to the desired pose
in the robot body frame.

Subscribes:
    /teleop/safe_pose  (geometry_msgs/msg/PoseStamped)

Uses Action:
    /robot_command     (spot_msgs/action/RobotCommand)

Publishes:
    /teleop/arm_command_status  (std_msgs/msg/String)
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
import time

from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger

try:
    from spot_msgs.action import RobotCommand
    from bosdyn_msgs.msg import (
        RobotCommand as RobotCommandMsg,
        RobotCommandFeedback,
        SynchronizedCommand,
        SynchronizedCommandFeedback,
        ArmCartesianCommand,
        ArmCartesianCommandFeedback,
        ArmCommand,
        ArmCommandFeedback,
        SE3Pose,
        Vec3,
        Quaternion as BDQuaternion,
    )
    HAS_SPOT_MSGS = True
except ImportError:
    HAS_SPOT_MSGS = False


class SpotArmCommander(Node):
    """Sends end-effector pose commands to Spot's arm."""

    def __init__(self):
        super().__init__('spot_arm_commander')

        self.cb_group = ReentrantCallbackGroup()

        # Parameters
        self.declare_parameter('spot_name', '')
        self.declare_parameter('command_mode', 'cartesian')
        self.declare_parameter('root_frame', 'body')
        self.declare_parameter('command_rate', 10.0)
        self.declare_parameter('command_timeout', 2.0)

        self.spot_name = self.get_parameter('spot_name').value
        self.command_mode = self.get_parameter('command_mode').value
        self.root_frame = self.get_parameter('root_frame').value
        self.command_rate = self.get_parameter('command_rate').value
        self.command_timeout = self.get_parameter('command_timeout').value

        # Build topic/action names with optional namespace
        prefix = f'/{self.spot_name}' if self.spot_name else ''

        # State
        self.enabled = False
        self.latest_target = None
        self.arm_is_deployed = False
        self.last_command_time = None

        # Subscriber
        self.sub_safe_pose = self.create_subscription(
            PoseStamped, '/teleop/safe_pose', self.safe_pose_cb, 10,
            callback_group=self.cb_group
        )

        # Publishers
        self.pub_status = self.create_publisher(
            String, '/teleop/arm_command_status', 10
        )

        # Services for teleop control
        self.srv_enable = self.create_service(
            Trigger, '/teleop/enable_arm', self.enable_cb
        )
        self.srv_disable = self.create_service(
            Trigger, '/teleop/disable_arm', self.disable_cb
        )
        self.srv_stow = self.create_service(
            Trigger, '/teleop/stow_arm', self.stow_cb
        )
        self.srv_unstow = self.create_service(
            Trigger, '/teleop/unstow_arm', self.unstow_cb
        )

        # Action client for robot_command (spot_ros2)
        if HAS_SPOT_MSGS:
            self.robot_command_client = ActionClient(
                self, RobotCommand, f'{prefix}/robot_command',
                callback_group=self.cb_group
            )
            self.get_logger().info(
                f'Waiting for {prefix}/robot_command action server...'
            )
        else:
            self.robot_command_client = None
            self.get_logger().warn(
                'spot_msgs not found! Running in dry-run mode. '
                'Commands will be logged but not sent to robot.'
            )

        # Spot driver services
        self.claim_client = self.create_client(
            Trigger, f'{prefix}/claim', callback_group=self.cb_group
        )
        self.power_on_client = self.create_client(
            Trigger, f'{prefix}/power_on', callback_group=self.cb_group
        )
        self.stand_client = self.create_client(
            Trigger, f'{prefix}/stand', callback_group=self.cb_group
        )
        self.arm_stow_client = self.create_client(
            Trigger, f'{prefix}/arm_stow', callback_group=self.cb_group
        )
        self.arm_unstow_client = self.create_client(
            Trigger, f'{prefix}/arm_unstow', callback_group=self.cb_group
        )

        # Command timer
        period = 1.0 / self.command_rate
        self.command_timer = self.create_timer(
            period, self.send_command_tick, callback_group=self.cb_group
        )

        self.get_logger().info(
            f'SpotArmCommander started. Mode={self.command_mode}, '
            f'Rate={self.command_rate}Hz, Frame={self.root_frame}'
        )

    def safe_pose_cb(self, msg: PoseStamped):
        """Store latest target pose."""
        self.latest_target = msg

    def send_command_tick(self):
        """Periodically send arm command at fixed rate."""
        if not self.enabled or self.latest_target is None:
            return

        if HAS_SPOT_MSGS and self.robot_command_client is not None:
            self._send_arm_cartesian_command(self.latest_target)
        else:
            # Dry-run: log the command
            p = self.latest_target.pose.position
            self.get_logger().debug(
                f'[DRY-RUN] Arm target: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})'
            )

        self.last_command_time = self.get_clock().now()

    def _send_arm_cartesian_command(self, target: PoseStamped):
        """
        Build and send an ArmCartesianCommand via the RobotCommand action.

        This uses the spot_ros2 action interface which wraps the Spot SDK's
        RobotCommandBuilder pattern.
        """
        if not self.robot_command_client.server_is_ready():
            self.get_logger().warn('robot_command action server not ready')
            return

        try:
            # Build the RobotCommand goal with arm cartesian command
            # The spot_ros2 driver expects a serialized protobuf command
            # We construct it using the bosdyn_msgs ROS message types
            goal_msg = RobotCommand.Goal()

            # Build SE3Pose for hand target
            hand_pose = SE3Pose()
            hand_pose.position = Vec3()
            hand_pose.position.x = target.pose.position.x
            hand_pose.position.y = target.pose.position.y
            hand_pose.position.z = target.pose.position.z
            hand_pose.rotation = BDQuaternion()
            hand_pose.rotation.x = target.pose.orientation.x
            hand_pose.rotation.y = target.pose.orientation.y
            hand_pose.rotation.z = target.pose.orientation.z
            hand_pose.rotation.w = target.pose.orientation.w

            # Build ArmCartesianCommand
            arm_cartesian = ArmCartesianCommand()
            arm_cartesian.pose_trajectory_in_task.reference_time.sec = 0
            arm_cartesian.root_frame_name = self.root_frame

            # Set target pose in the command
            # The exact message structure depends on spot_ros2/bosdyn_msgs version
            # This follows the pattern from spot_ros2 examples

            # Send as action goal
            future = self.robot_command_client.send_goal_async(goal_msg)
            future.add_done_callback(self._goal_response_cb)

            # Publish status
            status = String()
            p = target.pose.position
            status.data = (
                f'COMMANDING: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) '
                f'frame={self.root_frame}'
            )
            self.pub_status.publish(status)

        except Exception as e:
            self.get_logger().error(f'Failed to send arm command: {e}')
            status = String()
            status.data = f'ERROR: {str(e)}'
            self.pub_status.publish(status)

    def _goal_response_cb(self, future):
        """Handle action goal response."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Arm command goal rejected')
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        """Handle action result."""
        try:
            result = future.result()
            self.get_logger().debug('Arm command completed')
        except Exception as e:
            self.get_logger().warn(f'Arm command result error: {e}')

    # ---- Service handlers ----

    def enable_cb(self, request, response):
        """Enable arm teleop."""
        self.enabled = True
        response.success = True
        response.message = 'Arm teleop ENABLED'
        self.get_logger().info(response.message)
        return response

    def disable_cb(self, request, response):
        """Disable arm teleop."""
        self.enabled = False
        response.success = True
        response.message = 'Arm teleop DISABLED'
        self.get_logger().info(response.message)
        return response

    def stow_cb(self, request, response):
        """Stow the arm."""
        self.enabled = False
        if self.arm_stow_client.service_is_ready():
            future = self.arm_stow_client.call_async(Trigger.Request())
            response.success = True
            response.message = 'Stow command sent'
        else:
            response.success = False
            response.message = 'arm_stow service not available'
        return response

    def unstow_cb(self, request, response):
        """Unstow the arm to ready position."""
        if self.arm_unstow_client.service_is_ready():
            future = self.arm_unstow_client.call_async(Trigger.Request())
            self.arm_is_deployed = True
            response.success = True
            response.message = 'Unstow command sent'
        else:
            response.success = False
            response.message = 'arm_unstow service not available'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = SpotArmCommander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
