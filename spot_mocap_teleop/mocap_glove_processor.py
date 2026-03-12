#!/usr/bin/env python3
"""
mocap_glove_processor.py
========================
Reads OptiTrack rigid body data from MOCAP4ROS2 and extracts the glove
rigid body pose. Applies calibration offset and scaling to map from
the OptiTrack workspace to the Spot arm workspace.

Subscribes:
    /mocap4r2_optitrack/rigid_bodies  (mocap4r2_msgs/msg/RigidBodies)

Publishes:
    /teleop/raw_glove_pose   (geometry_msgs/msg/PoseStamped)
    /teleop/glove_velocity   (geometry_msgs/msg/TwistStamped)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np

from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from std_msgs.msg import Bool, Header
from std_srvs.srv import Trigger

try:
    from mocap4r2_msgs.msg import RigidBodies
    HAS_MOCAP_MSGS = True
except ImportError:
    HAS_MOCAP_MSGS = False


class MocapGloveProcessor(Node):
    """Processes OptiTrack glove rigid body and publishes scaled pose."""

    def __init__(self):
        super().__init__('mocap_glove_processor')

        # Parameters
        self.declare_parameter('glove_rigid_body_name', 'GloveRB')
        self.declare_parameter('mocap_frame_id', 'mocap_world')
        self.declare_parameter('glove_frame_id', 'glove_ee')
        self.declare_parameter('publish_rate', 100.0)
        self.declare_parameter('position_scale.x', 0.6)
        self.declare_parameter('position_scale.y', 0.6)
        self.declare_parameter('position_scale.z', 0.6)
        self.declare_parameter('calibration_offset.x', 0.0)
        self.declare_parameter('calibration_offset.y', 0.0)
        self.declare_parameter('calibration_offset.z', 0.0)

        self.rb_name = self.get_parameter('glove_rigid_body_name').value
        self.mocap_frame = self.get_parameter('mocap_frame_id').value
        self.glove_frame = self.get_parameter('glove_frame_id').value
        self.scale = np.array([
            self.get_parameter('position_scale.x').value,
            self.get_parameter('position_scale.y').value,
            self.get_parameter('position_scale.z').value,
        ])
        self.cal_offset = np.array([
            self.get_parameter('calibration_offset.x').value,
            self.get_parameter('calibration_offset.y').value,
            self.get_parameter('calibration_offset.z').value,
        ])

        # State
        self.is_calibrated = False
        self.calibration_pose = None
        self.last_pose = None
        self.last_time = None

        # QoS for low-latency
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        if HAS_MOCAP_MSGS:
            self.sub_rigid_bodies = self.create_subscription(
                RigidBodies,
                '/mocap4r2_optitrack/rigid_bodies',
                self.rigid_bodies_cb,
                qos
            )
        else:
            self.get_logger().warn(
                'mocap4r2_msgs not found. Using PoseStamped fallback on '
                '/mocap4r2_optitrack/rigid_body_pose'
            )
            self.sub_pose = self.create_subscription(
                PoseStamped,
                '/mocap4r2_optitrack/rigid_body_pose',
                self.pose_fallback_cb,
                qos
            )

        # Publishers
        self.pub_raw_pose = self.create_publisher(PoseStamped, '/teleop/raw_glove_pose', 10)
        self.pub_velocity = self.create_publisher(TwistStamped, '/teleop/glove_velocity', 10)
        self.pub_tracking = self.create_publisher(Bool, '/teleop/glove_tracking', 10)

        # Calibration service
        self.srv_calibrate = self.create_service(
            Trigger, '/teleop/calibrate_glove', self.calibrate_cb
        )

        self.get_logger().info(
            f'MocapGloveProcessor started. Looking for rigid body: "{self.rb_name}"'
        )

    def rigid_bodies_cb(self, msg):
        """Process rigid bodies from MOCAP4ROS2."""
        for rb in msg.rigidbodies:
            if rb.rigid_body_name == self.rb_name:
                pose = PoseStamped()
                pose.header = msg.header
                pose.header.frame_id = self.mocap_frame
                pose.pose = rb.pose
                self._process_pose(pose)

                # Publish tracking status
                tracking_msg = Bool()
                tracking_msg.data = True
                self.pub_tracking.publish(tracking_msg)
                return

        # Rigid body not found
        tracking_msg = Bool()
        tracking_msg.data = False
        self.pub_tracking.publish(tracking_msg)

    def pose_fallback_cb(self, msg):
        """Fallback for direct PoseStamped subscription."""
        self._process_pose(msg)

    def _process_pose(self, pose_msg: PoseStamped):
        """Apply calibration, scaling, and compute velocity."""
        pos = np.array([
            pose_msg.pose.position.x,
            pose_msg.pose.position.y,
            pose_msg.pose.position.z,
        ])

        # Apply calibration offset (subtract origin)
        if self.is_calibrated and self.calibration_pose is not None:
            pos = pos - self.calibration_pose
        pos = pos - self.cal_offset

        # Apply scaling
        scaled_pos = pos * self.scale

        # Build output pose
        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.glove_frame
        out.pose.position.x = float(scaled_pos[0])
        out.pose.position.y = float(scaled_pos[1])
        out.pose.position.z = float(scaled_pos[2])
        out.pose.orientation = pose_msg.pose.orientation

        self.pub_raw_pose.publish(out)

        # Compute velocity via finite difference
        now = self.get_clock().now()
        if self.last_pose is not None and self.last_time is not None:
            dt = (now - self.last_time).nanoseconds * 1e-9
            if dt > 0.001:
                vel = (scaled_pos - self.last_pose) / dt
                twist = TwistStamped()
                twist.header.stamp = now.to_msg()
                twist.header.frame_id = self.glove_frame
                twist.twist.linear.x = float(vel[0])
                twist.twist.linear.y = float(vel[1])
                twist.twist.linear.z = float(vel[2])
                self.pub_velocity.publish(twist)

        self.last_pose = scaled_pos.copy()
        self.last_time = now

    def calibrate_cb(self, request, response):
        """Set current glove position as the calibration origin."""
        if self.last_pose is not None:
            # Store the raw (unscaled) position as calibration origin
            self.calibration_pose = self.last_pose / self.scale + self.cal_offset
            self.is_calibrated = True
            response.success = True
            response.message = (
                f'Calibration set at [{self.calibration_pose[0]:.3f}, '
                f'{self.calibration_pose[1]:.3f}, {self.calibration_pose[2]:.3f}]'
            )
            self.get_logger().info(response.message)
        else:
            response.success = False
            response.message = 'No mocap data received yet. Cannot calibrate.'
            self.get_logger().warn(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = MocapGloveProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
