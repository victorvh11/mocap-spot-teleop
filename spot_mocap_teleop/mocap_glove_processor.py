#!/usr/bin/env python3
"""
mocap_glove_processor.py
========================
Maps hand movement in OptiTrack space to end-effector commands for Spot arm.

The core idea is delta-based motion mapping:
    1. Operator calls /teleop/calibrate_glove -> records glove position as ORIGIN
    2. Every subsequent frame computes DELTA = (current_glove - origin)
    3. DELTA is scaled to match robot workspace proportions
    4. Output = arm_home_pose + scaled_delta

This way, moving your hand 30cm forward from the calibration point moves
the end-effector (30cm * scale) forward from its home position.

Subscribes:
    /rigid_bodies  (mocap4r2_msgs/msg/RigidBodies)

Publishes:
    /teleop/raw_glove_pose   (geometry_msgs/msg/PoseStamped)  -- target EE pose
    /teleop/glove_velocity   (geometry_msgs/msg/TwistStamped)
    /teleop/glove_tracking   (std_msgs/msg/Bool)
    TF: mocap_world -> mocap/<rigid_body_name>  (all rigid bodies)
    TF: body -> glove_target  (the target EE pose in robot frame)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np

from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

try:
    from mocap4r2_msgs.msg import RigidBodies
    HAS_MOCAP_MSGS = True
except ImportError:
    HAS_MOCAP_MSGS = False


class MocapGloveProcessor(Node):
    """Maps glove mocap movement to Spot arm end-effector target poses."""

    def __init__(self):
        super().__init__('mocap_glove_processor')

        # ---- Parameters ----
        self.declare_parameter('glove_rigid_body_name', 'GloveRB')
        self.declare_parameter('mocap_frame_id', 'mocap_world')
        self.declare_parameter('publish_rate', 100.0)

        # Scale: ratio of robot workspace to operator workspace.
        # If scale=0.5 and you move your hand 40cm, EE moves 20cm.
        self.declare_parameter('position_scale.x', 0.5)
        self.declare_parameter('position_scale.y', 0.5)
        self.declare_parameter('position_scale.z', 0.5)

        # Arm home pose (in body frame): where the EE rests before teleop.
        # This is the "center" of the robot workspace. Deltas are added to this.
        # Spot arm ready position is approximately x=0.5, y=0, z=0 in body frame.
        self.declare_parameter('arm_home_pose.x', 0.5)
        self.declare_parameter('arm_home_pose.y', 0.0)
        self.declare_parameter('arm_home_pose.z', 0.0)

        # Axis mapping: OptiTrack frame may differ from Spot body frame.
        # sign flips: if mocap +Y = robot -Y, set sign_y = -1.0
        self.declare_parameter('axis_mapping.sign_x', 1.0)
        self.declare_parameter('axis_mapping.sign_y', 1.0)
        self.declare_parameter('axis_mapping.sign_z', 1.0)
        # Axis reorder: indices [0,1,2] mean X->X, Y->Y, Z->Z.
        # Use [0,2,1] to swap Y and Z, etc.
        self.declare_parameter('axis_mapping.order', [0, 1, 2])

        # Read parameters
        self.rb_name = self.get_parameter('glove_rigid_body_name').value
        self.mocap_frame = self.get_parameter('mocap_frame_id').value
        self.scale = np.array([
            self.get_parameter('position_scale.x').value,
            self.get_parameter('position_scale.y').value,
            self.get_parameter('position_scale.z').value,
        ])
        self.arm_home = np.array([
            self.get_parameter('arm_home_pose.x').value,
            self.get_parameter('arm_home_pose.y').value,
            self.get_parameter('arm_home_pose.z').value,
        ])
        self.axis_sign = np.array([
            self.get_parameter('axis_mapping.sign_x').value,
            self.get_parameter('axis_mapping.sign_y').value,
            self.get_parameter('axis_mapping.sign_z').value,
        ])
        order_param = self.get_parameter('axis_mapping.order').value
        self.axis_order = [int(i) for i in order_param]

        # ---- State ----
        self.is_calibrated = False
        self.glove_origin = None       # mocap position at calibration time
        self.last_raw_pos = None       # last raw mocap position (for calibration)
        self.last_output_pos = None    # last published target (for velocity)
        self.last_time = None

        # ---- TF broadcasters ----
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_frames()

        # ---- QoS ----
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ---- Subscribers ----
        if HAS_MOCAP_MSGS:
            self.sub_rigid_bodies = self.create_subscription(
                RigidBodies,
                '/rigid_bodies',
                self.rigid_bodies_cb,
                qos
            )
        else:
            self.get_logger().warn(
                'mocap4r2_msgs not found. Using PoseStamped fallback.'
            )
            self.sub_pose = self.create_subscription(
                PoseStamped,
                '/mocap4r2_optitrack/rigid_body_pose',
                self.pose_fallback_cb,
                qos
            )

        # ---- Publishers ----
        self.pub_target_pose = self.create_publisher(
            PoseStamped, '/teleop/raw_glove_pose', 10)
        self.pub_velocity = self.create_publisher(
            TwistStamped, '/teleop/glove_velocity', 10)
        self.pub_tracking = self.create_publisher(
            Bool, '/teleop/glove_tracking', 10)

        # ---- Services ----
        self.srv_calibrate = self.create_service(
            Trigger, '/teleop/calibrate_glove', self.calibrate_cb)

        self.get_logger().info(
            f'MocapGloveProcessor started.\n'
            f'  Rigid body: "{self.rb_name}"\n'
            f'  Scale: {self.scale}\n'
            f'  Arm home: {self.arm_home}\n'
            f'  Axis sign: {self.axis_sign}, order: {self.axis_order}\n'
            f'  Waiting for calibration (call /teleop/calibrate_glove)...'
        )

    # =====================================================================
    # Static TF setup
    # =====================================================================

    def _publish_static_frames(self):
        """Publish static TFs so frames exist in the TF tree."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = self.mocap_frame
        t.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform(t)

    # =====================================================================
    # Callbacks
    # =====================================================================

    def rigid_bodies_cb(self, msg):
        """Process incoming rigid bodies from MOCAP4ROS2."""
        found_glove = False

        for rb in msg.rigidbodies:
            # Broadcast every rigid body as a TF (for RViz)
            self._broadcast_rb_tf(rb)

            if rb.rigid_body_name == self.rb_name:
                found_glove = True
                raw_pos = np.array([
                    rb.pose.position.x,
                    rb.pose.position.y,
                    rb.pose.position.z,
                ])
                raw_quat = np.array([
                    rb.pose.orientation.x,
                    rb.pose.orientation.y,
                    rb.pose.orientation.z,
                    rb.pose.orientation.w,
                ])
                self._process_glove(raw_pos, raw_quat)

        tracking_msg = Bool()
        tracking_msg.data = found_glove
        self.pub_tracking.publish(tracking_msg)

    def pose_fallback_cb(self, msg):
        """Fallback for direct PoseStamped input."""
        raw_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        raw_quat = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ])
        self._process_glove(raw_pos, raw_quat)

    # =====================================================================
    # Core processing
    # =====================================================================

    def _process_glove(self, raw_pos: np.ndarray, raw_quat: np.ndarray):
        """
        Convert glove mocap position to arm EE target.

        Before calibration: stores raw data but publishes nothing.
        After calibration:
            delta = raw_pos - glove_origin
            delta_mapped = reorder axes + apply sign
            delta_scaled = delta_mapped * scale
            target_pos = arm_home + delta_scaled
        """
        now = self.get_clock().now()
        self.last_raw_pos = raw_pos.copy()

        if not self.is_calibrated:
            self.last_time = now
            return

        # --- Delta from calibration origin ---
        delta = raw_pos - self.glove_origin

        # --- Axis reordering (e.g. mocap Y -> robot Z) ---
        delta_reordered = np.array([delta[self.axis_order[i]] for i in range(3)])

        # --- Axis sign (e.g. mocap +Y = robot -Y) ---
        delta_signed = delta_reordered * self.axis_sign

        # --- Scale (operator workspace -> robot workspace) ---
        delta_scaled = delta_signed * self.scale

        # --- Target = home + scaled delta ---
        target_pos = self.arm_home + delta_scaled

        # --- Publish target pose in body frame ---
        out = PoseStamped()
        out.header.stamp = now.to_msg()
        out.header.frame_id = 'body'
        out.pose.position.x = float(target_pos[0])
        out.pose.position.y = float(target_pos[1])
        out.pose.position.z = float(target_pos[2])
        out.pose.orientation.x = float(raw_quat[0])
        out.pose.orientation.y = float(raw_quat[1])
        out.pose.orientation.z = float(raw_quat[2])
        out.pose.orientation.w = float(raw_quat[3])
        self.pub_target_pose.publish(out)

        # --- Broadcast TF: body -> glove_target ---
        tf_msg = TransformStamped()
        tf_msg.header.stamp = now.to_msg()
        tf_msg.header.frame_id = 'body'
        tf_msg.child_frame_id = 'glove_target'
        tf_msg.transform.translation.x = float(target_pos[0])
        tf_msg.transform.translation.y = float(target_pos[1])
        tf_msg.transform.translation.z = float(target_pos[2])
        tf_msg.transform.rotation = out.pose.orientation
        self.tf_broadcaster.sendTransform(tf_msg)

        # --- Velocity ---
        if self.last_output_pos is not None and self.last_time is not None:
            dt = (now - self.last_time).nanoseconds * 1e-9
            if dt > 0.001:
                vel = (target_pos - self.last_output_pos) / dt
                twist = TwistStamped()
                twist.header.stamp = now.to_msg()
                twist.header.frame_id = 'body'
                twist.twist.linear.x = float(vel[0])
                twist.twist.linear.y = float(vel[1])
                twist.twist.linear.z = float(vel[2])
                self.pub_velocity.publish(twist)

        self.last_output_pos = target_pos.copy()
        self.last_time = now

    # =====================================================================
    # TF helpers
    # =====================================================================

    def _broadcast_rb_tf(self, rb):
        """Broadcast a raw rigid body as TF for RViz visualization."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.mocap_frame
        t.child_frame_id = f'mocap/{rb.rigid_body_name}'
        t.transform.translation.x = rb.pose.position.x
        t.transform.translation.y = rb.pose.position.y
        t.transform.translation.z = rb.pose.position.z
        t.transform.rotation = rb.pose.orientation
        self.tf_broadcaster.sendTransform(t)

    # =====================================================================
    # Calibration
    # =====================================================================

    def calibrate_cb(self, request, response):
        """
        Record current glove position as the motion origin.

        After this, any movement relative to this position gets mapped
        to end-effector movement relative to arm_home_pose.
        """
        if self.last_raw_pos is None:
            response.success = False
            response.message = 'No mocap data received yet. Cannot calibrate.'
            self.get_logger().warn(response.message)
            return response

        self.glove_origin = self.last_raw_pos.copy()
        self.is_calibrated = True
        self.last_output_pos = self.arm_home.copy()

        response.success = True
        response.message = (
            f'Calibration OK. Glove origin: '
            f'[{self.glove_origin[0]:.4f}, '
            f'{self.glove_origin[1]:.4f}, '
            f'{self.glove_origin[2]:.4f}] (mocap frame). '
            f'EE starts at home: [{self.arm_home[0]:.3f}, '
            f'{self.arm_home[1]:.3f}, {self.arm_home[2]:.3f}] (body frame).'
        )
        self.get_logger().info(response.message)
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
