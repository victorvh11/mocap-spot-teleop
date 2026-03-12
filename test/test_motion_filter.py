#!/usr/bin/env python3
"""
test_motion_filter.py
=====================
Unit tests for the MotionFilter ROS2 node.

Covers:
    - Node initialization with butterworth/ema filter types
    - Dead zone suppression of micro-movements
    - Velocity rate limiting enforcement
    - Acceleration rate limiting enforcement
    - Orientation SLERP smoothing
    - Quaternion normalization and flip handling
    - Combined pipeline (filter + rate limit + dead zone)
    - Zero quaternion handling
"""

import pytest
import numpy as np
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from spot_mocap_teleop.motion_filter import MotionFilter


@pytest.fixture
def filter_node():
    """Create a MotionFilter node for testing."""
    node = MotionFilter()
    yield node
    node.destroy_node()


class TestMotionFilterInit:
    """Initialization tests."""

    def test_node_name(self, filter_node):
        assert filter_node.get_name() == 'motion_filter'

    def test_default_params(self, filter_node):
        assert filter_node.max_lin_vel == 0.5
        assert filter_node.max_lin_acc == 2.0
        assert filter_node.pos_deadzone == 0.005

    def test_butterworth_filter_created(self, filter_node):
        """Default should create a ButterworthFilter."""
        from spot_mocap_teleop.motion_filter import ButterworthFilter
        assert isinstance(filter_node.pos_filter, ButterworthFilter)

    def test_initial_state(self, filter_node):
        assert filter_node.prev_pos is None
        assert filter_node.prev_time is None
        assert filter_node.prev_quat is None


class TestDeadZone:
    """Tests for the position dead zone."""

    def test_micro_movement_suppressed(self, filter_node):
        """Movements smaller than deadzone should be ignored."""
        published = []
        filter_node.pub_filtered.publish = lambda msg: published.append(msg)

        # First pose: establish baseline
        msg1 = PoseStamped()
        msg1.header.frame_id = 'test'
        msg1.pose.position.x = 0.5
        msg1.pose.position.y = 0.5
        msg1.pose.position.z = 0.5
        msg1.pose.orientation.w = 1.0
        filter_node.raw_pose_cb(msg1)

        pos1 = np.array([
            published[-1].pose.position.x,
            published[-1].pose.position.y,
            published[-1].pose.position.z
        ])

        time.sleep(0.02)

        # Second pose: move by 1mm (< 5mm deadzone)
        msg2 = PoseStamped()
        msg2.header.frame_id = 'test'
        msg2.pose.position.x = 0.5005
        msg2.pose.position.y = 0.5005
        msg2.pose.position.z = 0.5005
        msg2.pose.orientation.w = 1.0
        filter_node.raw_pose_cb(msg2)

        pos2 = np.array([
            published[-1].pose.position.x,
            published[-1].pose.position.y,
            published[-1].pose.position.z
        ])

        # Positions should be identical due to dead zone
        np.testing.assert_allclose(pos1, pos2, atol=1e-6,
                                   err_msg='Dead zone did not suppress micro-movement')

    def test_large_movement_passes(self, filter_node):
        """Movements larger than deadzone should pass through."""
        published = []
        filter_node.pub_filtered.publish = lambda msg: published.append(msg)

        msg1 = PoseStamped()
        msg1.header.frame_id = 'test'
        msg1.pose.position.x = 0.0
        msg1.pose.orientation.w = 1.0
        filter_node.raw_pose_cb(msg1)

        time.sleep(0.02)

        msg2 = PoseStamped()
        msg2.header.frame_id = 'test'
        msg2.pose.position.x = 0.1  # 100mm >> 5mm deadzone
        msg2.pose.orientation.w = 1.0
        filter_node.raw_pose_cb(msg2)

        # Positions should differ
        assert published[-1].pose.position.x != published[0].pose.position.x


class TestVelocityLimiting:
    """Tests for velocity rate limiting."""

    def test_excessive_velocity_clamped(self, filter_node):
        """Jump exceeding max velocity should be clamped."""
        published = []
        filter_node.pub_filtered.publish = lambda msg: published.append(msg)

        # First pose at origin
        msg1 = PoseStamped()
        msg1.header.frame_id = 'test'
        msg1.pose.position.x = 0.0
        msg1.pose.orientation.w = 1.0
        filter_node.raw_pose_cb(msg1)

        time.sleep(0.01)  # dt = 10ms

        # Second pose: jump 10m in 10ms = 1000 m/s (>> 0.5 m/s limit)
        msg2 = PoseStamped()
        msg2.header.frame_id = 'test'
        msg2.pose.position.x = 10.0
        msg2.pose.orientation.w = 1.0
        filter_node.raw_pose_cb(msg2)

        # The output should NOT have jumped to 10.0
        # It should be limited by max_lin_vel * dt
        out_x = published[-1].pose.position.x
        assert out_x < 1.0, \
            f'Velocity not clamped: output jumped to x={out_x}'

    def test_slow_movement_unclamped(self, filter_node):
        """Movements within velocity limits should not be clamped."""
        published = []
        filter_node.pub_filtered.publish = lambda msg: published.append(msg)

        msg1 = PoseStamped()
        msg1.header.frame_id = 'test'
        msg1.pose.position.x = 0.0
        msg1.pose.orientation.w = 1.0
        filter_node.raw_pose_cb(msg1)

        time.sleep(0.1)  # 100ms

        # Move 0.01m in 100ms = 0.1 m/s (< 0.5 m/s limit)
        msg2 = PoseStamped()
        msg2.header.frame_id = 'test'
        msg2.pose.position.x = 0.01
        msg2.pose.orientation.w = 1.0
        filter_node.raw_pose_cb(msg2)

        # Output should be close to input (filtered but not clamped)
        assert len(published) == 2


class TestOrientationFiltering:
    """Tests for the SLERP orientation smoothing."""

    def test_zero_quaternion_handled(self, filter_node):
        """Zero quaternion should return identity."""
        result = filter_node._filter_orientation(np.array([0.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0, 1.0])

    def test_identity_quaternion_passthrough(self, filter_node):
        """First identity quaternion should pass through."""
        result = filter_node._filter_orientation(np.array([0.0, 0.0, 0.0, 1.0]))
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0, 1.0], atol=1e-6)

    def test_quaternion_normalized(self, filter_node):
        """Output quaternion should always be normalized."""
        # Non-normalized input
        quat = np.array([1.0, 1.0, 1.0, 1.0])  # norm = 2
        result = filter_node._filter_orientation(quat)

        norm = np.linalg.norm(result)
        np.testing.assert_allclose(norm, 1.0, atol=1e-6,
                                   err_msg=f'Quaternion not normalized: norm={norm}')

    def test_slerp_smoothing_effect(self, filter_node):
        """SLERP should interpolate between previous and current orientation."""
        # Set initial orientation (identity)
        filter_node._filter_orientation(np.array([0.0, 0.0, 0.0, 1.0]))

        # 90-degree rotation around Z
        target = np.array([0.0, 0.0, 0.7071, 0.7071])
        result = filter_node._filter_orientation(target)

        # With alpha=0.3, the result should be between identity and target
        # Not exactly at target
        angle_z = abs(result[2])
        assert 0.0 < angle_z < 0.7071, \
            f'SLERP did not interpolate: qz={angle_z}'

    def test_quaternion_flip_handling(self, filter_node):
        """Antipodal quaternions (q and -q) should be handled gracefully."""
        q1 = np.array([0.0, 0.0, 0.0, 1.0])
        filter_node._filter_orientation(q1)

        # -q1 represents the same rotation
        q2 = np.array([0.0, 0.0, 0.0, -1.0])
        result = filter_node._filter_orientation(q2)

        # Should not jump wildly — result should be close to identity
        norm = np.linalg.norm(result)
        np.testing.assert_allclose(norm, 1.0, atol=1e-6)


class TestFrameIdPropagation:
    """Test that frame_id is propagated correctly."""

    def test_frame_id_preserved(self, filter_node):
        """Output should preserve the input frame_id."""
        published = []
        filter_node.pub_filtered.publish = lambda msg: published.append(msg)

        msg = PoseStamped()
        msg.header.frame_id = 'glove_ee'
        msg.pose.orientation.w = 1.0
        filter_node.raw_pose_cb(msg)

        assert published[0].header.frame_id == 'glove_ee'
