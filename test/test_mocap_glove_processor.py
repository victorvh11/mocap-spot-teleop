#!/usr/bin/env python3
"""
test_mocap_glove_processor.py
=============================
Unit tests for the MocapGloveProcessor ROS2 node.

Covers:
    - Node initialization and parameter defaults
    - Position scaling (mocap -> robot workspace)
    - Calibration offset application
    - Calibration service behavior
    - Velocity computation via finite difference
    - Orientation passthrough
    - Frame ID assignment
    - Handling of uncalibrated vs calibrated state
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from spot_mocap_teleop.mocap_glove_processor import MocapGloveProcessor


@pytest.fixture
def processor():
    """Create a MocapGloveProcessor node for testing."""
    node = MocapGloveProcessor()
    yield node
    node.destroy_node()


class TestMocapGloveProcessorInit:
    """Initialization and parameter tests."""

    def test_node_name(self, processor):
        assert processor.get_name() == 'mocap_glove_processor'

    def test_default_parameters(self, processor):
        assert processor.rb_name == 'GloveRB'
        assert processor.mocap_frame == 'mocap_world'
        assert processor.glove_frame == 'glove_ee'
        np.testing.assert_array_equal(processor.scale, [0.6, 0.6, 0.6])
        np.testing.assert_array_equal(processor.cal_offset, [0.0, 0.0, 0.0])

    def test_initial_state_uncalibrated(self, processor):
        assert not processor.is_calibrated
        assert processor.calibration_pose is None
        assert processor.last_pose is None
        assert processor.last_time is None

    def test_publishers_created(self, processor):
        topic_names = [t[0] for t in processor.get_topic_names_and_types()]
        # The node should have created publishers for these topics
        assert processor.pub_raw_pose is not None
        assert processor.pub_velocity is not None
        assert processor.pub_tracking is not None


class TestProcessPose:
    """Tests for the _process_pose internal method."""

    def test_scaling_applied(self, processor):
        """Position should be multiplied by scale factors."""
        published = []
        processor.pub_raw_pose.publish = lambda msg: published.append(msg)

        pose = PoseStamped()
        pose.header.frame_id = 'mocap_world'
        pose.pose.position.x = 1.0
        pose.pose.position.y = 2.0
        pose.pose.position.z = 3.0
        pose.pose.orientation.w = 1.0

        processor._process_pose(pose)

        assert len(published) == 1
        out = published[0]
        np.testing.assert_allclose(out.pose.position.x, 0.6, atol=1e-6)
        np.testing.assert_allclose(out.pose.position.y, 1.2, atol=1e-6)
        np.testing.assert_allclose(out.pose.position.z, 1.8, atol=1e-6)

    def test_calibration_offset_subtracted(self, processor):
        """Calibration offset should be subtracted before scaling."""
        processor.cal_offset = np.array([1.0, 0.0, 0.0])
        published = []
        processor.pub_raw_pose.publish = lambda msg: published.append(msg)

        pose = PoseStamped()
        pose.pose.position.x = 2.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0

        processor._process_pose(pose)

        # (2.0 - 1.0) * 0.6 = 0.6
        np.testing.assert_allclose(published[0].pose.position.x, 0.6, atol=1e-6)

    def test_calibrated_pose_subtracted(self, processor):
        """When calibrated, calibration_pose should be subtracted from raw position."""
        processor.is_calibrated = True
        processor.calibration_pose = np.array([1.0, 1.0, 1.0])

        published = []
        processor.pub_raw_pose.publish = lambda msg: published.append(msg)

        pose = PoseStamped()
        pose.pose.position.x = 3.0
        pose.pose.position.y = 2.0
        pose.pose.position.z = 1.5
        pose.pose.orientation.w = 1.0

        processor._process_pose(pose)

        # (3.0 - 1.0 - 0.0) * 0.6 = 1.2
        np.testing.assert_allclose(published[0].pose.position.x, 1.2, atol=1e-6)
        # (2.0 - 1.0 - 0.0) * 0.6 = 0.6
        np.testing.assert_allclose(published[0].pose.position.y, 0.6, atol=1e-6)

    def test_orientation_passthrough(self, processor):
        """Orientation should be passed through unchanged."""
        published = []
        processor.pub_raw_pose.publish = lambda msg: published.append(msg)

        pose = PoseStamped()
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 0.0
        pose.pose.orientation.x = 0.1
        pose.pose.orientation.y = 0.2
        pose.pose.orientation.z = 0.3
        pose.pose.orientation.w = 0.9

        processor._process_pose(pose)

        out = published[0]
        assert out.pose.orientation.x == 0.1
        assert out.pose.orientation.y == 0.2
        assert out.pose.orientation.z == 0.3
        assert out.pose.orientation.w == 0.9

    def test_output_frame_id(self, processor):
        """Output frame should be the glove_frame_id parameter."""
        published = []
        processor.pub_raw_pose.publish = lambda msg: published.append(msg)

        pose = PoseStamped()
        pose.header.frame_id = 'mocap_world'
        pose.pose.orientation.w = 1.0

        processor._process_pose(pose)

        assert published[0].header.frame_id == 'glove_ee'

    def test_velocity_not_published_on_first_sample(self, processor):
        """No velocity should be published on the very first sample."""
        vel_published = []
        processor.pub_velocity.publish = lambda msg: vel_published.append(msg)
        processor.pub_raw_pose.publish = lambda msg: None

        pose = PoseStamped()
        pose.pose.orientation.w = 1.0

        processor._process_pose(pose)

        assert len(vel_published) == 0

    def test_velocity_published_on_subsequent_samples(self, processor):
        """Velocity should be published after the first sample."""
        vel_published = []
        processor.pub_velocity.publish = lambda msg: vel_published.append(msg)
        processor.pub_raw_pose.publish = lambda msg: None

        pose = PoseStamped()
        pose.pose.position.x = 0.0
        pose.pose.orientation.w = 1.0
        processor._process_pose(pose)

        # Wait a tiny bit for time difference
        import time
        time.sleep(0.01)

        pose2 = PoseStamped()
        pose2.pose.position.x = 1.0
        pose2.pose.orientation.w = 1.0
        processor._process_pose(pose2)

        assert len(vel_published) >= 1

    def test_zero_movement_zero_velocity(self, processor):
        """Identical poses should produce near-zero velocity."""
        vel_published = []
        processor.pub_velocity.publish = lambda msg: vel_published.append(msg)
        processor.pub_raw_pose.publish = lambda msg: None

        pose = PoseStamped()
        pose.pose.position.x = 1.0
        pose.pose.position.y = 1.0
        pose.pose.position.z = 1.0
        pose.pose.orientation.w = 1.0

        processor._process_pose(pose)
        import time; time.sleep(0.01)
        processor._process_pose(pose)

        if vel_published:
            v = vel_published[-1].twist.linear
            speed = np.sqrt(v.x**2 + v.y**2 + v.z**2)
            # Velocity should be very small (same position)
            assert speed < 0.01, f'Expected near-zero velocity, got {speed}'


class TestCalibrationService:
    """Tests for the /teleop/calibrate_glove service."""

    def test_calibration_fails_without_data(self, processor):
        """Calibration should fail if no mocap data has been received."""
        req = Trigger.Request()
        resp = Trigger.Response()
        result = processor.calibrate_cb(req, resp)

        assert not result.success
        assert 'No mocap data' in result.message

    def test_calibration_succeeds_with_data(self, processor):
        """Calibration should succeed after receiving mocap data."""
        # Simulate having received a pose
        processor.last_pose = np.array([0.3, 0.6, 0.9])

        req = Trigger.Request()
        resp = Trigger.Response()
        result = processor.calibrate_cb(req, resp)

        assert result.success
        assert processor.is_calibrated
        assert processor.calibration_pose is not None

    def test_calibration_stores_raw_position(self, processor):
        """Calibration should store the unscaled raw position."""
        processor.last_pose = np.array([0.6, 1.2, 1.8])  # scaled
        processor.scale = np.array([0.6, 0.6, 0.6])
        processor.cal_offset = np.array([0.0, 0.0, 0.0])

        req = Trigger.Request()
        resp = Trigger.Response()
        processor.calibrate_cb(req, resp)

        # Unscaled: [0.6/0.6, 1.2/0.6, 1.8/0.6] = [1.0, 2.0, 3.0]
        np.testing.assert_allclose(processor.calibration_pose, [1.0, 2.0, 3.0], atol=1e-6)


class TestScalingVariations:
    """Tests for different scaling configurations."""

    def test_asymmetric_scaling(self):
        """Different scale factors per axis should work correctly."""
        node = MocapGloveProcessor()
        node.scale = np.array([0.5, 1.0, 0.3])

        published = []
        node.pub_raw_pose.publish = lambda msg: published.append(msg)

        pose = PoseStamped()
        pose.pose.position.x = 2.0
        pose.pose.position.y = 2.0
        pose.pose.position.z = 2.0
        pose.pose.orientation.w = 1.0

        node._process_pose(pose)

        np.testing.assert_allclose(published[0].pose.position.x, 1.0, atol=1e-6)
        np.testing.assert_allclose(published[0].pose.position.y, 2.0, atol=1e-6)
        np.testing.assert_allclose(published[0].pose.position.z, 0.6, atol=1e-6)

        node.destroy_node()

    def test_unity_scaling_passthrough(self):
        """Scale=[1,1,1] should pass positions through."""
        node = MocapGloveProcessor()
        node.scale = np.array([1.0, 1.0, 1.0])

        published = []
        node.pub_raw_pose.publish = lambda msg: published.append(msg)

        pose = PoseStamped()
        pose.pose.position.x = 0.5
        pose.pose.position.y = -0.3
        pose.pose.position.z = 0.7
        pose.pose.orientation.w = 1.0

        node._process_pose(pose)

        np.testing.assert_allclose(published[0].pose.position.x, 0.5, atol=1e-6)
        np.testing.assert_allclose(published[0].pose.position.y, -0.3, atol=1e-6)
        np.testing.assert_allclose(published[0].pose.position.z, 0.7, atol=1e-6)

        node.destroy_node()
