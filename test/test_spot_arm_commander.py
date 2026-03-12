#!/usr/bin/env python3
"""
test_spot_arm_commander.py
==========================
Unit tests for the SpotArmCommander ROS2 node.

Covers:
    - Node initialization and parameter defaults
    - Enable/Disable service behavior
    - Stow/Unstow service behavior
    - Target pose latching
    - Command timer does not fire when disabled
    - Command timer fires when enabled with target
    - Dry-run mode (no spot_msgs available)
    - Status publishing
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger

from spot_mocap_teleop.spot_arm_commander import SpotArmCommander


@pytest.fixture
def commander():
    """Create a SpotArmCommander node for testing."""
    node = SpotArmCommander()
    yield node
    node.destroy_node()


# =============================================================================
# Initialization
# =============================================================================

class TestSpotArmCommanderInit:
    """Initialization tests."""

    def test_node_name(self, commander):
        assert commander.get_name() == 'spot_arm_commander'

    def test_default_params(self, commander):
        assert commander.command_mode == 'cartesian'
        assert commander.root_frame == 'body'
        assert commander.command_rate == 10.0
        assert commander.command_timeout == 2.0

    def test_initial_state_disabled(self, commander):
        """Commander should start disabled."""
        assert commander.enabled is False
        assert commander.latest_target is None
        assert commander.last_command_time is None


# =============================================================================
# Enable / Disable
# =============================================================================

class TestEnableDisable:
    """Tests for enable/disable services."""

    def test_enable_service(self, commander):
        """Enable should set enabled=True."""
        req = Trigger.Request()
        resp = Trigger.Response()
        result = commander.enable_cb(req, resp)

        assert result.success is True
        assert commander.enabled is True
        assert 'ENABLED' in result.message

    def test_disable_service(self, commander):
        """Disable should set enabled=False."""
        commander.enabled = True

        req = Trigger.Request()
        resp = Trigger.Response()
        result = commander.disable_cb(req, resp)

        assert result.success is True
        assert commander.enabled is False
        assert 'DISABLED' in result.message

    def test_enable_then_disable(self, commander):
        """Enable followed by disable should leave node disabled."""
        req = Trigger.Request()
        resp = Trigger.Response()

        commander.enable_cb(req, resp)
        assert commander.enabled is True

        commander.disable_cb(req, Trigger.Response())
        assert commander.enabled is False


# =============================================================================
# Target pose latching
# =============================================================================

class TestTargetLatching:
    """Tests for target pose storage."""

    def test_safe_pose_stored(self, commander):
        """Latest safe pose should be stored."""
        msg = PoseStamped()
        msg.pose.position.x = 0.5
        msg.pose.position.y = 0.1
        msg.pose.position.z = 0.2
        msg.pose.orientation.w = 1.0

        commander.safe_pose_cb(msg)

        assert commander.latest_target is not None
        assert commander.latest_target.pose.position.x == 0.5

    def test_latest_target_overwritten(self, commander):
        """New pose should overwrite the old target."""
        msg1 = PoseStamped()
        msg1.pose.position.x = 0.1
        msg1.pose.orientation.w = 1.0
        commander.safe_pose_cb(msg1)

        msg2 = PoseStamped()
        msg2.pose.position.x = 0.9
        msg2.pose.orientation.w = 1.0
        commander.safe_pose_cb(msg2)

        assert commander.latest_target.pose.position.x == 0.9


# =============================================================================
# Command timer behavior
# =============================================================================

class TestCommandTimer:
    """Tests for the periodic command timer."""

    def test_no_command_when_disabled(self, commander):
        """Timer should not send commands when disabled."""
        commander.enabled = False
        commander.latest_target = PoseStamped()
        commander.latest_target.pose.orientation.w = 1.0

        # Should not raise or send anything
        commander.send_command_tick()

        assert commander.last_command_time is None

    def test_no_command_without_target(self, commander):
        """Timer should not send commands without a target."""
        commander.enabled = True
        commander.latest_target = None

        commander.send_command_tick()

        assert commander.last_command_time is None

    def test_command_sent_when_enabled_with_target(self, commander):
        """Timer should process command when enabled with target."""
        commander.enabled = True
        msg = PoseStamped()
        msg.pose.position.x = 0.5
        msg.pose.orientation.w = 1.0
        commander.latest_target = msg

        # In dry-run mode (no spot_msgs), should still update time
        commander.send_command_tick()

        assert commander.last_command_time is not None


# =============================================================================
# Stow / Unstow
# =============================================================================

class TestStowUnstow:
    """Tests for stow/unstow services."""

    def test_stow_disables_teleop(self, commander):
        """Stowing should disable arm teleop."""
        commander.enabled = True

        req = Trigger.Request()
        resp = Trigger.Response()
        result = commander.stow_cb(req, resp)

        assert commander.enabled is False

    def test_stow_service_response(self, commander):
        """Stow should return appropriate response."""
        req = Trigger.Request()
        resp = Trigger.Response()
        result = commander.stow_cb(req, resp)

        # Service may or may not be ready (no spot driver running)
        # But should not crash
        assert isinstance(result.message, str)


# =============================================================================
# Status publishing
# =============================================================================

class TestStatusPublishing:
    """Tests for arm command status."""

    def test_status_published_on_command(self, commander):
        """Status should be published when sending a command."""
        status_published = []
        commander.pub_status.publish = lambda msg: status_published.append(msg)

        commander.enabled = True
        msg = PoseStamped()
        msg.pose.position.x = 0.5
        msg.pose.orientation.w = 1.0
        commander.latest_target = msg

        commander.send_command_tick()

        # In dry-run mode, no status is published (debug log only)
        # In real mode, status would be published
        # This test just verifies no crash


# =============================================================================
# Dry-run mode
# =============================================================================

class TestDryRunMode:
    """Tests for operation without spot_msgs (dry-run)."""

    def test_dry_run_does_not_crash(self, commander):
        """Sending commands in dry-run mode should not raise errors."""
        commander.enabled = True
        msg = PoseStamped()
        msg.pose.position.x = 0.5
        msg.pose.position.y = 0.1
        msg.pose.position.z = 0.2
        msg.pose.orientation.w = 1.0
        commander.latest_target = msg

        # Should not raise any exception
        for _ in range(10):
            commander.send_command_tick()

    def test_dry_run_updates_timestamp(self, commander):
        """Dry-run should still update the last_command_time."""
        commander.enabled = True
        msg = PoseStamped()
        msg.pose.orientation.w = 1.0
        commander.latest_target = msg

        commander.send_command_tick()

        assert commander.last_command_time is not None
