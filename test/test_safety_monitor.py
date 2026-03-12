#!/usr/bin/env python3
"""
test_safety_monitor.py
======================
Unit tests for the SafetyMonitor ROS2 node.

Covers:
    - Node initialization and parameter defaults
    - Velocity watchdog triggers on excessive speed
    - Velocity watchdog ignores safe speed
    - Velocity watchdog disabled when configured
    - Force limit triggers e-stop
    - Torque limit triggers e-stop
    - Force/torque below threshold passes safely
    - Workspace violation triggers e-stop
    - Workspace OK does not trigger
    - Tracking callback updates heartbeat timestamp
    - E-stop only triggers once (idempotent)
    - Reset service clears all violations
    - E-stop state persists until reset
    - Safe pose updates command heartbeat
"""

import pytest
import numpy as np
import time

import rclpy
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from spot_mocap_teleop.safety_monitor import SafetyMonitor


@pytest.fixture
def safety():
    """Create a SafetyMonitor node for testing."""
    node = SafetyMonitor()
    yield node
    node.destroy_node()


# =============================================================================
# Initialization
# =============================================================================

class TestSafetyMonitorInit:
    """Initialization tests."""

    def test_node_name(self, safety):
        assert safety.get_name() == 'safety_monitor'

    def test_default_params(self, safety):
        assert safety.mocap_timeout == 0.5
        assert safety.max_force == 50.0
        assert safety.max_torque == 20.0
        assert safety.max_vel == 1.0
        assert safety.recovery_action == 'stow'

    def test_initial_state_safe(self, safety):
        assert safety.is_safe is True
        assert safety.e_stop_active is False
        assert len(safety.safety_violations) == 0

    def test_enables_all_monitors_by_default(self, safety):
        assert safety.enable_force is True
        assert safety.enable_vel is True
        assert safety.enable_heartbeat is True
        assert safety.enable_workspace is True


# =============================================================================
# Velocity watchdog
# =============================================================================

class TestVelocityWatchdog:
    """Tests for velocity-based safety monitoring."""

    def test_excessive_velocity_triggers_estop(self, safety):
        """Velocity above limit should trigger e-stop."""
        msg = TwistStamped()
        msg.twist.linear.x = 2.0  # 2 m/s > 1.0 m/s limit
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0

        safety.velocity_cb(msg)

        assert safety.e_stop_active is True
        assert any('VELOCITY_EXCEEDED' in v for v in safety.safety_violations)

    def test_safe_velocity_no_estop(self, safety):
        """Velocity below limit should not trigger e-stop."""
        msg = TwistStamped()
        msg.twist.linear.x = 0.3
        msg.twist.linear.y = 0.2
        msg.twist.linear.z = 0.1
        # magnitude = sqrt(0.09+0.04+0.01) = 0.374 < 1.0

        safety.velocity_cb(msg)

        assert safety.e_stop_active is False

    def test_velocity_exactly_at_limit(self, safety):
        """Velocity exactly at limit should not trigger."""
        # 1.0 m/s exactly
        msg = TwistStamped()
        msg.twist.linear.x = 1.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0

        safety.velocity_cb(msg)

        assert safety.e_stop_active is False

    def test_velocity_watchdog_disabled(self, safety):
        """When disabled, excessive velocity should not trigger e-stop."""
        safety.enable_vel = False

        msg = TwistStamped()
        msg.twist.linear.x = 100.0

        safety.velocity_cb(msg)

        assert safety.e_stop_active is False

    def test_3d_velocity_magnitude(self, safety):
        """Velocity magnitude should be computed in 3D."""
        msg = TwistStamped()
        # Each axis = 0.6, magnitude = sqrt(3*0.36) = 1.039 > 1.0
        msg.twist.linear.x = 0.6
        msg.twist.linear.y = 0.6
        msg.twist.linear.z = 0.6

        safety.velocity_cb(msg)

        assert safety.e_stop_active is True


# =============================================================================
# Force/Torque limits
# =============================================================================

class TestForceTorqueLimits:
    """Tests for force and torque safety monitoring."""

    def test_excessive_force_triggers_estop(self, safety):
        """Force above threshold should trigger e-stop."""
        msg = WrenchStamped()
        msg.wrench.force.x = 60.0  # > 50N
        msg.wrench.force.y = 0.0
        msg.wrench.force.z = 0.0

        safety.wrench_cb(msg)

        assert safety.e_stop_active is True
        assert any('FORCE_EXCEEDED' in v for v in safety.safety_violations)

    def test_excessive_torque_triggers_estop(self, safety):
        """Torque above threshold should trigger e-stop."""
        msg = WrenchStamped()
        msg.wrench.torque.x = 25.0  # > 20Nm

        safety.wrench_cb(msg)

        assert safety.e_stop_active is True
        assert any('TORQUE_EXCEEDED' in v for v in safety.safety_violations)

    def test_safe_force_no_estop(self, safety):
        """Force below threshold should not trigger e-stop."""
        msg = WrenchStamped()
        msg.wrench.force.x = 10.0
        msg.wrench.force.y = 10.0
        msg.wrench.force.z = 10.0
        # magnitude = sqrt(300) = 17.3 < 50

        safety.wrench_cb(msg)

        assert safety.e_stop_active is False

    def test_force_disabled(self, safety):
        """When disabled, excessive force should not trigger e-stop."""
        safety.enable_force = False

        msg = WrenchStamped()
        msg.wrench.force.x = 1000.0

        safety.wrench_cb(msg)

        assert safety.e_stop_active is False

    def test_combined_force_and_torque(self, safety):
        """Both force and torque exceeded should trigger with both violations."""
        msg = WrenchStamped()
        msg.wrench.force.z = 60.0  # > 50N
        msg.wrench.torque.z = 25.0  # > 20Nm

        safety.wrench_cb(msg)

        assert safety.e_stop_active is True
        # Due to e-stop idempotency, only the first violation triggers
        assert len(safety.safety_violations) >= 1


# =============================================================================
# Workspace boundary monitoring
# =============================================================================

class TestWorkspaceMonitoring:
    """Tests for workspace boundary monitoring."""

    def test_out_of_workspace_triggers_estop(self, safety):
        """Workspace violation should trigger e-stop."""
        msg = Bool()
        msg.data = False  # Out of workspace

        safety.workspace_cb(msg)

        assert safety.e_stop_active is True
        assert any('WORKSPACE' in v for v in safety.safety_violations)

    def test_in_workspace_no_estop(self, safety):
        """In-workspace signal should not trigger e-stop."""
        msg = Bool()
        msg.data = True

        safety.workspace_cb(msg)

        assert safety.e_stop_active is False

    def test_workspace_monitoring_disabled(self, safety):
        """When disabled, workspace violation should not trigger."""
        safety.enable_workspace = False

        msg = Bool()
        msg.data = False

        safety.workspace_cb(msg)

        assert safety.e_stop_active is False


# =============================================================================
# Tracking heartbeat
# =============================================================================

class TestTrackingHeartbeat:
    """Tests for mocap tracking heartbeat monitoring."""

    def test_tracking_true_updates_timestamp(self, safety):
        """Tracking=True should update the last_mocap_time."""
        assert safety.last_mocap_time is None

        msg = Bool()
        msg.data = True
        safety.tracking_cb(msg)

        assert safety.last_mocap_time is not None

    def test_tracking_false_does_not_update(self, safety):
        """Tracking=False should not update the timestamp."""
        msg = Bool()
        msg.data = False
        safety.tracking_cb(msg)

        assert safety.last_mocap_time is None

    def test_safe_pose_updates_command_heartbeat(self, safety):
        """Receiving safe_pose should update command heartbeat."""
        assert safety.last_command_time is None

        msg = PoseStamped()
        safety.safe_pose_cb(msg)

        assert safety.last_command_time is not None


# =============================================================================
# E-stop behavior
# =============================================================================

class TestEStopBehavior:
    """Tests for e-stop triggering and idempotency."""

    def test_estop_only_triggers_once(self, safety):
        """Multiple violations should not re-trigger after e-stop is active."""
        msg1 = TwistStamped()
        msg1.twist.linear.x = 5.0
        safety.velocity_cb(msg1)

        violations_after_first = len(safety.safety_violations)

        msg2 = TwistStamped()
        msg2.twist.linear.x = 10.0
        safety.velocity_cb(msg2)

        # Should not add another violation (e-stop already active)
        assert len(safety.safety_violations) == violations_after_first

    def test_estop_persists(self, safety):
        """E-stop should persist even after safe data is received."""
        # Trigger e-stop
        msg = TwistStamped()
        msg.twist.linear.x = 5.0
        safety.velocity_cb(msg)

        assert safety.e_stop_active is True

        # Send safe velocity
        msg2 = TwistStamped()
        msg2.twist.linear.x = 0.1
        safety.velocity_cb(msg2)

        # Still in e-stop
        assert safety.e_stop_active is True


# =============================================================================
# Reset service
# =============================================================================

class TestResetService:
    """Tests for the /teleop/reset_safety service."""

    def test_reset_clears_estop(self, safety):
        """Reset should clear e-stop and all violations."""
        # Trigger e-stop first
        safety.e_stop_active = True
        safety.is_safe = False
        safety.safety_violations = ['FORCE_EXCEEDED', 'VELOCITY_EXCEEDED']

        req = Trigger.Request()
        resp = Trigger.Response()
        result = safety.reset_safety_cb(req, resp)

        assert result.success is True
        assert safety.e_stop_active is False
        assert safety.is_safe is True
        assert len(safety.safety_violations) == 0

    def test_reset_when_already_safe(self, safety):
        """Reset should succeed even when no violations exist."""
        req = Trigger.Request()
        resp = Trigger.Response()
        result = safety.reset_safety_cb(req, resp)

        assert result.success is True

    def test_can_trigger_again_after_reset(self, safety):
        """After reset, new violations should be able to trigger e-stop again."""
        # Trigger
        msg = TwistStamped()
        msg.twist.linear.x = 5.0
        safety.velocity_cb(msg)
        assert safety.e_stop_active is True

        # Reset
        req = Trigger.Request()
        resp = Trigger.Response()
        safety.reset_safety_cb(req, resp)
        assert safety.e_stop_active is False

        # Trigger again
        safety.velocity_cb(msg)
        assert safety.e_stop_active is True
        assert len(safety.safety_violations) == 1


# =============================================================================
# Monitor tick
# =============================================================================

class TestMonitorTick:
    """Tests for the periodic monitor_tick callback."""

    def test_ok_status_when_safe(self, safety):
        """Should publish 'OK' when no violations and no timeout."""
        published = []
        safety.pub_safety.publish = lambda msg: published.append(msg)
        safety.pub_estop.publish = lambda msg: None

        safety.monitor_tick()

        assert any('OK' in p.data for p in published)

    def test_estop_status_when_active(self, safety):
        """Should publish E_STOP_ACTIVE when e-stop is active."""
        safety.e_stop_active = True
        safety.safety_violations = ['TEST_VIOLATION']

        published = []
        safety.pub_safety.publish = lambda msg: published.append(msg)
        safety.pub_estop.publish = lambda msg: None

        safety.monitor_tick()

        assert any('E_STOP_ACTIVE' in p.data for p in published)

    def test_estop_bool_published(self, safety):
        """E-stop Bool should reflect current e-stop state."""
        estop_published = []
        safety.pub_safety.publish = lambda msg: None
        safety.pub_estop.publish = lambda msg: estop_published.append(msg)

        safety.monitor_tick()
        assert estop_published[-1].data is False

        safety.e_stop_active = True
        safety.monitor_tick()
        assert estop_published[-1].data is True
