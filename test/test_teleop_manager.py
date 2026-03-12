#!/usr/bin/env python3
"""
test_teleop_manager.py
======================
Unit tests for the TeleopManager ROS2 node.

Covers:
    - Node initialization and default state
    - State machine transitions (IDLE -> INIT -> READY -> TELEOP -> STOPPED)
    - E-stop callback sets E_STOP state
    - Tracking callback updates glove_tracked flag
    - Start teleop fails without tracking
    - Start teleop fails from invalid state
    - Stop teleop transitions to STOPPED
    - Emergency stop transitions to E_STOP
    - State publishing
    - Diagnostic publishing
"""

import pytest
from unittest.mock import MagicMock, patch
import time

import rclpy
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from spot_mocap_teleop.teleop_manager import TeleopManager, TeleopState


@pytest.fixture
def manager():
    """Create a TeleopManager node for testing."""
    node = TeleopManager()
    yield node
    node.destroy_node()


# =============================================================================
# Initialization
# =============================================================================

class TestTeleopManagerInit:
    """Initialization tests."""

    def test_node_name(self, manager):
        assert manager.get_name() == 'teleop_manager'

    def test_initial_state_idle(self, manager):
        assert manager.state == TeleopState.IDLE

    def test_glove_not_tracked_initially(self, manager):
        assert manager.glove_tracked is False


# =============================================================================
# Tracking callback
# =============================================================================

class TestTrackingCallback:
    """Tests for glove tracking state updates."""

    def test_tracking_true(self, manager):
        msg = Bool()
        msg.data = True
        manager.tracking_cb(msg)
        assert manager.glove_tracked is True

    def test_tracking_false(self, manager):
        manager.glove_tracked = True
        msg = Bool()
        msg.data = False
        manager.tracking_cb(msg)
        assert manager.glove_tracked is False


# =============================================================================
# E-stop callback
# =============================================================================

class TestEStopCallback:
    """Tests for e-stop handling."""

    def test_estop_during_teleop_changes_state(self, manager):
        """E-stop during teleop should transition to E_STOP state."""
        manager.state = TeleopState.TELEOPERATING

        msg = Bool()
        msg.data = True
        manager.estop_cb(msg)

        assert manager.state == TeleopState.E_STOP

    def test_estop_false_does_not_change_state(self, manager):
        """E-stop=False should not change state."""
        manager.state = TeleopState.TELEOPERATING

        msg = Bool()
        msg.data = False
        manager.estop_cb(msg)

        assert manager.state == TeleopState.TELEOPERATING

    def test_estop_during_idle_no_effect(self, manager):
        """E-stop during IDLE should not change state."""
        manager.state = TeleopState.IDLE

        msg = Bool()
        msg.data = True
        manager.estop_cb(msg)

        assert manager.state == TeleopState.IDLE

    def test_estop_during_ready_no_effect(self, manager):
        """E-stop during READY should not change state (not teleoperating)."""
        manager.state = TeleopState.READY

        msg = Bool()
        msg.data = True
        manager.estop_cb(msg)

        assert manager.state == TeleopState.READY


# =============================================================================
# Start teleop
# =============================================================================

class TestStartTeleop:
    """Tests for the start_teleop service."""

    def test_cannot_start_from_idle(self, manager):
        """Cannot start teleop directly from IDLE state."""
        manager.state = TeleopState.IDLE

        req = Trigger.Request()
        resp = Trigger.Response()
        result = manager.start_teleop_cb(req, resp)

        assert result.success is False
        assert 'Cannot start' in result.message

    def test_cannot_start_without_tracking(self, manager):
        """Cannot start teleop if glove is not tracked."""
        manager.state = TeleopState.READY
        manager.glove_tracked = False

        req = Trigger.Request()
        resp = Trigger.Response()
        result = manager.start_teleop_cb(req, resp)

        assert result.success is False
        assert 'not tracked' in result.message
        # State should revert to READY
        assert manager.state == TeleopState.READY

    def test_start_teleop_with_tracking(self, manager):
        """Start teleop should succeed from READY state with tracking."""
        manager.state = TeleopState.READY
        manager.glove_tracked = True

        # Mock all service clients so they appear ready
        for name, client in manager.teleop_clients.items():
            client.wait_for_service = MagicMock(return_value=True)
            client.service_is_ready = MagicMock(return_value=True)

            # Create mock future with result
            mock_future = MagicMock()
            mock_result = Trigger.Response()
            mock_result.success = True
            mock_result.message = 'OK'
            mock_future.result.return_value = mock_result
            mock_future.done.return_value = True
            client.call_async = MagicMock(return_value=mock_future)

        # Patch spin_until_future_complete
        with patch('rclpy.spin_until_future_complete'):
            req = Trigger.Request()
            resp = Trigger.Response()
            result = manager.start_teleop_cb(req, resp)

        assert result.success is True
        assert manager.state == TeleopState.TELEOPERATING

    def test_can_restart_from_stopped(self, manager):
        """Should be able to start teleop again from STOPPED state."""
        manager.state = TeleopState.STOPPED
        manager.glove_tracked = True

        for name, client in manager.teleop_clients.items():
            client.wait_for_service = MagicMock(return_value=True)
            client.service_is_ready = MagicMock(return_value=True)
            mock_future = MagicMock()
            mock_result = Trigger.Response()
            mock_result.success = True
            mock_result.message = 'OK'
            mock_future.result.return_value = mock_result
            mock_future.done.return_value = True
            client.call_async = MagicMock(return_value=mock_future)

        with patch('rclpy.spin_until_future_complete'):
            req = Trigger.Request()
            resp = Trigger.Response()
            result = manager.start_teleop_cb(req, resp)

        assert result.success is True
        assert manager.state == TeleopState.TELEOPERATING


# =============================================================================
# Stop teleop
# =============================================================================

class TestStopTeleop:
    """Tests for the stop_teleop service."""

    def test_stop_transitions_to_stopped(self, manager):
        """Stop should transition to STOPPED state."""
        manager.state = TeleopState.TELEOPERATING

        for name, client in manager.teleop_clients.items():
            client.wait_for_service = MagicMock(return_value=True)
            client.service_is_ready = MagicMock(return_value=True)
            mock_future = MagicMock()
            mock_result = Trigger.Response()
            mock_result.success = True
            mock_result.message = 'OK'
            mock_future.result.return_value = mock_result
            mock_future.done.return_value = True
            client.call_async = MagicMock(return_value=mock_future)

        with patch('rclpy.spin_until_future_complete'):
            req = Trigger.Request()
            resp = Trigger.Response()
            result = manager.stop_teleop_cb(req, resp)

        assert result.success is True
        assert manager.state == TeleopState.STOPPED


# =============================================================================
# Emergency stop
# =============================================================================

class TestEmergencyStop:
    """Tests for the emergency_stop service."""

    def test_emergency_stop_sets_state(self, manager):
        """Emergency stop should set E_STOP state."""
        manager.state = TeleopState.TELEOPERATING

        for name, client in manager.teleop_clients.items():
            client.wait_for_service = MagicMock(return_value=True)
            client.service_is_ready = MagicMock(return_value=True)
            mock_future = MagicMock()
            mock_result = Trigger.Response()
            mock_result.success = True
            mock_future.result.return_value = mock_result
            mock_future.done.return_value = True
            client.call_async = MagicMock(return_value=mock_future)

        with patch('rclpy.spin_until_future_complete'):
            req = Trigger.Request()
            resp = Trigger.Response()
            result = manager.emergency_stop_cb(req, resp)

        assert result.success is True
        assert manager.state == TeleopState.E_STOP


# =============================================================================
# State publishing
# =============================================================================

class TestStatePublishing:
    """Tests for periodic state publishing."""

    def test_publishes_current_state(self, manager):
        """Should publish the current state string."""
        published = []
        manager.pub_state.publish = lambda msg: published.append(msg)

        manager.state = TeleopState.READY
        manager.publish_state()

        assert len(published) == 1
        assert published[0].data == 'READY'

    def test_all_states_publishable(self, manager):
        """All TeleopState values should be publishable."""
        published = []
        manager.pub_state.publish = lambda msg: published.append(msg)

        for state in [TeleopState.IDLE, TeleopState.INITIALIZING,
                      TeleopState.CALIBRATING, TeleopState.READY,
                      TeleopState.TELEOPERATING, TeleopState.STOPPED,
                      TeleopState.ERROR, TeleopState.E_STOP]:
            manager.state = state
            manager.publish_state()

        assert len(published) == 8
        state_values = [p.data for p in published]
        assert 'IDLE' in state_values
        assert 'TELEOPERATING' in state_values
        assert 'E_STOP' in state_values


# =============================================================================
# Diagnostic publishing
# =============================================================================

class TestDiagnosticPublishing:
    """Tests for diagnostic message publishing."""

    def test_publish_diag(self, manager):
        """Diagnostic messages should include current state."""
        published = []
        manager.pub_diag.publish = lambda msg: published.append(msg)

        manager.state = TeleopState.READY
        manager._publish_diag('Test message')

        assert len(published) == 1
        assert '[READY]' in published[0].data
        assert 'Test message' in published[0].data


# =============================================================================
# TeleopState constants
# =============================================================================

class TestTeleopStateEnum:
    """Tests for state string constants."""

    def test_all_states_are_strings(self):
        for state in [TeleopState.IDLE, TeleopState.INITIALIZING,
                      TeleopState.CALIBRATING, TeleopState.READY,
                      TeleopState.TELEOPERATING, TeleopState.STOPPED,
                      TeleopState.ERROR, TeleopState.E_STOP]:
            assert isinstance(state, str)

    def test_states_are_unique(self):
        states = [TeleopState.IDLE, TeleopState.INITIALIZING,
                  TeleopState.CALIBRATING, TeleopState.READY,
                  TeleopState.TELEOPERATING, TeleopState.STOPPED,
                  TeleopState.ERROR, TeleopState.E_STOP]
        assert len(states) == len(set(states))
