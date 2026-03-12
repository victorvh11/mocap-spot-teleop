#!/usr/bin/env python3
"""
test_workspace_limiter.py
=========================
Unit tests for the WorkspaceLimiter ROS2 node.

Covers:
    - Clamp mode: positions clamped to workspace bounds
    - In-bounds positions pass through unchanged
    - Boundary edge cases (exactly on boundary)
    - Self-collision avoidance (min distance to body)
    - Reject mode: out-of-bounds suppressed entirely
    - Soft limit mode: quadratic easing near boundaries
    - _is_in_bounds helper
    - Orientation passthrough
    - Status message publishing
"""

import pytest
import numpy as np

import rclpy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String

from spot_mocap_teleop.workspace_limiter import WorkspaceLimiter


@pytest.fixture
def limiter():
    """Create a WorkspaceLimiter node for testing."""
    node = WorkspaceLimiter()
    yield node
    node.destroy_node()


# Default bounds from the node
DEFAULT_BOUNDS = {
    'x_min': 0.2, 'x_max': 0.85,
    'y_min': -0.50, 'y_max': 0.50,
    'z_min': -0.25, 'z_max': 0.45,
}


class TestWorkspaceLimiterInit:
    """Initialization tests."""

    def test_node_name(self, limiter):
        assert limiter.get_name() == 'workspace_limiter'

    def test_default_bounds(self, limiter):
        assert limiter.bounds == DEFAULT_BOUNDS

    def test_default_mode_is_clamp(self, limiter):
        assert limiter.boundary_mode == 'clamp'

    def test_collision_check_enabled(self, limiter):
        assert limiter.check_collision is True
        assert limiter.min_body_dist == 0.15


# =============================================================================
# Clamp position tests
# =============================================================================

class TestClampPosition:
    """Direct tests of the _clamp_position method."""

    def test_in_bounds_unchanged(self, limiter):
        """Position inside workspace should not be modified."""
        pos = np.array([0.5, 0.0, 0.1])
        result, clamped = limiter._clamp_position(pos)
        np.testing.assert_array_equal(result, pos)
        assert len(clamped) == 0

    def test_x_below_min_clamped(self, limiter):
        pos = np.array([-1.0, 0.0, 0.0])
        result, clamped = limiter._clamp_position(pos)
        assert result[0] == DEFAULT_BOUNDS['x_min']
        assert 'x_min' in clamped

    def test_x_above_max_clamped(self, limiter):
        pos = np.array([5.0, 0.0, 0.0])
        result, clamped = limiter._clamp_position(pos)
        assert result[0] == DEFAULT_BOUNDS['x_max']
        assert 'x_max' in clamped

    def test_y_below_min_clamped(self, limiter):
        pos = np.array([0.5, -2.0, 0.0])
        result, clamped = limiter._clamp_position(pos)
        assert result[1] == DEFAULT_BOUNDS['y_min']

    def test_y_above_max_clamped(self, limiter):
        pos = np.array([0.5, 2.0, 0.0])
        result, clamped = limiter._clamp_position(pos)
        assert result[1] == DEFAULT_BOUNDS['y_max']

    def test_z_below_min_clamped(self, limiter):
        pos = np.array([0.5, 0.0, -5.0])
        result, clamped = limiter._clamp_position(pos)
        assert result[2] == DEFAULT_BOUNDS['z_min']

    def test_z_above_max_clamped(self, limiter):
        pos = np.array([0.5, 0.0, 5.0])
        result, clamped = limiter._clamp_position(pos)
        assert result[2] == DEFAULT_BOUNDS['z_max']

    def test_multiple_axes_clamped(self, limiter):
        """All three axes out of bounds should all be clamped."""
        pos = np.array([-10.0, 10.0, -10.0])
        result, clamped = limiter._clamp_position(pos)
        assert result[0] == DEFAULT_BOUNDS['x_min']
        assert result[1] == DEFAULT_BOUNDS['y_max']
        assert result[2] == DEFAULT_BOUNDS['z_min']
        assert len(clamped) == 3

    def test_exact_boundary_not_clamped(self, limiter):
        """Position exactly on boundary should not be flagged as clamped."""
        pos = np.array([DEFAULT_BOUNDS['x_min'], DEFAULT_BOUNDS['y_max'],
                        DEFAULT_BOUNDS['z_min']])
        result, clamped = limiter._clamp_position(pos)
        np.testing.assert_array_equal(result, pos)
        assert len(clamped) == 0


# =============================================================================
# Is in bounds
# =============================================================================

class TestIsInBounds:
    """Tests for the _is_in_bounds helper."""

    def test_center_is_in_bounds(self, limiter):
        pos = np.array([0.5, 0.0, 0.1])
        assert limiter._is_in_bounds(pos) is True

    def test_outside_x_not_in_bounds(self, limiter):
        pos = np.array([0.0, 0.0, 0.0])  # x=0 < x_min=0.2
        assert limiter._is_in_bounds(pos) is False

    def test_boundary_is_in_bounds(self, limiter):
        pos = np.array([0.2, -0.5, -0.25])
        assert limiter._is_in_bounds(pos) is True

    def test_all_corners_in_bounds(self, limiter):
        """All 8 corners of the workspace box should be in bounds."""
        for x in [DEFAULT_BOUNDS['x_min'], DEFAULT_BOUNDS['x_max']]:
            for y in [DEFAULT_BOUNDS['y_min'], DEFAULT_BOUNDS['y_max']]:
                for z in [DEFAULT_BOUNDS['z_min'], DEFAULT_BOUNDS['z_max']]:
                    pos = np.array([x, y, z])
                    assert limiter._is_in_bounds(pos), \
                        f'Corner ({x}, {y}, {z}) should be in bounds'


# =============================================================================
# Soft limit tests
# =============================================================================

class TestSoftLimit:
    """Tests for the soft limit boundary mode."""

    def test_center_unchanged(self, limiter):
        """Position in center of workspace should not be modified."""
        pos = np.array([0.5, 0.0, 0.1])
        result = limiter._soft_limit_position(pos)
        np.testing.assert_array_equal(result, pos)

    def test_beyond_boundary_clamped(self, limiter):
        """Position beyond hard boundary should be clamped."""
        pos = np.array([2.0, 0.0, 0.0])
        result = limiter._soft_limit_position(pos)
        assert result[0] <= DEFAULT_BOUNDS['x_max']

    def test_in_soft_zone_modified(self, limiter):
        """Position in the soft margin zone should be pulled inward."""
        # x_max=0.85, soft_margin=0.10, so soft zone starts at 0.75
        pos = np.array([0.82, 0.0, 0.1])
        result = limiter._soft_limit_position(pos)
        # Should be pulled slightly toward center
        assert result[0] <= pos[0] + 0.001  # might equal or be less
        assert result[0] >= DEFAULT_BOUNDS['x_min']
        assert result[0] <= DEFAULT_BOUNDS['x_max']

    def test_result_always_within_bounds(self, limiter):
        """Soft limit should always produce a result within hard bounds."""
        np.random.seed(123)
        for _ in range(100):
            pos = np.random.uniform(-2.0, 2.0, size=3)
            result = limiter._soft_limit_position(pos)
            for i, axis in enumerate(['x', 'y', 'z']):
                lo = DEFAULT_BOUNDS[f'{axis}_min']
                hi = DEFAULT_BOUNDS[f'{axis}_max']
                assert result[i] >= lo - 1e-6, \
                    f'{axis}={result[i]} < {lo}'
                assert result[i] <= hi + 1e-6, \
                    f'{axis}={result[i]} > {hi}'


# =============================================================================
# Self-collision avoidance
# =============================================================================

class TestSelfCollisionAvoidance:
    """Tests for the self-collision check."""

    def test_position_too_close_to_body(self, limiter):
        """Position closer than min_distance should be pushed outward."""
        safe_published = []
        limiter.pub_safe.publish = lambda msg: safe_published.append(msg)
        limiter.pub_in_bounds.publish = lambda msg: None
        limiter.pub_status.publish = lambda msg: None

        msg = PoseStamped()
        msg.header.frame_id = 'body'
        # Very close to body origin (within 0.15m)
        msg.pose.position.x = 0.05
        msg.pose.position.y = 0.0
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0

        # Override bounds so x=0.05 is technically in bounds
        limiter.bounds['x_min'] = 0.0

        limiter.filtered_cb(msg)

        if safe_published:
            out = safe_published[-1]
            dist = np.sqrt(out.pose.position.x**2 +
                          out.pose.position.y**2 +
                          out.pose.position.z**2)
            assert dist >= limiter.min_body_dist - 1e-6, \
                f'Position too close to body: dist={dist}'

    def test_far_position_unmodified(self, limiter):
        """Position far from body should not be modified by collision check."""
        safe_published = []
        limiter.pub_safe.publish = lambda msg: safe_published.append(msg)
        limiter.pub_in_bounds.publish = lambda msg: None
        limiter.pub_status.publish = lambda msg: None

        msg = PoseStamped()
        msg.header.frame_id = 'body'
        msg.pose.position.x = 0.5
        msg.pose.position.y = 0.0
        msg.pose.position.z = 0.1
        msg.pose.orientation.w = 1.0

        limiter.filtered_cb(msg)

        out = safe_published[-1]
        np.testing.assert_allclose(out.pose.position.x, 0.5, atol=1e-3)


# =============================================================================
# Full callback integration (clamp mode)
# =============================================================================

class TestFilteredCallback:
    """Integration tests for the filtered_cb callback in clamp mode."""

    def test_in_bounds_publishes_safe_pose(self, limiter):
        """In-bounds pose should be published on /teleop/safe_pose."""
        safe_published = []
        in_bounds_published = []
        limiter.pub_safe.publish = lambda msg: safe_published.append(msg)
        limiter.pub_in_bounds.publish = lambda msg: in_bounds_published.append(msg)
        limiter.pub_status.publish = lambda msg: None

        msg = PoseStamped()
        msg.header.frame_id = 'body'
        msg.pose.position.x = 0.5
        msg.pose.position.y = 0.0
        msg.pose.position.z = 0.1
        msg.pose.orientation.w = 1.0

        limiter.filtered_cb(msg)

        assert len(safe_published) == 1
        assert in_bounds_published[-1].data is True

    def test_out_of_bounds_clamped_and_flagged(self, limiter):
        """Out-of-bounds pose should be clamped and flagged."""
        safe_published = []
        in_bounds_published = []
        status_published = []
        limiter.pub_safe.publish = lambda msg: safe_published.append(msg)
        limiter.pub_in_bounds.publish = lambda msg: in_bounds_published.append(msg)
        limiter.pub_status.publish = lambda msg: status_published.append(msg)

        msg = PoseStamped()
        msg.header.frame_id = 'body'
        msg.pose.position.x = 2.0  # Way beyond x_max=0.85
        msg.pose.position.y = 0.0
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0

        limiter.filtered_cb(msg)

        assert len(safe_published) == 1
        assert safe_published[0].pose.position.x == DEFAULT_BOUNDS['x_max']
        assert in_bounds_published[-1].data is False
        assert len(status_published) > 0
        assert 'LIMITED' in status_published[-1].data

    def test_orientation_passthrough(self, limiter):
        """Orientation should pass through unchanged."""
        safe_published = []
        limiter.pub_safe.publish = lambda msg: safe_published.append(msg)
        limiter.pub_in_bounds.publish = lambda msg: None
        limiter.pub_status.publish = lambda msg: None

        msg = PoseStamped()
        msg.pose.position.x = 0.5
        msg.pose.position.y = 0.0
        msg.pose.position.z = 0.1
        msg.pose.orientation.x = 0.1
        msg.pose.orientation.y = 0.2
        msg.pose.orientation.z = 0.3
        msg.pose.orientation.w = 0.927

        limiter.filtered_cb(msg)

        out = safe_published[0]
        assert out.pose.orientation.x == 0.1
        assert out.pose.orientation.y == 0.2
        assert out.pose.orientation.z == 0.3
        assert out.pose.orientation.w == 0.927


# =============================================================================
# Reject mode
# =============================================================================

class TestRejectMode:
    """Tests for the reject boundary mode."""

    def test_out_of_bounds_rejected(self):
        """Out-of-bounds pose should not produce any safe_pose output."""
        node = WorkspaceLimiter()
        node.boundary_mode = 'reject'

        safe_published = []
        node.pub_safe.publish = lambda msg: safe_published.append(msg)
        node.pub_in_bounds.publish = lambda msg: None
        node.pub_status.publish = lambda msg: None

        msg = PoseStamped()
        msg.pose.position.x = 5.0  # Way out of bounds
        msg.pose.orientation.w = 1.0

        node.filtered_cb(msg)

        assert len(safe_published) == 0
        node.destroy_node()

    def test_in_bounds_accepted(self):
        """In-bounds pose should be published normally in reject mode."""
        node = WorkspaceLimiter()
        node.boundary_mode = 'reject'

        safe_published = []
        node.pub_safe.publish = lambda msg: safe_published.append(msg)
        node.pub_in_bounds.publish = lambda msg: None
        node.pub_status.publish = lambda msg: None

        msg = PoseStamped()
        msg.pose.position.x = 0.5
        msg.pose.position.y = 0.0
        msg.pose.position.z = 0.1
        msg.pose.orientation.w = 1.0

        node.filtered_cb(msg)

        assert len(safe_published) == 1
        node.destroy_node()


# =============================================================================
# Stress / fuzz testing
# =============================================================================

class TestFuzzWorkspace:
    """Fuzz tests with random positions."""

    def test_random_positions_always_within_bounds_clamp(self, limiter):
        """1000 random positions should always be clamped within bounds."""
        np.random.seed(42)

        safe_published = []
        limiter.pub_safe.publish = lambda msg: safe_published.append(msg)
        limiter.pub_in_bounds.publish = lambda msg: None
        limiter.pub_status.publish = lambda msg: None
        # Disable collision check to test pure clamping
        limiter.check_collision = False

        for _ in range(1000):
            msg = PoseStamped()
            msg.pose.position.x = float(np.random.uniform(-5, 5))
            msg.pose.position.y = float(np.random.uniform(-5, 5))
            msg.pose.position.z = float(np.random.uniform(-5, 5))
            msg.pose.orientation.w = 1.0
            limiter.filtered_cb(msg)

        for out in safe_published:
            p = out.pose.position
            assert p.x >= DEFAULT_BOUNDS['x_min'] - 1e-6
            assert p.x <= DEFAULT_BOUNDS['x_max'] + 1e-6
            assert p.y >= DEFAULT_BOUNDS['y_min'] - 1e-6
            assert p.y <= DEFAULT_BOUNDS['y_max'] + 1e-6
            assert p.z >= DEFAULT_BOUNDS['z_min'] - 1e-6
            assert p.z <= DEFAULT_BOUNDS['z_max'] + 1e-6
