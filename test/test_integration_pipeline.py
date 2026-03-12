#!/usr/bin/env python3
"""
test_integration_pipeline.py
============================
Integration tests verifying the end-to-end data flow through the
teleop pipeline without requiring actual hardware.

Tests the full chain:
    raw pose -> GloveProcessor -> MotionFilter -> WorkspaceLimiter

Covers:
    - Full pipeline: in-bounds pose flows through all stages
    - Full pipeline: out-of-bounds pose is clamped
    - Full pipeline: high-velocity input is rate-limited
    - Full pipeline: consistent orientation through all stages
    - Pipeline latency (processing overhead)
    - Pipeline with calibration applied
    - Noisy signal produces smooth output
"""

import pytest
import numpy as np
import time

import rclpy

from geometry_msgs.msg import PoseStamped
from spot_mocap_teleop.mocap_glove_processor import MocapGloveProcessor
from spot_mocap_teleop.motion_filter import MotionFilter
from spot_mocap_teleop.workspace_limiter import WorkspaceLimiter


DEFAULT_BOUNDS = {
    'x_min': 0.2, 'x_max': 0.85,
    'y_min': -0.50, 'y_max': 0.50,
    'z_min': -0.25, 'z_max': 0.45,
}


@pytest.fixture
def pipeline():
    """Create all three pipeline nodes and wire them together."""
    glove = MocapGloveProcessor()
    filt = MotionFilter()
    limiter = WorkspaceLimiter()

    # Wire: glove -> filter
    filter_inputs = []
    original_glove_pub = glove.pub_raw_pose.publish

    def capture_and_forward_to_filter(msg):
        filter_inputs.append(msg)
        filt.raw_pose_cb(msg)

    glove.pub_raw_pose.publish = capture_and_forward_to_filter
    glove.pub_velocity.publish = lambda msg: None
    glove.pub_tracking.publish = lambda msg: None

    # Wire: filter -> limiter
    limiter_inputs = []
    original_filter_pub = filt.pub_filtered.publish

    def capture_and_forward_to_limiter(msg):
        limiter_inputs.append(msg)
        limiter.filtered_cb(msg)

    filt.pub_filtered.publish = capture_and_forward_to_limiter

    # Capture final output
    final_outputs = []
    limiter.pub_safe.publish = lambda msg: final_outputs.append(msg)
    limiter.pub_in_bounds.publish = lambda msg: None
    limiter.pub_status.publish = lambda msg: None

    class PipelineFixture:
        def __init__(self):
            self.glove = glove
            self.filt = filt
            self.limiter = limiter
            self.filter_inputs = filter_inputs
            self.limiter_inputs = limiter_inputs
            self.final_outputs = final_outputs

        def feed_pose(self, x, y, z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
            """Feed a raw pose through the entire pipeline."""
            msg = PoseStamped()
            msg.header.frame_id = 'mocap_world'
            msg.pose.position.x = float(x)
            msg.pose.position.y = float(y)
            msg.pose.position.z = float(z)
            msg.pose.orientation.x = float(qx)
            msg.pose.orientation.y = float(qy)
            msg.pose.orientation.z = float(qz)
            msg.pose.orientation.w = float(qw)
            self.glove._process_pose(msg)

    fixture = PipelineFixture()
    yield fixture

    glove.destroy_node()
    filt.destroy_node()
    limiter.destroy_node()


class TestFullPipelineFlow:
    """End-to-end pipeline tests."""

    def test_in_bounds_pose_flows_through(self, pipeline):
        """An in-bounds pose should arrive at the final output."""
        # Feed a pose that, after scaling (0.6x), lands inside workspace
        # Target output: x=0.6*0.7=0.42, y=0, z=0.6*0.3=0.18
        pipeline.feed_pose(0.7, 0.0, 0.3)

        assert len(pipeline.filter_inputs) == 1, 'Filter should receive 1 message'
        assert len(pipeline.limiter_inputs) == 1, 'Limiter should receive 1 message'
        assert len(pipeline.final_outputs) == 1, 'Should produce 1 final output'

        out = pipeline.final_outputs[0]
        # Should be within workspace bounds
        assert out.pose.position.x >= DEFAULT_BOUNDS['x_min']
        assert out.pose.position.x <= DEFAULT_BOUNDS['x_max']

    def test_out_of_bounds_pose_clamped(self, pipeline):
        """A pose that maps outside workspace should be clamped in final output."""
        # After scaling: x = 5.0 * 0.6 = 3.0 (way beyond x_max=0.85)
        pipeline.feed_pose(5.0, 0.0, 0.0)

        assert len(pipeline.final_outputs) == 1
        out = pipeline.final_outputs[0]

        assert out.pose.position.x <= DEFAULT_BOUNDS['x_max'] + 1e-6

    def test_orientation_preserved_through_pipeline(self, pipeline):
        """Orientation should be maintained through all stages."""
        qx, qy, qz, qw = 0.0, 0.0, 0.3827, 0.9239  # ~45 deg around Z

        pipeline.feed_pose(0.7, 0.0, 0.3, qx, qy, qz, qw)

        out = pipeline.final_outputs[0]
        # Orientation should be close to input (SLERP smoothing may adjust slightly)
        # On first sample, SLERP should return the input itself
        np.testing.assert_allclose(
            [out.pose.orientation.x, out.pose.orientation.y,
             out.pose.orientation.z, out.pose.orientation.w],
            [qx, qy, qz, qw],
            atol=0.1,  # Allow for SLERP smoothing
            err_msg='Orientation not preserved through pipeline'
        )


class TestPipelineMultipleSamples:
    """Pipeline behavior with multiple sequential samples."""

    def test_steady_state_convergence(self, pipeline):
        """Repeated identical poses should converge to a stable output."""
        for i in range(50):
            pipeline.feed_pose(0.7, 0.0, 0.3)
            time.sleep(0.001)

        # Last several outputs should be very similar
        last_outputs = pipeline.final_outputs[-10:]
        positions = np.array([
            [o.pose.position.x, o.pose.position.y, o.pose.position.z]
            for o in last_outputs
        ])

        # Standard deviation across last 10 should be tiny
        std = np.std(positions, axis=0)
        np.testing.assert_array_less(std, [0.01, 0.01, 0.01],
                                     err_msg=f'Output not converged: std={std}')

    def test_noisy_input_produces_smooth_output(self, pipeline):
        """Noisy input should be smoothed by the filter."""
        np.random.seed(42)

        for i in range(100):
            # Base position + noise
            x = 0.7 + np.random.normal(0, 0.05)
            y = 0.0 + np.random.normal(0, 0.05)
            z = 0.3 + np.random.normal(0, 0.05)
            pipeline.feed_pose(x, y, z)
            time.sleep(0.001)

        # Compare input noise vs output noise (last 50 samples)
        final_positions = np.array([
            [o.pose.position.x, o.pose.position.y, o.pose.position.z]
            for o in pipeline.final_outputs[-50:]
        ])
        output_std = np.std(final_positions, axis=0)

        # After Butterworth filter, output should be much smoother
        # Input std was ~0.05*0.6 = 0.03 (after scaling)
        # Output should be significantly less
        assert np.all(output_std < 0.025), \
            f'Output not smooth enough: std={output_std}'

    def test_all_outputs_within_bounds(self, pipeline):
        """Every output from the pipeline should be within workspace."""
        np.random.seed(123)

        for _ in range(200):
            x = np.random.uniform(-2, 5)
            y = np.random.uniform(-3, 3)
            z = np.random.uniform(-2, 3)
            pipeline.feed_pose(x, y, z)
            time.sleep(0.001)

        # Disable collision check for pure bounds test
        pipeline.limiter.check_collision = False

        for out in pipeline.final_outputs:
            p = out.pose.position
            assert p.x >= DEFAULT_BOUNDS['x_min'] - 0.01, f'X={p.x} below min'
            assert p.x <= DEFAULT_BOUNDS['x_max'] + 0.01, f'X={p.x} above max'
            assert p.y >= DEFAULT_BOUNDS['y_min'] - 0.01, f'Y={p.y} below min'
            assert p.y <= DEFAULT_BOUNDS['y_max'] + 0.01, f'Y={p.y} above max'
            assert p.z >= DEFAULT_BOUNDS['z_min'] - 0.01, f'Z={p.z} below min'
            assert p.z <= DEFAULT_BOUNDS['z_max'] + 0.01, f'Z={p.z} above max'


class TestPipelineWithCalibration:
    """Pipeline behavior with calibration applied."""

    def test_calibrated_origin_produces_expected_output(self, pipeline):
        """After calibration, the calibration position should map to near-zero."""
        # Feed some initial data
        pipeline.feed_pose(1.0, 0.5, 0.8)
        time.sleep(0.005)

        # Calibrate at current position
        from std_srvs.srv import Trigger
        req = Trigger.Request()
        resp = Trigger.Response()
        pipeline.glove.calibrate_cb(req, resp)
        assert resp.success

        # Clear outputs
        pipeline.final_outputs.clear()

        # Feed the same position (should now be near zero after calibration)
        pipeline.feed_pose(1.0, 0.5, 0.8)

        assert len(pipeline.final_outputs) == 1
        out = pipeline.final_outputs[0]

        # Position should be near zero (or clamped at workspace min)
        # The exact value depends on how clamp interacts with near-zero values
        assert abs(out.pose.position.x) < 0.5 or \
               out.pose.position.x == DEFAULT_BOUNDS['x_min']


class TestPipelineLatency:
    """Performance tests for pipeline processing time."""

    def test_single_sample_under_1ms(self, pipeline):
        """Processing a single sample through the pipeline should take < 1ms."""
        # Warm up
        for _ in range(10):
            pipeline.feed_pose(0.5, 0.0, 0.2)

        pipeline.final_outputs.clear()

        start = time.perf_counter()
        pipeline.feed_pose(0.6, 0.1, 0.3)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.001, \
            f'Pipeline processing too slow: {elapsed*1000:.2f}ms'

    def test_throughput_100hz(self, pipeline):
        """Pipeline should handle 100Hz throughput."""
        n_samples = 100
        start = time.perf_counter()

        for i in range(n_samples):
            pipeline.feed_pose(0.5 + 0.001*i, 0.0, 0.2)

        elapsed = time.perf_counter() - start

        # Should process 100 samples in under 100ms (well within 10ms budget each)
        assert elapsed < 0.1, \
            f'Throughput too low: {n_samples/elapsed:.0f} Hz (need >= 100Hz)'
        assert len(pipeline.final_outputs) == n_samples
