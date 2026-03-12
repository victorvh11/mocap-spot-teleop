#!/usr/bin/env python3
"""
test_filters.py - Unit tests for ButterworthFilter and ExponentialFilter.
Tests pure signal processing logic without requiring ROS2.
"""

import pytest
import numpy as np
from scipy.signal import butter, lfilter_zi, lfilter


# ============================================================================
# Reproduce the filter classes here (standalone, no ROS dependency)
# ============================================================================

class ButterworthFilter:
    def __init__(self, order, cutoff_hz, sample_rate_hz, n_channels=3):
        nyquist = sample_rate_hz / 2.0
        normalized_cutoff = min(cutoff_hz / nyquist, 0.99)
        self.b, self.a = butter(order, normalized_cutoff, btype='low')
        self.zi = [lfilter_zi(self.b, self.a) for _ in range(n_channels)]
        self.initialized = False

    def filter(self, data):
        if not self.initialized:
            for i in range(len(self.zi)):
                self.zi[i] = self.zi[i] * data[i]
            self.initialized = True
        out = np.zeros_like(data)
        for i in range(len(data)):
            y, self.zi[i] = lfilter(self.b, self.a, [data[i]], zi=self.zi[i])
            out[i] = y[0]
        return out


class ExponentialFilter:
    def __init__(self, alpha, n_channels=3):
        self.alpha = alpha
        self.state = None

    def filter(self, data):
        if self.state is None:
            self.state = data.copy()
            return data.copy()
        self.state = self.alpha * data + (1.0 - self.alpha) * self.state
        return self.state.copy()


# ============================================================================
# ButterworthFilter Tests
# ============================================================================

class TestButterworthFilter:
    def test_initialization(self):
        filt = ButterworthFilter(order=2, cutoff_hz=5.0, sample_rate_hz=100.0, n_channels=3)
        assert len(filt.zi) == 3
        assert not filt.initialized

    def test_first_sample_initializes(self):
        filt = ButterworthFilter(2, 5.0, 100.0, 3)
        filt.filter(np.array([1.0, 2.0, 3.0]))
        assert filt.initialized

    def test_step_response_converges(self):
        filt = ButterworthFilter(2, 5.0, 100.0, 3)
        step = np.array([1.0, -0.5, 2.0])
        for _ in range(200):
            result = filt.filter(step)
        np.testing.assert_allclose(result, step, atol=0.01)

    def test_rejects_high_frequency_noise(self):
        filt = ButterworthFilter(2, 5.0, 100.0, 1)
        fs = 100.0
        t = np.arange(0, 2.0, 1.0/fs)
        outputs = []
        for ti in t:
            sample = np.array([1.0 + 0.5 * np.sin(2 * np.pi * 40.0 * ti)])
            outputs.append(filt.filter(sample)[0])
        settled = np.array(outputs[-100:])
        residual = np.max(settled) - np.min(settled)
        assert residual < 0.05, f'Noise not rejected: amplitude={residual:.4f}'

    def test_passes_low_frequency_signal(self):
        filt = ButterworthFilter(2, 5.0, 100.0, 1)
        fs = 100.0
        t = np.arange(0, 4.0, 1.0/fs)
        outputs = []
        for ti in t:
            sample = np.array([np.sin(2 * np.pi * 1.0 * ti)])
            outputs.append(filt.filter(sample)[0])
        settled = np.array(outputs[-200:])
        amplitude = (np.max(settled) - np.min(settled)) / 2.0
        assert amplitude > 0.9, f'Low freq over-attenuated: {amplitude:.3f}'

    def test_channels_independent(self):
        filt = ButterworthFilter(2, 5.0, 100.0, 3)
        for _ in range(200):
            filt.filter(np.array([1.0, 0.0, -1.0]))
        result = filt.filter(np.array([1.0, 0.0, -1.0]))
        np.testing.assert_allclose(result[0], 1.0, atol=0.01)
        np.testing.assert_allclose(result[1], 0.0, atol=0.01)
        np.testing.assert_allclose(result[2], -1.0, atol=0.01)

    def test_cutoff_near_nyquist(self):
        filt = ButterworthFilter(2, cutoff_hz=55.0, sample_rate_hz=100.0, n_channels=1)
        result = filt.filter(np.array([1.0]))
        assert np.isfinite(result[0])

    def test_zero_input(self):
        filt = ButterworthFilter(2, 5.0, 100.0, 3)
        zero = np.zeros(3)
        for _ in range(50):
            result = filt.filter(zero)
        np.testing.assert_allclose(result, zero, atol=1e-10)


# ============================================================================
# ExponentialFilter Tests
# ============================================================================

class TestExponentialFilter:
    def test_first_sample_passthrough(self):
        filt = ExponentialFilter(alpha=0.5, n_channels=3)
        data = np.array([1.0, 2.0, 3.0])
        result = filt.filter(data)
        np.testing.assert_array_equal(result, data)

    def test_alpha_one_passthrough(self):
        filt = ExponentialFilter(alpha=1.0, n_channels=3)
        filt.filter(np.zeros(3))
        data = np.array([5.0, -3.0, 7.0])
        result = filt.filter(data)
        np.testing.assert_array_equal(result, data)

    def test_alpha_zero_holds_first(self):
        filt = ExponentialFilter(alpha=0.0, n_channels=3)
        first = np.array([1.0, 2.0, 3.0])
        filt.filter(first)
        for _ in range(50):
            result = filt.filter(np.array([99.0, 99.0, 99.0]))
        np.testing.assert_array_equal(result, first)

    def test_convergence(self):
        filt = ExponentialFilter(alpha=0.3, n_channels=2)
        target = np.array([5.0, -3.0])
        filt.filter(np.zeros(2))
        for _ in range(200):
            result = filt.filter(target)
        np.testing.assert_allclose(result, target, atol=1e-6)

    def test_smoothing_reduces_noise(self):
        filt = ExponentialFilter(alpha=0.1, n_channels=1)
        np.random.seed(42)
        outputs = []
        for _ in range(200):
            noisy = np.array([1.0 + np.random.normal(0, 0.5)])
            outputs.append(filt.filter(noisy)[0])
        output_std = np.std(outputs[-100:])
        assert output_std < 0.2, f'Not smooth: std={output_std:.3f}'

    def test_step_monotonic(self):
        filt = ExponentialFilter(alpha=0.2, n_channels=1)
        filt.filter(np.array([0.0]))
        outputs = []
        for _ in range(20):
            outputs.append(filt.filter(np.array([1.0]))[0])
        for i in range(1, len(outputs)):
            assert outputs[i] >= outputs[i-1] - 1e-10
