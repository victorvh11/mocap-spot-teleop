#!/usr/bin/env python3
"""conftest.py - Shared pytest fixtures for spot_mocap_teleop test suites."""

import pytest
import numpy as np

try:
    import rclpy
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False


@pytest.fixture(scope='session', autouse=True)
def ros2_context():
    if HAS_ROS2:
        rclpy.init()
        yield
        rclpy.shutdown()
    else:
        yield


def require_ros2(func):
    return pytest.mark.skipif(not HAS_ROS2, reason='ROS2 not available')(func)
