"""
spot_mocap_teleop.launch.py
============================
Main launch file for the Spot mocap teleoperation system.

Launches:
    1. mocap_glove_processor  - Reads OptiTrack, extracts glove pose
    2. motion_filter          - Low-pass filter + rate limiter
    3. workspace_limiter      - Enforces safe workspace bounds
    4. spot_arm_commander     - Sends commands to Spot arm
    5. safety_monitor         - Monitors safety constraints
    6. rosbag_recorder        - Records all data to rosbag
    7. teleop_manager         - Orchestrates the pipeline
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ---- Launch arguments ----
    spot_name_arg = DeclareLaunchArgument(
        'spot_name', default_value='',
        description='Spot robot namespace (empty if none)'
    )
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('spot_mocap_teleop'), 'config', 'teleop_params.yaml'
        ]),
        description='Path to teleop parameter file'
    )
    auto_record_arg = DeclareLaunchArgument(
        'auto_record', default_value='false',
        description='Auto-start rosbag recording'
    )

    config_file = LaunchConfiguration('config_file')
    spot_name = LaunchConfiguration('spot_name')

    # ---- Nodes ----

    mocap_glove_processor = Node(
        package='spot_mocap_teleop',
        executable='mocap_glove_processor',
        name='mocap_glove_processor',
        output='screen',
        parameters=[config_file],
        remappings=[],
    )

    motion_filter = Node(
        package='spot_mocap_teleop',
        executable='motion_filter',
        name='motion_filter',
        output='screen',
        parameters=[config_file],
    )

    workspace_limiter = Node(
        package='spot_mocap_teleop',
        executable='workspace_limiter',
        name='workspace_limiter',
        output='screen',
        parameters=[config_file],
    )

    spot_arm_commander = Node(
        package='spot_mocap_teleop',
        executable='spot_arm_commander',
        name='spot_arm_commander',
        output='screen',
        parameters=[config_file, {'spot_name': spot_name}],
    )

    safety_monitor = Node(
        package='spot_mocap_teleop',
        executable='safety_monitor',
        name='safety_monitor',
        output='screen',
        parameters=[config_file],
    )

    rosbag_recorder = Node(
        package='spot_mocap_teleop',
        executable='rosbag_recorder',
        name='rosbag_recorder',
        output='screen',
        parameters=[config_file],
    )

    teleop_manager = Node(
        package='spot_mocap_teleop',
        executable='teleop_manager',
        name='teleop_manager',
        output='screen',
        parameters=[config_file, {'spot_name': spot_name}],
    )

    return LaunchDescription([
        spot_name_arg,
        config_file_arg,
        auto_record_arg,
        mocap_glove_processor,
        motion_filter,
        workspace_limiter,
        spot_arm_commander,
        safety_monitor,
        rosbag_recorder,
        teleop_manager,
    ])
