from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'spot_mocap_teleop'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Teleoperation of Spot arm via OptiTrack motion capture glove',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mocap_glove_processor = spot_mocap_teleop.mocap_glove_processor:main',
            'motion_filter = spot_mocap_teleop.motion_filter:main',
            'workspace_limiter = spot_mocap_teleop.workspace_limiter:main',
            'spot_arm_commander = spot_mocap_teleop.spot_arm_commander:main',
            'safety_monitor = spot_mocap_teleop.safety_monitor:main',
            'rosbag_recorder = spot_mocap_teleop.rosbag_recorder:main',
            'teleop_manager = spot_mocap_teleop.teleop_manager:main',
        ],
    },
)
