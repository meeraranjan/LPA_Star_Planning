from setuptools import find_packages, setup
import os
package_name = 'greenhorn_nav'
from glob import glob

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'local_goal_selector = greenhorn_nav.local_goal_selector:main',
            'lpa_planner_node = greenhorn_nav.lpa_planner_node:main',
            'boat_simulator = greenhorn_nav.demo.boat_simulator:main',
            'map_publisher = greenhorn_nav.demo.map_publisher:main',
            'local_goal_visualizer = greenhorn_nav.demo.local_goal_visualizer:main'
        ],
    },
)
