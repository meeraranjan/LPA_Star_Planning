#!/usr/bin/env python3
"""
demo_system.launch.py
Launches full demo stack:
- LPA* planner (via lpa_planner_launch.py)
- map_publisher
- local_goal_selector
- local_goal_visualizer
- boat_simulator
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    pkg_share = FindPackageShare('greenhorn_nav').find('greenhorn_nav')

    # Parameter file for planner
    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_share, 'config', 'lpa_params.yaml'),
        description='Path to the LPA planner parameter file'
    )

    params_file = LaunchConfiguration('params_file')

    lpa_planner_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'lpa_planner.launch.py')
        ),
        launch_arguments={'params_file': params_file}.items()
    )
    declare_lgs_params_cmd = DeclareLaunchArgument(
        'local_goal_selector',
        default_value=os.path.join(pkg_share, 'config', 'local_goal_selector.yaml'),
        description='Path to the local goal selector parameter file'
    )   
    local_goal_selector = LaunchConfiguration('local_goal_selector')
    local_goal_selector_node = Node(
        package='greenhorn_nav',
        executable='local_goal_selector',
        name='local_goal_selector',
        output='screen',
        parameters=[LaunchConfiguration('local_goal_selector')]
    )

    map_publisher_node = Node(
        package='greenhorn_nav',
        executable='map_publisher',
        name='map_publisher',
        output='screen'
    )


    local_goal_visualizer_node = Node(
        package='greenhorn_nav',
        executable='local_goal_visualizer',
        name='local_goal_visualizer',
        output='screen'
    )

    boat_simulator_node = Node(
        package='greenhorn_nav',
        executable='boat_simulator',
        name='boat_simulator',
        output='screen'
    )

    # --- Final launch description ---
    return LaunchDescription([
        declare_params_file_cmd,
        declare_lgs_params_cmd,
        lpa_planner_launch,
        map_publisher_node,
        local_goal_selector_node,
        local_goal_visualizer_node,
        boat_simulator_node,
    ])
