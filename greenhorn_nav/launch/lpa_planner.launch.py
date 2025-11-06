from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

def generate_launch_description():
    ld = LaunchDescription()
    ld.add_action(DeclareLaunchArgument('params_file', default_value=str(THIS_DIR / '../config/lpa_params.yaml')))

    params_file = LaunchConfiguration('params_file')

    node = Node(
        package='greenhorn_nav',              
        executable='lpa_planner_node',      
        name='lpa_planner_node',
        output='screen',
        parameters=[params_file],
        remappings=[
            # ('/map', '/my_map'),
        ],
    )
    ld.add_action(node)
    return ld
