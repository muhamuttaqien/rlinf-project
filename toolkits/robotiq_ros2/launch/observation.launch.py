# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launch the observation publisher with serial access disabled by default."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    """Select a reviewed parameter file without changing other robot nodes."""
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=PathJoinSubstitution(
                    [
                        get_package_share_directory("rlinf_robotiq_observation"),
                        "config",
                        "observation.yaml",
                    ]
                ),
            ),
            Node(
                package="rlinf_robotiq_observation",
                executable="robotiq_observation",
                name="robotiq_observation",
                parameters=[LaunchConfiguration("config")],
                output="screen",
            ),
        ]
    )
