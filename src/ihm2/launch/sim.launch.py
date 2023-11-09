# Copyright (c) 2023, Tudor Oancea, Mattéo Berthet, Antonio Pisanello, Maximilian Gangloff, Joe Najm, Louis Gounot, Philippe Servant, Vincent Philippoz working for EPFL Racing Team Driverless
from launch import LaunchDescription
from launch.actions import *
from launch.conditions import *
from launch.launch_description_sources import *
from launch.substitutions import *
from launch_ros.actions import *
from launch_ros.substitutions import *


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                name="v_dyn",
                default_value="3.0",
                description="the longitudinal velocity threshold from which the dynamic model is used in simulation",
            ),
            DeclareLaunchArgument(
                name="T_max",
                default_value="1107.0",
                description="maximum torque (Nm)",
            ),
            DeclareLaunchArgument(
                name="delta_max",
                default_value="0.5",
                description="maximum steering angle (rad)",
            ),
            DeclareLaunchArgument(
                name="track_name_or_file",
                default_value="fsds_competition_1",
                description="name of the track or path to the track file",
            ),
            DeclareLaunchArgument(
                name="manual_control",
                default_value="true",
                description="whether the sim node should listen on /ihm2/target_controls (programmatic control) or /ihm2/alternative_target_controls (manual control)",
            ),
            Node(
                package="ihm2",
                executable="sim_node",
                output="screen",
                on_exit=Shutdown(),
                parameters=[
                    {
                        "v_dyn": LaunchConfiguration("v_dyn"),
                        "T_max": LaunchConfiguration("T_max"),
                        "delta_max": LaunchConfiguration("delta_max"),
                        "track_name_or_file": LaunchConfiguration("track_name_or_file"),
                        "manual_control": LaunchConfiguration("manual_control"),
                    }
                ],
            ),
        ]
    )


if __name__ == "__main__":
    generate_launch_description()
