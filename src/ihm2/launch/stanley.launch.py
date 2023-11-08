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
                name="k_P",
                default_value="100.0",
            ),
            DeclareLaunchArgument(
                name="k_I",
                default_value="0.0",
            ),
            DeclareLaunchArgument(
                name="k_psi",
                default_value="1.7",
            ),
            DeclareLaunchArgument(
                name="k_e",
                default_value="1.5",
            ),
            DeclareLaunchArgument(
                name="k_s",
                default_value="3.0",
            ),
            DeclareLaunchArgument(
                name="T_max",
                default_value="200.0",
                description="maximum torque (Nm)",
            ),
            DeclareLaunchArgument(
                name="delta_max",
                default_value="0.5",
                description="maximum steering angle (rad)",
            ),
            DeclareLaunchArgument(
                name="v_x_ref",
                default_value="5.0",
                description="reference longitudinal velocity (m/s)",
            ),
            DeclareLaunchArgument(
                name="track_name_or_file",
                default_value="fsds_competition_1",
                description="name of the track or path to the track file",
            ),
            DeclareLaunchArgument(
                name="phi_ref_preview_distance",
                default_value="0.0",
            ),
            Node(
                package="ihm2",
                executable="stanley_control_node",
                output="screen",
                on_exit=Shutdown(),
                parameters=[
                    {
                        "k_P": LaunchConfiguration("k_P"),
                        "k_I": LaunchConfiguration("k_I"),
                        "k_psi": LaunchConfiguration("k_psi"),
                        "k_e": LaunchConfiguration("k_e"),
                        "k_s": LaunchConfiguration("k_s"),
                        "T_max": LaunchConfiguration("T_max"),
                        "delta_max": LaunchConfiguration("delta_max"),
                        "phi_ref_preview_distance": LaunchConfiguration(
                            "phi_ref_preview_distance"
                        ),
                        "v_x_ref": LaunchConfiguration("v_x_ref"),
                        "track_name_or_file": LaunchConfiguration("track_name_or_file"),
                    }
                ],
            ),
            Node(
                package="ihm2",
                executable="sim_node",
                output="screen",
                on_exit=Shutdown(),
                parameters=[
                    {
                        "v_dyn": LaunchConfiguration("v_dyn"),
                        # "T_max": LaunchConfiguration("T_max"),
                        # "delta_max": LaunchConfiguration("delta_max"),
                        "track_name_or_file": LaunchConfiguration("track_name_or_file"),
                        "manual_control": False,
                    }
                ],
            ),
        ]
    )


if __name__ == "__main__":
    generate_launch_description()
