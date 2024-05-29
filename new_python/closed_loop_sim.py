from abc import ABC, abstractmethod
from collections import deque
from enum import Enum, auto
from time import perf_counter
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from acados_template import AcadosOcpSolver, AcadosSimSolver
from icecream import ic
from matplotlib.axes import Axes
from strongpods import PODS
from tqdm import trange


@PODS
class Trajectory:
    length: npt.NDArray[np.float64]
    X: npt.NDArray[np.float64]
    Y: npt.NDArray[np.float64]
    heading: npt.NDArray[np.float64]
    curvature: npt.NDArray[np.float64]
    closed: bool  # if true, the last point of each array above IS NOT the same as the first one


class SimModelVariant(Enum):
    KIN6 = auto()
    DYN6 = auto()
    KIN6_DYN6 = auto()
    DYN10 = auto()


def closed_loop(
    track_data,
    simulator_type,
    motion_planner_type: type,
    motion_tracker_type: type,
    low_level_controller_type: type,
    interation_end_callback: Callable,
    cleanup_callback: Callable,
):
    # initialize components for closed loop simulation
    simulator = ...
    track_data = ...
    motion_planner = ...
    motion_tracker = ...
    low_level_controller = ...

    # initialize car state and simulation variables
    car_state = ...

    # initialize logged data

    while True:
        # call subcomponents making up the control component: motion planer, motion tracker and low level controller
        motion_plan = motion_planner.plan(car_state)
        tracking_plan = motion_tracker.track(car_state, motion_plan)
        actuator_target = low_level_controller.get_actuator_target(tracking_plan)

        # call simulator to update state
        car_state = simulator.simulate(car_state, actuator_target)

        # optional logging after each time step (state, actuator_target, info on MP, MT and LLC, etc. )

    # optional cleanup (for dumping data to vile, visualizing stuff, etc.)
