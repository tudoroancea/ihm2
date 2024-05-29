from abc import ABC, abstractmethod
from enum import Enum, auto
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from acados_template import AcadosOcpOptions, AcadosOcpSolver, AcadosSimOpts
from casadi import MX
from constants import R_w, car_width, front_axle_track, l_F, l_R, rear_axle_track
from icecream import ic
from matplotlib.axes import Axes
from models import (
    fdyn6_model,
    fdyn10_model,
    fkin6_model,
    get_acados_model_from_explicit_dynamics,
    get_acados_model_from_implicit_dynamics,
)
from motion_planning import (
    NUMBER_SPLINE_INTERVALS,
    offline_motion_plan,
    triple_motion_plan_ref,
)
from mpc import ModelBounds, get_acados_ocp, get_acados_solver, get_ipopt_solver
from sim import generate_sim_solver
from tqdm import trange
from track_database.tracks import Track

control_variables_idx = {"torque_command": 0, "steering_command": 1}
state_variables_idx_cartesian = {
    "X": 0,
    "Y": 1,
    "yaw": 2,
    "longitudinal_velocity": 3,
    "lateral_velocity": 4,
    "yaw_rate": 5,
    "torque": 6,
    "steering": 7,
}
state_variables_idx_frenet = {
    "track_progess": 0,
    "lateral_error": 1,
    "yaw_error": 2,
    "longitudinal_velocity": 3,
    "lateral_velocity": 4,
    "yaw_rate": 5,
    "torque": 6,
    "steering": 7,
}


from collections import deque

model_bounds = ModelBounds(
    n_max=2.0,
    v_x_min=0.0,
    v_x_max=31.0,
    T_max=500.0,
    delta_max=0.5,
    T_dot_max=1e6,
    delta_dot_max=1.0,
    a_lat_max=5.0,
)


def project(
    car_pos: npt.NDArray[np.float64],
    car_pos_2: np.ndarray[np.float64],
    s_guess: float,
    s_tol: float = 0.5,
) -> tuple[float, int]:
    # extract all points in X_ref, Y_ref associated with s_ref values within s_guess ± s_tol
    # find the closest point to car_pos to find one segment extremity
    # compute the angles between car_pos, the closest point and the next/previous point
    # find the segment that contains the actual projection of car_pos on the reference path
    # compute the curvilinear abscissa of car_pos on the segment
    return (s_guess, 0)


class MyTrack:
    s_ref: np.ndarray
    X_ref: np.ndarray
    Y_ref: np.ndarray
    phi_ref: np.ndarray
    kappa_ref: np.ndarray


class Controller(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def compute_control(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        pass


class StanleyController(Controller):
    last_epsilons = deque(maxlen=400)
    k_P: float
    k_I: float
    k_n: float
    k_n: float
    k_psi: float
    k_kappa: float
    T_max: float
    delta_max: float
    dt: float

    def __init__(
        self,
        k_P=90.0,
        k_I=20.0,
        k_offset=1.0,
        k_n=5.5,
        k_psi=1.8,
        k_kappa=1.0,
        T_max=500.0,
        delta_max=0.5,
        dt=1 / 20,
    ) -> None:
        super().__init__()
        self.k_P = k_P
        self.k_I = k_I
        self.k_offset = k_offset
        self.k_n = k_n
        self.k_psi = k_psi
        self.k_kappa = k_kappa
        self.T_max = T_max
        self.delta_max = delta_max
        self.dt = dt

    def compute_control(
        self,
        n: float,
        psi: float,
        v_x: float,
        v_x_ref: float,
        kappa_ref: float,
    ) -> np.ndarray:
        # torque control
        epsilon = v_x_ref - v_x
        u_T = self.k_P * epsilon
        if len(self.last_epsilons) > 1:
            u_T += (
                self.k_I
                * (
                    np.sum(self.last_epsilons)
                    - self.last_epsilons[0]
                    - self.last_epsilons[-1]
                )
                * self.dt
            )
        self.last_epsilons.append(epsilon)
        # steering control
        u_delta = self.k_kappa * np.arctan(
            2 * np.tan(np.arcsin(kappa_ref * l_R))
        )  # feedforward
        u_delta += -self.k_psi * psi  # heading error compensation
        u_delta += -np.arctan(self.k_n * n / (2.0 + v_x))  # lateral error compensation
        # saturate control
        u_max = np.array([self.T_max, self.delta_max])
        return np.clip(np.array([u_T, u_delta]), -u_max, u_max)


class IHM2Controller(Controller):
    Nf: int
    dt: float
    s_target: float

    nx = 8
    nu = 2

    solver: AcadosOcpSolver

    x_pred: list[np.ndarray]
    u_pred: list[np.ndarray]

    def __init__(
        self,
        s_ref: np.ndarray,
        kappa_ref: np.ndarray,
        Nf: int = 40,
        dt: float = 1 / 20,
        s_target: float = 40.0,
        n_max: float = 2.0,
        v_x_max: float = 31.0,
        T_max: float = 500.0,
        delta_max: float = 0.5,
        T_dot_max: float = 1e6,
        delta_dot_max: float = 1.0,
        a_lat_max: float = 5.0,
        q_s: float = 1.0,
        q_n: float = 1.0,
        q_psi: float = 1.0,
        q_v_x: float = 1.0,
        q_v_y: float = 1.0,
        q_r: float = 1.0,
        q_T: float = 1.0,
        q_delta: float = 100.0,
        q_s_f: float = 1000.0,
        q_n_f: float = 100.0,
        q_psi_f: float = 100.0,
        q_v_x_f: float = 1.0,
        q_v_y_f: float = 1.0,
        q_r_f: float = 1.0,
        q_T_f: float = 1.0,
        q_delta_f: float = 100.0,
        q_T_dot: float = 0.0,
        q_delta_dot: float = 500.0,
    ) -> None:
        super().__init__()
        self.Nf = Nf
        self.dt = dt
        self.s_target = s_target

        model = get_acados_model_from_explicit_dynamics(
            name="ihm2_fkin6",
            continuous_model_fn=fkin6_model,
            x=MX.sym("x", self.nx),
            u=MX.sym("u", self.nu),
            p=MX.sym("p", 3 * 2 * NUMBER_SPLINE_INTERVALS),
        )
        ocp = get_acados_ocp(
            model, Nf, n_max, v_x_max, T_max, delta_max, T_dot_max, delta_dot_max
        )
        ocp_opts = AcadosOcpOptions()
        ocp_opts.tf = Nf * dt
        ocp_opts.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp_opts.nlp_solver_type = "SQP"
        ocp_opts.nlp_solver_max_iter = 2
        ocp_opts.hessian_approx = "GAUSS_NEWTON"
        ocp_opts.hpipm_mode = "SPEED_ABS"
        ocp_opts.integrator_type = "IRK"
        ocp_opts.sim_method_num_stages = 4
        ocp_opts.sim_method_num_steps = 1
        ocp_opts.globalization = "MERIT_BACKTRACKING"
        ocp_opts.print_level = 0
        self.solver = get_acados_solver(ocp, ocp_opts, "generated")

        # initialize prediction arrays ############################################################
        self.x_pred = [
            np.array([-6.0 + i * dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            for i in range(Nf + 1)
        ]
        self.u_pred = [np.array([model_bounds.T_max, 0.0]) for i in range(Nf)]

        # set all parameters for the mpc solver ###############################################
        p = np.append(s_ref, kappa_ref)
        for i in range(Nf + 1):
            # set parameters
            self.solver.set(i, "p", p)
            # set cost weights
            if i < Nf:
                self.solver.cost_set(
                    i,
                    "W",
                    np.diag(
                        np.array(
                            [
                                q_s,
                                q_n,
                                q_psi,
                                q_v_x,
                                q_v_y,
                                q_r,
                                q_T,
                                q_delta,
                                q_T,
                                q_delta,
                                q_T_dot,
                                q_delta_dot,
                            ]
                        )
                    ),
                )
            else:
                self.solver.cost_set(
                    i,
                    "W",
                    np.diag(
                        np.array(
                            [
                                q_s_f,
                                q_n_f,
                                q_psi_f,
                                q_v_x_f,
                                q_v_y_f,
                                q_r_f,
                                q_T_f,
                                q_delta_f,
                            ]
                        )
                    ),
                )

    def compute_control(self, x: np.ndarray) -> np.ndarray | None:
        # set current state
        self.solver.set(0, "lbx", x)
        self.solver.set(0, "ubx", x)

        # update reference
        s0 = x[0]
        for j in range(self.Nf):
            self.solver.set(
                j,
                "yref",
                np.array(
                    [s0 + self.s_target * j / self.Nf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ),
            )
        self.solver.set(
            self.Nf, "yref", np.array([s0 + self.s_target, 0, 0, 0, 0, 0, 0, 0])
        )

        # set initial guess
        for j in range(self.Nf - 1):
            self.solver.set(j, "x", self.x_pred[j + 1])
            self.solver.set(j, "u", self.u_pred[j + 1])
        self.solver.set(self.Nf - 1, "x", self.x_pred[self.Nf])
        self.solver.set(self.Nf, "x", self.x_pred[self.Nf])
        self.solver.set(self.Nf - 1, "u", np.zeros(2))

        # solve ocp
        status = self.solver.solve()
        if status not in {0, 2}:
            print()
            return None

        # extract prediction and append control input
        self.x_pred = [self.solver.get(i, "x") for i in range(self.Nf + 1)]
        self.u_pred = [self.solver.get(i, "u") for i in range(self.Nf)]

        return self.solver.get(0, "u")


class SimModelVariant(Enum):
    KIN6 = auto()
    DYN6 = auto()
    KIN6_DYN6 = auto()
    DYN10 = auto()


def main():
    dt = 1 / 20
    sim_model_variant = SimModelVariant.KIN6_DYN6

    # perform offline motion plan #########################################################
    track = Track("fsds_competition_1")
    motion_plan = triple_motion_plan_ref(offline_motion_plan(track))

    # generate model and mpc solver ######################################################
    start = perf_counter()
    model = get_acados_model_from_explicit_dynamics(
        name="ihm2_fkin6",
        continuous_model_fn=fkin6_model,
        x=MX.sym("x", 8),
        u=MX.sym("u", 2),
        p=MX.sym("p", 3 * 2 * NUMBER_SPLINE_INTERVALS),
    )

    # generate controller ################################################################
    stanley_controller = StanleyController(
        dt=dt,
        k_kappa=0.0,
        k_psi=1.0,
        k_n=4.0,
    )
    # ihm2_controller = IHM2Controller(
    #     s_ref=motion_plan.s_ref,
    #     kappa_ref=motion_plan.kappa_ref,
    #     Nf=40,
    #     dt=dt,
    #     s_target=40.0,
    #     n_max=np.min(track.track_widths) - car_width / 2,
    # )
    # ipopt_solver, lbx, ubx, lbg, ubg = get_ipopt_solver(
    #     fkin6_model,
    #     Nf,
    #     model_bounds,
    #     dt,
    #     motion_plan.s_ref,
    #     motion_plan.kappa_ref,
    #     np.diag(Q),
    #     np.diag(R),
    #     np.diag(Qf),
    # )
    # last_ipopt_solution = np.zeros_like(lbx)
    print(f"Generation of MPC solver took {perf_counter() - start} seconds.\n")

    # ipopt solver #######################################################################

    # generate simulation solver ##########################################################
    start = perf_counter()
    sim_opts = AcadosSimOpts()
    sim_opts.T = dt
    sim_opts.num_stages = 4
    sim_opts.num_steps = 100
    sim_opts.integrator_type = "IRK"
    sim_opts.collocation_type = "GAUSS_RADAU_IIA"
    sim_solver = generate_sim_solver(
        model, sim_opts, "generated", generate=True, build=True
    )
    nx6 = 8
    nx10 = 15
    nu6 = 2
    nu10 = 5
    model_fdyn6 = get_acados_model_from_implicit_dynamics(
        name="fdyn6",
        continuous_model_fn=fdyn6_model,
        x=MX.sym("x", nx6),
        u=MX.sym("u", nu6),
        p=MX.sym("p", 3 * 2 * NUMBER_SPLINE_INTERVALS),
    )
    sim_solver_fdyn6 = generate_sim_solver(
        model_fdyn6, sim_opts, "generated", generate=True, build=True
    )
    model_fdyn10 = get_acados_model_from_implicit_dynamics(
        name="fdyn10",
        continuous_model_fn=fdyn10_model,
        x=MX.sym("x", nx10),
        u=MX.sym("u", nu10),
        p=MX.sym("p", 3 * 2 * NUMBER_SPLINE_INTERVALS),
    )

    sim_solver_fdyn10 = generate_sim_solver(
        model_fdyn10, sim_opts, "generated", generate=True, build=True
    )
    print(f"Generation of simulation solver took {perf_counter() - start} seconds.\n")

    # set all parameters for the mpc solver ###############################################
    p = np.append(motion_plan.s_ref, motion_plan.kappa_ref)
    sim_solver.set("p", p)
    sim_solver_fdyn6.set("p", p)
    sim_solver_fdyn10.set("p", p)

    # initialize data arrays ############################################################
    x = [np.zeros(nx10 if sim_model_variant == SimModelVariant.DYN10 else nx6)]
    x[0][0] = -6.0  # initial track progress
    # x[0][1] = 0.1  # initial lateral position
    # x[0][3] = 1e-2 # initial longitudinal velocity
    u = []
    runtimes = []
    Nsim = int(25 / dt) + 1  # simulate 40 seconds

    # simulate #########################################################################
    start_sim = perf_counter()
    for i in trange(Nsim):
        # ipopt
        # ubx[:nx] = x[-1]
        # lbx[:nx] = x[-1]
        # res = ipopt_solver(x0=last_ipopt_solution, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        # runtimes.append(1000 * (perf_counter() - start))

        # extract prediction and append control input
        start = perf_counter()
        # new_u = ihm2_controller.compute_control(x[-1])
        # alternative stanley + feedforward control
        new_u = stanley_controller.compute_control(
            n=x[-1][1],
            psi=x[-1][2],
            v_x=x[-1][3],
            v_x_ref=5.0,
            kappa_ref=np.interp(x[-1][0], motion_plan.s_ref, motion_plan.kappa_ref),
        )
        if new_u is None:
            Nsim = i
            break
        u.append(new_u)
        runtimes.append(1000 * (perf_counter() - start))
        # mpc (ipopt) control
        # last_ipopt_solution = res["x"]
        # u.append(res["x"][8 * (Nf + 1) : 8 * (Nf + 1) + 2].full().ravel())

        # sim step
        try:
            match sim_model_variant:
                case SimModelVariant.KIN6:
                    new_x = sim_solver.simulate(x[-1], u[-1])
                case SimModelVariant.DYN6:
                    new_x = sim_solver_fdyn6.simulate(x[-1], u[-1])
                case SimModelVariant.KIN6_DYN6:
                    beta = np.arctan(0.5 * np.tan(x[-1][7]))
                    new_x = (
                        sim_solver.simulate(x[-1], u[-1])
                        if np.square(np.hypot(x[-1][3], x[-1][4])) * np.sin(beta) / l_R
                        <= 3.0
                        else sim_solver_fdyn6.simulate(x[-1], u[-1])
                    )
                case SimModelVariant.DYN10:
                    new_x = sim_solver_fdyn10.simulate(
                        x[-1],
                        np.array(
                            [
                                0.25 * u[-1][0],  # u_tau_FL
                                0.25 * u[-1][0],  # u_tau_FR
                                0.25 * u[-1][0],  # u_tau_RL
                                0.25 * u[-1][0],  # u_tau_RR
                                u[-1][1],  # u_delta
                            ]
                        ),
                    )
            if np.any(np.isnan(new_x)):
                raise Exception
            x.append(new_x)
        except Exception as e:
            print(
                f"simulation solver returned exception {e} in closed loop iteration {i}."
            )
            x.append(x[-1])
            Nsim = i + 1
            break

        # check if one lap is done and break and remove entries beyond
        if x[-1][0] > motion_plan.lap_length + 1.0:
            Nsim = i + 1
            break

    stop_sim = perf_counter()
    print(f"Simulation took {stop_sim - start_sim:.3f} s\n")

    x = np.array(x)
    u = np.array(u)

    def smooth_sgn(x: np.ndarray) -> np.ndarray:
        return np.tanh(1e1 * x)

    def smooth_abs(x: np.ndarray) -> np.ndarray:
        return smooth_sgn(x) * x

    def smooth_abs_nonzero(x: np.ndarray) -> np.ndarray:
        return smooth_abs(x) + 1e-3 * np.exp(-x * x)

    if sim_model_variant == SimModelVariant.DYN10:
        v_x = x[:, 3]
        v_y = x[:, 4]
        r = x[:, 5]
        delta = x[:, 14]
        omega_FL = x[:, 6]
        omega_FR = x[:, 7]
        omega_RL = x[:, 8]
        omega_RR = x[:, 9]
        v_x_FL = v_x - 0.5 * front_axle_track * r
        v_y_FL = v_y + l_F * r
        v_x_FR = v_x + 0.5 * front_axle_track * r
        v_y_FR = v_y + l_F * r
        v_lon_FL = np.cos(delta) * v_x_FL + np.sin(delta) * v_y_FL
        v_lon_FR = np.cos(delta) * v_x_FR + np.sin(delta) * v_y_FR
        v_lon_RL = v_x - 0.5 * rear_axle_track * r
        v_lon_RR = v_x + 0.5 * rear_axle_track * r
        v_lon_FL_smooth_abs = smooth_abs_nonzero(v_lon_FL)
        v_lon_FR_smooth_abs = smooth_abs_nonzero(v_lon_FR)
        v_lon_RL_smooth_abs = smooth_abs_nonzero(v_lon_RL)
        v_lon_RR_smooth_abs = smooth_abs_nonzero(v_lon_RR)
        s_FL = (omega_FL * R_w - v_lon_FL) / v_lon_FL_smooth_abs
        s_FR = (omega_FR * R_w - v_lon_FR) / v_lon_FR_smooth_abs
        s_RL = (omega_RL * R_w - v_lon_RL) / v_lon_RL_smooth_abs
        s_RR = (omega_RR * R_w - v_lon_RR) / v_lon_RR_smooth_abs

    # find where there are big jumps in s to find start and end of lap
    idx = np.argwhere(
        np.abs(np.diff(np.mod(x[:, 0], motion_plan.lap_length))) > 1.0
    ).ravel()
    if idx.size > 1:
        start_lap = idx[0]
        end_lap = idx[1]

        # Print some stats
        # runtimes = runtimes[10:]
        print(f"Lap time: {dt*(end_lap - start_lap):.3f} s")
        print(f"Average computation time: {np.mean(runtimes):.3f} ms")
        print(f"Maximum computation time: {np.max(runtimes):.3f} ms")
        print(f"Average speed: {np.mean(x[start_lap-1:end_lap, 3]):.3f} m/s")
    else:
        print("No complete lap was done.")

    #############################################################################################
    # Plot Results
    #############################################################################################
    orange = "#ff9b31"
    purple = "#7c00c6"
    yellow = "#d5c904"
    blue = "#1f77b4"
    green = "#51bf63"

    # compute trajectory points
    phi_ref = np.interp(x[:, 0], motion_plan.s_ref, motion_plan.phi_ref)
    norm_vectors = np.column_stack((-np.sin(phi_ref), np.cos(phi_ref)))
    X_cen = np.interp(x[:, 0], motion_plan.s_ref, motion_plan.X_ref)
    Y_cen = np.interp(x[:, 0], motion_plan.s_ref, motion_plan.Y_ref)
    X = X_cen + x[:, 1] * norm_vectors[:, 0]
    Y = Y_cen + x[:, 1] * norm_vectors[:, 1]

    # plot trajectory
    fig = plt.figure(figsize=(12, 8))
    axes = {}
    gridshape = (4 if sim_model_variant == SimModelVariant.DYN10 else 3, 3)
    axes["XY"] = plt.subplot2grid(gridshape, (0, 0), rowspan=2, fig=fig)
    axes["XY"].scatter(
        track.blue_cones[:, 0], track.blue_cones[:, 1], s=14, c=blue, marker="^"
    )

    axes["XY"].scatter(
        track.yellow_cones[:, 0], track.yellow_cones[:, 1], s=14, c=yellow, marker="^"
    )
    axes["XY"].scatter(
        track.big_orange_cones[:, 0],
        track.big_orange_cones[:, 1],
        s=28,
        c=orange,
        marker="^",
    )
    axes["XY"].plot(motion_plan.X_ref, motion_plan.Y_ref, "k")
    axes["XY"].plot(X, Y, color=green)
    axes["XY"].set_xlabel("X [m]")
    axes["XY"].set_ylabel("Y [m]")
    axes["XY"].set_aspect("equal")

    # create layout for the rest of the subplots
    t = np.linspace(0.0, Nsim * dt, Nsim + 1)
    layout = {
        "n": {
            "loc": (2, 0),
            "title": r"$n$ [m]",
            "data": {
                "min": -model_bounds.n_max + car_width / 2,
                "max": model_bounds.n_max - car_width / 2,
                "past_state": x[:, 1],
            },
        },
        "psi": {
            "loc": (0, 1),
            "title": r"$\psi$ [deg]",
            "data": {"min": np.nan, "max": np.nan, "past_state": np.rad2deg(x[:, 2])},
        },
        "v_x": {
            "loc": (1, 1),
            "title": r"$v_x$ [m/s]",
            "data": {
                "min": model_bounds.v_x_min,
                "max": model_bounds.v_x_max,
                "past_state": x[:, 3],
            },
        },
        "v_y": {
            "loc": (1, 2),
            "title": r"$v_y$ [m/s]",
            "data": {"min": np.nan, "max": np.nan, "past_state": x[:, 4]},
        },
        "r": {
            "loc": (0, 2),
            "title": r"$r$ [deg/s]",
            "data": {"min": np.nan, "max": np.nan, "past_state": np.rad2deg(x[:, 5])},
        },
        "T": {
            "loc": (2, 1),
            "title": r"$T$ [N]",
            "data": {
                "min": -model_bounds.T_max,
                "max": model_bounds.T_max,
            }
            | (
                {
                    "past_state": np.sum(x[:, 10:14], axis=1),
                    "past_control": np.sum(u[:, :4], axis=1),
                }
                if sim_model_variant == SimModelVariant.DYN10
                else {
                    "past_state": x[:, 6],
                    "past_control": u[:, 0],
                }
            ),
        },
        "delta": {
            "loc": (2, 2),
            "title": r"$\delta$ [rad]",
            "data": {
                "min": -model_bounds.delta_max,
                "max": model_bounds.delta_max,
                "past_state": x[:, -1],
                "past_control": u[:, -1],
            },
        },
    } | (
        {
            "omega": {
                "loc": (3, 0),
                "title": r"$\omega$ [rad/s]",
                "data": {
                    "min": np.nan,
                    "max": np.nan,
                    "FL": x[:, 6],
                    "FR": x[:, 7],
                    "RL": x[:, 8],
                    "RR": x[:, 9],
                },
            },
            "s": {
                "loc": (3, 1),
                "title": r"$s$ [1]",
                "data": {
                    "min": np.nan,
                    "max": np.nan,
                    "FL": s_FL,
                    "FR": s_FR,
                    "RL": s_RL,
                    "RR": s_RR,
                },
            },
        }
        if sim_model_variant == SimModelVariant.DYN10
        else {}
    )
    sharedx_ax = None
    for ax_name, ax_data in layout.items():
        if sharedx_ax is None:
            axes[ax_name]: Axes = plt.subplot2grid(gridshape, ax_data["loc"])
            sharedx_ax = axes[ax_name]
        else:
            axes[ax_name] = plt.subplot2grid(
                gridshape, ax_data["loc"], sharex=sharedx_ax
            )
        axes[ax_name].set_ylabel(ax_data["title"])
        axes[ax_name].set_xlabel("time [s]")
        axes[ax_name].axhline(ax_data["data"]["min"], color="r", linestyle="--")
        axes[ax_name].axhline(ax_data["data"]["max"], color="r", linestyle="--")
        if "past_state" in ax_data["data"]:
            axes[ax_name].plot(t, ax_data["data"]["past_state"], color=purple)
        if "past_control" in ax_data["data"]:
            axes[ax_name].step(
                t[:-1], ax_data["data"]["past_control"], color=orange, where="post"
            )
        if "FL" in ax_data["data"]:
            axes[ax_name].plot(t, ax_data["data"]["FL"])
            axes[ax_name].plot(t, ax_data["data"]["FR"])
            axes[ax_name].plot(t, ax_data["data"]["RL"])
            axes[ax_name].plot(t, ax_data["data"]["RR"])
            axes[ax_name].legend(["FL", "FR", "RL", "RR"])

    fig.tight_layout()

    # plot 1: all vars
    # plt.figure()
    # plt.subplot(3, 1, 1)
    # plt.plot(t, x[:, 0], label=r"$s$")
    # plt.plot(t, x[:, 1], label=r"$n$")
    # plt.plot(t, x[:, 2], label=r"$\psi$")
    # plt.plot(t, x[:, 3], label=r"$v_x$")
    # plt.plot(t, x[:, 4], label=r"$v_y$")
    # plt.plot(t, x[:, 5], label=r"$r$")
    # plt.ylabel("State variables")
    # plt.legend()

    # plt.subplot(3, 1, 2)
    # plt.step(t[:-1], u[:, 0], label=r"$u_T$", where="post")
    # plt.step(t[:-1], u[:, 1], label=r"$u_\delta$", where="post")
    # plt.plot(t, x[:, -2], label=r"$T$")
    # plt.plot(t, x[:, -1], label=r"$\delta$")
    # plt.ylabel("Control variables")
    # plt.legend()

    # plt.subplot(3, 1, 3)
    # a_lat = np.zeros(Nsim)
    # for i in range(Nsim):
    #     a_lat[i] = a_lat_fun(x[i], u[i])
    # ic(a_lat.shape, x.shape, u.shape)
    # plt.plot(t[:-1], a_lat, label=r"$a_{\text{lat}}$")
    # plt.ylabel("lateral accelerations")

    # plot 2: driven trajectory

    # plot 3: runtimes
    # plt.figure()
    # plt.boxplot(runtimes, vert=False)
    # plt.title("Computation time distribution")
    # plt.xlabel("Computation time [ms]")
    # plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
