from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from acados_template import AcadosOcpOptions, AcadosSimOpts
from casadi import MX
from constants import (
    Ba,
    Bs,
    C_downforce,
    C_m0,
    C_r0,
    C_r1,
    C_r2,
    Ca,
    Cs,
    Da,
    Ds,
    Ea,
    Es,
    I_w,
    I_z,
    K_tv,
    R_w,
    W,
    axle_track,
    front_axle_track,
    g,
    k_d,
    k_s,
    l_F,
    l_R,
    m,
    rear_axle_track,
    t_delta,
    t_T,
    wheelbase,
    z_CG,
)
from icecream import ic
from matplotlib.axes import Axes
from models import (
    fdyn6_model,
    fdyn10_model,
    fkin6_model,
    get_acados_model_from_explicit_dynamics,
    get_acados_model_from_implicit_dynamics,
    linearized_fkin6_model,
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

last_epsilons = deque(maxlen=400)


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


def stanley_control(
    n: float, psi: float, v_x: float, v_x_ref: float, dt: float
) -> np.ndarray:
    k_P = 90.0
    k_I = 20.0

    k_n = 1.5
    k_psi = 1.7
    k_offset = 3.0
    T = k_P * (v_x_ref - v_x)
    if len(last_epsilons) > 1:
        T += k_I * (np.sum(last_epsilons) - last_epsilons[0] - last_epsilons[-1]) * dt
    delta = -k_psi * psi - np.arctan(k_n * n / (k_offset + v_x))
    last_epsilons.append(v_x_ref - v_x)
    u_max = np.array([model_bounds.T_max, model_bounds.delta_max])
    return np.clip(np.array([T, delta]), -u_max, u_max)


def stanley_feedforward_control(
    n: float,
    psi: float,
    v_x: float,
    v_x_ref: float,
    kappa_ref: float,
    dt: float,
) -> np.ndarray:
    k_P = 90.0
    k_I = 20.0

    k_offset = 1.0
    k_n = 5.5
    k_psi = 1.8
    k_kappa = 1.0
    u_T = k_P * (v_x_ref - v_x)
    if len(last_epsilons) > 1:
        u_T += k_I * (np.sum(last_epsilons) - last_epsilons[0] - last_epsilons[-1]) * dt
    u_delta = (
        k_kappa * np.arctan(2 * np.tan(np.arcsin(kappa_ref * l_R)))
        + -k_psi * psi
        - np.arctan(k_n * n / (k_offset + v_x))
    )
    last_epsilons.append(v_x_ref - v_x)
    u_max = np.array([model_bounds.T_max, model_bounds.delta_max])
    return np.clip(np.array([u_T, u_delta]), -u_max, u_max)


def main():
    nx = 8
    nu = 2
    Nf = 40
    dt = 1 / 20
    Q = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0])
    Qf = np.array([1000.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0, 100.0])
    R = np.array([1.0, 100.0])
    Rdu = np.array([0.0, 500.0])
    s_ref_Nf = 40.0
    use_dyn10 = False

    # perform offline motion plan #########################################################
    track = Track("fsds_competition_1")
    motion_plan = triple_motion_plan_ref(offline_motion_plan(track))

    # generate model and mpc solver ######################################################
    start = perf_counter()
    model = get_acados_model_from_explicit_dynamics(
        name="ihm2_fkin6",
        # continuous_model_fn=linearized_fkin6_model,
        continuous_model_fn=fkin6_model,
        x=MX.sym("x", 8),
        u=MX.sym("u", 2),
        p=MX.sym("p", 3 * 2 * NUMBER_SPLINE_INTERVALS),
    )

    model_bounds.n_max = np.min(track.track_widths) - W / 2
    ocp = get_acados_ocp(model, Nf, model_bounds)
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
    mpc_solver = get_acados_solver(ocp, ocp_opts, "generated")
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
    sim_opts.num_steps = 50
    sim_opts.integrator_type = "IRK"
    sim_opts.collocation_type = "GAUSS_RADAU_IIA"
    sim_solver = generate_sim_solver(
        model, sim_opts, "generated", generate=True, build=True
    )
    model_fdyn6 = get_acados_model_from_implicit_dynamics(
        name="fdyn6",
        continuous_model_fn=fdyn6_model,
        x=MX.sym("x", 8),
        u=MX.sym("u", 2),
        p=MX.sym("p", 3 * 2 * NUMBER_SPLINE_INTERVALS),
    )
    sim_solver_fdyn6 = generate_sim_solver(
        model_fdyn6, sim_opts, "generated", generate=True, build=True
    )
    nx_dyn10 = 15
    nu_dyn10 = 5
    model_fdyn10 = get_acados_model_from_implicit_dynamics(
        name="fdyn10",
        continuous_model_fn=fdyn10_model,
        x=MX.sym("x", nx_dyn10),
        u=MX.sym("u", nu_dyn10),
        p=MX.sym("p", 3 * 2 * NUMBER_SPLINE_INTERVALS),
    )

    sim_solver_fdyn10 = generate_sim_solver(
        model_fdyn10, sim_opts, "generated", generate=True, build=True
    )
    print(f"Generation of simulation solver took {perf_counter() - start} seconds.\n")

    # set all parameters for the mpc solver ###############################################

    p = np.append(motion_plan.s_ref, motion_plan.kappa_ref)
    for i in range(Nf + 1):
        # set parameters
        mpc_solver.set(i, "p", p)
        # set cost weights
        if i < Nf:
            # mpc_solver.cost_set(i, "W", np.diag(np.concatenate((Q, R))))
            mpc_solver.cost_set(i, "W", np.diag(np.concatenate((Q, R, Rdu))))
        else:
            mpc_solver.cost_set(i, "W", np.diag(Qf))

    sim_solver.set("p", p)
    sim_solver_fdyn6.set("p", p)
    sim_solver_fdyn10.set("p", p)

    # warm start the solver for the first iteration #######################################
    # mpc_solver.set(0, "x", )

    # initialize data arrays ############################################################
    x = [np.zeros(nx_dyn10 if use_dyn10 else nx)]
    x[0][0] = -6.0  # initial track progress
    u = []
    # x_pred = [
    #     np.array([-6.0 + i * dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    #     for i in range(Nf + 1)
    # ]
    # u_pred = [np.array([model_bounds.T_max, 0.0]) for i in range(Nf)]
    runtimes = []
    Nsim = int(20 / dt) + 1  # simulate 40 seconds

    # simulate #########################################################################
    start_sim = perf_counter()
    for i in trange(Nsim):
        # set current state
        # mpc_solver.set(0, "lbx", x[-1])
        # mpc_solver.set(0, "ubx", x[-1])

        # update reference
        # s0 = x[-1][0]
        # for j in range(Nf):
        #     mpc_solver.set(
        #         j,
        #         "yref",
        #         # np.array([s0 + s_ref_Nf * j / Nf, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        #         np.array([s0 + s_ref_Nf * j / Nf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        #     )
        # mpc_solver.set(Nf, "yref", np.array([s0 + s_ref_Nf, 0, 0, 0, 0, 0, 0, 0]))

        # set initial guess
        # for j in range(Nf - 1):
        #     mpc_solver.set(j, "x", x_pred[j + 1])
        #     mpc_solver.set(j, "u", u_pred[j + 1])
        # mpc_solver.set(Nf - 1, "x", x_pred[Nf])
        # mpc_solver.set(Nf, "x", x_pred[Nf])
        # mpc_solver.set(Nf - 1, "u", np.zeros(2))

        # solve ocp
        # start = perf_counter()
        # status = mpc_solver.solve()
        # if status not in {0, 2}:
        #     print(f"mpc solver returned status {status} in closed loop iteration {i}.")
        #     Nsim = i
        #     break
        # ipopt
        # ubx[:nx] = x[-1]
        # lbx[:nx] = x[-1]
        # res = ipopt_solver(x0=last_ipopt_solution, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        # runtimes.append(1000 * (perf_counter() - start))

        # extract prediction and append control input
        # x_pred = np.vstack([mpc_solver.get(i, "x") for i in range(Nf + 1)])
        # u_pred = np.vstack([mpc_solver.get(i, "u") for i in range(Nf)])
        # u.append(mpc_solver.get(0, "u"))
        # alternative stanley + feedforward control
        start = perf_counter()
        u.append(
            stanley_feedforward_control(
                n=x[-1][1],
                psi=x[-1][2],
                v_x=x[-1][3],
                v_x_ref=15.0,
                kappa_ref=np.interp(x[-1][0], motion_plan.s_ref, motion_plan.kappa_ref),
                dt=dt,
            )
        )
        # u[-1][-1] = 0.0
        runtimes.append(1000 * (perf_counter() - start))
        # mpc (ipopt) control
        # last_ipopt_solution = res["x"]
        # u.append(res["x"][8 * (Nf + 1) : 8 * (Nf + 1) + 2].full().ravel())

        # sim step
        try:
            if use_dyn10:
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
            else:
                beta = np.arctan(0.5 * np.tan(x[-1][7]))
                new_x = (
                    sim_solver.simulate(x[-1], u[-1])
                    if np.square(np.hypot(x[-1][3], x[-1][4])) * np.sin(beta) / l_R
                    <= 3.0
                    else sim_solver_fdyn6.simulate(x[-1], u[-1])
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
        # alternative from open loop prediction
        # x.append(mpc_solver.get(1, "x"))

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

    if use_dyn10:
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

    norm_vectors = np.column_stack((-np.sin(x[:, 2]), np.cos(x[:, 2])))
    X_cen = np.interp(x[:, 0], motion_plan.s_ref, motion_plan.X_ref)
    Y_cen = np.interp(x[:, 0], motion_plan.s_ref, motion_plan.Y_ref)
    X = X_cen + x[:, 1] * norm_vectors[:, 0]
    Y = Y_cen + x[:, 1] * norm_vectors[:, 1]

    t = np.linspace(0.0, Nsim * dt, Nsim + 1)

    fig = plt.figure(figsize=(12, 8))
    axes = {}
    gridshape = (4 if use_dyn10 else 3, 3)
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

    layout = {
        "n": {
            "loc": (2, 0),
            "title": r"$n$ [m]",
            "data": {
                "min": -model_bounds.n_max,
                "max": model_bounds.n_max,
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
                if use_dyn10
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
        if use_dyn10
        else {}
    )
    sharedx_ax = None
    for ax_name, ax_data in layout.items():
        ic(ax_name)
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
