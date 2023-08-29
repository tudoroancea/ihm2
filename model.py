# Copyright (c) 2023. Tudor Oancea
from casadi import *
from acados_template import *
from constants import *
import scipy.linalg
from icecream import ic

__all__ = [
    "gen_kin_model",
    "get_ocp_solver",
    "get_sim_solver",
]


def gen_kin_model(
    kappa_ref: Function, right_width: Function, left_width: Function
) -> tuple[AcadosModel, Function]:
    # set up states & controls
    s = MX.sym("s")
    n = MX.sym("n")
    alpha = MX.sym("alpha")
    v = MX.sym("v")
    T = MX.sym("T")
    delta = MX.sym("delta")
    x = vertcat(s, n, alpha, v, T, delta)
    u_T = MX.sym("u_T")
    u_delta = MX.sym("u_delta")
    u = vertcat(u_T, u_delta)
    xdot = MX.sym("xdot", x.shape)

    # dynamics
    beta = C1 * delta
    F_x = (C_m0 - C_m1 * v) * T - (C_r0 + C_r2 * v * v) * tanh(1000 * v)
    a_long = F_x / m
    a_lat = a_long * sin(beta) + v * v * sin(beta) / l_R
    sdot_expr = (v * cos(alpha + beta)) / (1 - kappa_ref(s) * n)
    T_dot = (u_T - T) / t_T
    delta_dot = (u_delta - delta) / t_delta

    # constraints
    right_track_constraint = (
        n - 0.5 * L * sin(fabs(alpha)) + 0.5 * W * cos(alpha) - right_width(s)
    )
    left_track_constraint = (
        -n + 0.5 * L * sin(fabs(alpha)) + 0.5 * W * cos(alpha) - left_width(s)
    )

    # Define model
    model = AcadosModel()
    f_expl = vertcat(
        sdot_expr,
        v * sin(alpha + beta),
        v * C2 * delta - kappa_ref(s) * sdot_expr,
        a_long * cos(beta),
        T_dot,
        delta_dot,
    )
    model.xdot = xdot
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl

    model.x = x
    model.u = u
    model.name = "ihm2_kinematic_model"
    model.con_h_expr = vertcat(
        right_track_constraint,
        left_track_constraint,
        T_dot,
        delta_dot,
        a_lat,
    )
    model.con_h_expr_e = vertcat(
        right_track_constraint,
        left_track_constraint,
    )

    return model


def get_ocp_solver(model: AcadosModel) -> AcadosOcpSolver:
    # create ocp instance
    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = Nf
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    ny_e = nx

    # set cost
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Qe

    Vx = np.zeros((ny, nx))
    Vx[:nx] = np.eye(nx)
    Vx[-nu:, -nu:] = np.eye(2)
    ocp.cost.Vx = Vx
    Vu = np.zeros((ny, nu))
    Vu[-nu:] = -np.eye(nu)
    ocp.cost.Vu = Vu

    Vx_e = np.eye(ny_e)
    ocp.cost.Vx_e = Vx_e

    ocp.cost.zl = zl
    ocp.cost.zl_e = zl_e
    ocp.cost.zu = zu
    ocp.cost.zu_e = zu_e
    ocp.cost.Zl = Zl
    ocp.cost.Zl_e = Zl_e
    ocp.cost.Zu = Zu
    ocp.cost.Zu_e = Zu_e

    # set intial references (will be overrided either way so the actual value doesn't matter)
    # but need them for the dimensions
    ocp.cost.yref = np.random.randn(ny)
    ocp.cost.yref_e = np.random.randn(ny_e)

    # set intial condition (same as for yref)
    ocp.constraints.x0 = np.random.randn(nx)

    # setting constraints
    ocp.constraints.idxbx = np.array([2, 3, 4, 5])
    ocp.constraints.lbx = np.array([alpha_min, v_min, T_min, delta_min])
    ocp.constraints.ubx = np.array([alpha_max, v_max, T_max, delta_max])
    ocp.constraints.idxsbx = np.array([2, 3])
    ocp.constraints.idxbx_e = np.array([2, 3, 4, 5])
    ocp.constraints.lbx_e = np.array([alpha_min, v_min, T_min, delta_min])
    ocp.constraints.ubx_e = np.array([alpha_max, v_max, T_max, delta_max])
    ocp.constraints.idxsbx_e = np.array([2, 3])

    ocp.constraints.lh = np.array(
        [
            -1e3,
            -1e3,
            T_dot_min,
            delta_dot_min,
            a_lat_min,
        ]
    )
    ocp.constraints.uh = np.array(
        [
            0.0,
            0.0,
            T_dot_max,
            delta_dot_max,
            a_lat_max,
        ]
    )
    ocp.constraints.idxsh = np.array(range(nsh))
    ocp.constraints.lh_e = np.array(
        [
            -1e3,
            -1e3,
        ]
    )
    ocp.constraints.uh_e = np.array(
        [
            0.0,
            0.0,
        ]
    )
    ocp.constraints.idxsh_e = np.array(range(nsh_e))

    # set QP solver and integration
    ocp.solver_options = ocp_solver_opts
    ocp.code_export_directory = "ihm2_gen_code"

    # create solver
    return AcadosOcpSolver(ocp, json_file="ihm2_ocp.json", verbose=False)


def get_sim_solver(model: AcadosSimSolver) -> AcadosSimSolver:
    sim = AcadosSim()
    sim.model = model
    sim.solver_options = sim_solver_opts
    sim.code_export_directory = "ihm2_gen_code"
    return AcadosSimSolver(sim, json_file="ihm2_sim.json", verbose=False)
