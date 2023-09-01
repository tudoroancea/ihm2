# Copyright (c) 2023. Tudor Oancea
from casadi import *
from acados_template import *
import scipy.linalg
from icecream import ic
from time import perf_counter

__all__ = [
    "gen_model",
    "get_ocp_solver",
    "get_sim_solver",
]


## Race car parameters
g = 9.81
m = 190.0
I_z = 110.0
l_R = 1.22
l_F = 1.22
L = 3.0
W = 1.5
C_m0 = 8000.0 / 1600.0
C_m1 = 43.0 / 1600.0
C_r0 = 180.0
C_r2 = 0.7
B = 10.0
C = 1.38
D = 1.609
t_T = 1e-3  # time constant for throttle actuator
t_delta = 0.02  # time constant for steering actuator

# model bounds
v_min = 0.0
v_max = 31.0
alpha_min = -np.pi / 2
alpha_max = np.pi / 2
n_min = -1.5
n_max = 1.5
T_min = -1600.0
T_max = 1600.0
delta_min = -0.5
delta_max = 0.5
T_dot_min = -1e6
T_dot_max = 1e6
delta_dot_min = -2.0
delta_dot_max = 2.0
a_lat_min = -5.0
a_lat_max = 5.0


def gen_model(
    id: str,
    kappa_ref: Function = None,
    right_width: Function = None,
    left_width: Function = None,
) -> tuple[AcadosModel, Function]:
    assert id in {"kin4", "fkin4", "dyn6", "fdyn6"}
    is_frenet = id[0] == "f"
    is_dynamic = id[-1] == "6"
    if is_frenet:
        assert (
            kappa_ref is not None and right_width is not None and left_width is not None
        )

    # set up states & controls
    x = []
    if is_frenet:
        s = MX.sym("s")
        n = MX.sym("n")
        psi = MX.sym("psi")
        x += [s, n, psi]
    else:
        X = MX.sym("X")
        Y = MX.sym("Y")
        phi = MX.sym("phi")
        x += [X, Y, phi]
    if is_dynamic:
        v_x = MX.sym("v_x")
        v_y = MX.sym("v_y")
        r = MX.sym("r")
        x += [v_x, v_y, r]
    else:
        v = MX.sym("v")
        x += [v]
    T = MX.sym("T")
    delta = MX.sym("delta")
    x += [T, delta]
    x = vertcat(*x)
    u_T = MX.sym("u_T")
    u_delta = MX.sym("u_delta")
    u = vertcat(u_T, u_delta)
    xdot = MX.sym("xdot", x.shape)

    # longitudinal dynamics
    if is_dynamic:
        F_motor = (C_m0 - C_m1 * v_x) * T
        F_drag = -(C_r0 + C_r2 * v_x * v_x) * tanh(1000 * v_x)
    else:
        F_motor = (C_m0 - C_m1 * v) * T
        F_drag = -(C_r0 + C_r2 * v * v) * tanh(1000 * v)
    F_Rx = 0.5 * F_motor + F_drag
    F_Fx = 0.5 * F_motor
    # lateral dynamics
    if is_dynamic:
        F_Rz = m * g * l_R / (l_R + l_F)
        F_Fz = m * g * l_F / (l_R + l_F)
        alpha_R = atan2(v_y - l_R * r, v_x)
        alpha_F = atan2(v_y + l_F * r, v_x) - delta
        mu_Ry = D * sin(C * atan(B * alpha_R))
        mu_Fy = D * sin(C * atan(B * alpha_F))
        F_Ry = F_Rz * mu_Ry
        F_Fy = F_Fz * mu_Fy
    else:
        beta = atan(l_R * tan(delta) / (l_R + l_F))
        # accelerations
        a_lat = (-F_Rx * sin(beta) + F_Fx * sin(delta - beta)) / m + v * v * sin(
            beta
        ) / l_R
        a_long = (F_Rx * cos(beta) + F_Fx * cos(delta - beta)) / m

    # Define model
    model = AcadosModel()
    T_dot = (u_T - T) / t_T
    delta_dot = (u_delta - delta) / t_delta
    f_expl = []
    if is_frenet:
        if is_dynamic:
            sdot_expr = (v_x * cos(psi) - v_y * sin(psi)) / (1 - kappa_ref(s) * n)
            f_expl += [
                sdot_expr,
                v_x * sin(psi) + v_y * cos(psi),
                r - kappa_ref(s) * sdot_expr,
            ]
        else:
            sdot_expr = (v * cos(psi + beta)) / (1 - kappa_ref(s) * n)
            f_expl += [
                sdot_expr,
                v * sin(psi + beta),
                v * sin(beta) / (l_R + l_F) - kappa_ref(s) * sdot_expr,
            ]
    else:
        if is_dynamic:
            f_expl += [
                v_x * cos(phi) - v_y * sin(phi),
                v_x * sin(phi) + v_y * cos(phi),
                r,
            ]
        else:
            f_expl += [
                v * cos(phi + beta),
                v * sin(phi + beta),
                v * sin(beta) / (l_R + l_F),
            ]
    if is_dynamic:
        f_expl += [
            (F_Rx + F_Fx * cos(delta) - F_Fy * sin(delta)) / m + v_y * r,
            (F_Ry + F_Fx * sin(delta) + F_Fy * cos(delta)) / m - v_x * r,
            (l_F * (F_Fx * sin(delta) + F_Fy * cos(delta)) - l_R * F_Ry) / I_z,
        ]
    else:
        f_expl += [
            a_long,
        ]
    f_expl += [
        T_dot,
        delta_dot,
    ]
    f_expl = vertcat(*f_expl)
    model.xdot = xdot
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl

    model.x = x
    model.u = u
    model.name = f"ihm2_{id}"

    # constraints
    if is_frenet:
        right_track_constraint = (
            n - 0.5 * L * sin(fabs(psi)) + 0.5 * W * cos(psi) - right_width(s)
        )
        left_track_constraint = (
            -n + 0.5 * L * sin(fabs(psi)) + 0.5 * W * cos(psi) - left_width(s)
        )
        model.con_h_expr = vertcat(
            *(
                [
                    right_track_constraint,
                    left_track_constraint,
                    T_dot,
                    delta_dot,
                ]
                + ([] if is_dynamic else [a_lat])
            )
        )
        model.con_h_expr_e = vertcat(
            right_track_constraint,
            left_track_constraint,
        )

    return model


def get_ocp_solver(
    model: AcadosModel,
    opts: AcadosOcpOptions,
    Nf: int,
    dt: float,
    Q: np.ndarray,
    R: np.ndarray,
    Qe: np.ndarray,
    code_export_directory="ihm2_ocp_gen_code",
    cmake=False,
) -> AcadosOcpSolver:
    if not os.path.exists(code_export_directory):
        os.makedirs(code_export_directory)
    # create COLCON_IGNORE file in the directory
    with open(os.path.join(code_export_directory, "COLCON_IGNORE"), "w") as f:
        f.write("")

    # create ocp instance
    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = Nf
    nx = model.x.size()[0]
    ic(nx)
    nu = model.u.size()[0]
    nh = model.con_h_expr.size()[0]
    nh_e = model.con_h_expr_e.size()[0]
    ny = nx + nu
    ny_e = nx
    nsh = nh
    nsh_e = nh_e
    nsbx = 2
    nsbx_e = 2
    ns = nsh + nsbx
    ns_e = nsh_e + nsbx_e

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

    zl = 100 * np.ones((ns,))
    zu = 100 * np.ones((ns,))
    Zl = 100 * np.ones((ns,))
    Zu = 100 * np.ones((ns,))
    zl_e = 100 * np.ones((ns_e,))
    zu_e = 100 * np.ones((ns_e,))
    Zl_e = 100 * np.ones((ns_e,))
    Zu_e = 100 * np.ones((ns_e,))
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
    ocp.solver_options = opts
    ocp.code_export_directory = code_export_directory

    # create solver
    return AcadosOcpSolver(
        ocp,
        json_file="ihm2_ocp.json",
        cmake_builder=ocp_get_default_cmake_builder() if cmake else None,
        verbose=False,
    )


def get_sim_solver(
    model: AcadosSimSolver,
    opts: AcadosSimOpts,
    code_export_directory="ihm2_sim_gen_code",
    cmake=False,
) -> AcadosSimSolver:
    if not os.path.exists(code_export_directory):
        os.makedirs(code_export_directory)
    # create COLCON_IGNORE file in the directory
    with open(os.path.join(code_export_directory, "COLCON_IGNORE"), "w") as f:
        f.write("")

    sim = AcadosSim()
    sim.model = model
    sim.solver_options = opts
    sim.code_export_directory = code_export_directory
    builder = sim_get_default_cmake_builder()
    return AcadosSimSolver(
        sim,
        json_file="ihm2_sim.json",
        cmake_builder=builder if cmake else None,
        verbose=False,
    )


def generate_ocp_sim_code(csv_file="data/fsds_competition_2.csv"):
    data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
    s_ref = data[:, 0]
    X_ref = data[:, 1]
    Y_ref = data[:, 2]
    phi_ref = data[:, 3]
    kappa_ref = data[:, 4]
    right_width = data[:, 5]
    left_width = data[:, 6]
    total_length = s_ref[-1] + np.hypot(X_ref[-1] - X_ref[0], Y_ref[-1] - Y_ref[0])

    # duplicate data before and after the loop to avoid discontinuities
    s_ref_extended = np.hstack(
        (
            s_ref - total_length,
            s_ref,
            s_ref + total_length,
        )
    )
    kappa_ref_extended = np.hstack((kappa_ref, kappa_ref, kappa_ref))
    right_width_extended = np.hstack((right_width, right_width, right_width))
    left_width_extended = np.hstack((left_width, left_width, left_width))

    # create B-spline interpolants for kappa, right_width, left_width
    kappa_ref_expr = interpolant(
        "kappa_ref", "bspline", [s_ref_extended], kappa_ref_extended
    )
    right_width_expr = interpolant(
        "right_width", "bspline", [s_ref_extended], right_width_extended
    )
    left_width_expr = interpolant(
        "left_width", "bspline", [s_ref_extended], left_width_extended
    )

    # generate Frenet model for ocp
    model = gen_model("fkin4", kappa_ref_expr, right_width_expr, left_width_expr)

    # generate ocp solver
    # for fkin4: x = (s, n, psi, v, T, delta), u = (u_T, u_delta)
    Nf = 50  # number of discretization steps
    dt = 0.02  # sampling time
    Q = np.diag([1e-1, 5e-7, 5e-7, 5e-7, 5e-2, 2.5e-2])
    R = np.diag([5e-1, 1e-1])
    Qe = np.diag([1e-1, 2e-10, 2e-10, 2e-10, 1e-4, 4e-5])
    ocp_solver_opts = AcadosOcpOptions()
    ocp_solver_opts.tf = Nf * dt
    ocp_solver_opts.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp_solver_opts.nlp_solver_type = "SQP_RTI"
    ocp_solver_opts.hessian_approx = "GAUSS_NEWTON"
    ocp_solver_opts.hpipm_mode = "SPEED_ABS"
    ocp_solver_opts.integrator_type = "ERK"
    ocp_solver_opts.sim_method_num_stages = 4
    ocp_solver_opts.sim_method_num_steps = 1
    t0 = perf_counter()
    ocp_solver = get_ocp_solver(
        model,
        ocp_solver_opts,
        Nf,
        dt,
        Q,
        R,
        Qe,
        "src/ihm2/ihm2_ocp_gen_code",
        cmake=False,
    )
    t1 = perf_counter()
    print(f"OCP solver generation time: {t1 - t0:.3f} s")

    # generate regular model

    # generate sim solver
    sim_solver_opts = AcadosSimOpts()
    sim_solver_opts.T = 0.01
    sim_solver_opts.num_stages = 4
    sim_solver_opts.num_steps = 10
    sim_solver_opts.integrator_type = "IRK"
    sim_solver_opts.collocation_type = "GAUSS_RADAU_IIA"
    t0 = perf_counter()
    get_sim_solver(
        gen_model("kin4"),
        sim_solver_opts,
        "src/ihm2/ihm2_kin4_sim_gen_code",
        cmake=False,
    )
    get_sim_solver(
        gen_model("dyn6"),
        sim_solver_opts,
        "src/ihm2/ihm2_dyn6_sim_gen_code",
        cmake=False,
    )
    t1 = perf_counter()
    print(f"Sim solver generation time: {t1 - t0:.3f} s")


if __name__ == "__main__":
    generate_ocp_sim_code("data/fsds_competition_2.csv")
