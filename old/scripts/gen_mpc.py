# Copyright (c) 2023. Tudor Oancea
from time import perf_counter
from casadi import (
    cos,
    sin,
    tan,
    atan,
    interpolant,
    MX,
    vertcat,
    exp,
    sqrt,
    tanh,
)
import os
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosOcpOptions
import scipy.linalg as sla
import numpy as np

__all__ = []

## Race car parameters
g = 9.81  # gravity
m = 230.0  # mass
I_z = 137.583  # yaw moment of inertia
a = b = 1.24  # wheelbase
l_R = 0.7853  # distance from CoG to rear axle
l_F = 0.7853  # distance from CoG to front axle
l = 1.5706  # distance between the two axles
L = 3.19
W = 1.55
z_CG = 0.295  # height of CoG
# drivetrain parameters (simplified)
C_m0 = 4.950
C_r0 = 297.030
C_r1 = 16.665
C_r2 = 0.6784
# pacejka parameters
b1s = -6.75e-6
b2s = 1.35e-1
b3s = 1.2e-3
c1s = 1.86
d1s = 1.12e-4
d2s = 1.57
e1s = -5.38e-6
e2s = 1.11e-2
e3s = -4.26
b1a = 3.79e1
b2a = 5.28e2
c1a = 1.57
d1a = -2.03e-4
d2a = 1.77
e1a = -2.24e-3
e2a = 1.81

static_weight = 0.5 * m * g * l_F / l
BCDs = (b1s * static_weight**2 + b2s * static_weight) * np.exp(-b3s * static_weight)
Cs = c1s
Ds = d1s * static_weight + d2s
Es = e1s * static_weight**2 + e2s * static_weight + e3s
Bs = BCDs / (Cs * Ds)
BCDa = b1a * np.sin(2 * np.arctan(static_weight / b2a))
Ca = c1a
Da = d1a * static_weight + d2a
Ea = e1a * static_weight + e2a
Ba = BCDa / (Ca * Da)

# wheel parameters
R_w = 0.20809  # wheel radius
I_w = 0.3  # wheel inertia
k_d = 0.17  #
k_s = 15.0  #
# time constants of actuators
t_T = 1e-3  # time constant for throttle actuator
t_delta = 0.02  # time constant for steering actuator
# aerodynamic parameters
C_downforce = 3.96864
K_tv = 300.0


# derived geometric parameters
C = l_R / l
Ctilde = 1 / l

# model bounds
n_min = -2.5
n_max = 2.5
psi_min = -np.pi / 2
psi_max = np.pi / 2
v_x_min = 0.0
v_x_max = 31.0
T_min = -500.0
T_max = 500.0
delta_min = -0.5
delta_max = 0.5
T_dot_min = -1e6
T_dot_max = 1e6
delta_dot_min = -1.0
delta_dot_max = 1.0
a_lat_min = -5.0
a_lat_max = 5.0

sym_t = MX

Q = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
R = np.diag([1.0, 1.0])
Q_e = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
zl = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
zu = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
Zl = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
Zu = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
zl_e = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
zu_e = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
Zl_e = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
Zu_e = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])

dt = 0.05  # 20 Hz
Nf = 20  # 1 seconds horizon
opts = AcadosOcpOptions()
opts.tf = Nf * dt
opts.qp_solver = "PARTIAL_CONDENSING_HPIPM"
opts.nlp_solver_type = "SQP"
opts.nlp_solver_max_iter = 10
opts.hessian_approx = "GAUSS_NEWTON"
opts.hpipm_mode = "ROBUST"
opts.integrator_type = "IRK"
opts.sim_method_num_stages = 4
opts.sim_method_num_steps = 1


def smooth_sgn(x: sym_t) -> sym_t:
    return tanh(1e6 * x)


def smooth_dev(x: sym_t) -> sym_t:
    return x + 1e-6 * exp(-x * x)


def smooth_abs(x: sym_t) -> sym_t:
    return smooth_sgn(x) * x


def generate_fkin6_model(
    s_ref: np.ndarray, right_width: float, left_width: float
) -> AcadosModel:
    s = sym_t.sym("s")
    n = sym_t.sym("n")
    psi = sym_t.sym("psi")
    v_x = sym_t.sym("v_x")
    v_y = sym_t.sym("v_y")
    r = sym_t.sym("r")
    T = sym_t.sym("T")
    delta = sym_t.sym("delta")
    x = vertcat(s, n, psi, v_x, v_y, r, T, delta)
    u_T = sym_t.sym("u_T")
    u_delta = sym_t.sym("u_delta")
    u = vertcat(u_T, u_delta)
    xdot = sym_t.sym("xdot", x.shape)
    kappa_ref_values = sym_t.sym("kappa_ref_values", *s_ref.shape)
    p = kappa_ref_values

    # actuator dynamics
    delta_dot = (u_delta - delta) / t_delta
    T_dot = (u_T - T) / t_T

    # longitudinal dynamics
    F_motor = C_m0 * T
    F_drag = -(C_r0 + C_r1 * v_x + C_r2 * v_x * v_x) * smooth_sgn(v_x)
    F_Rx = 0.5 * F_motor + F_drag
    F_Fx = 0.5 * F_motor

    # lateral dynamics
    tandelta = tan(delta)
    beta = atan(C * tandelta)
    sinbeta = C * tandelta / sqrt(1 + C * C * tandelta * tandelta)
    cosbeta = 1 / sqrt(1 + C * C * tandelta * tandelta)
    beta_dot = (
        C * (1 + tandelta * tandelta) / (1 + C * C * tandelta * tandelta) * delta_dot
    )
    # accelerations
    v_dot = (F_Rx * cosbeta + F_Fx * cos(delta - beta)) / m
    a_lat = (-F_Rx * sinbeta + F_Fx * sin(delta - beta)) / m + (
        v_x * v_x + v_y * v_y
    ) * sinbeta / l_R

    # complete dynamics
    kappa_ref = interpolant("kappa_ref", "linear", [s_ref])
    sdot_expr = (v_x * cos(psi) - v_y * sin(psi)) / (
        1 + kappa_ref(s, kappa_ref_values) * n
    )
    v_y_dot = v_dot * sinbeta + beta_dot * v_x
    f_expl = vertcat(
        sdot_expr,  # s_dot
        v_x * sin(psi) + v_y * cos(psi),  # n_dot
        r + kappa_ref(s, kappa_ref_values) * sdot_expr,  # psi_dot
        v_dot * cosbeta - beta_dot * v_y,  # v_x_dot
        v_y_dot,  # v_y_dot
        l_R * v_y_dot,  # r_dot
        T_dot,
        delta_dot,
    )

    # create acados model
    model = AcadosModel()
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl
    model.x = x
    model.u = u
    model.p = p
    model.xdot = xdot
    model.name = "ihm2_fkin6"

    # right_track_constraint = (
    #     n - 0.5 * L * sin(fabs(psi)) + 0.5 * W * cos(psi) - right_width
    # )
    # left_track_constraint = (
    #     -n + 0.5 * L * sin(fabs(psi)) + 0.5 * W * cos(psi) - left_width
    # )
    # model.con_h_expr = vertcat(
    #     a_lat, right_track_constraint, left_track_constraint
    # )
    # model.con_h_expr_e = vertcat(a_lat, right_track_constraint, left_track_constraint)
    return model


def gen_mpc(model: AcadosModel, s_ref: np.ndarray):
    gen_code_dir = "src/ihm2/generated"
    if not os.path.exists(gen_code_dir):
        os.makedirs(gen_code_dir)

    # create ocp instance
    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = Nf
    ocp.dims.nx = nx = model.x.size()[0]
    ocp.dims.nu = nu = model.u.size()[0]

    # stage costs
    ocp.dims.ny = ny = nx + nu
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.W = sla.block_diag(Q, R)

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx] = np.eye(nx)
    ocp.cost.Vx[-nu:, -nu:] = np.eye(2)
    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[-nu:] = -np.eye(nu)

    ocp.dims.ny_e = ny_e = nx
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W_e = Q_e
    ocp.cost.Vx_e = np.eye(ny_e)

    # nonlinear constraints
    # ocp.dims.nh = model.con_h_expr.size()[0]

    # set intial references (will be overrided either way so the actual value doesn't matter)
    # but need them for the dimensions
    ocp.cost.yref = np.ones(ny)
    ocp.cost.yref_e = np.ones(ny_e)

    # set intial condition (same as for yref)
    ocp.constraints.x0 = np.ones(nx)

    # set paremeter values
    ocp.parameter_values = np.ones_like(s_ref)

    # setting constraints
    ocp.constraints.idxbx = np.array([1, 2, 3, 6, 7])
    ocp.constraints.lbx = np.array([n_min, psi_min, v_x_min, T_min, delta_min])
    ocp.constraints.ubx = np.array([n_max, psi_max, v_x_max, T_max, delta_max])
    # ocp.constraints.idxsbx = np.array([1, 2, 3])
    ocp.constraints.idxsbx = np.array([1, 3])
    ocp.dims.nsbx = ocp.constraints.idxsbx.size

    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.lbu = np.array([T_min, delta_min])
    ocp.constraints.ubu = np.array([T_max, delta_max])

    ocp.constraints.idxbx_e = np.array([1, 2, 3, 4, 5])
    ocp.constraints.lbx_e = np.array([n_min, psi_min, v_x_min, T_min, delta_min])
    ocp.constraints.ubx_e = np.array([n_max, psi_max, v_x_max, T_max, delta_max])
    # ocp.constraints.idxsbx_e = np.array([1, 2, 3])
    ocp.constraints.idxsbx_e = np.array([1, 3])
    ocp.dims.nsbx_e = ocp.constraints.idxsbx_e.size

    ocp.constraints.C = np.zeros((2, nx))
    ocp.constraints.C[0, -2] = -1.0
    ocp.constraints.C[1, -1] = -1.0
    ocp.constraints.D = np.zeros((2, nu))
    ocp.constraints.D[0, 0] = 1.0
    ocp.constraints.D[1, 1] = 1.0
    ocp.constraints.lg = np.array([t_T * T_dot_min, t_delta * delta_dot_min])
    ocp.constraints.ug = np.array([t_T * T_dot_max, t_delta * delta_dot_max])

    # ocp.constraints.idxsh = np.array([0, 1, 2])
    # ocp.constraints.lh = np.array(
    #     [
    #         a_lat_min,
    #         -1e3,
    #         -1e3,
    #         T_dot_min,
    #         delta_dot_min,
    #     ]
    # )
    # ocp.constraints.uh = np.array(
    #     [
    #         a_lat_max,
    #         0.0,
    #         0.0,
    #         T_dot_max,
    #         delta_dot_max,
    #     ]
    # )
    # ocp.dims.nsh = ocp.constraints.idxsh.size
    #
    # ocp.constraints.idxsh_e = np.array([0, 1, 2])
    # ocp.constraints.lh_e = np.array(
    #     [
    #         a_lat_min,
    #         -1e3,
    #         -1e3,
    #     ]
    # )
    # ocp.constraints.uh_e = np.array(
    #     [
    #         a_lat_max,
    #         0.0,
    #         0.0,
    #     ]
    # )
    # ocp.dims.nsh_e = ocp.constraints.idxsh_e.size

    # slack variables
    ocp.cost.zl = zl[:2]
    ocp.cost.zu = zu[:2]
    ocp.cost.Zl = Zl[:2]
    ocp.cost.Zu = Zu[:2]
    ocp.cost.zl_e = zl_e[:2]
    ocp.cost.zu_e = zu_e[:2]
    ocp.cost.Zl_e = Zl_e[:2]
    ocp.cost.Zu_e = Zu_e[:2]

    # set QP solver and integration
    ocp.solver_options = opts
    ocp.code_export_directory = gen_code_dir + "/ihm2_fkin6_mpc_gen_code"
    # create solver
    return AcadosOcpSolver(
        ocp,
        json_file=gen_code_dir + "/ihm2_fkin6_mpc.json",
        verbose=False,
    )


def main():
    print("**************************************************")
    print("* Generating acados OCP solver *******************")
    print("**************************************************\n")
    start = perf_counter()
    file = "src/ihm2/generated/tracks/fsds_competition_1.csv"
    data = np.loadtxt(file, delimiter=",", skiprows=1)
    s_ref = data[:, 0]
    right_width = data[0, -2]
    left_width = data[0, -1]
    model = generate_fkin6_model(s_ref, right_width, left_width)
    gen_mpc(model, s_ref)
    print(f"Generation took {perf_counter() - start} seconds.\n")


# def gen_initial_guess():
#     file = "src/ihm2/tracks/fsds_competition_1.csv"
#     data = np.loadtxt(file, delimiter=",", skiprows=1)
#     s_ref = data[:, 0]
#     kappa_ref_values = data[:, 1]
#     right_width = data[0, -2]
#     left_width = data[0, -1]
#     model = generate_fkin6_model(s_ref, right_width, left_width)

#     # discretize the explicit dynamics
#     x = model.x
#     u = model.u
#     p = model.p
#     f_cont = Function("f_cont", [x, u, p], [model.f_expl_expr])
#     k1 = f_cont(x, u, p)
#     k2 = f_cont(x + dt / 2 * k1, u, p)
#     k3 = f_cont(x + dt * k2, u, p)
#     k4 = f_cont(x + dt * k3, u, p)
#     x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
#     f_disc = Function("f_disc", [x, u, p], [x_next])

#     # unroll the dynamics
#     u = np.array([T_max, 0.0])
#     x_pred = [np.zeros(model.x.size()[0])]
#     x_pred[0][0] = -5.0
#     for i in range(Nf):
#         pass


if __name__ == "__main__":
    main()
