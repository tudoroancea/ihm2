# Copyright (c) 2023. Tudor Oancea

# This script generates two acados sim solver for two car models:
# - Kin6: a 6 DoF kinematic model
# - Dyn6: a 6 DoF dynamic model
# both are used in the simulation node to simulate the car's behavior
# (the kinematic for low velocities and the dynamic for high velocities)

from casadi import *
from acados_template import *
from icecream import ic
from time import perf_counter

__all__ = []


## Race car parameters
g = 9.81
m = 230.0
I_z = 137.583
l_R = 0.785
l_F = 0.785
C_m0 = 4.950
C_r0 = 297.030
C_r1 = 16.665
C_r2 = 0.6784
B = 11.15
C = 1.98
D = 1.67
E = 0.97
t_T = 1e-3  # time constant for throttle actuator
# t_delta = 0.02  # time constant for steering actuator
t_delta = 1e-3

# derived parameters
C = l_R / (l_R + l_F)
Ctilde = 1 / (l_R + l_F)

# model bounds
T_min = -1107.0
T_max = 1107.0
delta_min = -0.5
delta_max = 0.5
T_dot_min = -1e6
T_dot_max = 1e6
delta_dot_min = -2.0
delta_dot_max = 2.0


def smooth_sgn(x: MX) -> MX:
    return tanh(100000 * x)


def smooth_dev(x: MX) -> MX:
    return x + 1e-6 * exp(-x * x)


def gen_kin6_model() -> AcadosModel:
    X = MX.sym("X")
    Y = MX.sym("Y")
    phi = MX.sym("phi")
    v_x = MX.sym("v_x")
    v_y = MX.sym("v_y")
    r = MX.sym("r")
    T = MX.sym("T")
    delta = MX.sym("delta")
    x = vertcat(X, Y, phi, v_x, v_y, r, T, delta)
    u_T = MX.sym("u_T")
    u_delta = MX.sym("u_delta")
    u = vertcat(u_T, u_delta)
    xdot = MX.sym("xdot", x.shape)

    # actuator dynamics
    delta_dot = (u_delta - delta) / t_delta
    T_dot = (u_T - T) / t_T

    # longitudinal dynamics
    F_motor = C_m0 * T
    F_drag = -(C_r0 + C_r1 * v_x + C_r2 * v_x * v_x) * tanh(1000 * v_x)
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

    # complete dynamics
    f_expl = vertcat(
        v_x * cos(phi) - v_y * sin(phi),
        v_x * sin(phi) + v_y * cos(phi),
        r,
        v_dot * cosbeta - beta_dot * v_y,
        v_dot * sinbeta + beta_dot * v_x,
        (v_dot * sinbeta + beta_dot * v_x) / l_R,
        T_dot,
        delta_dot,
    )

    # create acados model
    model = AcadosModel()
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl
    model.x = x
    model.u = u
    model.xdot = xdot
    model.name = "ihm2_kin6"

    return model


def gen_dyn6_model() -> AcadosModel:
    X = MX.sym("X")
    Y = MX.sym("Y")
    phi = MX.sym("phi")
    v_x = MX.sym("v_x")
    v_y = MX.sym("v_y")
    r = MX.sym("r")
    T = MX.sym("T")
    delta = MX.sym("delta")
    x = vertcat(X, Y, phi, v_x, v_y, r, T, delta)
    u_T = MX.sym("u_T")
    u_delta = MX.sym("u_delta")
    u = vertcat(u_T, u_delta)
    xdot = MX.sym("xdot", x.shape)

    # actuator dynamics
    delta_dot = (u_delta - delta) / t_delta
    T_dot = (u_T - T) / t_T

    # longitudinal dynamics
    F_motor = C_m0 * T
    F_drag = -(C_r0 + C_r1 * v_x + C_r2 * v_x * v_x) * tanh(1000 * v_x)
    F_Rx = 0.5 * F_motor + F_drag
    F_Fx = 0.5 * F_motor

    # lateral dynamics
    F_Rz = m * g * l_F / (l_R + l_F)  # / 10
    F_Fz = m * g * l_R / (l_R + l_F)  # / 10
    # F_Rz = .0
    # F_Fz = .0
    ic(F_Rz, F_Fz)
    alpha_R = (
        smooth_sgn(v_x)
        * smooth_sgn(v_y)
        * atan2(smooth_dev(v_y - l_R * r), smooth_dev(v_x))
    )
    alpha_F = (
        smooth_sgn(v_x)
        * smooth_sgn(v_y)
        * atan2(smooth_dev(v_y + l_F * r), smooth_dev(v_x))
        - delta
    )
    alpha_F *= -1
    alpha_R *= -1
    mu_Ry = D * sin(C * atan(B * alpha_R - E * (B * alpha_R - atan(B * alpha_R))))
    mu_Fy = D * sin(C * atan(B * alpha_F - E * (B * alpha_F - atan(B * alpha_F))))
    F_Ry = F_Rz * mu_Ry
    F_Fy = F_Fz * mu_Fy

    # complete dynamics
    f_expl = vertcat(
        v_x * cos(phi) - v_y * sin(phi),
        v_x * sin(phi) + v_y * cos(phi),
        r,
        (F_Rx + F_Fx * cos(delta) - F_Fy * sin(delta)) / m + v_y * r,
        (F_Ry + F_Fx * sin(delta) + F_Fy * cos(delta)) / m - v_x * r,
        (l_F * (F_Fx * sin(delta) + F_Fy * cos(delta)) - l_R * F_Ry) / I_z,
        T_dot,
        delta_dot,
    )

    # create acados model
    model = AcadosModel()
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl
    model.x = x
    model.u = u
    model.xdot = xdot
    model.name = "ihm2_dyn6"

    return model


def generate_sim_solver(
    model: AcadosSimSolver,
    opts: AcadosSimOpts,
    code_export_directory="",
    json_file="",
) -> AcadosSimSolver:
    sim = AcadosSim()
    sim.model = model
    sim.solver_options = opts
    sim.code_export_directory = code_export_directory
    return AcadosSimSolver(sim, json_file=json_file, verbose=False)


def main():
    gen_code_dir = "src/ihm2/generated"
    if not os.path.exists(gen_code_dir):
        os.makedirs(gen_code_dir)

    sim_solver_opts = AcadosSimOpts()
    sim_solver_opts.T = 0.01
    sim_solver_opts.num_stages = 4
    sim_solver_opts.num_steps = 10
    sim_solver_opts.integrator_type = "IRK"
    sim_solver_opts.collocation_type = "GAUSS_RADAU_IIA"

    # kin6 model
    t0 = perf_counter()
    generate_sim_solver(
        gen_kin6_model(),
        sim_solver_opts,
        gen_code_dir + "/ihm2_kin6_sim_gen_code",
        gen_code_dir + "/ihm2_kin6_sim.json",
    )
    t1 = perf_counter()
    print(f"Generating Kin6 sim solver took: {t1 - t0:.3f} s")

    # dyn6 model
    t0 = perf_counter()
    generate_sim_solver(
        gen_dyn6_model(),
        sim_solver_opts,
        gen_code_dir + "/ihm2_dyn6_sim_gen_code",
        gen_code_dir + "/ihm2_dyn6_sim.json",
    )
    t1 = perf_counter()
    print(f"Generating Dyn6 sim solver took: {t1 - t0:.3f} s")


if __name__ == "__main__":
    main()
