# Copyright (c) 2023. Tudor Oancea

# This script generates two acados sim solver for two car models:
# - Kin6: a 6 DoF kinematic model
# - Dyn6: a 6 DoF dynamic model
# both are used in the simulation node to simulate the car's behavior
# (the kinematic for low velocities and the dynamic for high velocities)

from casadi import (
    cos,
    sin,
    tan,
    atan,
    MX,
    vertcat,
    exp,
    sqrt,
    tanh,
    atan2,
    hypot,
)
import numpy as np
import os
from acados_template import AcadosModel, AcadosSim, AcadosSimSolver, AcadosSimOpts
from time import perf_counter

__all__ = []


## Race car parameters
g = 9.81  # gravity
m = 230.0  # mass
I_z = 137.583  # yaw moment of inertia
a = b = 1.24  # wheelbase
l_R = 0.7853  # distance from CoG to rear axle
l_F = 0.7853  # distance from CoG to front axle
l = 1.5706  # distance between the two axles
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


# derived parameters
C = l_R / l
Ctilde = 1 / l

# model bounds
T_min = -1107.0
T_max = 1107.0
delta_min = -0.5
delta_max = 0.5
T_dot_min = -1e6
T_dot_max = 1e6
delta_dot_min = -2.0
delta_dot_max = 2.0

sym_t = MX


def smooth_sgn(x: sym_t) -> sym_t:
    return tanh(1e6 * x)


def smooth_dev(x: sym_t) -> sym_t:
    return x + 1e-6 * exp(-x * x)


def smooth_abs(x: sym_t) -> sym_t:
    return smooth_sgn(x) * x


def gen_kin6_model() -> AcadosModel:
    X = sym_t.sym("X")
    Y = sym_t.sym("Y")
    phi = sym_t.sym("phi")
    v_x = sym_t.sym("v_x")
    v_y = sym_t.sym("v_y")
    r = sym_t.sym("r")
    T = sym_t.sym("T")
    delta = sym_t.sym("delta")
    x = vertcat(X, Y, phi, v_x, v_y, r, T, delta)
    u_T = sym_t.sym("u_T")
    u_delta = sym_t.sym("u_delta")
    u = vertcat(u_T, u_delta)
    xdot = sym_t.sym("xdot", x.shape)

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
    X = sym_t.sym("X")
    Y = sym_t.sym("Y")
    phi = sym_t.sym("phi")
    v_x = sym_t.sym("v_x")
    v_y = sym_t.sym("v_y")
    r = sym_t.sym("r")
    T = sym_t.sym("T")
    delta = sym_t.sym("delta")
    x = vertcat(X, Y, phi, v_x, v_y, r, T, delta)

    u_T = sym_t.sym("u_T")
    u_delta = sym_t.sym("u_delta")
    u = vertcat(u_T, u_delta)

    X_dot = sym_t.sym("X_dot")
    Y_dot = sym_t.sym("Y_dot")
    phi_dot = sym_t.sym("phi_dot")
    v_x_dot = sym_t.sym("v_x_dot")
    v_y_dot = sym_t.sym("v_y_dot")
    r_dot = sym_t.sym("r_dot")
    T_dot = sym_t.sym("T_dot")
    delta_dot = sym_t.sym("delta_dot")
    xdot = vertcat(X_dot, Y_dot, phi_dot, v_x_dot, v_y_dot, r_dot, T_dot, delta_dot)

    a_x = v_x_dot - v_y * r
    a_y = v_y_dot + v_x * r

    # lateral dynamics
    F_downforce = 0.5 * C_downforce * v_x * v_x
    static_weight = 0.5 * m * g * l_F / l
    longitudinal_weight_transfer = 0.5 * m * a_x * z_CG / l
    lateral_weight_transfer = 0.5 * m * a_y * z_CG / a
    F_z_FL = -(
        static_weight
        - longitudinal_weight_transfer
        + lateral_weight_transfer
        + 0.25 * F_downforce
    )
    F_z_FR = -(
        static_weight
        - longitudinal_weight_transfer
        - lateral_weight_transfer
        + 0.25 * F_downforce
    )
    F_z_RL = -(
        static_weight
        + longitudinal_weight_transfer
        + lateral_weight_transfer
        + 0.25 * F_downforce
    )
    F_z_RR = -(
        static_weight
        + longitudinal_weight_transfer
        - lateral_weight_transfer
        + 0.25 * F_downforce
    )
    v_x_FL = v_x - 0.5 * a * r
    v_x_FR = v_x + 0.5 * a * r
    v_x_RL = v_x - 0.5 * b * r
    v_x_RR = v_x + 0.5 * b * r
    v_y_FL = v_y + l_F * r
    v_y_FR = v_y + l_F * r
    v_y_RL = v_y - l_R * r
    v_y_RR = v_y - l_R * r
    alpha_FL = atan2(smooth_dev(v_y_FL), smooth_dev(v_x_FL)) - delta
    alpha_FR = atan2(smooth_dev(v_y_FR), smooth_dev(v_x_FR)) - delta
    alpha_RL = atan2(smooth_dev(v_y_RL), smooth_dev(v_x_RL))
    alpha_RR = atan2(smooth_dev(v_y_RR), smooth_dev(v_x_RR))
    mu_y_FL = Da * sin(
        Ca * atan(Ba * alpha_FL - Ea * (Ba * alpha_FL - atan(Ba * alpha_FL)))
    )
    mu_y_FR = Da * sin(
        Ca * atan(Ba * alpha_FR - Ea * (Ba * alpha_FR - atan(Ba * alpha_FR)))
    )
    mu_y_RL = Da * sin(
        Ca * atan(Ba * alpha_RL - Ea * (Ba * alpha_RL - atan(Ba * alpha_RL)))
    )
    mu_y_RR = Da * sin(
        Ca * atan(Ba * alpha_RR - Ea * (Ba * alpha_RR - atan(Ba * alpha_RR)))
    )
    F_y_FL = F_z_FL * mu_y_FL
    F_y_FR = F_z_FR * mu_y_FR
    F_y_RL = F_z_RL * mu_y_RL
    F_y_RR = F_z_RR * mu_y_RR

    # longitudinal dynamics
    F_drag = -(C_r0 + C_r1 * v_x + C_r2 * v_x * v_x) * tanh(1000 * v_x)
    tandelta = tan(delta)
    sinbeta = C * tandelta / sqrt(1 + C * C * tandelta * tandelta)
    r_kin = v_x * sinbeta / l_R
    delta_tau = K_tv * (r_kin - r)
    tau_FL = (T - delta_tau) * F_z_FL / (-m * g - 0.25 * F_downforce)
    tau_FR = (T + delta_tau) * F_z_FR / (-m * g - 0.25 * F_downforce)
    tau_RL = (T - delta_tau) * F_z_RL / (-m * g - 0.25 * F_downforce)
    tau_RR = (T + delta_tau) * F_z_RR / (-m * g - 0.25 * F_downforce)
    F_x_FL = C_m0 * tau_FL
    F_x_FR = C_m0 * tau_FR
    F_x_RL = C_m0 * tau_RL
    F_x_RR = C_m0 * tau_RR

    # complete dynamics
    f_impl = vertcat(
        X_dot - (v_x * cos(phi) - v_y * sin(phi)),
        Y_dot - (v_x * sin(phi) + v_y * cos(phi)),
        phi_dot - r,
        m * a_x
        - (
            (F_x_FR + F_x_FL) * cos(delta)
            - (F_y_FR + F_y_FL) * sin(delta)
            + F_x_RR
            + F_x_RL
            + F_drag
        ),
        m * a_y
        - (
            (F_x_FR + F_x_FL) * sin(delta)
            + (F_y_FR + F_y_FL) * cos(delta)
            + F_y_RR
            + F_y_RL
        ),
        I_z * r_dot
        - (
            (F_x_FR * cos(delta) - F_y_FR * sin(delta)) * a / 2
            + (F_x_FR * sin(delta) + F_y_FR * cos(delta)) * l_F
            - (F_x_FL * cos(delta) - F_y_FL * sin(delta)) * a / 2
            + (F_x_FL * sin(delta) + F_y_FL * cos(delta)) * l_F
            + F_x_RR * b / 2
            - F_y_RR * l_R
            - F_x_RL * b / 2
            - F_y_RL * l_R
        ),
        T_dot - (u_T - T) / t_T,
        delta_dot - (u_delta - delta) / t_delta,
    )

    # create acados model
    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.x = x
    model.u = u
    model.xdot = xdot
    model.name = "ihm2_dyn6"

    return model


def gen_dyn10_model() -> AcadosModel:
    # states
    X = sym_t.sym("X")
    Y = sym_t.sym("Y")
    phi = sym_t.sym("phi")
    v_x = sym_t.sym("v_x")
    v_y = sym_t.sym("v_y")
    r = sym_t.sym("r")
    T = sym_t.sym("T")
    delta = sym_t.sym("delta")
    omega_FL = sym_t.sym("omega_FL")
    omega_FR = sym_t.sym("omega_FR")
    omega_RL = sym_t.sym("omega_RL")
    omega_RR = sym_t.sym("omega_RR")
    x = vertcat(
        X, Y, phi, v_x, v_y, r, omega_FL, omega_FR, omega_RL, omega_RR, T, delta
    )

    # controls
    u_T = sym_t.sym("u_T")
    u_delta = sym_t.sym("u_delta")
    u = vertcat(u_T, u_delta)

    # states derivatives
    X_dot = sym_t.sym("X_dot")
    Y_dot = sym_t.sym("Y_dot")
    phi_dot = sym_t.sym("phi_dot")
    v_x_dot = sym_t.sym("v_x_dot")
    v_y_dot = sym_t.sym("v_y_dot")
    r_dot = sym_t.sym("r_dot")
    omega_FL_dot = sym_t.sym("omega_FL_dot")
    omega_FR_dot = sym_t.sym("omega_FR_dot")
    omega_RL_dot = sym_t.sym("omega_RL_dot")
    omega_RR_dot = sym_t.sym("omega_RR_dot")
    T_dot = sym_t.sym("T_dot")
    delta_dot = sym_t.sym("delta_dot")
    x_dot = vertcat(
        X_dot,
        Y_dot,
        phi_dot,
        v_x_dot,
        v_y_dot,
        r_dot,
        omega_FL_dot,
        omega_FR_dot,
        omega_RL_dot,
        omega_RR_dot,
        T_dot,
        delta_dot,
    )

    a_x = v_x_dot - v_y * r
    a_y = v_y_dot + v_x * r
    F_drag = -C_r0 * tanh(1000 * v_x) + C_r1 * v_x + C_r2 * v_x * v_x
    F_downforce = 0.5 * C_downforce * v_x * v_x
    static_weight = 0.5 * m * g * l_F / l
    longitudinal_weight_transfer = 0.5 * m * a_x * z_CG / l
    lateral_weight_transfer = 0.5 * m * a_y * z_CG / a
    F_z_FL = -(
        static_weight
        - longitudinal_weight_transfer
        + lateral_weight_transfer
        + 0.25 * F_downforce
    )
    F_z_FR = -(
        static_weight
        - longitudinal_weight_transfer
        - lateral_weight_transfer
        + 0.25 * F_downforce
    )
    F_z_RL = -(
        static_weight
        + longitudinal_weight_transfer
        + lateral_weight_transfer
        + 0.25 * F_downforce
    )
    F_z_RR = -(
        static_weight
        + longitudinal_weight_transfer
        - lateral_weight_transfer
        + 0.25 * F_downforce
    )
    v_x_FL = v_x - 0.5 * a * r
    v_x_FR = v_x + 0.5 * a * r
    v_x_RL = v_x - 0.5 * b * r
    v_x_RR = v_x + 0.5 * b * r
    v_y_FL = v_y + l_F * r
    v_y_FR = v_y + l_F * r
    v_y_RL = v_y - l_R * r
    v_y_RR = v_y - l_R * r
    alpha_FL = atan2(v_y_FL, v_x_FL) - delta
    alpha_FR = atan2(v_y_FR, v_x_FR) - delta
    alpha_RL = atan2(v_y_RL, v_x_RL)
    alpha_RR = atan2(v_y_RR, v_x_RR)
    mu_lat_FL = Da * sin(
        Ca * atan(Ba * alpha_FL - Ea * (Ba * alpha_FL - atan(Ba * alpha_FL)))
    )
    mu_lat_FR = Da * sin(
        Ca * atan(Ba * alpha_FR - Ea * (Ba * alpha_FR - atan(Ba * alpha_FR)))
    )
    mu_lat_RL = Da * sin(
        Ca * atan(Ba * alpha_RL - Ea * (Ba * alpha_RL - atan(Ba * alpha_RL)))
    )
    mu_lat_RR = Da * sin(
        Ca * atan(Ba * alpha_RR - Ea * (Ba * alpha_RR - atan(Ba * alpha_RR)))
    )
    F_lat_star_FL = F_z_FL * mu_lat_FL
    F_lat_star_FR = F_z_FR * mu_lat_FR
    F_lat_star_RL = F_z_RL * mu_lat_RL
    F_lat_star_RR = F_z_RR * mu_lat_RR

    s_FL = omega_FL * R_w / v_x_FL - 1.0
    s_FR = omega_FR * R_w / v_x_FR - 1.0
    s_RL = omega_RL * R_w / v_x_RL - 1.0
    s_RR = omega_RR * R_w / v_x_RR - 1.0

    mu_lon_FL = Ds * sin(Cs * atan(Bs * s_FL - Es * (Bs * s_FL - atan(Bs * s_FL))))
    mu_lon_FR = Ds * sin(Cs * atan(Bs * s_FR - Es * (Bs * s_FR - atan(Bs * s_FR))))
    mu_lon_RL = Ds * sin(Cs * atan(Bs * s_RL - Es * (Bs * s_RL - atan(Bs * s_RL))))
    mu_lon_RR = Ds * sin(Cs * atan(Bs * s_RR - Es * (Bs * s_RR - atan(Bs * s_RR))))
    F_lon_star_FL = F_z_FL * mu_lon_FL
    F_lon_star_FR = F_z_FR * mu_lon_FR
    F_lon_star_RL = F_z_RL * mu_lon_RL
    F_lon_star_RR = F_z_RR * mu_lon_RR

    F_lon_FL = F_lon_star_FL * smooth_abs(s_FL) / hypot(s_FL, tan(alpha_FL))
    F_lon_FR = F_lon_star_FR * smooth_abs(s_FR) / hypot(s_FR, tan(alpha_FR))
    F_lon_RL = F_lon_star_RL * smooth_abs(s_RL) / hypot(s_RL, tan(alpha_RL))
    F_lon_RR = F_lon_star_RR * smooth_abs(s_RR) / hypot(s_RR, tan(alpha_RR))
    F_lat_FL = F_lat_star_FL * smooth_abs(tan(alpha_FL)) / hypot(s_FL, tan(alpha_FL))
    F_lat_FR = F_lat_star_FR * smooth_abs(tan(alpha_FR)) / hypot(s_FR, tan(alpha_FR))
    F_lat_RL = F_lat_star_RL * smooth_abs(tan(alpha_RL)) / hypot(s_RL, tan(alpha_RL))
    F_lat_RR = F_lat_star_RR * smooth_abs(tan(alpha_RR)) / hypot(s_RR, tan(alpha_RR))

    f_impl_expr = vertcat(
        X_dot - v_x * cos(phi) + v_y * sin(phi),
        Y_dot - v_x * sin(phi) - v_y * cos(phi),
        phi_dot - r,
        m * a_x
        - (
            F_drag
            + cos(delta) * (F_lon_FL + F_lon_FR)
            - sin(delta) * (F_lat_FL + F_lat_FR)
            + F_lon_RL
            + F_lon_RR
        ),
        m * a_y
        - (
            sin(delta) * (F_lon_FL + F_lon_FR)
            + cos(delta) * (F_lat_FL + F_lat_FR)
            + F_lat_RL
            + F_lat_RR
        ),
        I_z * r_dot
        - (
            (F_lon_FR * cos(delta) - F_lat_FR * sin(delta)) * a / 2
            + (F_lon_FR * sin(delta) + F_lat_FR * cos(delta)) * l_F
            - (F_lon_FL * cos(delta) - F_lat_FL * sin(delta)) * a / 2
            + (F_lon_FL * sin(delta) + F_lat_FL * cos(delta)) * l_F
            + F_lon_RR * b / 2
            - F_lat_RR * l_R
            - F_lon_RL * b / 2
            - F_lat_RL * l_R
        ),
        I_w * omega_FL_dot - (0.25 * T - k_d * omega_FL - k_s - R_w * F_lon_FL),
        I_w * omega_FR_dot - (0.25 * T - k_d * omega_FR - k_s - R_w * F_lon_FR),
        I_w * omega_RL_dot - (0.25 * T - k_d * omega_RL - k_s - R_w * F_lon_RL),
        I_w * omega_RR_dot - (0.25 * T - k_d * omega_RR - k_s - R_w * F_lon_RR),
        T_dot - (u_T - T) / t_T,
        delta_dot - (u_delta - delta) / t_delta,
    )

    model = AcadosModel()
    model.x = x
    model.xdot = x_dot
    model.u = u
    model.name = "ihm2_dyn10"
    model.f_impl_expr = f_impl_expr

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
    print("**************************************************")
    print("* Generating track files *************************")
    print("**************************************************\n")

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

    # dyn10 model
    # t0 = perf_counter()
    # generate_sim_solver(
    #     gen_dyn10_model(),
    #     sim_solver_opts,
    #     gen_code_dir + "/ihm2_dyn10_sim_gen_code",
    #     gen_code_dir + "/ihm2_dyn10_sim.json",
    # )
    # t1 = perf_counter()
    # print(f"Generating Dyn10 sim solver took: {t1 - t0:.3f} s")

    print("")


if __name__ == "__main__":
    main()
