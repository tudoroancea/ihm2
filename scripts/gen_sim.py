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
g = 9.81  # gravity
m = 230.0  # mass
I_z = 137.583  # yaw moment of inertia
a = b = 1.24  # wheelbase
l_R = 0.785  # distance from CoG to rear axle
l_F = 0.785  # distance from CoG to front axle
l = 1.570  # distance between the two axles
z_CG = 0.295  # height of CoG
# drivetrain parameters (simplified)
C_m0 = 4.950
C_r0 = 297.030
C_r1 = 16.665
C_r2 = 0.6784
# pacejka parameters
B = 11.15
C = 1.98
D = 1.67
E = 0.97
b1s = -6.75e-6
b2s = 1.35e-1
b3s = 1.2e-3
c1s = 1.86
d1s = 1.12e-4
d2s = 1.57
e1s = --5.38e-6
e2s = 1.11e-2
e3s = -4.26
b1a = 3.79e1
b2a = 5.28e2
c1a = 1.57
d1a = -2.03e-4
d2a = 1.77
e1a = -2.24e-3
e2a = 1.81
# wheel parameters
R_w = 0.202  # wheel radius
I_w = 0.3  # wheel inertia
k_d = 0.17  #
k_s = 15.0  #
# time constants of actuators
t_T = 1e-3  # time constant for throttle actuator
# t_delta = 0.02  # time constant for steering actuator
t_delta = 1e-3
# aerodynamic parameters
C_downforce = 3.96864


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

    # longitudinal dynamics
    F_motor = C_m0 * T
    F_drag = -(C_r0 + C_r1 * v_x + C_r2 * v_x * v_x) * tanh(1000 * v_x)
    F_Rx = 0.5 * F_motor + F_drag
    F_Fx = 0.5 * F_motor

    # lateral dynamics
    F_downforce = 0.5 * C_downforce * v_x * v_x
    F_Rz = m * g * l_F / l + 0.5 * F_downforce + m * a_x * z_CG / l
    F_Fz = m * g * l_R / l + 0.5 * F_downforce - m * a_x * z_CG / l
    alpha_R = -atan2(smooth_dev(v_y - l_R * r), smooth_dev(v_x))
    alpha_F = delta - atan2(smooth_dev(v_y + l_F * r), smooth_dev(v_x))
    mu_Ry = D * sin(C * atan(B * alpha_R - E * (B * alpha_R - atan(B * alpha_R))))
    mu_Fy = D * sin(C * atan(B * alpha_F - E * (B * alpha_F - atan(B * alpha_F))))
    F_Ry = F_Rz * mu_Ry
    F_Fy = F_Fz * mu_Fy

    # complete dynamics
    f_impl = vertcat(
        X_dot - (v_x * cos(phi) - v_y * sin(phi)),
        Y_dot - (v_x * sin(phi) + v_y * cos(phi)),
        phi_dot - r,
        m * a_x - (F_Rx + F_Fx * cos(delta) - F_Fy * sin(delta)),
        m * a_y - (F_Ry + F_Fx * sin(delta) + F_Fy * cos(delta)),
        I_z * r_dot - (l_F * (F_Fx * sin(delta) + F_Fy * cos(delta)) - l_R * F_Ry),
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
    F_downforce = C_downforce * v_x * v_x
    F_z_FL = (
        0.5 * m * g * l_R / l
        + 0.5 * m * a_y * z_CG / a
        - 0.5 * m * a_x * z_CG / l
        + 0.25 * F_downforce
    )
    F_z_FR = (
        0.5 * m * g * l_R / l
        - 0.5 * m * a_y * z_CG / a
        - 0.5 * m * a_x * z_CG / l
        + 0.25 * F_downforce
    )
    F_z_RL = (
        0.5 * m * g * l_F / l
        + 0.5 * m * a_y * z_CG / a
        + 0.5 * m * a_x * z_CG / l
        + 0.25 * F_downforce
    )
    F_z_RR = (
        0.5 * m * g * l_F / l
        - 0.5 * m * a_y * z_CG / a
        + 0.5 * m * a_x * z_CG / l
        + 0.25 * F_downforce
    )
    F_z = {"FL": F_z_FL, "FR": F_z_FR, "RL": F_z_RL, "RR": F_z_RR}
    omega = {"FL": omega_FL, "FR": omega_FR, "RL": omega_RL, "RR": omega_RR}

    BCDs = {
        k: (b1s * F_z[k] * F_z[k] + b2s * F_z[k]) * exp(-b3s * F_z[k])
        for k in F_z.keys()
    }
    Cs = {k: c1s for k in F_z.keys()}
    Ds = {k: d1s * F_z[k] + d2s for k in F_z.keys()}
    Es = {k: e1s * F_z[k] * F_z[k] + e2s * F_z[k] + e3s for k in F_z.keys()}
    Bs = {k: BCDs[k] / (Cs[k] * Ds[k]) for k in F_z.keys()}
    BCDa = {k: b1a * sin(2 * atan(F_z[k] / b2a)) for k in F_z.keys()}
    Ca = {k: c1a for k in F_z.keys()}
    Da = {k: d1a * F_z[k] + d2a for k in F_z.keys()}
    Ba = {k: BCDa[k] / (Ca[k] * Da[k]) for k in F_z.keys()}
    Ea = {k: e1a * F_z[k] + e2a for k in F_z.keys()}
    s = {k: omega[k] * R_w / v_x - 1.0 for k in omega.keys()}
    F_lon_star = {
        k: Ds[k]
        * sin(Cs[k] * atan(Bs[k] * s[k] - Es[k] * (Bs[k] * s[k] - atan(Bs[k] * s[k]))))
        for k in F_z.keys()
    }
    alpha = {
        "FL": delta
        - smooth_sgn(v_x - 0.5 * a * r)
        * atan2(smooth_dev(v_y + l_F * r), smooth_dev(v_x - 0.5 * a * r)),
        "FR": delta
        - smooth_sgn(v_x + 0.5 * a * r)
        * atan2(smooth_dev(v_y + l_F * r), smooth_dev(v_x + 0.5 * a * r)),
        "RL": -smooth_sgn(v_x - 0.5 * b * r)
        * atan2(smooth_dev(v_y - l_R * r), smooth_dev(v_x - 0.5 * b * r)),
        "RR": -smooth_sgn(v_x + 0.5 * b * r)
        * atan2(smooth_dev(v_y - l_R * r), smooth_dev(v_x + 0.5 * b * r)),
    }
    F_lat_star = {
        k: Da[k]
        * sin(Ca[k] * atan(Ba[k] * alpha[k] - Ea[k] * (Ba[k] * alpha[k] - atan(Ba[k]))))
        for k in F_z.keys()
    }

    F_lon = {
        k: fabs(s[k])
        * F_lon_star[k]
        / sqrt(s[k] * s[k] + tan(alpha[k]) * tan(alpha[k]))
        for k in F_z.keys()
    }
    F_lon_FL = F_lon["FL"]
    F_lon_FR = F_lon["FR"]
    F_lon_RL = F_lon["RL"]
    F_lon_RR = F_lon["RR"]
    F_lat = {
        k: fabs(tan(alpha[k]))
        * F_lat_star[k]
        / sqrt(alpha[k] * alpha[k] + tan(s[k]) * tan(s[k]))
        for k in F_z.keys()
    }
    F_lat_FL = F_lat["FL"]
    F_lat_FR = F_lat["FR"]
    F_lat_RL = F_lat["RL"]
    F_lat_RR = F_lat["RR"]

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
    t0 = perf_counter()
    generate_sim_solver(
        gen_dyn10_model(),
        sim_solver_opts,
        gen_code_dir + "/ihm2_dyn10_sim_gen_code",
        gen_code_dir + "/ihm2_dyn10_sim.json",
    )
    t1 = perf_counter()
    print(f"Generating Dyn10 sim solver took: {t1 - t0:.3f} s")


if __name__ == "__main__":
    main()
