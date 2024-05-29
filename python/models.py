# Copyright (c) 2024. Tudor Oancea
from typing import Callable

from acados_template import AcadosModel
from casadi import (
    MX,
    SX,
    Function,
    atan,
    atan2,
    cos,
    hypot,
    interpolant,
    sin,
    sqrt,
    tan,
    tanh,
    vertcat,
)
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
    axle_track,
    front_axle_track,
    g,
    k_d,
    k_s,
    l_F,
    l_R,
    m,
    rear_axle_track,
    rear_weight_distribution,
    t_delta,
    t_T,
    wheelbase,
    z_CG,
)
from icecream import ic
from utils import smooth_abs, smooth_abs_nonzero, smooth_sgn

__all__ = [
    "kin6_model",
    "dyn6_model",
    "fkin6_model",
    "get_acados_model_from_explicit_dynamics",
    "get_acados_model_from_implicit_dynamics",
]


####################################################################################################
# models
####################################################################################################

alpha = SX.sym("alpha")
lat_pacejka = Function(
    "lat_pacejka",
    [alpha],
    [Da * sin(Ca * atan(Ba * alpha - Ea * (Ba * alpha - atan(Ba * alpha))))],
)
s = SX.sym("s")
lon_pacejka = Function(
    "lon_pacejka",
    [s],
    [Ds * sin(Cs * atan(Bs * s - Es * (Bs * s - atan(Bs * s))))],
)
del alpha, s


class Model:
    nx: int
    nu: int
    np: int
    f_cont: Callable[[SX, SX, SX], SX]
    f_disc: Callable[[SX, SX, SX], SX]
    auxiliary_vars: dict[str, SX]


def get_drag_force(v_x: SX | MX) -> SX | MX:
    """the drag force (aerodynamic and rolling resistance)"""
    return -(C_r0 + C_r1 * v_x + C_r2 * v_x * v_x) * smooth_sgn(v_x)


def get_motor_force(T: SX | MX) -> SX | MX:
    return C_m0 * T


def get_pose_derivative_cartesian_6(phi: SX, v_x: SX, v_y: SX, r: SX) -> SX:
    return vertcat(
        v_x * cos(phi) - v_y * sin(phi),
        v_x * sin(phi) + v_y * cos(phi),
        r,
    )


def get_pose_derivative_cartesian_4(phi: SX, beta: SX, v: SX) -> SX:
    return vertcat(
        v * cos(phi + beta),
        v * sin(phi + beta),
        v * sin(beta) / l_R,
    )


def kin4_model(x: SX, u: SX, _: SX) -> SX:
    """
    x = (X, Y, phi, v, T, delta)
    u = (u_T, u_delta)
    """
    # extract state and control variables
    phi = x[2]
    v = x[3]
    T = x[4]
    delta = x[5]
    u_T = u[0]
    u_delta = u[1]

    # actuator dynamics
    delta_dot = (u_delta - delta) / t_delta
    T_dot = (u_T - T) / t_T

    # lateral dynamics
    tandelta = tan(delta)
    # beta = atan(rear_weight_distribution * tandelta)
    beta = rear_weight_distribution * delta
    beta_dot = (
        rear_weight_distribution
        * (1 + tandelta * tandelta)
        / (
            1
            + rear_weight_distribution * rear_weight_distribution * tandelta * tandelta
        )
        * delta_dot
    )

    # longitudinal forces applied to the wheels
    F_motor = get_motor_force(T)
    F_drag = get_drag_force(v * cos(beta))
    F_lon_R = 0.5 * F_motor + F_drag
    F_lon_F = 0.5 * F_motor

    # accelerations
    v_dot = (F_lon_R * cos(beta) + F_lon_F * cos(delta - beta)) / m

    return vertcat(
        v * cos(phi + beta),
        v * sin(phi + beta),
        v * sin(beta) / l_R - beta_dot,
        v_dot,
        T_dot,
        delta_dot,
    )


def kin6_model(x: SX, u: SX, _: SX) -> SX:
    # extract state and control variables
    X = x[0]
    Y = x[1]
    phi = x[2]
    v_x = x[3]
    v_y = x[4]
    r = x[5]
    T = x[6]
    delta = x[7]
    u_T = u[0]
    u_delta = u[1]

    # actuator dynamics
    delta_dot = (u_delta - delta) / t_delta
    T_dot = (u_T - T) / t_T

    # longitudinal dynamics
    F_motor = C_m0 * T  # the traction force
    F_drag = -(C_r0 + C_r1 * v_x + C_r2 * v_x * v_x) * smooth_sgn(
        v_x
    )  # the drag force (aerodynamic and rolling resistance)
    F_Rx = 0.5 * F_motor + F_drag  # the force applied at the rear wheels
    F_Fx = 0.5 * F_motor  # the force applied at the front wheels

    # lateral dynamics
    tandelta = tan(delta)
    beta = atan(rear_weight_distribution * tandelta)
    sinbeta = (
        rear_weight_distribution
        * tandelta
        / sqrt(
            1
            + rear_weight_distribution * rear_weight_distribution * tandelta * tandelta
        )
    )
    cosbeta = 1 / sqrt(
        1 + rear_weight_distribution * rear_weight_distribution * tandelta * tandelta
    )
    beta_dot = (
        rear_weight_distribution
        * (1 + tandelta * tandelta)
        / (
            1
            + rear_weight_distribution * rear_weight_distribution * tandelta * tandelta
        )
        * delta_dot
    )

    # accelerations
    v_dot = (F_Rx * cosbeta + F_Fx * cos(delta - beta)) / m

    return vertcat(
        v_x * cos(phi) - v_y * sin(phi),
        v_x * sin(phi) + v_y * cos(phi),
        r,
        v_dot * cosbeta - beta_dot * v_y,
        v_dot * sinbeta + beta_dot * v_x,
        (v_dot * sinbeta + beta_dot * v_x) / l_R,
        T_dot,
        delta_dot,
    )


def fkin6_model(x: MX, u: MX, p: MX) -> MX:
    # check inputs
    assert p.shape[0] % 2 == 0

    s = x[0]
    n = x[1]
    psi = x[2]
    v_x = x[3]
    v_y = x[4]
    r = x[5]
    T = x[6]
    delta = x[7]
    u_T = u[0]
    u_delta = u[1]
    spline_dims = p.shape[0] // 2
    s_ref = p[:spline_dims]
    kappa_ref = p[spline_dims:]

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
    beta = atan(rear_weight_distribution * tandelta)
    # sinbeta = (
    #     rear_weight_distribution
    #     * tandelta
    #     / sqrt(
    #         1
    #         + rear_weight_distribution * rear_weight_distribution * tandelta * tandelta
    #     )
    # )
    # cosbeta = 1 / sqrt(
    #     1 + rear_weight_distribution * rear_weight_distribution * tandelta * tandelta
    # )
    cos_beta = cos(beta)
    sin_beta = sin(beta)
    beta_dot = (
        rear_weight_distribution
        * (1 + tandelta * tandelta)
        / (
            1
            + rear_weight_distribution * rear_weight_distribution * tandelta * tandelta
        )
        * delta_dot
    )

    # accelerations
    v_dot = (F_Rx * cos_beta + F_Fx * cos(delta - beta)) / m

    # complete dynamics
    kappa_ref_fn = interpolant(
        "kappa_ref", "linear", [spline_dims], 1, {"lookup_mode": "exact"}
    )
    s_dot = (v_x * cos(psi) - v_y * sin(psi)) / (
        1 + kappa_ref_fn(s, s_ref, kappa_ref) * n
    )
    v_y_dot = v_dot * sin_beta + beta_dot * v_x

    return vertcat(
        s_dot,  # s_dot
        v_x * sin(psi) + v_y * cos(psi),  # n_dot
        r - kappa_ref_fn(s, s_ref, kappa_ref) * s_dot,  # psi_dot
        v_dot * cos_beta - beta_dot * v_y,  # v_x_dot
        v_y_dot,  # v_y_dot
        l_R * v_y_dot - beta_dot,  # r_dot
        T_dot,  # T_dot
        delta_dot,  # delta_dot
    )


def dyn6_model(x: SX, u: SX, _: SX) -> SX:
    # extract state and control variables
    phi = x[2]
    v_x = x[3]
    v_y = x[4]
    r = x[5]
    T = x[6]
    delta = x[7]
    u_T = u[0]
    u_delta = u[1]

    # derivative of the states (used for the implicit dynamic formulation)
    X_dot = SX.sym("X_dot")
    Y_dot = SX.sym("Y_dot")
    phi_dot = SX.sym("phi_dot")
    v_x_dot = SX.sym("v_x_dot")
    v_y_dot = SX.sym("v_y_dot")
    r_dot = SX.sym("r_dot")
    T_dot = SX.sym("T_dot")
    delta_dot = SX.sym("delta_dot")

    # accelerations
    a_x = v_x_dot - v_y * r
    a_y = v_y_dot + v_x * r

    # vertical tire forces
    F_downforce = 0.5 * C_downforce * v_x * v_x  # always positive
    static_weight = 0.5 * m * g * l_F / wheelbase
    longitudinal_weight_transfer = 0.5 * m * a_x * z_CG / wheelbase
    lateral_weight_transfer = 0.5 * m * a_y * z_CG / axle_track
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

    # longitudinal and lateral velocity of each wheel (in its own reference frame)
    v_x_FL = v_x - 0.5 * axle_track * r
    v_x_FR = v_x + 0.5 * axle_track * r
    v_x_RL = v_x - 0.5 * axle_track * r
    v_x_RR = v_x + 0.5 * axle_track * r
    v_y_FL = v_y + l_F * r
    v_y_FR = v_y + l_F * r
    v_y_RL = v_y - l_R * r
    v_y_RR = v_y - l_R * r

    # lateral dynamics
    alpha_FL = atan2(v_y_FL, v_x_FL) - delta
    alpha_FR = atan2(v_y_FR, v_x_FR) - delta
    alpha_RL = atan2(v_y_RL, v_x_RL)
    alpha_RR = atan2(v_y_RR, v_x_RR)
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
    sinbeta = (
        rear_weight_distribution
        * tandelta
        / sqrt(
            1
            + rear_weight_distribution * rear_weight_distribution * tandelta * tandelta
        )
    )
    r_kin = v_x * sinbeta / l_R
    delta_tau = K_tv * (r_kin - r)  # torque vectoring gain
    tau_FL = (T - delta_tau) * F_z_FL / (-m * g - 0.25 * F_downforce)
    tau_FR = (T + delta_tau) * F_z_FR / (-m * g - 0.25 * F_downforce)
    tau_RL = (T - delta_tau) * F_z_RL / (-m * g - 0.25 * F_downforce)
    tau_RR = (T + delta_tau) * F_z_RR / (-m * g - 0.25 * F_downforce)
    F_x_FL = C_m0 * tau_FL
    F_x_FR = C_m0 * tau_FR
    F_x_RL = C_m0 * tau_RL
    F_x_RR = C_m0 * tau_RR

    # complete dynamics
    return vertcat(
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
            (F_x_FR * cos(delta) - F_y_FR * sin(delta)) * axle_track / 2
            + (F_x_FR * sin(delta) + F_y_FR * cos(delta)) * l_F
            - (F_x_FL * cos(delta) - F_y_FL * sin(delta)) * axle_track / 2
            + (F_x_FL * sin(delta) + F_y_FL * cos(delta)) * l_F
            + F_x_RR * axle_track / 2
            - F_y_RR * l_R
            - F_x_RL * axle_track / 2
            - F_y_RL * l_R
        ),
        T_dot - (u_T - T) / t_T,
        delta_dot - (u_delta - delta) / t_delta,
    )


def fdyn6_model(xdot: MX, x: MX, u: MX, p: MX) -> MX:
    # check inputs
    assert p.shape[0] % 2 == 0

    # extract state and control variables
    s = x[0]
    n = x[1]
    psi = x[2]
    v_x = x[3]
    v_y = x[4]
    r = x[5]
    T = x[6]
    delta = x[7]

    u_T = u[0]
    u_delta = u[1]

    # derivative of the states (used for the implicit dynamic formulation)
    s_dot = xdot[0]
    n_dot = xdot[1]
    psi_dot = xdot[2]
    v_x_dot = xdot[3]
    v_y_dot = xdot[4]
    r_dot = xdot[5]
    T_dot = xdot[6]
    delta_dot = xdot[7]

    # accelerations
    a_x = v_x_dot - v_y * r
    a_y = v_y_dot + v_x * r

    # vertical tire forces
    F_downforce = 0.5 * C_downforce * v_x * v_x  # always positive
    static_weight = 0.5 * m * g * l_F / wheelbase
    longitudinal_weight_transfer = 0.5 * m * a_x * z_CG / wheelbase
    lateral_weight_transfer = 0.5 * m * a_y * z_CG / axle_track

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

    # longitudinal and lateral velocity of each wheel (in its own reference frame)
    v_x_FL = v_x - 0.5 * front_axle_track * r
    v_x_FR = v_x + 0.5 * front_axle_track * r
    v_y_FL = v_y + l_F * r
    v_y_FR = v_y + l_F * r
    v_lon_FL = cos(delta) * v_x_FL + sin(delta) * v_y_FL
    v_lon_FR = cos(delta) * v_x_FR + sin(delta) * v_y_FR
    v_lat_FL = -sin(delta) * v_x_FL + cos(delta) * v_y_FL
    v_lat_FR = -sin(delta) * v_x_FR + cos(delta) * v_y_FR
    v_lon_RL = v_x - 0.5 * rear_axle_track * r
    v_lon_RR = v_x + 0.5 * rear_axle_track * r
    v_lat_RL = v_y - l_R * r
    v_lat_RR = v_y - l_R * r
    # smooth abs value of the velocity
    v_lon_FL_smooth_abs = smooth_abs_nonzero(v_lon_FL)
    v_lon_FR_smooth_abs = smooth_abs_nonzero(v_lon_FR)
    v_lon_RL_smooth_abs = smooth_abs_nonzero(v_lon_RL)
    v_lon_RR_smooth_abs = smooth_abs_nonzero(v_lon_RR)

    # slip angles
    alpha_FL = atan2(v_lat_FL, v_lon_FL_smooth_abs)
    alpha_FR = atan2(v_lat_FR, v_lon_FR_smooth_abs)
    alpha_RL = atan2(v_lat_RL, v_lon_RL_smooth_abs)
    alpha_RR = atan2(v_lat_RR, v_lon_RR_smooth_abs)

    # Pacejka formula for lateral forces
    F_lat_FL = F_z_FL * lat_pacejka(alpha_RR)
    F_lat_FR = F_z_FR * lat_pacejka(alpha_RL)
    F_lat_RL = F_z_RL * lat_pacejka(alpha_FR)
    F_lat_RR = F_z_RR * lat_pacejka(alpha_FL)

    # longitudinal dynamics
    F_drag = -(C_r0 + C_r1 * v_x + C_r2 * v_x * v_x) * smooth_sgn(v_x)
    beta = atan(rear_weight_distribution * tan(delta))
    r_kin = hypot(v_x, v_y) * sin(beta) / l_R
    delta_tau = K_tv * (r_kin - r)  # torque vectoring gain
    tau_FL = (T - delta_tau) * F_z_FL / (-m * g - 0.25 * F_downforce)
    tau_FR = (T + delta_tau) * F_z_FR / (-m * g - 0.25 * F_downforce)
    tau_RL = (T - delta_tau) * F_z_RL / (-m * g - 0.25 * F_downforce)
    tau_RR = (T + delta_tau) * F_z_RR / (-m * g - 0.25 * F_downforce)
    F_lon_FL = C_m0 * tau_FL
    F_lon_FR = C_m0 * tau_FR
    F_lon_RL = C_m0 * tau_RL
    F_lon_RR = C_m0 * tau_RR

    # frenet frame dynamics
    spline_dims = p.shape[0] // 2
    s_ref = p[:spline_dims]
    kappa_ref = p[spline_dims:]
    kappa_ref_fn = interpolant(
        "kappa_ref", "linear", [spline_dims], 1, {"lookup_mode": "exact"}
    )
    s_dot_expr = (v_x * cos(psi) - v_y * sin(psi)) / (
        1 + kappa_ref_fn(s, s_ref, kappa_ref) * n
    )

    # complete dynamics
    return vertcat(
        s_dot - s_dot_expr,  # s_dot
        n_dot - (v_x * sin(psi) + v_y * cos(psi)),  # n_dot
        psi_dot - (r - kappa_ref_fn(s, s_ref, kappa_ref) * s_dot_expr),  # psi_dot
        m * a_x
        - (
            (F_lon_FR + F_lon_FL) * cos(delta)
            - (F_lat_FR + F_lat_FL) * sin(delta)
            + F_lon_RR
            + F_lon_RL
            + F_drag
        ),
        m * a_y
        - (
            (F_lon_FR + F_lon_FL) * sin(delta)
            + (F_lat_FR + F_lat_FL) * cos(delta)
            + F_lat_RR
            + F_lat_RL
        ),
        I_z * r_dot
        - (
            (F_lon_FR * cos(delta) - F_lat_FR * sin(delta)) * axle_track / 2
            + (F_lon_FR * sin(delta) + F_lat_FR * cos(delta)) * l_F
            - (F_lon_FL * cos(delta) - F_lat_FL * sin(delta)) * axle_track / 2
            + (F_lon_FL * sin(delta) + F_lat_FL * cos(delta)) * l_F
            + F_lon_RR * axle_track / 2
            - F_lat_RR * l_R
            - F_lon_RL * axle_track / 2
            - F_lat_RL * l_R
        ),
        T_dot - (u_T - T) / t_T,
        delta_dot - (u_delta - delta) / t_delta,
    )


def fdyn10_model(xdot: MX, x: MX, u: MX, p: MX) -> AcadosModel:
    # check inputs
    assert p.shape[0] % 2 == 0

    # extract state and control variables
    s = x[0]
    n = x[1]
    psi = x[2]
    v_x = x[3]
    v_y = x[4]
    r = x[5]
    omega_FL = x[6]
    omega_FR = x[7]
    omega_RL = x[8]
    omega_RR = x[9]
    tau_FL = x[10]
    tau_FR = x[11]
    tau_RL = x[12]
    tau_RR = x[13]
    delta = x[14]

    u_tau_FL = u[0]
    u_tau_FR = u[1]
    u_tau_RL = u[2]
    u_tau_RR = u[3]
    u_delta = u[4]

    # derivative of the states (used for the implicit dynamic formulation)
    s_dot = xdot[0]
    n_dot = xdot[1]
    psi_dot = xdot[2]
    v_x_dot = xdot[3]
    v_y_dot = xdot[4]
    r_dot = xdot[5]
    omega_FL_dot = xdot[6]
    omega_FR_dot = xdot[7]
    omega_RL_dot = xdot[8]
    omega_RR_dot = xdot[9]
    tau_FL_dot = xdot[10]
    tau_FR_dot = xdot[11]
    tau_RL_dot = xdot[12]
    tau_RR_dot = xdot[13]
    delta_dot = xdot[14]

    # longitudinal and lateral dynamics
    a_x = v_x_dot - v_y * r
    a_y = v_y_dot + v_x * r

    # aerodynamic drag and rolling resistance
    F_drag = -(C_r0 + C_r1 * v_x + C_r2 * v_x * v_x) * tanh(1000 * v_x)

    # vertical tire forces
    F_downforce = 0.5 * C_downforce * v_x * v_x
    static_weight = 0.5 * m * g * l_F / wheelbase
    longitudinal_weight_transfer = 0.5 * m * a_x * z_CG / wheelbase
    lateral_weight_transfer = 0.5 * m * a_y * z_CG / front_axle_track

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

    # longitudinal and lateral tire velocities
    v_x_FL = v_x - 0.5 * front_axle_track * r
    v_y_FL = v_y + l_F * r
    v_x_FR = v_x + 0.5 * front_axle_track * r
    v_y_FR = v_y + l_F * r
    v_lon_FL = cos(delta) * v_x_FL + sin(delta) * v_y_FL
    v_lon_FR = cos(delta) * v_x_FR + sin(delta) * v_y_FR
    v_lat_FL = -sin(delta) * v_x_FL + cos(delta) * v_y_FL
    v_lat_FR = -sin(delta) * v_x_FR + cos(delta) * v_y_FR
    v_lon_RL = v_x - 0.5 * rear_axle_track * r
    v_lon_RR = v_x + 0.5 * rear_axle_track * r
    v_lat_RL = v_y - l_R * r
    v_lat_RR = v_y - l_R * r
    # smooth abs value of the velocity
    v_lon_FL_smooth_abs = smooth_abs_nonzero(v_lon_FL)
    v_lon_FR_smooth_abs = smooth_abs_nonzero(v_lon_FR)
    v_lon_RL_smooth_abs = smooth_abs_nonzero(v_lon_RL)
    v_lon_RR_smooth_abs = smooth_abs_nonzero(v_lon_RR)

    # slip angles
    alpha_FL = atan2(v_lat_FL, v_lon_FL_smooth_abs)
    alpha_FR = atan2(v_lat_FR, v_lon_FR_smooth_abs)
    alpha_RL = atan2(v_lat_RL, v_lon_RL_smooth_abs)
    alpha_RR = atan2(v_lat_RR, v_lon_RR_smooth_abs)

    # Pacejka formula for lateral forces
    F_lat_star_FL = F_z_FL * lat_pacejka(alpha_FL)
    F_lat_star_FR = F_z_FR * lat_pacejka(alpha_FR)
    F_lat_star_RL = F_z_RL * lat_pacejka(alpha_RL)
    F_lat_star_RR = F_z_RR * lat_pacejka(alpha_RR)

    # slip ratios
    s_FL = (omega_FL * R_w / v_lon_FL_smooth_abs) - 1.0
    s_FR = (omega_FR * R_w / v_lon_FR_smooth_abs) - 1.0
    s_RL = (omega_RL * R_w / v_lon_RL_smooth_abs) - 1.0
    s_RR = (omega_RR * R_w / v_lon_RR_smooth_abs) - 1.0

    # Pacejka formula for longitudinal forces
    F_lon_star_FL = -F_z_FL * lon_pacejka(s_FL)
    F_lon_star_FR = -F_z_FR * lon_pacejka(s_FR)
    F_lon_star_RL = -F_z_RL * lon_pacejka(s_RL)
    F_lon_star_RR = -F_z_RR * lon_pacejka(s_RR)

    # tire forces correlation
    F_lon_FL = F_lon_star_FL
    F_lon_FR = F_lon_star_FR
    F_lon_RL = F_lon_star_RL
    F_lon_RR = F_lon_star_RR
    F_lat_FL = F_lat_star_FL
    F_lat_FR = F_lat_star_FR
    F_lat_RL = F_lat_star_RL
    F_lat_RR = F_lat_star_RR

    # frenet frame dynamics
    spline_dims = p.shape[0] // 2
    s_ref = p[:spline_dims]
    kappa_ref = p[spline_dims:]
    kappa_ref_fn = interpolant(
        "kappa_ref", "linear", [spline_dims], 1, {"lookup_mode": "exact"}
    )
    s_dot_expr = (v_x * cos(psi) - v_y * sin(psi)) / (
        1 + kappa_ref_fn(s, s_ref, kappa_ref) * n
    )

    # complete dynamics
    return vertcat(
        # pose dynamics
        s_dot - s_dot_expr,
        n_dot - (v_x * sin(psi) + v_y * cos(psi)),
        psi_dot - (r - kappa_ref_fn(s, s_ref, kappa_ref) * s_dot_expr),
        # velocity dynamics
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
        # yaw rate dynamics
        I_z * r_dot
        - (
            (F_lon_FR * cos(delta) - F_lat_FR * sin(delta)) * front_axle_track / 2
            + (F_lon_FR * sin(delta) + F_lat_FR * cos(delta)) * l_F
            - (F_lon_FL * cos(delta) - F_lat_FL * sin(delta)) * front_axle_track / 2
            + (F_lon_FL * sin(delta) + F_lat_FL * cos(delta)) * l_F
            + F_lon_RR * rear_axle_track / 2
            - F_lat_RR * l_R
            - F_lon_RL * rear_axle_track / 2
            - F_lat_RL * l_R
        ),
        # wheel speed dynamics
        I_w * omega_FL_dot - (tau_FL - (k_d * omega_FL + k_s + R_w * F_lon_FL)),
        I_w * omega_FR_dot - (tau_FR - (k_d * omega_FR + k_s + R_w * F_lon_FR)),
        I_w * omega_RL_dot - (tau_RL - (k_d * omega_RL + k_s + R_w * F_lon_RL)),
        I_w * omega_RR_dot - (tau_RR - (k_d * omega_RR + k_s + R_w * F_lon_RR)),
        # torque dynamics
        tau_FL_dot - (u_tau_FL - tau_FL) / t_T,
        tau_FR_dot - (u_tau_FR - tau_FR) / t_T,
        tau_RL_dot - (u_tau_RL - tau_RL) / t_T,
        tau_RR_dot - (u_tau_RR - tau_RR) / t_T,
        # sterring dynamics
        delta_dot - (u_delta - delta) / t_delta,
    )


####################################################################################################
# acados model conversion
####################################################################################################


def get_acados_model_from_explicit_dynamics(
    name: str,
    continuous_model_fn: Callable[[SX | MX, SX | MX, SX | MX], SX | MX],
    x: SX | MX,
    u: SX | MX,
    p: SX | MX,
) -> AcadosModel:
    assert type(x) == type(u) == type(p)
    model = AcadosModel()
    model.name = name
    model.x = x
    model.u = u
    model.p = p
    model.xdot = type(x).sym("xdot", *x.shape)
    model.f_expl_expr = continuous_model_fn(x, u, p)
    model.f_impl_expr = model.xdot - model.f_expl_expr
    return model


def get_acados_model_from_implicit_dynamics(
    name: str,
    continuous_model_fn: Callable[[SX | MX, SX | MX, SX | MX], SX | MX],
    x: SX | MX,
    u: SX | MX,
    p: SX | MX,
) -> AcadosModel:
    assert type(x) == type(u) == type(p)
    model = AcadosModel()
    model.name = name
    model.x = x
    model.u = u
    model.p = p
    model.xdot = type(x).sym("xdot", *x.shape)
    model.f_impl_expr = continuous_model_fn(model.xdot, x, u, p)
    return model
