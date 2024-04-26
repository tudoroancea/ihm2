# Copyright (c) 2024. Tudor Oancea
from typing import Callable

from acados_template import AcadosModel
from casadi import (
    MX,
    SX,
    atan,
    atan2,
    conditional,
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


class Model:
    nx: int
    nu: int
    np: int
    f_cont: Callable[[SX, SX, SX], SX]
    f_disc: Callable[[SX, SX, SX], SX]
    auxiliary_vars: dict[str, SX]


def kin4_model(x: SX, u: SX, _: SX) -> SX:
    """
    x = (X, Y, phi, v, T, delta)
    u = (u_T, u_delta)
    """
    tpr = l_R / wheelbase  # TODO: rename this variable

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
    beta = atan(tpr * tandelta)
    cosbeta = 1 / sqrt(1 + tpr * tpr * tandelta * tandelta)
    beta_dot = (
        tpr
        * (1 + tandelta * tandelta)
        / (1 + tpr * tpr * tandelta * tandelta)
        * delta_dot
    )
    # longitudinal dynamics
    v_x = v * cosbeta
    F_motor = C_m0 * T  # the traction force
    F_drag = -(C_r0 + C_r1 * v_x + C_r2 * v_x * v_x) * smooth_sgn(
        v_x
    )  # the drag force (aerodynamic and rolling resistance)
    F_Rx = 0.5 * F_motor + F_drag  # the force applied at the rear wheels
    F_Fx = 0.5 * F_motor  # the force applied at the front wheels

    # accelerations
    v_dot = (F_Rx * cosbeta + F_Fx * cos(delta - beta)) / m

    return vertcat(
        v * cos(phi + beta),
        v * sin(phi + beta),
        v * sin(beta) / l_R - beta_dot,
        v_dot,
        T_dot,
        delta_dot,
    )


def kin6_model(x: SX, u: SX, _: SX) -> SX:
    tpr = l_R / wheelbase  # TODO: rename this variable

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
    beta = atan(tpr * tandelta)
    sinbeta = tpr * tandelta / sqrt(1 + tpr * tpr * tandelta * tandelta)
    cosbeta = 1 / sqrt(1 + tpr * tpr * tandelta * tandelta)
    beta_dot = (
        tpr
        * (1 + tandelta * tandelta)
        / (1 + tpr * tpr * tandelta * tandelta)
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


def dyn6_model(x: SX, u: SX, _: SX) -> SX:
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
    tpr = l_R / wheelbase  # TODO: rename this variable
    sinbeta = tpr * tandelta / sqrt(1 + tpr * tpr * tandelta * tandelta)
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


def fkin6_model(x: MX, u: MX, p: MX) -> MX:
    # check inputs
    assert p.shape[0] % 2 == 0

    rear_distribution = l_R / wheelbase

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
    beta = atan(rear_distribution * tandelta)
    sinbeta = (
        rear_distribution
        * tandelta
        / sqrt(1 + rear_distribution * rear_distribution * tandelta * tandelta)
    )
    cosbeta = 1 / sqrt(1 + rear_distribution * rear_distribution * tandelta * tandelta)
    beta_dot = (
        rear_distribution
        * (1 + tandelta * tandelta)
        / (1 + rear_distribution * rear_distribution * tandelta * tandelta)
        * delta_dot
    )

    # accelerations
    v_dot = (F_Rx * cosbeta + F_Fx * cos(delta - beta)) / m

    # complete dynamics
    kappa_ref_fn = interpolant(
        "kappa_ref", "linear", [spline_dims], 1, {"lookup_mode": "exact"}
    )
    s_dot = (v_x * cos(psi) - v_y * sin(psi)) / (
        1 + kappa_ref_fn(s, s_ref, kappa_ref) * n
    )
    v_y_dot = v_dot * sinbeta + beta_dot * v_x

    return vertcat(
        s_dot,  # s_dot
        v_x * sin(psi) + v_y * cos(psi),  # n_dot
        r - kappa_ref_fn(s, s_ref, kappa_ref) * s_dot,  # psi_dot
        v_dot * cosbeta - beta_dot * v_y,  # v_x_dot
        v_y_dot,  # v_y_dot
        l_R * v_y_dot - beta_dot,  # r_dot
        T_dot,  # T_dot
        delta_dot,  # delta_dot
    )


def linearized_fkin6_model(x: MX, u: MX, p: MX) -> MX:
    # check inputs
    assert p.shape[0] % 2 == 0

    rear_distribution = l_R / wheelbase

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

    # lateral dynamics
    beta = rear_distribution * delta
    beta_dot = rear_distribution * delta_dot

    # longitudinal dynamics
    F_motor = C_m0 * T
    F_drag = -(C_r0 + C_r1 * v_x + C_r2 * v_x * v_x) * smooth_sgn(v_x)
    F_Rx = 0.5 * F_motor + F_drag
    F_Fx = 0.5 * F_motor

    # complete dynamics
    kappa_ref_fn = interpolant(
        "kappa_ref", "linear", [spline_dims], 1, {"lookup_mode": "exact"}
    )
    s_dot = (v_x * cos(psi) - v_y * sin(psi)) / (
        1 + kappa_ref_fn(s, s_ref, kappa_ref) * n
    )

    v_dot = (F_Rx * cos(beta) + F_Fx * cos(delta - beta)) / m
    v_y_dot = v_dot * sin(beta) + beta_dot * v_x
    return vertcat(
        s_dot,  # s_dot
        v_x * sin(psi) + v_y * cos(psi),  # n_dot
        r + kappa_ref_fn(s, s_ref, kappa_ref) * s_dot,  # psi_dot
        v_dot * cos(beta) - beta_dot * v_y,  # v_x_dot
        v_y_dot,  # v_y_dot
        l_R * v_y_dot - beta_dot,  # r_dot
        T_dot,  # T_dot
        delta_dot,  # delta_dot
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
    tpr = l_R / wheelbase  # TODO: rename this variable
    sinbeta = tpr * tandelta / sqrt(1 + tpr * tpr * tandelta * tandelta)
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
    ic(static_weight)
    longitudinal_weight_transfer = 0.5 * m * a_x * z_CG / wheelbase
    lateral_weight_transfer = 0.5 * m * a_y * z_CG / front_axle_track

    F_z_FL = (
        static_weight
        - longitudinal_weight_transfer
        + lateral_weight_transfer
        + 0.25 * F_downforce
    )
    F_z_FR = (
        static_weight
        - longitudinal_weight_transfer
        - lateral_weight_transfer
        + 0.25 * F_downforce
    )
    F_z_RL = (
        static_weight
        + longitudinal_weight_transfer
        + lateral_weight_transfer
        + 0.25 * F_downforce
    )
    F_z_RR = (
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
    v_lon_FL_smooth_abs = smooth_abs_nonzero(v_lon_FL)
    v_lon_FR_smooth_abs = smooth_abs_nonzero(v_lon_FR)
    v_lon_RL_smooth_abs = smooth_abs_nonzero(v_lon_RL)
    v_lon_RR_smooth_abs = smooth_abs_nonzero(v_lon_RR)

    # slip angles
    alpha_FL = atan2(v_lat_FL, v_lon_FL_smooth_abs) - delta
    alpha_FR = atan2(v_lat_FR, v_lon_FR_smooth_abs) - delta
    alpha_RL = atan2(v_lat_RL, v_lon_RL_smooth_abs)
    alpha_RR = atan2(v_lat_RR, v_lon_RR_smooth_abs)

    # Pacejka formula for lateral forces
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
    F_lat_star_FL = -F_z_FL * mu_lat_FL
    F_lat_star_FR = -F_z_FR * mu_lat_FR
    F_lat_star_RL = -F_z_RL * mu_lat_RL
    F_lat_star_RR = -F_z_RR * mu_lat_RR

    # slip ratios
    s_FL = (omega_FL * R_w - v_lon_FL) / v_lon_FL_smooth_abs
    s_FR = (omega_FR * R_w - v_lon_FR) / v_lon_FR_smooth_abs
    s_RL = (omega_RL * R_w - v_lon_RL) / v_lon_RL_smooth_abs
    s_RR = (omega_RR * R_w - v_lon_RR) / v_lon_RR_smooth_abs

    # Pacejka formula for longitudinal forces
    mu_lon_FL = Ds * sin(Cs * atan(Bs * s_FL - Es * (Bs * s_FL - atan(Bs * s_FL))))
    mu_lon_FR = Ds * sin(Cs * atan(Bs * s_FR - Es * (Bs * s_FR - atan(Bs * s_FR))))
    mu_lon_RL = Ds * sin(Cs * atan(Bs * s_RL - Es * (Bs * s_RL - atan(Bs * s_RL))))
    mu_lon_RR = Ds * sin(Cs * atan(Bs * s_RR - Es * (Bs * s_RR - atan(Bs * s_RR))))
    F_lon_star_FL = F_z_FL * mu_lon_FL
    F_lon_star_FR = F_z_FR * mu_lon_FR
    F_lon_star_RL = F_z_RL * mu_lon_RL
    F_lon_star_RR = F_z_RR * mu_lon_RR

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
        s_dot - s_dot_expr,
        n_dot - (v_x * sin(psi) + v_y * cos(psi)),
        psi_dot - (r - kappa_ref_fn(s, s_ref, kappa_ref) * s_dot_expr),
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
            (F_lon_FR * cos(delta) - F_lat_FR * sin(delta)) * front_axle_track / 2
            + (F_lon_FR * sin(delta) + F_lat_FR * cos(delta)) * l_F
            - (F_lon_FL * cos(delta) - F_lat_FL * sin(delta)) * front_axle_track / 2
            + (F_lon_FL * sin(delta) + F_lat_FL * cos(delta)) * l_F
            + F_lon_RR * rear_axle_track / 2
            - F_lat_RR * l_R
            - F_lon_RL * rear_axle_track / 2
            - F_lat_RL * l_R
        ),
        I_w * omega_FL_dot
        - (
            tau_FL - (k_d * omega_FL + k_s + R_w * F_lon_FL)
            # - 0.5 * (1 + tanh(omega_FL - 4.0)) * (k_d * omega_FL + k_s + R_w * F_lon_FL)
        ),
        I_w * omega_FR_dot
        - (
            tau_FR - (k_d * omega_FR + k_s + R_w * F_lon_FR)
            # - 0.5 * (1 + tanh(omega_FR - 4.0)) * (k_d * omega_FR + k_s + R_w * F_lon_FR)
        ),
        I_w * omega_RL_dot
        - (
            tau_RL - (k_d * omega_RL + k_s + R_w * F_lon_RL)
            # - 0.5 * (1 + tanh(omega_RL - 4.0)) * (k_d * omega_RL + k_s + R_w * F_lon_RL)
        ),
        I_w * omega_RR_dot
        - (
            tau_RR - (k_d * omega_RR + k_s + R_w * F_lon_RR)
            # - 0.5 * (1 + tanh(omega_RR - 4.0)) * (k_d * omega_RR + k_s + R_w * F_lon_RR)
        ),
        tau_FL_dot - (u_tau_FL - tau_FL) / t_T,
        tau_FR_dot - (u_tau_FR - tau_FR) / t_T,
        tau_RL_dot - (u_tau_RL - tau_RL) / t_T,
        tau_RR_dot - (u_tau_RR - tau_RR) / t_T,
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
