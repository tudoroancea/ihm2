# Copyright (c) 2024. Tudor Oancea
import os
from typing import Callable

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpOptions, AcadosOcpSolver
from casadi import MX, SX, Function, nlpsol, reshape, sum1, sum2, vertcat
from icecream import ic

# from motion_planning import NUMBER_SPLINE_INTERVALS
from pydantic import BaseModel

__all__ = ["ModelBounds", "get_acados_ocp", "get_acados_solver", "get_ipopt_solver"]

from constants import t_delta, t_T


class ModelBounds(BaseModel):
    n_max: float
    v_x_min: float
    v_x_max: float
    T_max: float
    delta_max: float
    T_dot_max: float
    delta_dot_max: float
    a_lat_max: float


def get_acados_ocp(model: AcadosModel, Nf: int, model_bounds: ModelBounds) -> AcadosOcp:
    # create ocp instance ###############################################################
    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = Nf
    ocp.dims.nx = nx = model.x.shape[0]
    ocp.dims.nu = nu = model.u.shape[0]
    ocp.dims.np = model.p.shape[0]

    # costs #######################################################################
    # stage costs
    ocp.dims.ny = ny = nx + nu + nu
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.W = np.eye(ny)
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx] = np.eye(nx)
    ocp.cost.Vx[-nu:, -nu:] = np.eye(nu)

    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[-2 * nu : -nu] = np.eye(nu)
    ocp.cost.Vu[-nu:] = -np.eye(nu)

    # terminal costs
    ocp.dims.ny_e = ny_e = nx
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W_e = np.eye(ny_e)
    ocp.cost.Vx_e = np.eye(ny_e)

    # set intial references (will be overrided either way so the actual value doesn't matter)
    # but need them for the dimensions
    ocp.cost.yref = np.ones(ny)
    ocp.cost.yref_e = np.ones(ny_e)

    # intial condition #########################################################
    ocp.constraints.x0 = np.ones(nx)

    # paremeter values #########################################################
    ocp.parameter_values = np.ones(model.p.shape)

    # box constraints #########################################################
    # state box constraints
    ocp.constraints.idxbx = np.array([1, 3, 6, 7])
    ocp.constraints.lbx = np.array(
        [
            -model_bounds.n_max,
            model_bounds.v_x_min,
            -model_bounds.T_max,
            -model_bounds.delta_max,
        ]
    )
    ocp.constraints.ubx = np.array(
        [
            model_bounds.n_max,
            model_bounds.v_x_max,
            model_bounds.T_max,
            model_bounds.delta_max,
        ]
    )
    ocp.constraints.idxbx_e = np.array([1, 3, 4, 5])
    ocp.constraints.lbx_e = np.array(
        [
            -model_bounds.n_max,
            -model_bounds.v_x_max,
            -model_bounds.T_max,
            -model_bounds.delta_max,
        ]
    )
    ocp.constraints.ubx_e = np.array(
        [
            model_bounds.n_max,
            model_bounds.v_x_max,
            model_bounds.T_max,
            model_bounds.delta_max,
        ]
    )

    # control box constraints
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.lbu = np.array([-model_bounds.T_max, -model_bounds.delta_max])
    ocp.constraints.ubu = np.array([model_bounds.T_max, model_bounds.delta_max])

    # linear constraints #########################################################
    ocp.constraints.C = np.zeros((2, nx))
    ocp.constraints.C[0, -2] = -1.0
    ocp.constraints.C[1, -1] = -1.0
    ocp.constraints.D = np.zeros((2, nu))
    ocp.constraints.D[0, 0] = 1.0
    ocp.constraints.D[1, 1] = 1.0
    ocp.constraints.lg = np.array(
        [t_T * -model_bounds.T_dot_max, t_delta * -model_bounds.delta_dot_max]
    )
    ocp.constraints.ug = np.array(
        [t_T * model_bounds.T_dot_max, t_delta * model_bounds.delta_dot_max]
    )

    return ocp


def get_acados_solver(
    ocp: AcadosOcp, opts: AcadosOcpOptions, gen_code_dir: str
) -> AcadosOcpSolver:
    ocp.solver_options = opts
    if not os.path.exists(gen_code_dir):
        os.makedirs(gen_code_dir)
    ocp.code_export_directory = gen_code_dir + "/ihm2_fkin6_mpc_gen_code"
    return AcadosOcpSolver(
        ocp, json_file=gen_code_dir + "/ihm2_fkin6_mpc.json", verbose=False
    )


def get_ipopt_solver(
    continuous_model_fn: Callable[[SX, SX, SX], SX],
    Nf: int,
    model_bounds: ModelBounds,
    dt: float,
    s_ref: np.ndarray,
    kappa_ref: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    Qf: np.ndarray,
) -> Function:
    # p = MX.sym("p", 6 * NUMBER_SPLINE_INTERVALS)
    # s_ref = p[: 3 * NUMBER_SPLINE_INTERVALS]
    # kappa_ref = p[3 * NUMBER_SPLINE_INTERVALS :]
    p = np.append(s_ref, kappa_ref)

    # discretize dynamics
    x = MX.sym("x", 8, 1)
    u = MX.sym("u", 2, 1)
    k1 = continuous_model_fn(x, u, p)
    k2 = continuous_model_fn(x + dt / 2 * k1, u, p)
    k3 = continuous_model_fn(x + dt * k2, u, p)
    k4 = continuous_model_fn(x + dt * k3, u, p)
    discrete_model = Function("f", [x, u], [x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)])
    discrete_model_parallel = discrete_model.map(Nf, "unroll")
    ic(discrete_model, discrete_model_parallel)

    # create the optimization problem
    state_vars = MX.sym("state_vars", 8, Nf + 1)
    control_vars = MX.sym("control_vars", 2, Nf)

    optimization_variables = vertcat(
        reshape(state_vars, (8 * (Nf + 1), 1)), reshape(control_vars, (2 * Nf, 1))
    )

    # create the cost function
    # parallel_quad_form = Function("parallel_quad_form", [])
    Q = MX(Q)
    R = MX(R)
    Qf = MX(Qf)
    cost = (
        sum1(sum2(state_vars[:, :-1] * (Q @ state_vars[:, :-1])))
        + sum1(sum2(control_vars * (R @ control_vars)))
        + sum1(state_vars[:, -1] * (Qf @ state_vars[:, -1]))
    )
    linear_constraints_vars = reshape(
        state_vars[[6, 7], :-1] - control_vars, (2 * Nf, 1)
    )
    equality_constraints = reshape(
        discrete_model_parallel(state_vars[:, :-1], control_vars) - state_vars[:, 1:],
        (8 * Nf, 1),
    )

    # pre-populate ug, lg
    lstate = np.full(state_vars.shape, -np.inf)
    lstate[1] = -model_bounds.n_max
    lstate[3] = model_bounds.v_x_min
    lstate[6] = -model_bounds.T_max
    lstate[7] = -model_bounds.delta_max
    ustate = np.full(state_vars.shape, np.inf)
    ustate[1] = model_bounds.n_max
    ustate[3] = model_bounds.v_x_max
    ustate[6] = model_bounds.T_max
    ustate[7] = model_bounds.delta_max
    lcontrol = np.full(control_vars.shape, -np.inf)
    lcontrol[0] = -model_bounds.T_max
    lcontrol[1] = -model_bounds.delta_max
    ucontrol = np.full(control_vars.shape, np.inf)
    ucontrol[0] = model_bounds.T_max
    ucontrol[1] = model_bounds.delta_max
    lbx = np.concatenate((np.ravel(lstate, "F"), np.ravel(lcontrol, "F")))
    ubx = np.concatenate((np.ravel(ustate, "F"), np.ravel(ucontrol, "F")))

    ulin = np.zeros((2, Nf))
    llin = np.zeros((2, Nf))
    ulin[0] = t_T * model_bounds.T_dot_max
    ulin[1] = t_delta * model_bounds.delta_dot_max
    llin[0] = -t_T * model_bounds.T_dot_max
    llin[1] = -t_delta * model_bounds.delta_dot_max
    lbg = np.concatenate((np.zeros(8 * Nf), np.ravel(llin, "F")))
    ubg = np.concatenate((np.zeros(8 * Nf), np.ravel(ulin, "F")))

    return (
        nlpsol(
            "solver",
            "ipopt",
            {
                "x": optimization_variables,
                "f": cost,
                "g": vertcat(equality_constraints, linear_constraints_vars),
            },
            {
                "print_time": 0,
                "ipopt": {"print_level": 0, "sb": "yes"},
            },
        ),
        lbx,
        ubx,
        lbg,
        ubg,
    )


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
