from abc import ABC, abstractmethod
from time import perf_counter
import platform

import matplotlib.axes
import matplotlib.lines
import matplotlib.pyplot as plt
import matplotlib.widgets
import numpy as np
import numpy.typing as npt
from acados_template import AcadosModel, AcadosOcp, AcadosOcpOptions, AcadosOcpSolver
from casadi import SX, Function, cos, nlpsol, sin, tanh, vertcat
from icecream import ic
from qpsolvers import available_solvers, solve_qp
from scipy.linalg import block_diag
from scipy.sparse import csc_array
from scipy.sparse import eye as speye
from scipy.sparse import kron as spkron
from tqdm import trange

# car mass and geometry
m = 230.0  # mass
wheelbase = 1.5706  # distance between the two axles
# drivetrain parameters (simplified)
C_m0 = 4.950
C_r0 = 297.030
C_r1 = 16.665
C_r2 = 0.6784

Nf = 40
nx = 4
nu = 2
dt = 1 / 20

T_max = 500.0
delta_max = 0.5

np.set_printoptions(precision=3, suppress=True, linewidth=200)
FloatArray = npt.NDArray[np.float64]


def teds_projection(x: FloatArray, a: float) -> FloatArray:
    """Projection of x onto the interval [a, a + 2*pi)"""
    return np.mod(x - a, 2 * np.pi) + a


def unwrap_to_pi(x: FloatArray) -> FloatArray:
    """remove discontinuities caused by wrapToPi"""
    diffs = np.diff(x)
    diffs[diffs > 1.5 * np.pi] -= 2 * np.pi
    diffs[diffs < -1.5 * np.pi] += 2 * np.pi
    return np.insert(x[0] + np.cumsum(diffs), 0, x[0])


def get_continuous_dynamics() -> Function:
    # state and control variables
    X = SX.sym("X")
    Y = SX.sym("Y")
    phi = SX.sym("phi")
    v = SX.sym("v")
    T = SX.sym("T")
    delta = SX.sym("delta")
    x = vertcat(X, Y, phi, v)
    u = vertcat(T, delta)

    # auxiliary variables
    beta = 0.5 * delta  # slip angle
    v_x = v * cos(beta)  # longitudinal velocity
    l_R = 0.5 * wheelbase

    # assemble bicycle dynamics
    return Function(
        "continuous_dynamics",
        [x, u],
        [
            vertcat(
                v * cos(phi + beta),
                v * sin(phi + beta),
                v * sin(beta) / l_R,
                (C_m0 * T - (C_r0 + C_r1 * v_x + C_r2 * v_x**2) * tanh(10 * v_x)) / m,
            )
        ],
    )


def get_discrete_dynamics() -> Function:
    x = SX.sym("x", nx)
    u = SX.sym("u", nu)
    f = get_continuous_dynamics()
    k1 = f(x, u)
    k2 = f(x + dt / 2 * k1, u)
    k3 = f(x + dt / 2 * k2, u)
    k4 = f(x + dt * k3, u)
    return Function(
        "discrete_dynamics", [x, u], [x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)]
    )


def get_acados_model() -> AcadosModel:
    model = AcadosModel()
    model.name = "kin4"
    model.x = SX.sym("x", nx)
    model.u = SX.sym("u", nu)
    model.f_expl_expr = get_continuous_dynamics()(model.x, model.u)
    model.xdot = SX.sym("xdot", nx)
    model.f_impl_expr = model.xdot - model.f_expl_expr
    return model


class Controller(ABC):
    @abstractmethod
    def control(
        self,
        X: float,
        Y: float,
        phi: float,
        v: float,
        X_ref: FloatArray,
        Y_ref: FloatArray,
        phi_ref: FloatArray,
        v_ref: FloatArray,
    ) -> tuple[FloatArray, FloatArray, float]:
        pass


class NMPCController(Controller):
    ocp: AcadosOcp
    solver: AcadosOcpSolver
    discrete_dynamics: Function
    Q: FloatArray
    R: FloatArray
    Qf: FloatArray
    last_prediction_x: FloatArray
    last_prediction_u: FloatArray

    def __init__(
        self,
        q_lon: float = 1.0,
        q_lat: float = 1.0,
        q_phi: float = 1.0,
        q_v: float = 1.0,
        r_T: float = 1.0,
        r_delta: float = 1.0,
        track_width: float = 1.5,
        ocp_opts: AcadosOcpOptions = AcadosOcpOptions(),
    ):
        self.Q = np.diag([q_lon, q_lat, q_phi, q_v])
        self.Qf = 100 * self.Q
        self.R = np.diag([r_T, r_delta])

        ocp = AcadosOcp()
        ocp.solver_options = ocp_opts
        ocp.model = get_acados_model()

        ocp.dims.N = Nf
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.np = 0
        ocp.dims.ny = nx + nu
        ocp.dims.ny_e = nx
        ocp.dims.nbu = nu
        ocp.dims.ng = 1
        ocp.dims.ng_e = 1
        ocp.dims.nsg = 1
        ocp.dims.nsg_e = 1
        ocp.solver_options.tf = Nf * dt

        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.Vx = np.vstack((np.eye(nx), np.zeros((nu, nx))))
        ocp.cost.Vu = np.vstack((np.zeros((nx, nu)), np.eye(nu)))
        ocp.cost.W = np.eye(nx + nu)  # will be overwritten later
        ocp.cost.yref = np.zeros(nx + nu)  # will be overwritten later
        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.Vx_e = np.eye(nx)
        ocp.cost.W_e = np.eye(nx)  # will be overwritten later
        ocp.cost.yref_e = np.zeros(nx)  # will be overwritten later

        ocp.constraints.x0 = np.zeros(nx)  # will be overwritten later

        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.lbu = np.array([-T_max, -delta_max])
        ocp.constraints.ubu = np.array([T_max, delta_max])
        ocp.constraints.C = np.zeros((1, nx))  # will be overwritten later
        ocp.constraints.D = np.zeros((1, nu))
        ocp.constraints.C_e = np.zeros((1, nx))  # will be overwritten later
        ocp.constraints.ug = np.array([track_width])
        ocp.constraints.lg = np.array([track_width])
        ocp.constraints.ug_e = np.array([track_width])
        ocp.constraints.lg_e = np.array([track_width])
        ocp.constraints.idxsg = np.array([0])
        ocp.constraints.idxsg_e = np.array([0])

        ocp.cost.Zl_e = np.ones(1)
        ocp.cost.Zu_e = np.ones(1)
        ocp.cost.zl_e = np.ones(1)
        ocp.cost.zu_e = np.ones(1)
        ocp.cost.Zl = np.ones(1)
        ocp.cost.Zu = np.ones(1)
        ocp.cost.zl = np.ones(1)
        ocp.cost.zu = np.ones(1)

        self.ocp = ocp
        self.solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json", verbose=False)
        self.discrete_dynamics = get_discrete_dynamics()

        self.last_prediction_x = np.zeros((Nf + 1, nx))
        self.last_prediction_u = np.zeros((Nf, nu))
        self.last_prediction_u[:, 0] = T_max
        self.last_prediction_x[:, 2] = np.pi / 2
        self.last_prediction_x[:, 3] = C_m0 * T_max / m * dt * np.arange(Nf + 1)

    def control(
        self,
        X: float,
        Y: float,
        phi: float,
        v: float,
        X_ref: FloatArray,
        Y_ref: FloatArray,
        phi_ref: FloatArray,
        v_ref: FloatArray,
    ) -> tuple[FloatArray, FloatArray, float]:
        # set initial state
        x0 = np.array([X, Y, phi, v])
        self.solver.constraints_set(0, "lbx", x0)
        self.solver.constraints_set(0, "ubx", x0)

        # shift last prediction
        self.solver.set(0, "x", x0)
        # for j in range(Nf - 1):
        #     self.solver.set(j + 1, "x", self.last_prediction_x[j + 2, :])
        #     self.solver.set(j, "u", (self.last_prediction_u[j + 1, :]))
        # self.solver.set(Nf, "x", self.last_prediction_x[-1, :])
        for j in range(Nf):
            self.solver.set(
                j, "x", np.array([X_ref[j], Y_ref[j], phi_ref[j], v_ref[j]])
            )
            self.solver.set(j, "u", self.last_prediction_u[j, :])
        self.solver.set(
            Nf, "x", np.array([X_ref[Nf], Y_ref[Nf], phi_ref[Nf], v_ref[Nf]])
        )

        # change the phi_ref as well as the last_prediction to be in the same range as
        # phi (more exactly in the range [phi - pi, phi + pi))
        offset = phi - np.pi
        self.last_prediction_x[:, 2] = teds_projection(
            self.last_prediction_x[:, 2], offset
        )

        # compute the rotation matrices for the reference
        Rot = np.zeros((Nf + 1, nx, nx))
        Rot[:, 0, 0] = np.cos(phi_ref)
        Rot[:, 0, 1] = np.sin(phi_ref)
        Rot[:, 1, 0] = -np.sin(phi_ref)
        Rot[:, 1, 1] = np.cos(phi_ref)

        for i in range(Nf + 1):
            self.solver.constraints_set(
                i, "C", np.array([[Rot[i, 1, 0], Rot[i, 1, 1], 0.0, 0.0]])
            )
            if i < Nf:
                self.solver.cost_set(i, "W", block_diag(self.Q, self.R), api="new")
                # self.solver.cost_set(
                #     i, "W", block_diag(Rot[i].T @ self.Q @ Rot[i], self.R), api="new"
                # )
                self.solver.cost_set(
                    i,
                    "yref",
                    np.array([X_ref[i], Y_ref[i], phi_ref[i], v_ref[i], 0.0, 0.0]),
                    api="new",
                )
            else:
                self.solver.cost_set(i, "W", self.Qf, api="new")
                # self.solver.cost_set(i, "W", Rot[i] @ self.Qf @ Rot[i].T, api="new")
                self.solver.cost_set(
                    i,
                    "yref",
                    np.array([X_ref[i], Y_ref[i], phi_ref[i], v_ref[i]]),
                    api="new",
                )

        # solve the optimization problem
        start = perf_counter()
        exitflag = self.solver.solve()
        stop = perf_counter()
        exitmsg = {
            0: "success",
            1: "failure",
            2: "maximum number of iterations reached",
            3: "minimum step size in QP solver reached",
            4: "QP solver failed",
        }[exitflag]

        if exitflag not in {0, 2}:
            raise ValueError(exitmsg)

        # extract the first optimal input
        self.last_prediction_x[0, :] = x0
        for i in range(Nf):
            self.last_prediction_x[i + 1, :] = self.solver.get(i, "x")
            self.last_prediction_u[i, :] = self.solver.get(i, "u")

        e = np.squeeze(
            Rot[:, :2, :2]
            @ (self.last_prediction_x[:, :2] - np.array([X_ref, Y_ref]).T)[
                :, :, np.newaxis
            ]
        )
        ic(e)
        return (
            np.copy(self.last_prediction_x),
            np.copy(self.last_prediction_u),
            stop - start,
        )


class NMPCControllerIpopt(Controller):
    discrete_dynamics: Function
    solver: Function
    q_lon: float
    q_lat: float
    q_phi: float
    q_v: float
    r_T: float
    r_delta: float
    q_lon_f: float
    q_lat_f: float
    q_phi_f: float
    q_v_f: float

    def __init__(
        self,
        q_lon: float = 1.0,
        q_lat: float = 1.0,
        q_phi: float = 1.0,
        q_v: float = 1.0,
        r_T: float = 1.0,
        r_delta: float = 1.0,
        q_lon_f: float = 1.0,
        q_lat_f: float = 1.0,
        q_phi_f: float = 1.0,
        q_v_f: float = 1.0,
        **kwargs,
    ):
        # create discrete dynamics
        self.discrete_dynamics = get_discrete_dynamics()

        # create costs weights
        self.q_lon = q_lon
        self.q_lat = q_lat
        self.q_phi = q_phi
        self.q_v = q_v
        self.r_T = r_T
        self.r_delta = r_delta
        self.q_lon_f = q_lon_f
        self.q_lat_f = q_lat_f
        self.q_phi_f = q_phi_f
        self.q_v_f = q_v_f

        # optimization variables
        X_ref = SX.sym("X_ref", Nf + 1)
        Y_ref = SX.sym("Y_ref", Nf + 1)
        phi_ref = SX.sym("phi_ref", Nf + 1)
        v_ref = SX.sym("v_ref", Nf + 1)
        x = [SX.sym(f"x_{i}", nx) for i in range(Nf + 1)]
        u = [SX.sym(f"u_{i}", nu) for i in range(Nf)]
        parameters = vertcat(X_ref, Y_ref, phi_ref, v_ref)
        optimization_variables = vertcat(*x, *u)

        # construct cost function
        cost_function = 0.0
        for i in range(Nf):
            if i > 0:
                cp = cos(phi_ref[i])
                sp = sin(phi_ref[i])
                X = x[i][0]
                Y = x[i][1]
                phi = x[i][2]
                v = x[i][3]
                e_lon = cp * (X - X_ref[i]) + sp * (Y - Y_ref[i])
                e_lat = -sp * (X - X_ref[i]) + cp * (Y - Y_ref[i])
                cost_function += (
                    self.q_lon * e_lon**2
                    + self.q_lat * e_lat**2
                    + self.q_phi * (phi - phi_ref[i]) ** 2
                    + self.q_v * (v - v_ref[i]) ** 2
                )
            T = u[i][0]
            delta = u[i][1]
            cost_function += self.r_T * T**2 + self.r_delta * delta**2

        cp = cos(phi_ref[Nf])
        sp = sin(phi_ref[Nf])
        X = x[Nf][0]
        Y = x[Nf][1]
        phi = x[Nf][2]
        v = x[Nf][3]
        e_lon = cp * (X - X_ref[Nf]) + sp * (Y - Y_ref[Nf])
        e_lat = -sp * (X - X_ref[Nf]) + cp * (Y - Y_ref[Nf])
        cost_function += (
            self.q_lon_f * e_lon**2
            + self.q_lat_f * e_lat**2
            + self.q_phi_f * (phi - phi_ref[Nf]) ** 2
            + self.q_v_f * (v - v_ref[Nf]) ** 2
        )

        # equality constraints
        eq_constraints = vertcat(
            *[self.discrete_dynamics(x[i], u[i]) - x[i + 1] for i in range(Nf)]
        )
        self.lbg = np.zeros(eq_constraints.shape[0])
        self.ubg = np.zeros(eq_constraints.shape[0])

        # simple bounds
        self.lbx = np.concatenate(
            (
                np.tile(np.array([-np.inf, -np.inf, -np.inf, -np.inf]), Nf + 1),
                np.tile(np.array([-T_max, -delta_max]), Nf),
            )
        )
        self.ubx = np.concatenate(
            (
                np.tile(np.array([np.inf, np.inf, np.inf, np.inf]), Nf + 1),
                np.tile(np.array([T_max, delta_max]), Nf),
            )
        )

        # assemble solver
        self.solver = nlpsol(
            "nmpc",
            "ipopt",
            {
                "x": optimization_variables,
                "f": cost_function,
                "g": eq_constraints,
                "p": parameters,
            },
            {
                "print_time": 0,
                "ipopt": {"sb": "yes", "print_level": 0},
            } | (
                {} if platform.system() == "Linux" else {
                    "jit": True,
                    "jit_options": {"flags": ["-O3 -march=native"], "verbose": False},
                }
            ),
        )
        ic(self.solver)

    def control(
        self,
        X: float,
        Y: float,
        phi: float,
        v: float,
        X_ref: FloatArray,
        Y_ref: FloatArray,
        phi_ref: FloatArray,
        v_ref: FloatArray,
    ) -> tuple[FloatArray, FloatArray, float]:
        # set initial state
        x0 = np.array([X, Y, phi, v])
        self.lbx[:nx] = x0
        self.ubx[:nx] = x0

        # create guess
        # x = [x0]
        # u = []
        # for i in range(Nf):
        #     u.append(np.array([90.0 * (v_ref[i] - x[-1][3]), 0.0]))
        #     x.append(self.discrete_dynamics(x[-1], u[-1]).full().ravel())
        # initial_guess = np.concatenate(x + u)
        initial_guess = np.concatenate(
            (
                np.reshape(
                    np.column_stack((X_ref, Y_ref, phi_ref, v_ref)), (Nf + 1) * nx
                ),
                np.zeros(Nf * nu),
            )
        )
        # initial_guess = np.zeros((Nf + 1) * nx + Nf * nu)

        parameters = np.concatenate((X_ref, Y_ref, phi_ref, v_ref))
        # solve the optimization problem
        start = perf_counter()
        sol = self.solver(
            x0=initial_guess,
            p=parameters,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
        )
        stop = perf_counter()
        runtime = stop - start

        # extract solution
        opt_variables = sol["x"].full().ravel()
        last_prediction_x = opt_variables[: (Nf + 1) * nx].reshape(Nf + 1, nx).copy()
        last_prediction_u = opt_variables[(Nf + 1) * nx :].reshape(Nf, nu).copy()
        # last_prediction_x[0, :] = x0

        # check exit flag
        stats = self.solver.stats()
        if not stats["success"]:
            ic(stats)
            raise ValueError(stats["return_status"])

        return last_prediction_x, last_prediction_u, runtime


def _c(ca, i, j, p, q):
    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = np.linalg.norm(p[i] - q[j])
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(ca, i - 1, 0, p, q), np.linalg.norm(p[i] - q[j]))
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(ca, 0, j - 1, p, q), np.linalg.norm(p[i] - q[j]))
    elif i > 0 and j > 0:
        ca[i, j] = max(
            min(
                _c(ca, i - 1, j, p, q),
                _c(ca, i - 1, j - 1, p, q),
                _c(ca, i, j - 1, p, q),
            ),
            np.linalg.norm(p[i] - q[j]),
        )
    else:
        ca[i, j] = float("inf")

    return ca[i, j]


def frdist(p, q):
    """
    Computes the discrete Fréchet distance between
    two curves. The Fréchet distance between two curves in a
    metric space is a measure of the similarity between the curves.
    The discrete Fréchet distance may be used for approximately computing
    the Fréchet distance between two arbitrary curves,
    as an alternative to using the exact Fréchet distance between a polygonal
    approximation of the curves or an approximation of this value.

    This is a Python 3.* implementation of the algorithm produced
    in Eiter, T. and Mannila, H., 1994. Computing discrete Fréchet distance.
    Tech. Report CD-TR 94/64, Information Systems Department, Technical
    University of Vienna.
    http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf

    Function dF(P, Q): real;
        input: polygonal curves P = (u1, . . . , up) and Q = (v1, . . . , vq).
        return: δdF (P, Q)
        ca : array [1..p, 1..q] of real;
        function c(i, j): real;
            begin
                if ca(i, j) > −1 then return ca(i, j)
                elsif i = 1 and j = 1 then ca(i, j) := d(u1, v1)
                elsif i > 1 and j = 1 then ca(i, j) := max{ c(i − 1, 1), d(ui, v1) }
                elsif i = 1 and j > 1 then ca(i, j) := max{ c(1, j − 1), d(u1, vj) }
                elsif i > 1 and j > 1 then ca(i, j) :=
                max{ min(c(i − 1, j), c(i − 1, j − 1), c(i, j − 1)), d(ui, vj ) }
                else ca(i, j) = ∞
                return ca(i, j);
            end; /* function c */

        begin
            for i = 1 to p do for j = 1 to q do ca(i, j) := −1.0;
            return c(p, q);
        end.

    Parameters
    ----------
    P : Input curve - two dimensional array of points
    Q : Input curve - two dimensional array of points

    Returns
    -------
    dist: float64
        The discrete Fréchet distance between curves `P` and `Q`.

    Examples
    --------
    >>> from frechetdist import frdist
    >>> P=[[1,1], [2,1], [2,2]]
    >>> Q=[[2,2], [0,1], [2,4]]
    >>> frdist(P,Q)
    >>> 2.0
    >>> P=[[1,1], [2,1], [2,2]]
    >>> Q=[[1,1], [2,1], [2,2]]
    >>> frdist(P,Q)
    >>> 0
    """
    p = np.array(p, np.float64)
    q = np.array(q, np.float64)

    len_p = len(p)
    len_q = len(q)

    if len_p == 0 or len_q == 0:
        raise ValueError("Input curves are empty.")

    ca = np.ones((len_p, len_q), dtype=np.float64) * -1

    dist = _c(ca, len_p - 1, len_q - 1, p, q)
    return dist


def create_dpc_dataset():
    # sample points uniformly on all paths
    pass


def split_dataset():
    pass


class DPCController(Controller):
    def __init__(self):
        pass

    def control(
        self,
        X: float,
        Y: float,
        phi: float,
        v: float,
        X_ref: FloatArray,
        Y_ref: FloatArray,
        phi_ref: FloatArray,
        v_ref: FloatArray,
    ) -> tuple[FloatArray, FloatArray, float]:
        pass


NUMBER_SPLINE_INTERVALS = 500


def fit_spline(
    path: FloatArray,
    curv_weight: float = 1.0,
    return_errs: bool = False,
    qp_solver: str = "proxqp",
) -> tuple[FloatArray, FloatArray]:
    """
    computes the coefficients of each spline portion of the path.
    > Note: the path is assumed to be closed but the first and last points are NOT the same.

    :param path: Nx2 array of points
    :param curv_weight: weight of the curvature term in the cost function
    :return_errs:
    :qp_solver:
    :returns p_X, p_Y: Nx4 arrays of coefficients of the splines in the x and y directions
                       (each row correspond to a_i, b_i, c_i, d_i coefficients of the i-th spline portion)
    """
    assert (
        len(path.shape) == 2 and path.shape[1] == 2
    ), f"path must have shape (N,2) but has shape {path.shape}"
    assert (
        qp_solver in available_solvers
    ), f"qp_solver must be one of the available solvers: {available_solvers}"

    # precompute all the QP data
    N = path.shape[0]
    delta_s = np.linalg.norm(path[1:] - path[:-1], axis=1)
    delta_s = np.append(delta_s, np.linalg.norm(path[0] - path[-1]))
    rho = np.zeros(N)
    rho[:-1] = delta_s[:-1] / delta_s[1:]
    rho[-1] = delta_s[-1] / delta_s[0]
    IN = speye(N, format="csc")
    A = spkron(
        IN,
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 0.0, 2.0, 6.0],
            ]
        ),
        format="csc",
    ) + csc_array(
        (
            np.concatenate((-np.ones(N), -rho, -2 * rho**2)),
            (
                np.concatenate(
                    (3 * np.arange(N), 1 + 3 * np.arange(N), 2 + 3 * np.arange(N))
                ),
                np.concatenate(
                    (
                        np.roll(4 * np.arange(N), -1),
                        np.roll(1 + 4 * np.arange(N), -1),
                        np.roll(2 + 4 * np.arange(N), -1),
                    )
                ),
            ),
        ),
        shape=(3 * N, 4 * N),
    )
    B = spkron(IN, np.array([[1.0, 0.0, 0.0, 0.0]]), format="csc")
    C = csc_array(
        (
            np.concatenate((2 / np.square(delta_s), 6 / np.square(delta_s))),
            (
                np.concatenate((np.arange(N), np.arange(N))),
                np.concatenate((2 + 4 * np.arange(N), 3 + 4 * np.arange(N))),
            ),
        ),
        shape=(N, 4 * N),
    )
    P = B.T @ B + curv_weight * C.T @ C + 1e-10 * speye(4 * N, format="csc")
    q = -B.T @ path
    b = np.zeros(3 * N)

    if qp_solver in {"quadprog", "ecos"}:
        A = A.toarray()
        B = B.toarray()
        C = C.toarray()
        P = P.toarray()

    # solve the QP for X and Y separately
    p_X = solve_qp(P=P, q=q[:, 0], A=A, b=b, solver=qp_solver)
    p_Y = solve_qp(P=P, q=q[:, 1], A=A, b=b, solver=qp_solver)

    # compute interpolation error on X and Y
    X_err = B @ p_X - path[:, 0]
    Y_err = B @ p_Y - path[:, 1]

    # reshape to (N,4) arrays
    p_X = np.reshape(p_X, (N, 4))
    p_Y = np.reshape(p_Y, (N, 4))

    if return_errs:
        return p_X, p_Y, X_err, Y_err
    else:
        return p_X, p_Y


def check_spline_coeffs_dims(coeffs_X: FloatArray, coeffs_Y: FloatArray):
    assert (
        len(coeffs_X.shape) == 2 and coeffs_X.shape[1] == 4
    ), f"coeffs_X must have shape (N,4) but has shape {coeffs_X.shape}"
    assert (
        len(coeffs_Y.shape) == 2 and coeffs_Y.shape[1] == 4
    ), f"coeffs_Y must have shape (N,4) but has shape {coeffs_Y.shape}"
    assert (
        coeffs_X.shape[0] == coeffs_Y.shape[0]
    ), f"coeffs_X and coeffs_Y must have the same length but have lengths {coeffs_X.shape[0]} and {coeffs_Y.shape[0]}"


def compute_spline_interval_lengths(
    coeffs_X: FloatArray, coeffs_Y: FloatArray, no_interp_points=100
):
    """
    computes the lengths of each spline portion of the path.
    > Note: Here the closeness of the part does not matter, it is contained in the coefficients

    :param coeff_X: Nx4 array of coefficients of the splines in the x direction (as returned by calc_splines)
    :param coeff_Y: Nx4 array of coefficients of the splines in the y direction (as returned by calc_splines)
    :param delta_s: number of points to use on each spline portion for the interpolation
    """
    check_spline_coeffs_dims(coeffs_X, coeffs_Y)

    N = coeffs_X.shape[0]

    t_steps = np.linspace(0.0, 1.0, no_interp_points)[np.newaxis, :]
    interp_points = np.zeros((no_interp_points, N, 2))

    interp_points[:, :, 0] = coeffs_X[:, 0]
    interp_points[:, :, 1] = coeffs_Y[:, 0]

    coeffs_X = coeffs_X[:, np.newaxis, :]
    coeffs_Y = coeffs_Y[:, np.newaxis, :]

    interp_points = interp_points.transpose(1, 0, 2)

    interp_points[:, :, 0] += coeffs_X[:, :, 1] @ t_steps
    interp_points[:, :, 0] += coeffs_X[:, :, 2] @ np.power(t_steps, 2)
    interp_points[:, :, 0] += coeffs_X[:, :, 3] @ np.power(t_steps, 3)

    interp_points[:, :, 1] += coeffs_Y[:, :, 1] @ t_steps
    interp_points[:, :, 1] += coeffs_Y[:, :, 2] @ np.power(t_steps, 2)
    interp_points[:, :, 1] += coeffs_Y[:, :, 3] @ np.power(t_steps, 3)

    delta_s = np.sum(
        np.sqrt(np.sum(np.power(np.diff(interp_points, axis=1), 2), axis=2)), axis=1
    )
    assert delta_s.shape == (N,), f"{delta_s.shape}"
    return delta_s


def uniformly_sample_spline(
    coeffs_X: FloatArray,
    coeffs_Y: FloatArray,
    delta_s: FloatArray,
    n_samples: int,
):
    """
    uniformly n_samples equidistant points along the path defined by the splines.
    The first point will always be the initial point of the first spline portion, and
    the last point will NOT be the initial point of the first spline portion.

    :param coeffs_X: Nx4 array of coefficients of the splines in the x direction (as returned by calc_splines)
    :param coeffs_Y: Nx4 array of coefficients of the splines in the y direction (as returned by calc_splines)
    :param spline_lengths: N array of lengths of the spline portions (as returned by calc_spline_lengths)
    :param n_samples: number of points to sample

    :return X_interp: n_samples array of X coordinates along the path
    :return Y_interp: n_samples array of Y coordinates along the path
    :return idx_interp: n_samples array of indices of the spline portions that host the points
    :return t_interp: n_samples array of t values of the points within their respective spline portions
    :return s_interp: n_samples array of distances along the path of the points
    """
    s = np.cumsum(delta_s)
    s_interp = np.linspace(0.0, s[-1], n_samples, endpoint=False)

    # find the spline that hosts the current interpolation point
    idx_interp = np.argmax(s_interp[:, np.newaxis] < s, axis=1)

    t_interp = np.zeros(n_samples)  # save t values
    X_interp = np.zeros(n_samples)  # raceline coords
    Y_interp = np.zeros(n_samples)  # raceline coords

    # get spline t value depending on the progress within the current element
    t_interp[idx_interp > 0] = (
        s_interp[idx_interp > 0] - s[idx_interp - 1][idx_interp > 0]
    ) / delta_s[idx_interp][idx_interp > 0]
    t_interp[idx_interp == 0] = s_interp[idx_interp == 0] / delta_s[0]

    # calculate coords
    X_interp = (
        coeffs_X[idx_interp, 0]
        + coeffs_X[idx_interp, 1] * t_interp
        + coeffs_X[idx_interp, 2] * np.power(t_interp, 2)
        + coeffs_X[idx_interp, 3] * np.power(t_interp, 3)
    )

    Y_interp = (
        coeffs_Y[idx_interp, 0]
        + coeffs_Y[idx_interp, 1] * t_interp
        + coeffs_Y[idx_interp, 2] * np.power(t_interp, 2)
        + coeffs_Y[idx_interp, 3] * np.power(t_interp, 3)
    )

    return X_interp, Y_interp, idx_interp, t_interp, s_interp


def get_heading(
    coeffs_X: FloatArray,
    coeffs_Y: FloatArray,
    idx_interp: FloatArray,
    t_interp: FloatArray,
) -> FloatArray:
    """
    analytically computes the heading and the curvature at each point along the path
    specified by idx_interp and t_interp.

    :param coeffs_X: Nx4 array of coefficients of the splines in the x direction (as returned by calc_splines)
    :param coeffs_Y: Nx4 array of coefficients of the splines in the y direction (as returned by calc_splines)
    :param idx_interp: n_samples array of indices of the spline portions that host the points
    :param t_interp: n_samples array of t values of the points within their respective spline portions
    """
    check_spline_coeffs_dims(coeffs_X, coeffs_Y)

    # we don't divide by delta_s[idx_interp] here because this term will cancel out
    # in arctan2 either way
    x_d = (
        coeffs_X[idx_interp, 1]
        + 2 * coeffs_X[idx_interp, 2] * t_interp
        + 3 * coeffs_X[idx_interp, 3] * np.square(t_interp)
    )
    y_d = (
        coeffs_Y[idx_interp, 1]
        + 2 * coeffs_Y[idx_interp, 2] * t_interp
        + 3 * coeffs_Y[idx_interp, 3] * np.square(t_interp)
    )
    phi = np.arctan2(y_d, x_d)

    return phi


def get_curvature(
    coeffs_X: FloatArray,
    coeffs_Y: FloatArray,
    idx_interp: FloatArray,
    t_interp: FloatArray,
) -> FloatArray:
    # same here with the division by delta_s[idx_interp] ** 2
    x_d = (
        coeffs_X[idx_interp, 1]
        + 2 * coeffs_X[idx_interp, 2] * t_interp
        + 3 * coeffs_X[idx_interp, 3] * np.square(t_interp)
    )
    y_d = (
        coeffs_Y[idx_interp, 1]
        + 2 * coeffs_Y[idx_interp, 2] * t_interp
        + 3 * coeffs_Y[idx_interp, 3] * np.square(t_interp)
    )
    x_dd = 2 * coeffs_X[idx_interp, 2] + 6 * coeffs_X[idx_interp, 3] * t_interp
    y_dd = 2 * coeffs_Y[idx_interp, 2] + 6 * coeffs_Y[idx_interp, 3] * t_interp
    kappa = (x_d * y_dd - y_d * x_dd) / np.power(x_d**2 + y_d**2, 1.5)
    return kappa


class MotionPlanner:
    def __init__(
        self,
        center_line: FloatArray,
        n_samples: int = NUMBER_SPLINE_INTERVALS,
        v_ref=5.0,
    ):
        coeffs_X, coeffs_Y = fit_spline(
            path=center_line,
            curv_weight=2.0,
            qp_solver="proxqp",
            return_errs=False,
        )
        delta_s = compute_spline_interval_lengths(coeffs_X=coeffs_X, coeffs_Y=coeffs_Y)
        X_ref, Y_ref, idx_interp, t_interp, s_ref = uniformly_sample_spline(
            coeffs_X=coeffs_X,
            coeffs_Y=coeffs_Y,
            delta_s=delta_s,
            n_samples=n_samples,
        )
        # kappa_ref = get_curvature(
        #     coeffs_X=coeffs_X,
        #     coeffs_Y=coeffs_Y,
        #     idx_interp=idx_interp,
        #     t_interp=t_interp,
        # )
        phi_ref = get_heading(
            coeffs_X=coeffs_X,
            coeffs_Y=coeffs_Y,
            idx_interp=idx_interp,
            t_interp=t_interp,
        )
        # v_ref = np.minimum(v_max, np.sqrt(a_lat_max / np.abs(kappa_ref)))

        lap_length = s_ref[-1] + np.hypot(X_ref[-1] - X_ref[0], Y_ref[-1] - Y_ref[0])
        s_diff = np.append(
            np.diff(s_ref), np.hypot(X_ref[-1] - X_ref[0], Y_ref[-1] - Y_ref[0])
        )
        t_diff = s_diff / v_ref
        t_ref_extra = np.insert(np.cumsum(t_diff), 0, 0.0)
        lap_time = np.copy(t_ref_extra[-1])
        t_ref = t_ref_extra[:-1]

        self.lap_length = lap_length
        self.lap_time = lap_time
        self.s_ref = np.concatenate((s_ref - lap_length, s_ref, s_ref + lap_length))
        self.t_ref = np.concatenate((t_ref - lap_time, t_ref, t_ref + lap_time))
        self.X_ref = np.concatenate((X_ref, X_ref, X_ref))
        self.Y_ref = np.concatenate((Y_ref, Y_ref, Y_ref))
        self.phi_ref = unwrap_to_pi(np.concatenate((phi_ref, phi_ref, phi_ref)))
        self.v_ref = v_ref

    def project(
        self, X: float, Y: float, s_guess: float, tolerance: float = 10.0
    ) -> float:
        # extract all the points in X_ref, Y_ref assiciated with s_ref values within s_guess +- tolerance
        id_low = np.searchsorted(self.s_ref, s_guess - tolerance)
        id_up = np.searchsorted(self.s_ref, s_guess + tolerance)
        local_traj = np.array([self.X_ref[id_low:id_up], self.Y_ref[id_low:id_up]]).T

        # find the closes point to (X,Y) to find one segment extremity
        distances = np.linalg.norm(local_traj - np.array([X, Y]), axis=1)
        id_min = np.argmin(distances)

        # compute the angles between (X,Y), the closest point, and the next and previous points to find the second segment extremity
        def angle3pt(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        angle_prev = angle3pt(
            np.array([X, Y]), local_traj[id_min], local_traj[id_min - 1]
        )
        angle_next = angle3pt(
            np.array([X, Y]), local_traj[id_min], local_traj[id_min + 1]
        )
        if angle_prev < angle_next:
            a = local_traj[id_min - 1]
            b = local_traj[id_min]
            sa = self.s_ref[id_low + id_min - 1]
            sb = self.s_ref[id_low + id_min]
        else:
            a = local_traj[id_min]
            b = local_traj[id_min + 1]
            sa = self.s_ref[id_low + id_min]
            sb = self.s_ref[id_low + id_min + 1]

        # project (X,Y) on the segment [a,b]
        ab = b - a
        lamda = np.dot(np.array([X, Y]) - a, ab) / np.dot(ab, ab)

        return sa + lamda * (sb - sa)

    def plan(
        self, X: float, Y: float, phi: float, v: float, s_guess: float
    ) -> tuple[float, FloatArray, FloatArray, FloatArray, FloatArray]:
        # project current position on the reference trajectory and extract reference time of passage
        s0 = self.project(X, Y, s_guess)
        t0 = np.interp(s0, self.s_ref, self.t_ref)
        # sample reference values uniformly in time
        t_ref = dt * np.arange(Nf + 1) + t0
        s_ref = np.interp(t_ref, self.t_ref, self.s_ref)
        X_ref = np.interp(s_ref, self.s_ref, self.X_ref)
        Y_ref = np.interp(s_ref, self.s_ref, self.Y_ref)
        phi_ref = np.interp(s_ref, self.s_ref, self.phi_ref)
        v_ref = self.v_ref * np.ones(Nf + 1)
        # post-process the reference heading to make sure it is in the range [phi - pi, phi + pi))
        phi_ref = teds_projection(phi_ref, phi - np.pi)
        return s0, X_ref, Y_ref, phi_ref, v_ref

    def plot_motion_plan(
        self,
        center_line: FloatArray,
        blue_cones: FloatArray,
        yellow_cones: FloatArray,
        big_orange_cones: FloatArray,
        small_orange_cones: FloatArray,
        plot_title: str = "",
    ) -> None:
        plt.figure()
        plt.plot(self.s_ref, self.phi_ref, label="headings")
        plt.legend()
        plt.xlabel("track progress [m]")
        plt.ylabel("heading [rad]")
        plt.title(plot_title + " : reference heading/yaw profile")
        plt.tight_layout()

        plt.figure()
        plot_cones(
            blue_cones,
            yellow_cones,
            big_orange_cones,
            small_orange_cones,
            show=False,
        )
        plt.plot(self.X_ref, self.Y_ref, label="reference trajectory")
        plt.scatter(
            center_line[:, 0],
            center_line[:, 1],
            s=14,
            c="k",
            marker="x",
            label="center line",
        )
        plt.legend()
        plt.title(plot_title + " : reference trajectory")
        plt.tight_layout()


def load_center_line(filename: str) -> tuple[FloatArray, FloatArray]:
    """
    Loads the center line stored in CSV file specified by filename. This file must have
    the following format:
        X,Y,right_width,left_width
    Returns the center line as a numpy array of shape (N, 2) and the corresponding
    (right and left) track widths as a numpy array of shape (N,2).
    """
    arr = np.genfromtxt(filename, delimiter=",", dtype=float, skip_header=1)
    return arr[:, :2], arr[:, 2:]


def load_cones(
    filename: str,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """
    Loads the cones stored in CSV file specified by filename. This file must have the
    following format:
        cone_type,X,Y,Z,std_X,std_Y,std_Z,right,left
    The returned arrays correspond to (in this order) the blue cones, yellow cones, big
    orange cones, small orange cones (possibly empty), right cones and left cones (all
    colors .
    """
    arr = np.genfromtxt(filename, delimiter=",", dtype=str, skip_header=1)
    blue_cones = arr[arr[:, 0] == "blue"][:, 1:3].astype(float)
    yellow_cones = arr[arr[:, 0] == "yellow"][:, 1:3].astype(float)
    big_orange_cones = arr[arr[:, 0] == "big_orange"][:, 1:3].astype(float)
    small_orange_cones = arr[arr[:, 0] == "small_orange"][:, 1:3].astype(float)
    right_cones = arr[arr[:, 7] == "1"][:, 1:3].astype(float)
    left_cones = arr[arr[:, 8] == "1"][:, 1:3].astype(float)
    return (
        blue_cones,
        yellow_cones,
        big_orange_cones,
        small_orange_cones,
        right_cones,
        left_cones,
    )


def plot_cones(
    blue_cones,
    yellow_cones,
    big_orange_cones,
    small_orange_cones,
    origin=np.zeros(2),
    show=True,
):
    plt.scatter(blue_cones[:, 0], blue_cones[:, 1], s=14, c="b", marker="^")
    plt.scatter(yellow_cones[:, 0], yellow_cones[:, 1], s=14, c="y", marker="^")
    plt.scatter(
        big_orange_cones[:, 0], big_orange_cones[:, 1], s=28, c="orange", marker="^"
    )
    try:
        plt.scatter(
            small_orange_cones[:, 0],
            small_orange_cones[:, 1],
            s=7,
            c="orange",
            marker="^",
        )
    except IndexError:
        pass
    plt.scatter(origin[0], origin[1], c="g", marker="x")
    plt.axis("equal")
    plt.tight_layout()
    if show:
        plt.show()


def closed_loop(controller: Controller, data_file: str = "closed_loop_data.npz"):
    """
    we store all the open loop predictions into big arrays that we dump into npz files
    we dump x_ref (nx x (Nf+1)), x_pred (nx x (Nf+1)), u_pred (nu x Nf)
    the current state is always the first element in x_ref

    with this dumped data we can:
    1. plot it with a slider
    2. train a new neural control policy using either DPC or imitation learning
    """

    Tsim = 60.0
    Nsim = int(Tsim / dt) + 1
    v_ref = 5.0
    x_current = np.array([0.0, 0.0, np.pi / 2, 0.0])
    s_guess = 0.0
    all_x_ref = []
    all_x_pred = []
    all_u_pred = []
    all_runtimes = []
    discrete_dynamics = get_discrete_dynamics()

    # import track data
    center_line, _ = load_center_line("../data/fsds_competition_1/center_line.csv")
    blue_cones, yellow_cones, big_orange_cones, small_orange_cones, _, _ = load_cones(
        "../data/fsds_competition_1/cones.csv"
    )

    # create motion planner
    motion_planner = MotionPlanner(center_line, v_ref=v_ref)
    motion_planner.plot_motion_plan(
        center_line,
        blue_cones,
        yellow_cones,
        big_orange_cones,
        small_orange_cones,
        "Motion Planner",
    )

    for i in trange(Nsim):
        X = x_current[0]
        Y = x_current[1]
        phi = x_current[2]
        v = x_current[3]
        # construct the reference trajectory
        s_guess, X_ref, Y_ref, phi_ref, v_ref = motion_planner.plan(
            X, Y, phi, v, s_guess
        )
        # add data to arrays
        all_x_ref.append(np.column_stack((X_ref, Y_ref, phi_ref, v_ref)))
        # call controller
        try:
            x_pred, u_pred, runtime = controller.control(
                X, Y, phi, v, X_ref, Y_ref, phi_ref, v_ref
            )
        except ValueError as e:
            print(f"Error in iteration {i}: {e}")
            break
        u_current = u_pred[0]
        # ic(u_current)
        # add data to arrays
        all_runtimes.append(runtime)
        all_x_pred.append(x_pred.copy())
        all_u_pred.append(u_pred.copy())
        # simulate next state
        x_current = discrete_dynamics(x_current, u_current).full().ravel()
        # check if we have completed a lap
        if s_guess > motion_planner.lap_length:
            print(f"Completed a lap in {i} iterations")
            break

    all_runtimes = np.array(all_runtimes)
    all_x_ref = np.array(all_x_ref)
    all_x_pred = np.array(all_x_pred)
    all_u_pred = np.array(all_u_pred)

    # save data to npz file
    np.savez(
        data_file,
        x_ref=all_x_ref,
        x_pred=all_x_pred,
        u_pred=all_u_pred,
        runtimes=all_runtimes,
        center_line=np.column_stack((motion_planner.X_ref, motion_planner.Y_ref)),
        blue_cones=blue_cones,
        yellow_cones=yellow_cones,
        big_orange_cones=big_orange_cones,
    )


def visualize(
    x_ref: FloatArray,
    x_pred: FloatArray,
    u_pred: FloatArray,
    runtimes: FloatArray,
    center_line: FloatArray,
    blue_cones: FloatArray,
    yellow_cones: FloatArray,
    big_orange_cones: FloatArray,
    output_file: str = "",
    show: bool = True,
):
    """
    Creates 2 plots:
    1. a plot to display the evolution of the states and controls over time.
       It is constituted of the following subplots:
       +-------------------+-------------------+----------------+
       |                   | velocity v (m/s)  | trottle T (N)  |
       | trajectory XY (m) +-------------------+----------------+
       |                   | heading phi (deg) | steering (deg) |
       +-------------------+-------------------+----------------+
       underneath these subplots, a slider will allow to move through the time steps and to visualize the
       references given to the controller, as well as the predictions made by the controller.
    2. another plot to display the runtimes distribution (scatter plot superposed with a boxplot)
    """
    # plot runtime distribution
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(1000 * runtimes, vert=False)
    ax.set_xlabel("runtime [ms]")
    ax.set_yticks([])
    ax.set_title("Runtime distribution")

    # the shapes should be:
    # x_ref : (Nsim, Nf+1, nx)
    # x_pred : (Nsim, Nf+1, nx)
    # u_pred : (Nsim, Nf, nx)
    # it can happen that an error occured during the run and there is one x_ref more than x_pred and u_pred
    assert x_pred.shape[0] == u_pred.shape[0]
    if x_ref.shape[0] == x_pred.shape[0]:
        controller_failed = False
    elif x_ref.shape[0] == x_pred.shape[0] + 1:
        controller_failed = True
    else:
        raise ValueError(
            f"x_ref has shape {x_ref.shape} and x_pred has shape {x_pred.shape}"
        )
    Nsim = x_ref.shape[0]
    # assert x_ref.shape == (Nsim, Nf + 1, nx)
    # assert x_pred.shape == (Nsim, Nf + 1, nx)
    # assert u_pred.shape == (Nsim, Nf, nu)

    # create grid plot
    gridshape = (2, 3)
    fig = plt.figure(figsize=(20, 9))
    axes: dict[str, matplotlib.axes.Axes] = {}
    lines: dict[str, dict[str, matplotlib.lines.Line2D]] = {}

    # define plot data
    plot_data = {
        "XY": {
            "loc": (0, 0),
            "xlabel": r"$X$ [m]",
            "ylabel": r"$Y$ [m]",
            "data": {
                "center_line": center_line,
                "blue_cones": blue_cones,
                "yellow_cones": yellow_cones,
                "big_orange_cones": big_orange_cones,
                "past": x_pred[:, 0, :2],
                "ref": x_ref[:, :, :2],
                "pred": x_pred[:, :, :2],
            },
        },
        "v": {
            "loc": (0, 1),
            "ylabel": r"$v$ [m/s]",
            "data": {
                "past": x_pred[:, 0, 3],
                "ref": x_ref[:, :, 3],
                "pred": x_pred[:, :, 3],
            },
        },
        "phi": {
            "loc": (1, 1),
            "ylabel": r"$\varphi$ [°]",
            "data": {
                "past": np.rad2deg(x_pred[:, 0, 2]),
                "ref": np.rad2deg(x_ref[:, :, 2]),
                "pred": np.rad2deg(x_pred[:, :, 2]),
            },
        },
        "T": {
            "loc": (0, 2),
            "ylabel": r"$T$ [N]",
            "data": {
                "past": u_pred[:, 0, 0],
                "pred": np.concatenate((u_pred[:, :, 0], u_pred[:, -1:, 0]), axis=1),
            },
        },
        "delta": {
            "loc": (1, 2),
            "ylabel": r"$\delta$ [°]",
            "data": {
                "past": np.rad2deg(u_pred[:, 0, 1]),
                "pred": np.rad2deg(
                    np.concatenate((u_pred[:, :, 1], u_pred[:, -1:, 1]), axis=1)
                ),
            },
        },
    }
    # custom matplotlib colors
    green = "#51BF63"
    orange = "#ff9b31"
    blue = "#1f77b4"
    red = "#ff5733"
    purple = "#7c00c6"
    yellow = "#d5c904"

    # initialize axes and lines
    # TODO: add shared x axis for 1d subplots
    for subplot_name, subplot_info in plot_data.items():
        if subplot_name == "XY":
            # create axes
            axes[subplot_name] = plt.subplot2grid(
                gridshape, subplot_info["loc"], rowspan=2
            )
            # plot additional data that will not be updated
            axes[subplot_name].scatter(blue_cones[:, 0], blue_cones[:, 1], s=14, c=blue)
            axes[subplot_name].scatter(
                yellow_cones[:, 0], yellow_cones[:, 1], s=14, c=yellow
            )
            axes[subplot_name].scatter(
                big_orange_cones[:, 0], big_orange_cones[:, 1], s=28, c=orange
            )
            axes[subplot_name].plot(center_line[:, 0], center_line[:, 1], c="k")
            # plot data that will be update using the slider and store the lines
            lines[subplot_name] = {
                "past": axes[subplot_name].plot(
                    subplot_info["data"]["past"][:, 0],
                    subplot_info["data"]["past"][:, 1],
                    c=green,
                )[0],
                # at first we don't display references or predictions (only once we
                # activate the slider), so we just provide nan array with appropriate shape
                "ref": axes[subplot_name].plot(
                    np.full((Nf + 1,), np.nan), np.full((Nf + 1,), np.nan), c="cyan"
                )[0],
                "pred": axes[subplot_name].plot(
                    np.full((Nf + 1,), np.nan), np.full((Nf + 1,), np.nan), c=red
                )[0],
            }
            # set aspect ratio to be equal (because we display a map)
            axes[subplot_name].set_aspect("equal")
        else:
            # create axes
            axes[subplot_name] = plt.subplot2grid(
                gridshape, subplot_info["loc"], rowspan=1
            )
            # plot data that will be update using the slider and store the lines
            lines[subplot_name] = (
                {
                    "past": axes[subplot_name].plot(
                        dt * np.arange(subplot_info["data"]["past"].shape[0]),
                        subplot_info["data"]["past"],
                        c=green,
                    )[0],
                    "ref": axes[subplot_name].plot(np.full((Nf,), np.nan), c="cyan")[0],
                    "pred": axes[subplot_name].plot(np.full((Nf,), np.nan), c=red)[0],
                }
                if subplot_name not in {"T", "delta"}
                else {
                    "past": axes[subplot_name].step(
                        dt * np.arange(subplot_info["data"]["past"].shape[0]),
                        subplot_info["data"]["past"],
                        c=green,
                        where="post",
                    )[0],
                    "pred": axes[subplot_name].step(
                        np.arange(Nf), np.full((Nf,), np.nan), c=red, where="post"
                    )[0],
                }
            )

        # if we defined some, add labels to the axes
        if "xlabel" in subplot_info:
            axes[subplot_name].set_xlabel(subplot_info["xlabel"])
        if "ylabel" in subplot_info:
            axes[subplot_name].set_ylabel(subplot_info["ylabel"])

    fig.tight_layout()

    # save plot to file
    if output_file != "":
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    # define update function
    def update(it):
        # compute time vectors for past, reference and prediction of 1d plots
        t_past = dt * np.arange(it + 1)
        t_ref = dt * np.arange(it, it + Nf + 1)
        if it == Nsim - 1 and controller_failed:
            # we don't have any state or control predictions to plot
            t_pred = np.full((Nf + 1), np.nan)
        else:
            t_pred = dt * np.arange(it, it + Nf + 1)

        # plot everything
        for subplot_name, subplot_info in plot_data.items():
            if subplot_name == "XY":
                lines[subplot_name]["past"].set_data(
                    subplot_info["data"]["past"][: it + 1, 0],
                    subplot_info["data"]["past"][: it + 1, 1],
                )
                lines[subplot_name]["pred"].set_data(
                    subplot_info["data"]["pred"][it, :, 0],
                    subplot_info["data"]["pred"][it, :, 1],
                )
                all_points = subplot_info["data"]["pred"][it]
                if not controller_failed or it < Nsim - 1:
                    lines[subplot_name]["ref"].set_data(
                        subplot_info["data"]["ref"][it, :, 0],
                        subplot_info["data"]["ref"][it, :, 1],
                    )
                    all_points = np.concatenate(
                        (all_points, subplot_info["data"]["ref"][it])
                    )
            else:
                lines[subplot_name]["past"].set_data(
                    t_past, subplot_info["data"]["past"][: it + 1]
                )
                lines[subplot_name]["pred"].set_data(
                    t_pred, subplot_info["data"]["pred"][it]
                )
                if "ref" in subplot_info["data"]:
                    # we only plot reference for state variables
                    lines[subplot_name]["ref"].set_data(
                        t_ref, subplot_info["data"]["ref"][it]
                    )
                # recompute the ax.dataLim
                axes[subplot_name].relim()
                # update ax.viewLim using the new dataLim
                axes[subplot_name].autoscale_view()

    # create slider
    slider_ax = fig.add_axes([0.125, 0.02, 0.775, 0.03])
    slider = matplotlib.widgets.Slider(
        ax=slider_ax,
        label="sim iteration",
        valmin=0,
        valmax=Nsim - 1,
        valinit=Nsim - 1,
        valstep=1,
        valfmt="%d",
    )
    slider.on_changed(update)

    # show plot
    if show:
        plt.show()


def visualize_file(filename: str):
    data = np.load(filename)
    visualize(**data, output_file="bruh.png", show=True)


if __name__ == "__main__":
    # ocp_opts = AcadosOcpOptions()
    # ocp_opts.tf = Nf * dt
    # ocp_opts.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    # ocp_opts.nlp_solver_type = "SQP"
    # ocp_opts.nlp_solver_max_iter = 1
    # ocp_opts.hessian_approx = "EXACT"
    # ocp_opts.hpipm_mode = "ROBUST"
    # ocp_opts.integrator_type = "ERK"
    # ocp_opts.sim_method_num_stages = 4
    # ocp_opts.sim_method_num_steps = 1
    # ocp_opts.globalization = "MERIT_BACKTRACKING"
    # ocp_opts.print_level = 0
    closed_loop(
        controller=NMPCControllerIpopt(
            q_lon=10.0,
            q_lat=20.0,
            q_phi=50.0,
            q_v=20.0,
            r_T=0.01,
            r_delta=2.0,
            q_lon_f=1000.0,
            q_lat_f=1000.0,
            q_phi_f=500.0,
            q_v_f=1000.0,
        ),
        data_file="closed_loop_data.npz",
    )
    visualize_file("closed_loop_data.npz")
