from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import numpy.typing as npt
from acados_template import AcadosOcpSolver
from networkx import omega
from strongpods import PODS


@PODS
class PacejkaCoefficients:
    b1: float
    b2: float
    b3: float
    c1: float
    d1: float
    d2: float
    e1: float
    e2: float
    e3: float
    B: float
    C: float
    D: float
    BCD: float

    @property
    def cornering_stiffness(self):
        return self.BCD


@PODS
class TireParams:
    radius: float
    inertia: float
    longitudinal_pacejka_coefficients: PacejkaCoefficients
    lateral_pacejka_coefficients: PacejkaCoefficients


@PODS
class CarGeometry:
    length: float
    width: float
    wheelbase: float
    track: float
    cog_to_rear_axle: float
    cog_to_front_axle: float
    cog_height: float


@PODS
class DrivetrainParams:
    C_m0: float
    C_r0: float
    C_r1: float
    C_r2: float


@PODS
class AerodynamicParams:
    downforce_coeff: float


@PODS
class ActuatorParams:
    wheel_torque_max: float
    total_torque_max: float
    steering_max: float
    steering_rate_max: float
    steering_time_constant: float


@PODS
class CarParams:
    mass: float
    yaw_inertia: float
    geometry: CarGeometry
    tire_params: TireParams
    drivetrain_params: DrivetrainParams
    aerodynamic_params: AerodynamicParams
    actuator_params: ActuatorParams


@PODS
class CarState:
    X: float
    Y: float
    phi: float
    v_x: float
    v_y: float
    r: float
    omega_FR: float
    omega_FL: float
    omega_RR: float
    omega_RL: float
    tau_FR: float
    tau_FL: float
    tau_RR: float
    tau_RL: float
    delta: float

    @property
    def position(self):
        return np.array([self.X, self.Y])

    @property
    def pose(self):
        return np.array([self.X, self.Y, self.phi])

    @property
    def v(self):
        return np.hypot(self.v_x, self.v_y)

    @property
    def T(self):
        return self.tau_FR + self.tau_FL + self.tau_RR + self.tau_RL


@PODS
class MotionPlan:
    X: npt.NDArray[np.float64]
    Y: npt.NDArray[np.float64]
    phi: npt.NDArray[np.float64]
    v_x: npt.NDArray[np.float64]
    v_y: npt.NDArray[np.float64]
    r: npt.NDArray[np.float64]


@PODS
class TrackingPlan:
    pass


class Controller(ABC):
    id: str
    car_params: CarParams

    @PODS
    class Config:
        horizon_size: int
        sampling_time: float

        @property
        def controller_type(self) -> type:
            return Controller

    config: Config

    def __init__(self, car_params: CarParams) -> None:
        super().__init__()
        self.car_params = car_params

    @abstractmethod
    def compute_control(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        pass

    @property
    def sampling_time(self):
        return self.config.sampling_time


class StanleyController(Controller):
    id = "stanley"

    @PODS
    class Config(Controller.Config):
        k_P: float
        k_I: float
        k_offset: float
        k_n: float
        k_psi: float
        k_kappa: float

    default_config = Config(
        k_P=90.0,
        k_I=20.0,
        k_offset=1.0,
        k_n=5.5,
        k_psi=1.8,
        k_kappa=1.0,
    )
    config: Config

    last_epsilons = deque(maxlen=400)
    T_max: float
    delta_max: float
    dt: float

    def __init__(
        self,
        car_params: CarParams,
        config: Config = default_config,
        T_max=500.0,
        delta_max=0.5,
    ) -> None:
        super().__init__(car_params=car_params)
        self.config = config
        self.T_max = min(T_max, self.car_params.actuator_params.total_torque_max)
        self.delta_max = min(delta_max, self.car_params.actuator_params.steering_max)

    def compute_control(
        self,
        n: float,
        psi: float,
        v_x: float,
        v_x_ref: float,
        kappa_ref: float,
    ) -> np.ndarray:
        # torque control
        epsilon = v_x_ref - v_x
        u_T = self.config.k_P * epsilon
        if len(self.last_epsilons) > 1:
            u_T += (
                self.config.k_I
                * (
                    np.sum(self.last_epsilons)
                    - self.last_epsilons[0]
                    - self.last_epsilons[-1]
                )
                * self.sampling_time
            )
        self.last_epsilons.append(epsilon)
        # steering control
        l_R = self.car_params.geometry.cog_to_rear_axle
        u_delta = self.config.k_kappa * np.arctan(
            2 * np.tan(np.arcsin(kappa_ref * l_R))
        )  # feedforward
        u_delta += -self.config.k_psi * psi  # heading error compensation
        u_delta += -np.arctan(
            self.config.k_n * n / (2.0 + v_x)
        )  # lateral error compensation
        # saturate control
        u_max = np.array([self.T_max, self.delta_max])
        return np.clip(np.array([u_T, u_delta]), -u_max, u_max)


class IHM2Controller(Controller):
    Nf: int
    dt: float
    s_target: float

    nx = 8
    nu = 2

    solver: AcadosOcpSolver

    x_pred: list[np.ndarray]
    u_pred: list[np.ndarray]

    def __init__(
        self,
        s_ref: np.ndarray,
        kappa_ref: np.ndarray,
        Nf: int = 40,
        dt: float = 1 / 20,
        s_target: float = 40.0,
        n_max: float = 2.0,
        v_x_max: float = 31.0,
        T_max: float = 500.0,
        delta_max: float = 0.5,
        T_dot_max: float = 1e6,
        delta_dot_max: float = 1.0,
        a_lat_max: float = 5.0,
        q_s: float = 1.0,
        q_n: float = 1.0,
        q_psi: float = 1.0,
        q_v_x: float = 1.0,
        q_v_y: float = 1.0,
        q_r: float = 1.0,
        q_T: float = 1.0,
        q_delta: float = 100.0,
        q_s_f: float = 1000.0,
        q_n_f: float = 100.0,
        q_psi_f: float = 100.0,
        q_v_x_f: float = 1.0,
        q_v_y_f: float = 1.0,
        q_r_f: float = 1.0,
        q_T_f: float = 1.0,
        q_delta_f: float = 100.0,
        q_T_dot: float = 0.0,
        q_delta_dot: float = 500.0,
    ) -> None:
        super().__init__()
        self.Nf = Nf
        self.dt = dt
        self.s_target = s_target

        model = get_acados_model_from_explicit_dynamics(
            name="ihm2_fkin6",
            continuous_model_fn=fkin6_model,
            x=MX.sym("x", self.nx),
            u=MX.sym("u", self.nu),
            p=MX.sym("p", 3 * 2 * NUMBER_SPLINE_INTERVALS),
        )
        ocp = get_acados_ocp(
            model, Nf, n_max, v_x_max, T_max, delta_max, T_dot_max, delta_dot_max
        )
        ocp_opts = AcadosOcpOptions()
        ocp_opts.tf = Nf * dt
        ocp_opts.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp_opts.nlp_solver_type = "SQP"
        ocp_opts.nlp_solver_max_iter = 2
        ocp_opts.hessian_approx = "GAUSS_NEWTON"
        ocp_opts.hpipm_mode = "SPEED_ABS"
        ocp_opts.integrator_type = "IRK"
        ocp_opts.sim_method_num_stages = 4
        ocp_opts.sim_method_num_steps = 1
        ocp_opts.globalization = "MERIT_BACKTRACKING"
        ocp_opts.print_level = 0
        self.solver = get_acados_solver(ocp, ocp_opts, "generated")

        # initialize prediction arrays ############################################################
        self.x_pred = [
            np.array([-6.0 + i * dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            for i in range(Nf + 1)
        ]
        self.u_pred = [np.array([model_bounds.T_max, 0.0]) for i in range(Nf)]

        # set all parameters for the mpc solver ###############################################
        p = np.append(s_ref, kappa_ref)
        for i in range(Nf + 1):
            # set parameters
            self.solver.set(i, "p", p)
            # set cost weights
            if i < Nf:
                self.solver.cost_set(
                    i,
                    "W",
                    np.diag(
                        np.array(
                            [
                                q_s,
                                q_n,
                                q_psi,
                                q_v_x,
                                q_v_y,
                                q_r,
                                q_T,
                                q_delta,
                                q_T,
                                q_delta,
                                q_T_dot,
                                q_delta_dot,
                            ]
                        )
                    ),
                )
            else:
                self.solver.cost_set(
                    i,
                    "W",
                    np.diag(
                        np.array(
                            [
                                q_s_f,
                                q_n_f,
                                q_psi_f,
                                q_v_x_f,
                                q_v_y_f,
                                q_r_f,
                                q_T_f,
                                q_delta_f,
                            ]
                        )
                    ),
                )

    def compute_control(self, x: np.ndarray) -> np.ndarray | None:
        # set current state
        self.solver.set(0, "lbx", x)
        self.solver.set(0, "ubx", x)

        # update reference
        s0 = x[0]
        for j in range(self.Nf):
            self.solver.set(
                j,
                "yref",
                np.array(
                    [s0 + self.s_target * j / self.Nf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ),
            )
        self.solver.set(
            self.Nf, "yref", np.array([s0 + self.s_target, 0, 0, 0, 0, 0, 0, 0])
        )

        # set initial guess
        for j in range(self.Nf - 1):
            self.solver.set(j, "x", self.x_pred[j + 1])
            self.solver.set(j, "u", self.u_pred[j + 1])
        self.solver.set(self.Nf - 1, "x", self.x_pred[self.Nf])
        self.solver.set(self.Nf, "x", self.x_pred[self.Nf])
        self.solver.set(self.Nf - 1, "u", np.zeros(2))

        # solve ocp
        status = self.solver.solve()
        if status not in {0, 2}:
            print()
            return None

        # extract prediction and append control input
        self.x_pred = [self.solver.get(i, "x") for i in range(self.Nf + 1)]
        self.u_pred = [self.solver.get(i, "u") for i in range(self.Nf)]

        return self.solver.get(0, "u")
