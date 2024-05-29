import numpy as np
from strongpods import PODS
from acados_template import AcadosSimSolver, AcadosSimOpts
from abc import ABC, abstractmethod
from typing import Callable
from enum import Enum, auto

# coordinate system -> c/f
# kinematic/dynamic -> kin/dyn
# DOFs -> 4/6/10
# bicycle/4-wheel -> bicycle/fourwheels
# additional_flags ->
models = {
    "ckin4",  # cartesian kinematic bicycle model with 4 DOFs (X,Y,phi,v)
    "ckin6",  # cartesian kinematic bicycle model with 6 DOFs (X,Y,phi,v_x,v_y,r)
    "fkin4",  # frenet kinematic bicycle model with 4 DOFs (s,n,psi,v)
    "fkin6",  # frenet kinematic bicycle model with 6 DOFs (s,n,psi,v_x,v_y,r)
    "cdyn6",  # cartesian dynamic bicycle model with 6 DOFs (X,Y,phi,v_x,v_y,r)
    "cdyn6_4wheels"  # cartesian dynamic 4-wheel model with 6 DOFs (X,Y,phi,v_x,v_y,r)
    "fdyn6",  # frenet dynamic bicycle model with 6 DOFs (X,Y,phi,v_x,v_y,r)
    "cdyn10",  # cartesian dynamic bicycle model with 10 DOFs (X,Y,phi,v_x,v_y,r,omega_FL,omega_FR,omega_RL,omega_RR)
}


@PODS
class SimulatorConfig:
    sampling_time: float
    integrator_type: str = "IRK"
    colloaction_type: str = "GAUSS_RADAU_IIA"


class SimulatorModel(Enum):
    KIN6 = auto()
    DYN6 = auto()
    KIN6_DYN6 = auto()
    DYN10 = auto()


class Simulator(ABC):
    config: SimulatorConfig

    # def simulate(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    @abstractmethod
    def simulate(self, X, Y, phi, v_x, v_y, r, T, delta) -> np.ndarray:
        pass


class SingleModelSimulator(Simulator):
    solver: AcadosSimSolver
    pass


class MultiModelSimulator(Simulator):
    solvers: dict[str, AcadosSimSolver]
    condition: Callable[[np.ndarray], str]

    def __init__(self, condition: Callable[[np.ndarray], str]) -> None:
        pass

    def simulate(self, X, Y, phi, v_x, v_y, r, T, delta) -> np.ndarray:
        model = self.condition(np.sqrt(v_x**2 + v_y**2))
        return self.solvers[model].simulate(X, Y, phi, v_x, v_y, r, T, delta)


# sim_opts = AcadosSimOpts()
# sim_opts.T = dt
# sim_opts.num_stages = 4
# sim_opts.num_steps = 100
# sim_opts.integrator_type = "IRK"
# sim_opts.collocation_type = "GAUSS_RADAU_IIA"
