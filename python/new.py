from constants import l_R, l_F, t_delta
import numpy as np
from casadi import (
    SX,
    Function,
    jacobian,
    vertcat,
    mtimes,
    nlpsol,
    inf,
    norm_2,
    atan,
    atan2,
    tan,
)


n = SX.sym("n")
psi = SX.sym("psi")
v = SX.sym("v")
kappa_ref = SX.sym("kappa_ref")
delta = SX.sym("delta")
u_delta = SX.sym("u_delta")

x = vertcat(n, psi, delta)
u = u_delta
p = vertcat(v, kappa_ref)

beta = atan(0.5 * tan(delta))
cont_dyn = vertcat(
    v * sin(psi + beta),
    v * sin(beta) / l_R - kappa_ref * v * cos(psi + beta) / (1 - kappa_ref * n),
    (u_delta - delta) / t_delta,
)
