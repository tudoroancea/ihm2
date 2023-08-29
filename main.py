# Copyright (c) 2023. Tudor Oancea
import numpy as np
from casadi import *
import matplotlib.pyplot as plt
from icecream import ic
from model import *


def main():
    data = np.loadtxt("data/fsds_competition_2.csv", delimiter=",", skiprows=1)
    s_ref = data[:, 0]
    X_ref = data[:, 1]
    Y_ref = data[:, 2]
    phi_ref = data[:, 3]
    kappa_ref = data[:, 4]
    right_width = data[:, 5]
    left_width = data[:, 6]
    total_length = s_ref[-1] + np.hypot(X_ref[-1] - X_ref[0], Y_ref[-1] - Y_ref[0])

    # duplicate data before and after the loop to avoid discontinuities
    s_ref_extended = np.hstack(
        (
            s_ref - total_length,
            s_ref,
            s_ref + total_length,
        )
    )
    kappa_ref_extended = np.hstack((kappa_ref, kappa_ref, kappa_ref))
    right_width_extended = np.hstack((right_width, right_width, right_width))
    left_width_extended = np.hstack((left_width, left_width, left_width))
    kappa_ref_expr = interpolant(
        "kappa_ref", "bspline", [s_ref_extended], kappa_ref_extended
    )
    right_width_expr = interpolant(
        "right_width", "bspline", [s_ref_extended], right_width_extended
    )
    left_width_expr = interpolant(
        "left_width", "bspline", [s_ref_extended], left_width_extended
    )
    model = gen_kin_model(kappa_ref_expr, right_width_expr, left_width_expr)
    ocp_solver = get_ocp_solver(model)
    sim_solver = get_sim_solver(model)

    # run 1 lap simulation =====================================================
    x_init = np.array([-6.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # x_sim = np.zeros((Nsim, ))
    # x_sim[0, :] = x_init
    # u_sim = np.zeros((Nsim, nu))


if __name__ == "__main__":
    main()
