from time import perf_counter
import numpy as np
from casadi import interpolant
from acados_template import AcadosOcpOptions, AcadosSimOpts

from generate_acaods_interface import (
    generate_model,
    generate_ocp_solver,
    generate_sim_solver,
)
from scripts.gen_track_file import generate_track_file


def main():
    # generate ocp solvers for each track ================================================
    Nf = 40  # number of discretization steps
    dt = 0.05  # sampling time
    ocp_solver_opts = AcadosOcpOptions()
    ocp_solver_opts.tf = Nf * dt
    ocp_solver_opts.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp_solver_opts.nlp_solver_type = "SQP_RTI"
    ocp_solver_opts.hessian_approx = "GAUSS_NEWTON"
    ocp_solver_opts.hpipm_mode = "SPEED_ABS"
    ocp_solver_opts.integrator_type = "ERK"
    ocp_solver_opts.sim_method_num_stages = 4
    ocp_solver_opts.sim_method_num_steps = 1
    t0 = perf_counter()
    for track in [
        "fsds_competition_1",
        "fsds_competition_2",
        "fsds_competition_3",
    ]:
        file = f"src/ihm2/tracks/{track}.csv"
        # generate track file
        generate_track_file(track, file, plot=False)
        # load it again and extract reference data for model
        data = np.loadtxt(file, delimiter=",", skiprows=1)
        s_ref = data[:, 0]
        kappa_ref = data[:, 4]
        right_width = data[:, 5]
        left_width = data[:, 6]
        # create B-spline interpolants for kappa, right_width, left_width
        kappa_ref_expr = interpolant("kappa_ref", "linear", [s_ref], kappa_ref)
        right_width_expr = interpolant("right_width", "linear", [s_ref], right_width)
        left_width_expr = interpolant("left_width", "linear", [s_ref], left_width)
        # generate the ocp solver
        generate_ocp_solver(
            model=generate_model(
                "fkin4",
                kappa_ref_expr,
                right_width_expr,
                left_width_expr,
                f"ihm2_fkin4_{track}",
            ),
            opts=ocp_solver_opts,
            Nf=Nf,
            Q=np.diag([1e-1, 5e-7, 5e-7, 5e-7, 5e-2, 2.5e-2]),
            R=np.diag([5e-1, 1e-1]),
            Qe=np.diag([1e-1, 2e-10, 2e-10, 2e-10, 1e-4, 4e-5]),
            code_export_directory=f"src/ihm2/ocp/{track}",
            use_cmake=False,
        )
    t1 = perf_counter()
    print(f"OCP solver generation time: {t1 - t0:.3f} s")

    # generate sim solvers for each model ===================================
    sim_solver_opts = AcadosSimOpts()
    sim_solver_opts.T = 0.01
    sim_solver_opts.num_stages = 4
    sim_solver_opts.num_steps = 10
    sim_solver_opts.integrator_type = "IRK"
    sim_solver_opts.collocation_type = "GAUSS_RADAU_IIA"
    t0 = perf_counter()
    for model_id in ["kin4", "dyn6"]:
        generate_sim_solver(
            generate_model(model_id),
            sim_solver_opts,
            f"src/ihm2/sim/{model_id}",
            use_cmake=False,
        )
    t1 = perf_counter()
    print(f"Sim solvers generation time: {t1 - t0:.3f} s")


if __name__ == "__main__":
    main()
