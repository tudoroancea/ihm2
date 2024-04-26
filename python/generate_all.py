# Copyright (c) 2024. Tudor Oancea
import os
from time import perf_counter

from acados_template import AcadosOcpOptions, AcadosSimOpts
from casadi import SX
from motion_planning import NUMBER_SPLINE_INTERVALS, generate_track_data_file
from mpc import ModelBounds, get_acados_ocp, get_acados_solver
from sim import generate_sim_solver

from python.models import (
    dyn6_model,
    fkin6_model,
    get_acados_model_from_explicit_dynamics,
    get_acados_model_from_implicit_dynamics,
    kin6_model,
)


def main():
    print("**************************************************")
    print("* Generating acados sim solvers ******************")
    print("**************************************************\n")

    gen_code_dir = "generated"
    if not os.path.exists(gen_code_dir):
        os.makedirs(gen_code_dir)

    sim_solver_opts = AcadosSimOpts()
    sim_solver_opts.T = 0.01
    sim_solver_opts.num_stages = 4
    sim_solver_opts.num_steps = 1
    sim_solver_opts.integrator_type = "IRK"
    sim_solver_opts.collocation_type = "GAUSS_RADAU_IIA"

    # kin6 model
    t0 = perf_counter()
    generate_sim_solver(
        get_acados_model_from_explicit_dynamics(
            "ihm2_kin6", kin6_model, SX.sym("x", 8), SX.sym("u", 2), SX.sym("p", 0)
        ),
        sim_solver_opts,
        gen_code_dir + "/ihm2_kin6_sim_gen_code",
        gen_code_dir + "/ihm2_kin6_sim.json",
    )
    t1 = perf_counter()
    print(f"Generating Kin6 sim solver took: {t1 - t0:.3f} s")

    # dyn6 model
    t0 = perf_counter()
    generate_sim_solver(
        get_acados_model_from_implicit_dynamics(
            "ihm2_dyn6", dyn6_model, SX.sym("x", 8), SX.sym("u", 2), SX.sym("p", 0)
        ),
        sim_solver_opts,
        gen_code_dir + "/ihm2_dyn6_sim_gen_code",
        gen_code_dir + "/ihm2_dyn6_sim.json",
    )
    t1 = perf_counter()
    print(f"Generating Dyn6 sim solver took: {t1 - t0:.3f} s")

    print("")

    print("**************************************************")
    print("* Generating track data **************************")
    print("**************************************************\n")

    for track_name in [
        "fsds_competition_1",
        "fsds_competition_2",
        "fsds_competition_3",
    ]:
        t0 = perf_counter()
        generate_track_data_file(
            track_name, gen_code_dir + "/track_data/" + track_name + ".csv"
        )
        t1 = perf_counter()
        print(f"Generation of track data for track {track_name} took {t1-t0} seconds.")

    print("")

    print("**************************************************")
    print("* Generating acados mpc solvers ******************")
    print("**************************************************\n")

    start = perf_counter()
    ocp = get_acados_ocp(
        model=get_acados_model_from_explicit_dynamics(
            name="ihm2_fkin6",
            continuous_model_fn=fkin6_model,
            x=SX.sym("x", 8),
            u=SX.sym("u", 2),
            p=SX.sym("p", NUMBER_SPLINE_INTERVALS * 2),
        ),
        Nf=20,
        model_bounds=ModelBounds(
            n_max=2.5,
            v_x_min=0.0,
            v_x_max=31.0,
            T_max=500.0,
            delta_max=0.5,
            T_dot_max=1e6,
            delta_dot_max=1.0,
            a_lat_max=5.0,
        ),
    )
    opts = AcadosOcpOptions()
    opts.tf = 1.0
    opts.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    opts.nlp_solver_type = "SQP"
    opts.nlp_solver_max_iter = 10
    opts.hessian_approx = "GAUSS_NEWTON"
    opts.hpipm_mode = "ROBUST"
    opts.integrator_type = "IRK"
    opts.sim_method_num_stages = 4
    opts.sim_method_num_steps = 1
    get_acados_solver(ocp, opts, gen_code_dir)
    print(f"Generation of MPC solver took {perf_counter() - start} seconds.\n")


if __name__ == "__main__":
    main()
