# Copyright (c) 2024. Tudor Oancea
import os

import numpy as np
from acados_template import AcadosModel, AcadosSim, AcadosSimOpts, AcadosSimSolver
from icecream import ic


def generate_sim_solver(
    model: AcadosModel,
    opts: AcadosSimOpts,
    gen_code_dir: str,
    **kwargs,
) -> AcadosSimSolver:
    if not os.path.exists(gen_code_dir):
        os.makedirs(gen_code_dir)

    sim = AcadosSim()
    sim.model = model
    sim.solver_options = opts
    sim.code_export_directory = gen_code_dir + "/ihm2_fkin6_sim_gen_code"
    sim.parameter_values = np.ones(model.p.shape)
    return AcadosSimSolver(
        sim, json_file=gen_code_dir + "/ihm2_fkin6_sim.json", verbose=False, **kwargs
    )


default_sim_solver_opts = AcadosSimOpts()
default_sim_solver_opts.T = 0.01
default_sim_solver_opts.num_stages = 4
default_sim_solver_opts.num_steps = 1
default_sim_solver_opts.integrator_type = "IRK"
default_sim_solver_opts.collocation_type = "GAUSS_RADAU_IIA"
