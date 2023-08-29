import numpy as np
from acados_template import AcadosOcpOptions, AcadosSimOpts

Nf = 50  # number of discretization steps
dt = 0.02  # sampling time
Tsim = 10.0  # maximum simulation time[s]
Nsim = int(Tsim / dt) + 1
sref_N = 3.0  # reference for final reference progress

nh = 5
nh_e = 2
nsbx = 2
nsbx_e = 2
nsh = nh
nsh_e = nh_e
ns = nsh + nsbx
ns_e = nsh_e + nsbx_e

## Race car parameters
m = 190.0
I_z = 110.0
l_R = 1.22
l_F = 1.22
L = 3.0
W = 1.5
C1 = l_R / (l_R + l_F)
C2 = 1 / (l_R + l_F)
C_m0 = 2000.0
C_m1 = 43.0
C_r0 = 180.0
C_r2 = 0.7
B = 10.0
C = 1.38
D = 1.609
t_T = 1e-3  # time constant for throttle actuator
t_delta = 0.02  # time constant for steering actuator

# model bounds
v_min = 0.0
v_max = 31.0
alpha_min = -np.pi / 2
alpha_max = np.pi / 2
n_min = -1.5
n_max = 1.5
T_min = -2000.0
T_max = 2000.0
delta_min = -0.5
delta_max = 0.5
T_dot_min = -1e6
T_dot_max = 1e6
delta_dot_min = -2.0
delta_dot_max = 2.0
a_lat_min = -5.0
a_lat_max = 5.0

# initial state
x0 = np.array([-6.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# costs
# x = (s, n, alpha, v, T, delta), u = (Tdot, deltadot)
Q = np.diag([1e-1, 5e-7, 5e-7, 5e-7, 5e-2, 2.5e-2])
R = np.diag([5e-1, 1e-1])
Qe = np.diag([1e-1, 2e-10, 2e-10, 2e-10, 1e-4, 4e-5])
zl = 100 * np.ones((ns,))
zu = 100 * np.ones((ns,))
Zl = 100 * np.ones((ns,))
Zu = 100 * np.ones((ns,))
zl_e = 100 * np.ones((ns_e,))
zu_e = 100 * np.ones((ns_e,))
Zl_e = 100 * np.ones((ns_e,))
Zu_e = 100 * np.ones((ns_e,))

ocp_solver_opts = AcadosOcpOptions()
ocp_solver_opts.tf = Nf * dt
ocp_solver_opts.qp_solver = "PARTIAL_CONDENSING_HPIPM"
ocp_solver_opts.nlp_solver_type = "SQP_RTI"
ocp_solver_opts.hessian_approx = "GAUSS_NEWTON"
ocp_solver_opts.hpipm_mode = "SPEED_ABS"
ocp_solver_opts.integrator_type = "ERK"
ocp_solver_opts.sim_method_num_stages = 4
ocp_solver_opts.sim_method_num_steps = 1

sim_solver_opts = AcadosSimOpts()
sim_solver_opts.T = dt
sim_solver_opts.num_stages = 4
sim_solver_opts.num_steps = 10
sim_solver_opts.integrator_type = "IRK"
sim_solver_opts.collocation_type = "GAUSS_RADAU_IIA"
