# Copyright (c) 2023. Tudor Oancea
import numpy as np
from acados_template import AcadosOcpOptions, AcadosSimOpts

Nf = 50  # number of discretization steps
dt = 0.02  # sampling time

nsbx = 2
nsbx_e = 2

## Race car parameters
g = 9.81
m = 190.0
I_z = 110.0
l_R = 1.22
l_F = 1.22
L = 3.0
W = 1.5
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
