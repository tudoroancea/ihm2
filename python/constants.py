import numpy as np

__all__ = [
    "g",
    "m",
    "I_z",
    "z_CG",
    "front_axle_track",
    "rear_axle_track",
    "l_R",
    "l_F",
    "wheelbase",
    "front_axle_track",
    "rear_axle_track",
    "axle_track",
    "L",
    "W",
    "C_m0",
    "C_r0",
    "C_r1",
    "C_r2",
    "b1s",
    "b2s",
    "b3s",
    "c1s",
    "d1s",
    "d2s",
    "e1s",
    "e2s",
    "e3s",
    "b1a",
    "b2a",
    "c1a",
    "d1a",
    "d2a",
    "e1a",
    "e2a",
    "BCDs",
]

g = 9.81  # gravity
m = 230.0  # mass
I_z = 137.583  # yaw moment of inertia
z_CG = 0.295  # height of center of gravity
front_axle_track = rear_axle_track = axle_track = 1.24  # wheelbase
l_R = 0.7853  # distance from CoG to rear axle
l_F = 0.7853  # distance from CoG to front axle
wheelbase = 1.5706  # distance between the two axles
L = 3.19  # length of the car
W = 1.55  # width of the car

# drivetrain parameters (simplified)
C_m0 = 4.950
C_r0 = 297.030
C_r1 = 16.665
C_r2 = 0.6784

# Pacejka base parameters
b1s = -6.75e-6
b2s = 1.35e-1
b3s = 1.2e-3
c1s = 1.86
d1s = 1.12e-4
d2s = 1.57
e1s = -5.38e-6
e2s = 1.11e-2
e3s = -4.26
b1a = 3.79e1
b2a = 5.28e2
c1a = 1.57
d1a = -2.03e-4
d2a = 1.77
e1a = -2.24e-3
e2a = 1.81

# Pacejka parameters (constant version)
static_weight = 0.5 * m * g * l_F / wheelbase
BCDs = (b1s * static_weight**2 + b2s * static_weight) * np.exp(-b3s * static_weight)
Cs = c1s
Ds = d1s * static_weight + d2s
Es = e1s * static_weight**2 + e2s * static_weight + e3s
Bs = BCDs / (Cs * Ds)
BCDa = b1a * np.sin(2 * np.arctan(static_weight / b2a))
Ca = c1a
Da = d1a * static_weight + d2a
Ea = e1a * static_weight + e2a
Ba = BCDa / (Ca * Da)
from icecream import ic

ic(
    Bs,
    Cs,
    Ds,
    Es,
)

# wheel parameters (ony used in dyn10 model)
R_w = 0.20809  # wheel radius
I_w = 0.3  # wheel inertia
k_d = 0.17  #
k_s = 15.0  #

# time constants of actuators
t_T = 1e-3  # time constant for throttle actuator
t_delta = 0.02  # time constant for steering actuator

# aerodynamic parameters
C_downforce = 3.96864

# torque vectoring gains
K_tv = 300.0
