import numpy as np
from casadi import MX, SX, exp, tanh

__all__ = [
    "smooth_sgn",
    "smooth_dev",
    "smooth_abs",
    "smooth_abs_nonzero",
    "teds_projection",
    "wrap_to_pi",
    "unwrap_to_pi",
]

####################################################################################################
# modelling utilities
####################################################################################################


def smooth_dev(x: SX | MX) -> SX | MX:
    return x + 1e-6 * exp(-x * x)


def smooth_sgn(x: SX | MX) -> SX | MX:
    return tanh(1e1 * x)


def smooth_abs(x: SX | MX) -> SX | MX:
    return smooth_sgn(x) * x


def smooth_abs_nonzero(x: SX | MX):
    return smooth_abs(x) + 1e-3 * exp(-x * x)


def teds_projection(x: np.ndarray | float, a: float):
    """Projection of x onto the interval [a, a + 2*pi)"""
    return np.mod(x - a, 2 * np.pi) + a


def wrap_to_pi(x: np.ndarray | float):
    """Wrap angles to [-pi, pi)"""
    return teds_projection(x, -np.pi)


def unwrap_to_pi(x: np.ndarray | float):
    """remove discontinuities caused by wrapToPi"""
    diffs = np.diff(x)
    diffs[diffs > 1.5 * np.pi] -= 2 * np.pi
    diffs[diffs < -1.5 * np.pi] += 2 * np.pi
    return np.insert(x[0] + np.cumsum(diffs), 0, x[0])
