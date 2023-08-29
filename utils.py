import numpy as np


def teds_projection(x, a):
    """Projection of x onto the interval [a, a + 2*pi)"""
    return np.mod(x - a, 2 * np.pi) + a


def wrapToPi(x):
    """Wrap angles to [-pi, pi)"""
    return teds_projection(x, -np.pi)


def unwrapToPi(x):
    # remove discontinuities caused by wrapToPi
    diffs = np.diff(x)
    diffs[diffs > 1.5 * np.pi] -= 2 * np.pi
    diffs[diffs < -1.5 * np.pi] += 2 * np.pi
    return np.insert(x[0] + np.cumsum(diffs), 0, x[0])
