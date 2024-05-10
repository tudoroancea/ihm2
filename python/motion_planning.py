# Copyright (c) 2024. Tudor Oancea
import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from constants import l_R
from qpsolvers import available_solvers, solve_qp
from scipy.sparse import csc_array
from scipy.sparse import eye as speye
from scipy.sparse import kron as spkron
from track_database import Track
from track_database.utils import plot_cones
from utils import unwrap_to_pi

__all__ = [
    "NUMBER_SPLINE_INTERVALS",
    "MotionPlan",
    "offline_motion_plan",
    "plot_motion_plan",
    "generate_track_data_file",
    "triple_motion_plan_ref",
]
NUMBER_SPLINE_INTERVALS = 500


def fit_spline(
    path: np.ndarray,
    curv_weight: float = 1.0,
    return_errs: bool = False,
    qp_solver: str = "proxqp",
) -> tuple[np.ndarray, np.ndarray]:
    """
    computes the coefficients of each spline portion of the path.
    > Note: the path is assumed to be closed but the first and last points are NOT the same.

    :param path: Nx2 array of points
    :param curv_weight: weight of the curvature term in the cost function
    :return_errs:
    :qp_solver:
    :returns p_X, p_Y: Nx4 arrays of coefficients of the splines in the x and y directions
                       (each row correspond to a_i, b_i, c_i, d_i coefficients of the i-th spline portion)
    """
    assert (
        len(path.shape) == 2 and path.shape[1] == 2
    ), f"path must have shape (N,2) but has shape {path.shape}"
    assert (
        qp_solver in available_solvers
    ), f"qp_solver must be one of the available solvers: {available_solvers}"

    # precompute all the QP data
    N = path.shape[0]
    delta_s = np.linalg.norm(path[1:] - path[:-1], axis=1)
    delta_s = np.append(delta_s, np.linalg.norm(path[0] - path[-1]))
    rho = np.zeros(N)
    rho[:-1] = delta_s[:-1] / delta_s[1:]
    rho[-1] = delta_s[-1] / delta_s[0]
    IN = speye(N, format="csc")
    A = spkron(
        IN,
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 0.0, 2.0, 6.0],
            ]
        ),
        format="csc",
    ) + csc_array(
        (
            np.concatenate((-np.ones(N), -rho, -2 * rho**2)),
            (
                np.concatenate(
                    (3 * np.arange(N), 1 + 3 * np.arange(N), 2 + 3 * np.arange(N))
                ),
                np.concatenate(
                    (
                        np.roll(4 * np.arange(N), -1),
                        np.roll(1 + 4 * np.arange(N), -1),
                        np.roll(2 + 4 * np.arange(N), -1),
                    )
                ),
            ),
        ),
        shape=(3 * N, 4 * N),
    )
    B = spkron(IN, np.array([[1.0, 0.0, 0.0, 0.0]]), format="csc")
    C = csc_array(
        (
            np.concatenate((2 / np.square(delta_s), 6 / np.square(delta_s))),
            (
                np.concatenate((np.arange(N), np.arange(N))),
                np.concatenate((2 + 4 * np.arange(N), 3 + 4 * np.arange(N))),
            ),
        ),
        shape=(N, 4 * N),
    )
    P = B.T @ B + curv_weight * C.T @ C + 1e-10 * speye(4 * N, format="csc")
    q = -B.T @ path
    b = np.zeros(3 * N)

    if qp_solver in {"quadprog", "ecos"}:
        A = A.toarray()
        B = B.toarray()
        C = C.toarray()
        P = P.toarray()

    # solve the QP for X and Y separately
    p_X = solve_qp(P=P, q=q[:, 0], A=A, b=b, solver=qp_solver)
    p_Y = solve_qp(P=P, q=q[:, 1], A=A, b=b, solver=qp_solver)

    # compute interpolation error on X and Y
    X_err = B @ p_X - path[:, 0]
    Y_err = B @ p_Y - path[:, 1]

    # reshape to (N,4) arrays
    p_X = np.reshape(p_X, (N, 4))
    p_Y = np.reshape(p_Y, (N, 4))

    if return_errs:
        return p_X, p_Y, X_err, Y_err
    else:
        return p_X, p_Y


def check_spline_coeffs_dims(coeffs_X: np.ndarray, coeffs_Y: np.ndarray):
    assert (
        len(coeffs_X.shape) == 2 and coeffs_X.shape[1] == 4
    ), f"coeffs_X must have shape (N,4) but has shape {coeffs_X.shape}"
    assert (
        len(coeffs_Y.shape) == 2 and coeffs_Y.shape[1] == 4
    ), f"coeffs_Y must have shape (N,4) but has shape {coeffs_Y.shape}"
    assert (
        coeffs_X.shape[0] == coeffs_Y.shape[0]
    ), f"coeffs_X and coeffs_Y must have the same length but have lengths {coeffs_X.shape[0]} and {coeffs_Y.shape[0]}"


def compute_spline_interval_lengths(
    coeffs_X: np.ndarray, coeffs_Y: np.ndarray, no_interp_points=100
):
    """
    computes the lengths of each spline portion of the path.
    > Note: Here the closeness of the part does not matter, it is contained in the coefficients

    :param coeff_X: Nx4 array of coefficients of the splines in the x direction (as returned by calc_splines)
    :param coeff_Y: Nx4 array of coefficients of the splines in the y direction (as returned by calc_splines)
    :param delta_s: number of points to use on each spline portion for the interpolation
    """
    check_spline_coeffs_dims(coeffs_X, coeffs_Y)

    N = coeffs_X.shape[0]

    t_steps = np.linspace(0.0, 1.0, no_interp_points)[np.newaxis, :]
    interp_points = np.zeros((no_interp_points, N, 2))

    interp_points[:, :, 0] = coeffs_X[:, 0]
    interp_points[:, :, 1] = coeffs_Y[:, 0]

    coeffs_X = coeffs_X[:, np.newaxis, :]
    coeffs_Y = coeffs_Y[:, np.newaxis, :]

    interp_points = interp_points.transpose(1, 0, 2)

    interp_points[:, :, 0] += coeffs_X[:, :, 1] @ t_steps
    interp_points[:, :, 0] += coeffs_X[:, :, 2] @ np.power(t_steps, 2)
    interp_points[:, :, 0] += coeffs_X[:, :, 3] @ np.power(t_steps, 3)

    interp_points[:, :, 1] += coeffs_Y[:, :, 1] @ t_steps
    interp_points[:, :, 1] += coeffs_Y[:, :, 2] @ np.power(t_steps, 2)
    interp_points[:, :, 1] += coeffs_Y[:, :, 3] @ np.power(t_steps, 3)

    delta_s = np.sum(
        np.sqrt(np.sum(np.power(np.diff(interp_points, axis=1), 2), axis=2)), axis=1
    )
    assert delta_s.shape == (N,), f"{delta_s.shape}"
    return delta_s


def uniformly_sample_spline(
    coeffs_X: np.ndarray,
    coeffs_Y: np.ndarray,
    delta_s: np.ndarray,
    n_samples: int,
):
    """
    uniformly n_samples equidistant points along the path defined by the splines.
    The first point will always be the initial point of the first spline portion, and
    the last point will NOT be the initial point of the first spline portion.

    :param coeffs_X: Nx4 array of coefficients of the splines in the x direction (as returned by calc_splines)
    :param coeffs_Y: Nx4 array of coefficients of the splines in the y direction (as returned by calc_splines)
    :param spline_lengths: N array of lengths of the spline portions (as returned by calc_spline_lengths)
    :param n_samples: number of points to sample

    :return XY_interp: n_samplesx2 array of points along the path
    :return idx_interp: n_samples array of indices of the spline portions that host the points
    :return t_interp: n_samples array of t values of the points within their respective spline portions
    :return s_interp: n_samples array of distances along the path of the points
    """
    s = np.cumsum(delta_s)
    s_interp = np.linspace(0.0, s[-1], n_samples, endpoint=False)

    # find the spline that hosts the current interpolation point
    idx_interp = np.argmax(s_interp[:, np.newaxis] < s, axis=1)

    t_interp = np.zeros(n_samples)  # save t values
    X_interp = np.zeros(n_samples)  # raceline coords
    Y_interp = np.zeros(n_samples)  # raceline coords

    # get spline t value depending on the progress within the current element
    t_interp[idx_interp > 0] = (
        s_interp[idx_interp > 0] - s[idx_interp - 1][idx_interp > 0]
    ) / delta_s[idx_interp][idx_interp > 0]
    t_interp[idx_interp == 0] = s_interp[idx_interp == 0] / delta_s[0]

    # calculate coords
    X_interp = (
        coeffs_X[idx_interp, 0]
        + coeffs_X[idx_interp, 1] * t_interp
        + coeffs_X[idx_interp, 2] * np.power(t_interp, 2)
        + coeffs_X[idx_interp, 3] * np.power(t_interp, 3)
    )

    Y_interp = (
        coeffs_Y[idx_interp, 0]
        + coeffs_Y[idx_interp, 1] * t_interp
        + coeffs_Y[idx_interp, 2] * np.power(t_interp, 2)
        + coeffs_Y[idx_interp, 3] * np.power(t_interp, 3)
    )

    return X_interp, Y_interp, idx_interp, t_interp, s_interp


def get_heading(
    coeffs_X: np.ndarray,
    coeffs_Y: np.ndarray,
    idx_interp: np.ndarray,
    t_interp: np.ndarray,
) -> np.ndarray:
    """
    analytically computes the heading and the curvature at each point along the path
    specified by idx_interp and t_interp.

    :param coeffs_X: Nx4 array of coefficients of the splines in the x direction (as returned by calc_splines)
    :param coeffs_Y: Nx4 array of coefficients of the splines in the y direction (as returned by calc_splines)
    :param idx_interp: n_samples array of indices of the spline portions that host the points
    :param t_interp: n_samples array of t values of the points within their respective spline portions
    """
    check_spline_coeffs_dims(coeffs_X, coeffs_Y)

    # we don't divide by delta_s[idx_interp] here because this term will cancel out
    # in arctan2 either way
    x_d = (
        coeffs_X[idx_interp, 1]
        + 2 * coeffs_X[idx_interp, 2] * t_interp
        + 3 * coeffs_X[idx_interp, 3] * np.square(t_interp)
    )
    y_d = (
        coeffs_Y[idx_interp, 1]
        + 2 * coeffs_Y[idx_interp, 2] * t_interp
        + 3 * coeffs_Y[idx_interp, 3] * np.square(t_interp)
    )
    phi = np.arctan2(y_d, x_d)

    return phi


def get_curvature(
    coeffs_X: np.ndarray,
    coeffs_Y: np.ndarray,
    idx_interp: np.ndarray,
    t_interp: np.ndarray,
) -> np.ndarray:
    # same here with the division by delta_s[idx_interp] ** 2
    x_d = (
        coeffs_X[idx_interp, 1]
        + 2 * coeffs_X[idx_interp, 2] * t_interp
        + 3 * coeffs_X[idx_interp, 3] * np.square(t_interp)
    )
    y_d = (
        coeffs_Y[idx_interp, 1]
        + 2 * coeffs_Y[idx_interp, 2] * t_interp
        + 3 * coeffs_Y[idx_interp, 3] * np.square(t_interp)
    )
    x_dd = 2 * coeffs_X[idx_interp, 2] + 6 * coeffs_X[idx_interp, 3] * t_interp
    y_dd = 2 * coeffs_Y[idx_interp, 2] + 6 * coeffs_Y[idx_interp, 3] * t_interp
    kappa = (x_d * y_dd - y_d * x_dd) / np.power(x_d**2 + y_d**2, 1.5)
    return kappa


@dataclass
class MotionPlan:
    s_ref: np.ndarray
    X_ref: np.ndarray
    Y_ref: np.ndarray
    phi_ref: np.ndarray
    kappa_ref: np.ndarray
    right_widths: np.ndarray
    left_widths: np.ndarray
    lap_length: float


def plot_motion_plan(
    motion_plan: MotionPlan, track: Track, plot_title: str = ""
) -> None:
    plt.figure()
    plt.plot(motion_plan.s_ref, unwrap_to_pi(motion_plan.phi_ref), label="headings")
    plt.legend()
    plt.xlabel("track progress [m]")
    plt.ylabel("heading [rad]")
    plt.title(plot_title + " : reference heading/yaw profile")
    plt.tight_layout()

    plt.figure()
    plt.plot(motion_plan.s_ref, motion_plan.kappa_ref, label="curvatures")
    plt.xlabel("track progress [m]")
    plt.ylabel("curvature [1/m]")
    plt.legend()
    plt.title(plot_title + " : reference curvature profile")
    plt.tight_layout()

    plt.figure()
    plot_cones(
        track.blue_cones,
        track.yellow_cones,
        track.big_orange_cones,
        track.small_orange_cones,
        show=False,
    )
    plt.plot(motion_plan.X_ref, motion_plan.Y_ref, label="reference trajectory")
    plt.scatter(
        track.center_line[:, 0],
        track.center_line[:, 1],
        s=14,
        c="k",
        marker="x",
        label="center line",
    )
    plt.legend()
    plt.title(plot_title + " : reference trajectory")
    plt.tight_layout()


def offline_motion_plan(
    track: str | Track,
    n_samples: int = NUMBER_SPLINE_INTERVALS,
) -> MotionPlan:
    # s_ref, X_ref, Y_ref, phi_ref, kappa_ref, right_widths, left_widths
    # before: we fit the splines and re-sample points more finely distributed than
    # the original track points (roughly every 0.1m), all with the same function call
    #
    # now: we fit the splines ourselves, then resample a fixed number of points
    # (per the regulations, FS tracks have total lengths between 200m and 500m,
    # we therefore sample 500/0.1=5000 points to always have at most roughly 10cm
    # between two consecutive points)
    if isinstance(track, str):
        track = Track(track)

    coeffs_X, coeffs_Y = fit_spline(
        path=track.center_line,
        curv_weight=2.0,
        qp_solver=sys.argv[1] if len(sys.argv) > 1 else "proxqp",
        return_errs=False,
    )
    delta_s = compute_spline_interval_lengths(coeffs_X=coeffs_X, coeffs_Y=coeffs_Y)
    X_ref, Y_ref, idx_interp, t_interp, s_ref = uniformly_sample_spline(
        coeffs_X=coeffs_X,
        coeffs_Y=coeffs_Y,
        delta_s=delta_s,
        n_samples=n_samples,
    )
    kappa_ref = get_curvature(
        coeffs_X=coeffs_X,
        coeffs_Y=coeffs_Y,
        idx_interp=idx_interp,
        t_interp=t_interp,
    )
    phi_ref = get_heading(
        coeffs_X=coeffs_X,
        coeffs_Y=coeffs_Y,
        idx_interp=idx_interp,
        t_interp=t_interp,
    ) - np.arcsin(l_R * kappa_ref)
    right_widths = np.tile(np.min(track.track_widths[:, 0]), n_samples)
    left_widths = np.tile(np.min(track.track_widths[:, 1]), n_samples)

    lap_length = s_ref[-1] + np.hypot(X_ref[-1] - X_ref[0], Y_ref[-1] - Y_ref[0])

    return MotionPlan(
        s_ref=s_ref,
        X_ref=X_ref,
        Y_ref=Y_ref,
        phi_ref=phi_ref,
        kappa_ref=kappa_ref,
        right_widths=right_widths,
        left_widths=left_widths,
        lap_length=lap_length,
    )


def triple_motion_plan_ref(motion_plan: MotionPlan) -> MotionPlan:
    motion_plan.s_ref = np.hstack(
        (
            motion_plan.s_ref - motion_plan.lap_length,
            motion_plan.s_ref,
            motion_plan.s_ref + motion_plan.lap_length,
        )
    )
    motion_plan.X_ref = np.hstack(
        (motion_plan.X_ref, motion_plan.X_ref, motion_plan.X_ref)
    )
    motion_plan.Y_ref = np.hstack(
        (motion_plan.Y_ref, motion_plan.Y_ref, motion_plan.Y_ref)
    )
    motion_plan.phi_ref = np.hstack(
        (motion_plan.phi_ref, motion_plan.phi_ref, motion_plan.phi_ref)
    )
    motion_plan.kappa_ref = np.hstack(
        (motion_plan.kappa_ref, motion_plan.kappa_ref, motion_plan.kappa_ref)
    )
    motion_plan.right_widths = np.hstack(
        (motion_plan.right_widths, motion_plan.right_widths, motion_plan.right_widths)
    )
    motion_plan.left_widths = np.hstack(
        (motion_plan.left_widths, motion_plan.left_widths, motion_plan.left_widths)
    )
    return motion_plan


def generate_track_data_file(track_name: str, outfile: str) -> None:
    motion_plan = offline_motion_plan(track_name)

    parent_dir = os.path.dirname(outfile)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    with open(outfile, "w") as f:
        f.write("s_ref,X_ref,Y_ref,phi_ref,kappa_ref,right_width,left_width\n")
    with open(outfile, "a") as f:
        np.savetxt(
            f,
            np.column_stack(
                (
                    motion_plan.s_ref,
                    motion_plan.X_ref,
                    motion_plan.Y_ref,
                    motion_plan.phi_ref,
                    motion_plan.kappa_ref,
                    motion_plan.right_widths,
                    motion_plan.left_widths,
                )
            ),
            delimiter=",",
            fmt="%.6f",
        )


def main() -> None:
    track_name = "fsds_competition_1"
    track = Track(track_name)
    motion_plan = offline_motion_plan(track)
    plot_motion_plan(motion_plan, track, plot_title=track_name)
    plt.show()


if __name__ == "__main__":
    main()
