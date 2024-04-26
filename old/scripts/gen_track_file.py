# Copyright (c) 2024. Tudor Oancea
import os
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
from track_database import Track
from track_database.utils import plot_cones
from scipy.sparse import csc_array, eye as speye, kron as spkron
from qpsolvers import solve_qp, available_solvers
import sys


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


def my_calc_splines(
    path: np.ndarray, q: float = 1.0, return_errs: bool = False, qp_solver="proxqp"
):
    """
    computes the coefficients of each spline portion of the path.
    > Note: the path is assumed to be closed but the first and last points are NOT the same

    :param path: Nx2 array of points
    :param q: weight of the curvature term in the cost function
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
    P = B.T @ B + q * C.T @ C + 1e-10 * speye(4 * N, format="csc")
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


def my_calc_spline_lengths(
    coeffs_X: np.ndarray, coeffs_Y: np.ndarray, no_interp_points=100
):
    """
    computes the lengths of each spline portion of the path.
    > Note: Here the closeness of the part does not matter, it is contained in the coefficients

    :param coeff_X: Nx4 array of coefficients of the splines in the x direction (as returned by my_calc_splines)
    :param coeff_Y: Nx4 array of coefficients of the splines in the y direction (as returned by my_calc_splines)
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


def my_interp_splines(
    coeffs_X: np.ndarray,
    coeffs_Y: np.ndarray,
    delta_s: np.ndarray,
    n_samples: int,
):
    """
    uniformly n_samples equidistant points along the path defined by the splines.
    The first point will always be the initial point of the first spline portion, and
    the last point will NOT be the initial point of the first spline portion.

    :param coeffs_X: Nx4 array of coefficients of the splines in the x direction (as returned by my_calc_splines)
    :param coeffs_Y: Nx4 array of coefficients of the splines in the y direction (as returned by my_calc_splines)
    :param spline_lengths: N array of lengths of the spline portions (as returned by my_calc_spline_lengths)
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


def calc_heading_curvature(
    coeffs_X: np.ndarray,
    coeffs_Y: np.ndarray,
    idx_interp: np.ndarray,
    t_interp: np.ndarray,
):
    """
    analytically computes the heading and the curvature at each point along the path
    specified by idx_interp and t_interp.

    :param coeffs_X: Nx4 array of coefficients of the splines in the x direction (as returned by my_calc_splines)
    :param coeffs_Y: Nx4 array of coefficients of the splines in the y direction (as returned by my_calc_splines)
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
    # same here with the division by delta_s[idx_interp] ** 2
    x_dd = 2 * coeffs_X[idx_interp, 2] + 6 * coeffs_X[idx_interp, 3] * t_interp
    y_dd = 2 * coeffs_Y[idx_interp, 2] + 6 * coeffs_Y[idx_interp, 3] * t_interp
    kappa = (x_d * y_dd - y_d * x_dd) / np.power(x_d**2 + y_d**2, 1.5)

    return phi, kappa


def generate_track_file(
    track_name="fsds_competition_1",
    outfile="src/ihm2/generated/tracks/fsds_competition_1.csv",
    plot=False,
    n_samples=5000,
):
    # before: we fit the splines and re-sample points more finely distributed than
    # the original track points (roughly every 0.1m), all with the same function call
    #
    # now: we fit the splines ourselves, then resample a fixed number of points
    # (per the regulations, FS tracks have total lengths between 200m and 500m,
    # we therefore sample 500/0.1=5000 points to always have at most roughly 10cm
    # between two consecutive points)
    track = Track(track_name)
    coeffs_X, coeffs_Y = my_calc_splines(
        path=track.center_line,
        q=2.0,
        qp_solver=sys.argv[1] if len(sys.argv) > 1 else "proxqp",
    )
    delta_s = my_calc_spline_lengths(coeffs_X=coeffs_X, coeffs_Y=coeffs_Y)
    X_ref, Y_ref, idx_interp, t_interp, s_ref = my_interp_splines(
        coeffs_X=coeffs_X,
        coeffs_Y=coeffs_Y,
        delta_s=delta_s,
        n_samples=n_samples,
    )
    phi_ref, kappa_ref = calc_heading_curvature(
        coeffs_X=coeffs_X,
        coeffs_Y=coeffs_Y,
        idx_interp=idx_interp,
        t_interp=t_interp,
    )
    right_left_widths = np.tile(np.min(track.track_widths, axis=1), (n_samples, 1))

    if plot:
        plt.figure()
        plt.plot(s_ref, unwrapToPi(phi_ref), label="headings")
        plt.legend()
        plt.figure()
        plt.plot(s_ref, kappa_ref, label="curvatures")
        plt.legend()

        plt.figure()
        plot_cones(
            track.blue_cones,
            track.yellow_cones,
            track.big_orange_cones,
            track.small_orange_cones,
            show=False,
        )
        plt.plot(X_ref, Y_ref, label="reference")
        plt.scatter(
            track.center_line[:, 0],
            track.center_line[:, 1],
            s=14,
            c="k",
            marker="x",
            label="center line",
        )

    total_length = s_ref[-1] + np.hypot(
        XY_ref[-1, 0] - XY_ref[0, 0], XY_ref[-1, 1] - XY_ref[0, 1]
    )
    s_ref = np.hstack((s_ref - total_length, s_ref, s_ref + total_length))
    XY_ref = np.vstack((XY_ref, XY_ref, XY_ref))
    phi_ref = np.hstack((phi_ref, phi_ref, phi_ref))
    kappa_ref = np.hstack((kappa_ref, kappa_ref, kappa_ref))
    right_left_widths = np.vstack(
        (right_left_widths, right_left_widths, right_left_widths)
    )

    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    np.savetxt(
        outfile,
        np.array(
            (
                s_ref,
                XY_ref[:, 0],
                XY_ref[:, 1],
                phi_ref,
                kappa_ref,
                right_left_widths[:, 0],
                right_left_widths[:, 1],
            )
        ).T,
        delimiter=",",
        fmt="%.6f",
    )
    # open csv file and add header
    with open(outfile, "r") as f:
        lines = f.readlines()
        lines.insert(
            0,
            "s_ref,X_ref,Y_ref,phi_ref,kappa_ref,right_width,left_width\n",
        )
    with open(outfile, "w") as f:
        f.writelines(lines)

    if plot:
        plt.show()


def main():
    print("**************************************************")
    print("* Generating track files *************************")
    print("**************************************************\n")
    for track_name in [
        "fsds_competition_1",
        "fsds_competition_2",
        "fsds_competition_3",
    ]:
        start = perf_counter()
        generate_track_file(
            track_name,
            "src/ihm2/generated/tracks/" + track_name + ".csv",
            # plot=True,
        )
        print(
            f"Generation of track {track_name} took {perf_counter() - start} seconds."
        )

    print("")


if __name__ == "__main__":
    main()
