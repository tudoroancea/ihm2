# Copyright (c) 2023. Tudor Oancea
import os
import matplotlib.pyplot as plt
import numpy as np
import trajectory_planning_helpers as tph
from icecream import ic
from track_database import Track
from track_database.utils import plot_cones


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


def generate_track_file(
    track_name="fsds_competition_1",
    outfile="src/ihm2/tracks/fsds_competition_1.csv",
    plot=False,
):
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))

    track = Track(track_name)
    (
        XY_ref,
        _,
        coeffs_X,
        coeffs_Y,
        _,
        spline_idx,
        t_vals,
        s_ref,
        _,
        _,
    ) = tph.create_raceline(
        refline=track.center_line,
        normvectors=track.center_line,  # since we choose alpha = 0, the actual normvectors don't matter, we just need the correct shape
        alpha=np.zeros(track.center_line.shape[0]),
        stepsize_interp=0.1,
        closed=True,
    )
    total_length = s_ref[-1] + np.hypot(
        XY_ref[-1, 0] - XY_ref[0, 0], XY_ref[-1, 1] - XY_ref[0, 1]
    )
    phi_ref, kappa_ref = tph.calc_head_curv_an(
        coeffs_x=coeffs_X,
        coeffs_y=coeffs_Y,
        ind_spls=spline_idx,
        t_spls=t_vals,
        calc_curv=True,
    )
    right_left_widths = tph.interp_track_widths(
        w_track=track.track_widths,
        spline_inds=spline_idx,
        t_values=t_vals,
    )

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
        plt.plot(XY_ref[:, 0], XY_ref[:, 1], label="reference")
        plt.scatter(
            track.center_line[:, 0],
            track.center_line[:, 1],
            s=14,
            c="k",
            marker="x",
            label="center line",
        )

    s_ref = np.hstack((s_ref - total_length, s_ref, s_ref + total_length))
    XY_ref = np.vstack((XY_ref, XY_ref, XY_ref))
    phi_ref = np.hstack((phi_ref, phi_ref, phi_ref))
    kappa_ref = np.hstack((kappa_ref, kappa_ref, kappa_ref))
    right_left_widths = np.vstack(
        (right_left_widths, right_left_widths, right_left_widths)
    )
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
    for track_name in [
        "fsds_competition_1",
        "fsds_competition_2",
        "fsds_competition_3",
        "autoX_Vaudoise_Sponso",
    ]:
        generate_track_file(track_name, "src/ihm2/tracks/" + track_name + ".csv")


if __name__ == "__main__":
    main()
