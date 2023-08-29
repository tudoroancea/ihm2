# Copyright (c) 2023. Tudor Oancea
import numpy as np
import matplotlib.pyplot as plt
from track_database import Track
from track_database.utils import plot_cones
import trajectory_planning_helpers as tph
from .utils import unwrapToPi


def main():
    track_name = "fsds_competition_2"
    track = Track(track_name)

    _, _, _, original_center_normvectors = tph.calc_splines(
        path=track.center_line, closed=True
    )
    (
        reference_points,
        _,
        reference_coeffs_x,
        reference_coeffs_y,
        _,
        reference_new_center_spline_idx,
        reference_new_center_t_values,
        reference_new_center_s_values,
        _,
        _,
    ) = tph.create_raceline(
        refline=track.center_line,
        normvectors=original_center_normvectors,
        alpha=np.zeros(track.center_line.shape[0]),
        stepsize_interp=0.1,
        closed=True,
    )
    headings, curvatures = tph.calc_head_curv_an(
        coeffs_x=reference_coeffs_x,
        coeffs_y=reference_coeffs_y,
        ind_spls=reference_new_center_spline_idx,
        t_spls=reference_new_center_t_values,
        calc_curv=True,
    )
    new_widths = tph.interp_track_widths(
        track.track_widths,
        reference_new_center_spline_idx,
        reference_new_center_t_values,
    )
    s_ref = reference_new_center_s_values
    X_ref = reference_points[:, 0]
    Y_ref = reference_points[:, 1]
    phi_ref = headings
    kappa_ref = curvatures
    np.savetxt(
        f"data/{track_name}.csv",
        np.array(
            (
                s_ref,
                X_ref,
                Y_ref,
                phi_ref,
                kappa_ref,
                new_widths[:, 0],
                new_widths[:, 1],
            )
        ).T,
        delimiter=",",
        header="s_ref,X_ref,Y_ref,phi_ref,kappa_ref,right_width,left_width",
        fmt="%.6f",
    )

    plt.figure()
    plt.plot(s_ref, unwrapToPi(headings), label="headings")
    plt.legend()
    plt.figure()
    plt.plot(s_ref, curvatures, label="curvatures")
    plt.legend()

    plt.figure()
    plot_cones(
        track.blue_cones,
        track.yellow_cones,
        track.big_orange_cones,
        track.small_orange_cones,
        show=False,
    )
    plt.plot(reference_points[:, 0], reference_points[:, 1], label="reference")
    plt.scatter(
        track.center_line[:, 0],
        track.center_line[:, 1],
        s=14,
        c="k",
        marker="x",
        label="center line",
    )
    plt.show()


if __name__ == "__main__":
    main()
