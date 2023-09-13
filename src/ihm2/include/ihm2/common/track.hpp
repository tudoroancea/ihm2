// Copyright (c) 2023. Tudor Oancea
#ifndef TRACK_HPP
#define TRACK_HPP

#include "eigen3/Eigen/Eigen"
#include <cmath>

// step 1: read the CSV file with the track data (s_ref, X_ref, Y_ref, phi_ref, kappa_ref, right_width, left_width)
// step 2: compute linear interpolation for each quantity vs s_ref
// step 3: project the car's position on the spline and evaluate each quantity
class Track {
private:
    Eigen::VectorXd s_ref, X_ref, Y_ref, phi_ref, kappa_ref, right_width, left_width;
    Eigen::VectorXd delta_s;
    Eigen::MatrixX2d coeffs_X, coeffs_Y, coeffs_phi, coeffs_kappa, coeffs_right_width, coeffs_left_width;

    void interp(const Eigen::MatrixXd& coeffs, double s, double& value, int ind = -1);

public:
    Track(const std::string& csv_file);

    void project(
            const Eigen::Vector2d& car_pos,
            double s_guess,
            double s_tol,
            double* s,
            double* X_ref,
            double* Y_ref,
            double* phi_ref,
            double* kappa_ref = nullptr,
            double* right_width = nullptr,
            double* left_width = nullptr);
};


#endif  // TRACK_HPP
