// Copyright (c) 2023. Tudor Oancea
#ifndef TRACKS_HPP
#define TRACKS_HPP

#include "eigen3/Eigen/Dense"
#include "ihm2/common/cone_color.hpp"
#include <string>
#include <unordered_map>

// functions to load/save cones and center line from the track_database package ===========================================

std::unordered_map<ConeColor, Eigen::MatrixX2d> load_cones(const std::string& track_name_or_file);

void save_cones(
        const std::string& filename,
        const std::unordered_map<ConeColor, Eigen::MatrixX2d>& cones_map);

void load_center_line(
        const std::string& track_name_or_file,
        Eigen::MatrixX2d& center_line,
        Eigen::MatrixX2d& track_widths);

void save_center_line(
        const std::string& filename,
        const Eigen::MatrixX2d& center_line,
        const Eigen::MatrixX2d& track_widths);


// class used to wrap the track files generated in python =================================================================

// enum class AvailableTrack : char {
//     FSDS_COMPETITION_1,
//     FSDS_COMPETITION_2,
//     FSDS_COMPETITION_3,
// };

// AvailableTrack available_track_from_string(const std::string& str);
// std::string available_track_to_string(const AvailableTrack& track);

class Track {
private:
    Eigen::VectorXd s_ref, X_ref, Y_ref, phi_ref, kappa_ref, right_width, left_width;
    Eigen::VectorXd delta_s;
    Eigen::MatrixX2d coeffs_X, coeffs_Y, coeffs_phi, coeffs_kappa, coeffs_right_width, coeffs_left_width;

    void interp(const Eigen::MatrixXd& coeffs, double s, double& value, int ind = -1) const;

public:
    explicit Track(const std::string& csv_file);

    void project(
            const Eigen::Vector2d& car_pos,
            double s_guess,
            double s_tol,
            double* s = nullptr,
            double* X_ref_proj = nullptr,
            double* Y_ref_proj = nullptr,
            double* phi_ref_proj = nullptr,
            double* phi_ref_preview = nullptr,
            double* kappa_ref_proj = nullptr,
            double* right_width_proj = nullptr,
            double* left_width_proj = nullptr);

    void frenet_to_cartesian(const double& s, const double& n, double& X, double& Y) const;
    void frenet_to_cartesian(const Eigen::VectorXd& s, const Eigen::VectorXd& n, Eigen::VectorXd& X, Eigen::VectorXd& Y) const;

    double length() const;
    size_t size() const;

    double* get_s_ref();
    double* get_kappa_ref();
    double* get_X_ref();
    double* get_Y_ref();
};


#endif  // TRACKS_HPP
