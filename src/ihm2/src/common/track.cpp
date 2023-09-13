// Copyright (c) 2023. Tudor Oancea
#include "ihm2/common/track.hpp"
#include "ihm2/common/math.hpp"
#include "ihm2/external/rapidcsv.hpp"
#include <cmath>

Track::Track(const std::string& csv_file) {
    rapidcsv::Document doc(csv_file);
    size_t row_count = doc.GetRowCount();
    s_ref.resize(row_count);
    X_ref.resize(row_count);
    Y_ref.resize(row_count);
    phi_ref.resize(row_count);
    kappa_ref.resize(row_count);
    right_width.resize(row_count);
    left_width.resize(row_count);
    for (size_t i = 0; i < row_count; ++i) {
        s_ref(i) = doc.GetCell<double>("s_ref", i);
        X_ref(i) = doc.GetCell<double>("X_ref", i);
        Y_ref(i) = doc.GetCell<double>("Y_ref", i);
        phi_ref(i) = doc.GetCell<double>("phi_ref", i);
        kappa_ref(i) = doc.GetCell<double>("kappa_ref", i);
        right_width(i) = doc.GetCell<double>("right_width", i);
        left_width(i) = doc.GetCell<double>("left_width", i);
    }

    delta_s = s_ref.tail(s_ref.size() - 1) - s_ref.head(s_ref.size() - 1);

    // fit linear splines
    coeffs_X.resize(row_count - 1, 2);
    coeffs_Y.resize(row_count - 1, 2);
    coeffs_phi.resize(row_count - 1, 2);
    coeffs_kappa.resize(row_count - 1, 2);
    coeffs_right_width.resize(row_count - 1, 2);
    coeffs_left_width.resize(row_count - 1, 2);

    for (size_t i = 0; i < row_count - 1; ++i) {
        coeffs_X(i, 0) = X_ref(i);
        coeffs_X(i, 1) = (X_ref(i + 1) - X_ref(i)) / delta_s(i);

        coeffs_Y(i, 0) = Y_ref(i);
        coeffs_Y(i, 1) = (Y_ref(i + 1) - Y_ref(i)) / delta_s(i);

        coeffs_phi(i, 0) = phi_ref(i);
        coeffs_phi(i, 1) = (phi_ref(i + 1) - phi_ref(i)) / delta_s(i);

        coeffs_kappa(i, 0) = kappa_ref(i);
        coeffs_kappa(i, 1) = (kappa_ref(i + 1) - kappa_ref(i)) / delta_s(i);

        coeffs_right_width(i, 0) = right_width(i);
        coeffs_right_width(i, 1) = (right_width(i + 1) - right_width(i)) / delta_s(i);

        coeffs_left_width(i, 0) = left_width(i);
        coeffs_left_width(i, 1) = (left_width(i + 1) - left_width(i)) / delta_s(i);
    }
}

size_t locate_index(const Eigen::VectorXd& v, double x) {
    return std::upper_bound(v.data(), v.data() + v.size(), x) - v.data() - 1;
}

double angle3pt(const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Eigen::Vector2d& c) {
    return wrap_to_pi(std::atan2(c(1) - b(1), c(0) - b(0)) - std::atan2(a(1) - b(1), a(0) - b(0)));
}

void Track::interp(const Eigen::MatrixXd& coeffs, double s, double& value, int ind) {
    // find i such that s_ref[i] <= s < s_ref[i+1]
    if (ind < 0) {
        ind = locate_index(s_ref, s);
    }
    // find the value of the spline at s
    value = coeffs(ind, 0) + coeffs(ind, 1) * (s - s_ref(ind));
}

void Track::project(
        const Eigen::Vector2d& car_pos,
        double s_guess,
        double s_tol,
        double* s_proj,
        double* X_ref_proj,
        double* Y_ref_proj,
        double* phi_ref_proj,
        double* kappa_ref_proj,
        double* right_width_proj,
        double* left_width_proj) {
    // extract all the points in X_ref, Y_ref associated with s_ref values within s_guess +- s_tol
    double s_low = std::max(s_guess - s_tol, s_ref(0)),
           s_up = std::min(s_guess + s_tol, s_ref(s_ref.size() - 1));
    size_t id_low = locate_index(s_ref, s_low), id_up = locate_index(s_ref, s_up);
    if (id_low > 0) {
        --id_low;
    }
    if (id_up < s_ref.size() - 1) {
        ++id_up;
    }
    Eigen::ArrayX2d local_traj = Eigen::ArrayX2d::Zero(id_up - id_low + 1, 2);  // problem with difference of size_ints ?
    local_traj.col(0) = X_ref.segment(id_low, id_up - id_low + 1);
    local_traj.col(1) = Y_ref.segment(id_low, id_up - id_low + 1);

    // find the closest point to car_pos to find one segment extremity
    Eigen::VectorXd sqdist = (local_traj.col(0) - car_pos(0)).square() + (local_traj.col(1) - car_pos(1)).square();
    size_t id_min, id_prev, id_next;
    sqdist.minCoeff(&id_min);
    id_prev = id_min - 1;
    id_next = id_min + 1;
    // TODO: what happens if id_min == 0 or id_min == local_traj.rows() - 1 ?
    // This should not happen though

    // compute the angles between car_pos, the closest point and the next and previous point to find the second segment extremity
    double angle_prev = std::fabs(angle3pt(local_traj.row(id_min), car_pos, local_traj.row(id_prev))),
           angle_next = std::fabs(angle3pt(local_traj.row(id_min), car_pos, local_traj.row(id_next)));
    Eigen::Vector2d a, b;
    double sa, sb;
    if (angle_prev > angle_next) {
        a = local_traj.row(id_prev);
        b = local_traj.row(id_min);
        sa = s_ref(id_prev + id_low);
        sb = s_ref(id_min + id_low);
    } else {
        a = local_traj.row(id_min);
        b = local_traj.row(id_next);
        sa = s_ref(id_min + id_low);
        sb = s_ref(id_next + id_low);
    }

    // project car_pos on the segment and retrieve lambda
    double dx = b(0) - a(0), dy = b(1) - a(1);
    double lambda = ((car_pos(0) - a(0)) * dx + (car_pos(1) - a(1)) * dy) / (dx * dx + dy * dy);

    // compute the interpolated values (with non null pointers) at lambda using the index of the closest point
    *s_proj = sa + lambda * (sb - sa);
    *X_ref_proj = a(0) + lambda * (b(0) - a(0));
    *Y_ref_proj = a(1) + lambda * (b(1) - a(1));
    if (phi_ref_proj != nullptr) {
        interp(coeffs_phi, *s_proj, *phi_ref_proj, id_min + id_low);
    }
    if (kappa_ref_proj != nullptr) {
        interp(coeffs_kappa, *s_proj, *kappa_ref_proj, id_min + id_low);
    }
    if (right_width_proj != nullptr) {
        interp(coeffs_right_width, *s_proj, *right_width_proj, id_min + id_low);
    }
    if (left_width_proj != nullptr) {
        interp(coeffs_left_width, *s_proj, *left_width_proj, id_min + id_low);
    }
}