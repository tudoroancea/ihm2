// Copyright (c) 2023. Tudor Oancea
#include "ihm2/common/tracks.hpp"
#include "ihm2/common/math.hpp"
#include "ihm2/external/icecream.hpp"
#include "ihm2/external/rapidcsv.hpp"
#include <cmath>
#include <filesystem>
#include <vector>

// functions to load/save cones and center line from the track_database package ===========================================
std::string validate_track_name_or_file(const std::string& track_name_or_file, std::string tdb_suffix = "_cones.csv") {
#ifdef TRACK_DATABASE_PATH
    std::filesystem::path tdb_track(TRACK_DATABASE_PATH), other_track(track_name_or_file), actual_track;
    tdb_track /= (track_name_or_file + "/" + track_name_or_file + tdb_suffix);
    other_track = std::filesystem::absolute(other_track);
    if (std::filesystem::exists(tdb_track)) {
        actual_track = tdb_track;
    } else if (std::filesystem::exists(other_track)) {
        actual_track = other_track;
    } else {
        throw std::runtime_error("track " + track_name_or_file + " not found neither in TRACK_DATABASE_PATH nor in the current directory, tried:\n\r" + tdb_track.string() + "\n\r" + other_track.string());
    }

    if (!std::filesystem::exists(actual_track)) {
        throw std::runtime_error("file " + actual_track.string() + " does not exist");
    }
    if (!(actual_track.string().substr(actual_track.string().size() - 4) == ".csv")) {
        throw std::runtime_error("file " + actual_track.string() + " is not a CSV file");
    }
    return actual_track.string();
#else
    throw std::runtime_error("TRACK_DATABASE_PATH not defined");
#endif
}

std::unordered_map<ConeColor, Eigen::MatrixX2d> load_cones(const std::string& track_name_or_file) {
    // if the file is a CSV file, load it directly
    rapidcsv::Document cones(validate_track_name_or_file(track_name_or_file, "_cones.csv"));
    // get the positions of the cones in columns X and Y and the corresponding type in column cone_type
    std::vector<double> cones_x = cones.GetColumn<double>("X");
    std::vector<double> cones_y = cones.GetColumn<double>("Y");
    std::vector<std::string> cones_type = cones.GetColumn<std::string>("cone_type");
    std::unordered_map<ConeColor, Eigen::MatrixX2d> cones_map;
    for (size_t i = 0; i < cones_x.size(); ++i) {
        ConeColor type = cone_color_from_string(cones_type[i]);
        if (cones_map.find(type) == cones_map.end()) {
            cones_map[type] = Eigen::MatrixX2d::Zero(0, 2);
        }
        cones_map[type].conservativeResize(cones_map[type].rows() + 1, 2);
        cones_map[type].bottomRows<1>() << cones_x[i], cones_y[i];
    }
    return cones_map;
}


void save_cones(
        const std::string& filename,
        const std::unordered_map<ConeColor, Eigen::MatrixX2d>& cones_map) {
    std::ofstream f(filename);
    f << "cone_type,X,Y,Z,std_X,std_Y,std_Z,right,left\n";

    for (auto it = cones_map.begin(); it != cones_map.end(); ++it) {
        for (Eigen::Index id(0); id < it->second.rows(); ++id) {
            f << cone_color_to_string(it->first)
              << ","
              << it->second(id, 0)
              << ","
              << it->second(id, 1)
              << ",0.0,0.0,0.0,0.0,"
              << (it->first == ConeColor::YELLOW || it->second(id, 0) > 0)
              << ","
              << (it->first == ConeColor::BLUE || it->second(id, 0) < 0)
              << "\n";
        }
    }
}

void load_center_line(
        const std::string& track_name_or_file,
        Eigen::MatrixX2d& center_line,
        Eigen::MatrixX2d& track_widths) {
    rapidcsv::Document center_line_csv(validate_track_name_or_file(track_name_or_file, "_center_line.csv"));
    size_t row_count = center_line_csv.GetRowCount();
    center_line.resize(row_count, 2);
    track_widths.resize(row_count, 2);
    for (size_t i = 0; i < row_count; ++i) {
        center_line(i, 0) = center_line_csv.GetCell<double>("x", i);
        center_line(i, 1) = center_line_csv.GetCell<double>("y", i);
        track_widths(i, 0) = center_line_csv.GetCell<double>("right_width", i);
        track_widths(i, 1) = center_line_csv.GetCell<double>("left_width", i);
    }
}

void save_center_line(
        const std::string& filename,
        const Eigen::MatrixX2d& center_line,
        const Eigen::MatrixX2d& track_widths) {
    std::ofstream f(filename);
    f << "x,y,right_width,left_width\n";
    for (Eigen::Index i = 0; i < center_line.rows(); ++i) {
        f << center_line(i, 0) << "," << center_line(i, 1) << "," << track_widths(i, 0) << "," << track_widths(i, 1) << "\n";
    }
}

// class used to wrap the track files generated in python =================================================================

// AvailableTrack available_track_from_string(const std::string& str) {
//     if (str == "fsds_competition_1") {
//         return AvailableTrack::FSDS_COMPETITION_1;
//     } else if (str == "fsds_competition_2") {
//         return AvailableTrack::FSDS_COMPETITION_2;
//     } else if (str == "fsds_competition_3") {
//         return AvailableTrack::FSDS_COMPETITION_3;
//     } else {
//         throw std::runtime_error("unknown track name " + str);
//     }
// }

// std::string available_track_to_string(const AvailableTrack& track) {
//     switch (track) {
//         case AvailableTrack::FSDS_COMPETITION_1:
//             return "fsds_competition_1";
//         case AvailableTrack::FSDS_COMPETITION_2:
//             return "fsds_competition_2";
//         case AvailableTrack::FSDS_COMPETITION_3:
//             return "fsds_competition_3";
//         default:
//             throw std::runtime_error("unknown track name");
//     }
// }


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

void Track::interp(const Eigen::MatrixXd& coeffs, double s, double& value, int ind) const {
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
        double* phi_ref_preview,
        double* kappa_ref_proj,
        double* right_width_proj,
        double* left_width_proj) {
    // extract all the points in X_ref, Y_ref associated with s_ref values within s_guess +- s_tol
    double s_low = std::max(s_guess - s_tol, s_ref(0)),
           s_up = std::min(s_guess + s_tol, s_ref(s_ref.size() - 1));
    long long id_low = locate_index(s_ref, s_low), id_up = locate_index(s_ref, s_up);
    if (id_low > 0) {
        --id_low;
    }
    if (id_up < static_cast<long long>(s_ref.size()) - 1) {
        ++id_up;
    }
    Eigen::ArrayX2d local_traj = Eigen::ArrayX2d::Zero(id_up - id_low + 1, 2);  // problem with difference of size_ints ?
    local_traj.col(0) = X_ref.segment(id_low, id_up - id_low + 1);
    local_traj.col(1) = Y_ref.segment(id_low, id_up - id_low + 1);

    // find the closest point to car_pos to find one segment extremity
    Eigen::VectorXd sqdist = (local_traj.col(0) - car_pos(0)).square() + (local_traj.col(1) - car_pos(1)).square();
    long long id_min, id_prev, id_next;
    sqdist.minCoeff(&id_min);
    id_prev = id_min - 1;
    if (id_min == 0) {
        id_prev = local_traj.rows() - 1;
    }
    id_next = id_min + 1;
    if (id_min == static_cast<long long>(local_traj.rows()) - 1) {
        id_next = 0;
    }
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
    if (s_proj != nullptr) {
        *s_proj = sa + lambda * (sb - sa);
    }
    if (X_ref_proj != nullptr) {
        *X_ref_proj = a(0) + lambda * (b(0) - a(0));
    }
    if (Y_ref_proj != nullptr) {
        *Y_ref_proj = a(1) + lambda * (b(1) - a(1));
    }
    if (phi_ref_proj != nullptr) {
        if (phi_ref_preview != nullptr) {
            interp(coeffs_phi, *s_proj + *phi_ref_preview, *phi_ref_proj, id_min + id_low);
        } else {
            interp(coeffs_phi, *s_proj, *phi_ref_proj, id_min + id_low);
        }
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

void Track::frenet_to_cartesian(const double& s, const double& n, double& X, double& Y) const {
    double X_ref, Y_ref, phi_ref;
    this->interp(coeffs_phi, s, phi_ref);
    this->interp(coeffs_X, s, X_ref);
    this->interp(coeffs_Y, s, Y_ref);
    Eigen::Vector2d normal(-std::sin(phi_ref), std::cos(phi_ref));
    X = X_ref + n * normal(0);
    Y = Y_ref + n * normal(1);
}
void Track::frenet_to_cartesian(const Eigen::VectorXd& s, const Eigen::VectorXd& n, Eigen::VectorXd& X, Eigen::VectorXd& Y) const {
    X.resize(s.size());
    Y.resize(s.size());
    for (Eigen::Index i = 0; i < s.size(); ++i) {
        this->frenet_to_cartesian(s(i), n(i), X(i), Y(i));
    }
}

double Track::length() const { return -s_ref(0); }
size_t Track::size() const { return s_ref.size(); }
double* Track::get_s_ref() { return s_ref.data(); }
double* Track::get_kappa_ref() { return kappa_ref.data(); }
double* Track::get_X_ref() { return X_ref.data(); }
double* Track::get_Y_ref() { return Y_ref.data(); }
