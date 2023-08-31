// Copyright (c) 2023. Tudor Oancea
#include "ihm2/common/track_database.hpp"
#include "ihm2/external/rapidcsv.hpp"
#include <filesystem>
#include <vector>

// using namespace std;
// using namespace Eigen;


std::string validate_track_name_or_file(const std::string& track_name_or_file) {
#ifdef TRACK_DATABASE_PATH
    std::filesystem::path tdb_track(TRACK_DATABASE_PATH), other_track(track_name_or_file), actual_track;
    tdb_track /= (track_name_or_file + "/" + track_name_or_file + ".csv");
    other_track = std::filesystem::absolute(other_track);
    if (std::filesystem::exists(tdb_track)) {
        actual_track = tdb_track;
    } else if (std::filesystem::exists(other_track)) {
        actual_track = other_track;
    } else {
        throw std::runtime_error("track " + track_name_or_file + " not found neither in TRACK_DATABASE_PATH nor in the current directory");
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
    rapidcsv::Document cones(validate_track_name_or_file(track_name_or_file));
    // get the positions of the cones in columns X and Y and the corresponding type in column cone_type
    std::vector<double> cones_x = cones.GetColumn<double>("X");
    std::vector<double> cones_y = cones.GetColumn<double>("Y");
    std::vector<std::string> cones_type = cones.GetColumn<std::string>("cone_type");
    std::unordered_map<ConeColor, Eigen::MatrixX2d> cones_map;
    for (size_t i = 0; i < cones_x.size(); ++i) {
        ConeColor type = from_string(cones_type[i]);
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
            f << to_string(it->first)
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
    rapidcsv::Document center_line_csv(validate_track_name_or_file(track_name_or_file));
    size_t row_count = center_line_csv.GetRowCount();
    center_line.resize(row_count, 2);
    track_widths.resize(row_count, 2);
    for (size_t i=0; i < row_count; ++i) {
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
