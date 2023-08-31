// Copyright (c) 2023. Tudor Oancea
#ifndef TRACK_DATABASE_HPP
#define TRACK_DATABASE_HPP

#include "eigen3/Eigen/Dense"
#include "ihm2/common/cone_color.hpp"
#include <string>
#include <unordered_map>
#include <vector>

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

#endif  // TRACK_DATABASE_HPP
