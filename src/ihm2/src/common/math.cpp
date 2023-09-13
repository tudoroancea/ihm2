// Copyright (c) 2023. Tudor Oancea
#include "ihm2/common/math.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include <cmath>

double clip(double n, double lower, double upper) {
    return std::max(lower, std::min(n, upper));
}

Eigen::VectorXd atan2(const Eigen::VectorXd& y, const Eigen::VectorXd& x) {
    Eigen::VectorXd res(y.size());
    for (int i = 0; i < y.size(); i++) {
        res(i) = std::atan2(y(i), x(i));
    }
    return res;
}

double wrap_to_pi(double x) {
    double tpr = std::fmod(x + M_PI, 2 * M_PI);
    if (tpr < 0) {
        tpr += 2 * M_PI;  // necessary because of how fmod works in C++ (it always return a remainder with the same sign as the dividend)
    }
    return tpr - M_PI;
}

Eigen::VectorXd wrap_to_pi(const Eigen::VectorXd& x) {
    Eigen::VectorXd res(x.size());
    for (int i = 0; i < x.size(); i++) {
        res(i) = wrap_to_pi(x(i));
    }
    return res;
}

float rad2deg(float rad) {
    return rad * 57.29577951308232f;
}

double rad2deg(double rad) {
    return rad * 57.29577951308232;
}

float deg2rad(float deg) {
    return deg * 0.017453292519943295f;
}

double deg2rad(double deg) {
    return deg * 0.017453292519943295;
}

geometry_msgs::msg::Quaternion rpy_to_quaternion(double roll, double pitch, double yaw) {
    tf2::Quaternion q;
    q.setRPY(roll, pitch, yaw);
    return tf2::toMsg(q);
}
