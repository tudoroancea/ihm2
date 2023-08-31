#include "ihm2/common/marker_color.hpp"

std_msgs::msg::ColorRGBA marker_colors(const std::string& color) {
    std_msgs::msg::ColorRGBA color_msg;
    color_msg.a = 1.0;
    if (color == "red") {
        color_msg.r = 1.0;
    } else if (color == "green") {
        color_msg.g = 1.0;
    } else if (color == "blue") {
        color_msg.b = 1.0;
    } else if (color == "yellow") {
        color_msg.r = 1.0;
        color_msg.g = 1.0;
    } else if (color == "orange") {
        color_msg.r = 1.0;
        color_msg.g = 0.5;
    } else if (color == "purple") {
        color_msg.r = 0.5;
        color_msg.b = 0.5;
    } else if (color == "magenta") {
        color_msg.r = 1.0;
        color_msg.b = 1.0;
    } else if (color == "cyan") {
        color_msg.g = 1.0;
        color_msg.b = 1.0;
    } else if (color == "light_blue") {
        color_msg.g = 0.5;
        color_msg.b = 1.0;
    } else if (color == "dark_blue") {
        color_msg.b = 0.5;
    } else if (color == "brown") {
        color_msg.r = 0.5;
        color_msg.g = 0.25;
    } else if (color == "white") {
        color_msg.r = 1.0;
        color_msg.g = 1.0;
        color_msg.b = 1.0;
    } else if (color == "gray") {
        color_msg.r = 0.5;
        color_msg.g = 0.5;
        color_msg.b = 0.5;
    } else if (color == "light_gray") {
        color_msg.r = 0.75;
        color_msg.g = 0.75;
        color_msg.b = 0.75;
    } else if (color == "dark_gray") {
        color_msg.r = 0.25;
        color_msg.g = 0.25;
        color_msg.b = 0.25;
    } else {
        color_msg.r = 1.0;
        color_msg.g = 1.0;
        color_msg.b = 1.0;
    }
    return color_msg;
}