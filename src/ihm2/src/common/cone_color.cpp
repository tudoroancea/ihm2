// Copyright (c) 2023. Tudor Oancea
#include "ihm2/common/cone_color.hpp"
#include <string>

bool is_orange(const ConeColor& c) {
    return c == ConeColor::BIG_ORANGE || c == ConeColor::SMALL_ORANGE;
}

ConeColor from_string(const std::string& s) {
    if (s == "blue") {
        return ConeColor::BLUE;
    } else if (s == "yellow") {
        return ConeColor::YELLOW;
    } else if (s == "big_orange") {
        return ConeColor::BIG_ORANGE;
    } else if (s == "small_orange") {
        return ConeColor::SMALL_ORANGE;
    } else {
        throw std::runtime_error("invalid cone color " + s);
    }
}

std::string to_string(const ConeColor& c) {
    if (c == ConeColor::BLUE) {
        return "blue";
    } else if (c == ConeColor::YELLOW) {
        return "yellow";
    } else if (c == ConeColor::BIG_ORANGE) {
        return "big_orange";
    } else if (c == ConeColor::SMALL_ORANGE) {
        return "small_orange";
    } else {
        throw std::runtime_error("invalid cone color");
    }
}
