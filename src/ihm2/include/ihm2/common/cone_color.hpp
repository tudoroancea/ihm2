// Copyright (c) 2023. Tudor Oancea
#ifndef CONE_COLOR_HPP
#define CONE_COLOR_HPP
#include <string>

enum class ConeColor : char {
    BLUE,
    YELLOW,
    BIG_ORANGE,
    SMALL_ORANGE,
};

bool is_orange(const ConeColor& c);

ConeColor cone_color_from_string(const std::string& s);

std::string cone_color_to_string(const ConeColor& c);

#endif  // CONE_COLOR_HPP