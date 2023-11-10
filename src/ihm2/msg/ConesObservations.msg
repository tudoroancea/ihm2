# Copyright (c) 2023. Tudor Oancea
std_msgs/Header header
uint8 BLUE = 0
uint8 YELLOW = 1
uint8 SMALL_ORANGE = 2
uint8 BIG_ORANGE = 3
float64[] rho
float64[] theta
uint8[] colors
float64[] rho_uncertainty # std dev of the rho (range) measurement
float64[] theta_uncertainty # std dev of the theta (bearing) measurement
float64[] colors_confidence # confidence of the color measurement, between 0 and 1
