# Copyright (c) 2023. Tudor Oancea
cmake_minimum_required(VERSION 3.8)
project(ihm2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -pthread -O2")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(statistics_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(stereo_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
ament_auto_find_build_dependencies()

# find acados
option(ACADOS_PATH "Path to acados installation" $ENV{ACADOS_SOURCE_DIR})

if(NOT ACADOS_PATH)
  message(FATAL_ERROR "ACADOS_PATH not set")
endif()

message(STATUS "ACADOS_PATH: ${ACADOS_PATH}")

# generate custom message and service types
set(msg_files
  "msg/Controls.msg"
)
set(srv_files)
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES std_msgs # Add packages that above messages depend on
)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)

# generate sim node
ament_auto_add_executable(sim_node src/sim_node.cpp)
target_link_libraries(sim_node "${cpp_typesupport_target}")
find_library(IHM2_SIM_LIB acados_sim_solver_ihm2_kin4 PATHS ihm2_sim_gen_code)
message(STATUS "IHM2_SIM_LIB: ${IHM2_SIM_LIB}")
target_link_libraries(sim_node ${IHM2_SIM_LIB})
target_include_directories(sim_node PUBLIC
  "${ACADOS_PATH}/include"
  "${ACADOS_PATH}/include/acados"
  "ihm2_sim_gen_code"
  "ihm2_sim_gen_code/ihm2_kin4_model"
)

# install everything in the ihm2 package
ament_auto_package(INSTALL_TO_SHARE)