# Copyright (c) 2023. Tudor Oancea
cmake_minimum_required(VERSION 3.8)
project(ihm2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra -Wpedantic -pthread -O2 -std=c++17 -Wno-deprecated-declarations")
set(CMAKE_BUILD_TYPE Debug)

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

# find other libraries: eigen3, acados
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)

option(ACADOS_PATH "Path to acados installation" $ENV{ACADOS_SOURCE_DIR})

if(NOT ACADOS_PATH)
  message(FATAL_ERROR "ACADOS_PATH not set")
endif()

message(STATUS "ACADOS_PATH: ${ACADOS_PATH}")
find_library(ACADOS_LIB acados PATHS "${ACADOS_PATH}/lib")

# define paths for tracks
add_compile_definitions(TRACK_DATABASE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../track-database/track_database/data")
add_compile_definitions(TRACKS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tracks")

# ocp and sim json
add_compile_definitions(OCP_JSON_PATH=" ${CMAKE_CURRENT_SOURCE_DIR}/../../ihm2_ocp.json ")

# generate custom message and service types
set(msg_files
  "msg/Controls.msg"
  "msg/ConesObservations.msg"
)
set(srv_files
  "srv/String.srv"
)
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES std_msgs # Add packages that above messages depend on
)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)

# ihm2 common
ament_auto_add_library(ihm2_common SHARED
  include/ihm2/external/rapidcsv.hpp
  include/ihm2/external/icecream.hpp
  include/ihm2/common/cone_color.hpp
  include/ihm2/common/marker_color.hpp
  include/ihm2/common/math.hpp
  include/ihm2/common/tracks.hpp
  src/common/cone_color.cpp
  src/common/marker_color.cpp
  src/common/math.cpp
  src/common/tracks.cpp
)
target_link_libraries(ihm2_common Eigen3::Eigen nlohmann_json::nlohmann_json)

# generate sim node
find_library(IHM2_KIN6_SIM_LIB acados_sim_solver_ihm2_kin6 PATHS "generated/ihm2_kin6_sim_gen_code")
find_library(IHM2_DYN6_SIM_LIB acados_sim_solver_ihm2_dyn6 PATHS "generated/ihm2_dyn6_sim_gen_code")
message(STATUS "IHM2_KIN6_SIM_LIB: ${IHM2_KIN6_SIM_LIB}")
message(STATUS "IHM2_DYN6_SIM_LIB: ${IHM2_DYN6_SIM_LIB}")
ament_auto_add_executable(sim_node src/sim_node.cpp)
target_link_libraries(sim_node
  "${cpp_typesupport_target}" # for custom message and service types defined in this package
  ${ACADOS_LIB} # acados library
  ${IHM2_KIN6_SIM_LIB} # acados sim solver generated code
  ${IHM2_DYN6_SIM_LIB} # acados sim solver generated code
  ihm2_common # ihm2 common library
)
target_include_directories(sim_node PUBLIC
  "${ACADOS_PATH}/include"
  "${ACADOS_PATH}/include/acados"
  "generated/ihm2_kin6_sim_gen_code"
  "generated/ihm2_kin6_sim_gen_code/ihm2_kin6_model"
  "generated/ihm2_dyn6_sim_gen_code"
  "generated/ihm2_dyn6_sim_gen_code/ihm2_dyn6_model"
)

# stanley control node
ament_auto_add_executable(stanley_control_node src/stanley_control_node.cpp)
target_link_libraries(stanley_control_node
  "${cpp_typesupport_target}" # for custom message and service types defined in this package
  ihm2_common # ihm2 common library
)

# launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)
install(DIRECTORY
  meshes
  DESTINATION share/${PROJECT_NAME}/
)

# generate control node
# find_library(IHM2_FKIN4_OCP_LIB acados_ocp_solver_ihm2_fkin4 PATHS ihm2_ocp_gen_code)
# message(STATUS " IHM2_FKIN4_OCP_LIB: ${IHM2_FKIN4_OCP_LIB} ")
# ament_auto_add_executable(control_node src/control_node.cpp)
# target_link_libraries(control_node
# " ${cpp_typesupport_target} " # for custom message and service types defined in this package
# ${ACADOS_LIB} # acados library
# ${IHM2_FKIN4_OCP_LIB} # acados ocp solver generated code
# ihm2_common # ihm2 common library
# )
# target_include_directories(control_node PUBLIC
# " ${ACADOS_PATH}/include "
# " ${ACADOS_PATH}/include/acados "
# " ${ACADOS_PATH}/include/acados_c "
# " ${ACADOS_PATH}/include/blasfeo/include "
# " ${ACADOS_PATH}/include/hpipm/include "
# " ihm2_ocp_gen_code "
# " ihm2_ocp_gen_code/ihm2_fkin4_model "
# " ihm2_ocp_gen_code/ihm2_fkin4_constraints "
# " ihm2_ocp_gen_code/ihm2_fkin4_cost "
# )

# install everything in the ihm2 package
ament_auto_package(INSTALL_TO_SHARE)