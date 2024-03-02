# Copyright (c) 2023. Tudor Oancea
cmake_minimum_required(VERSION 3.8)
project(ihm2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra -Wpedantic -pthread -O2 -std=c++17 -Wno-deprecated-declarations")
set(CMAKE_BUILD_TYPE Debug)

# find ROS 2 dependencies
find_package(ament_cmake_auto REQUIRED)
find_package(rosidl_default_generators REQUIRED)
ament_auto_find_build_dependencies()

# find other libraries: eigen3, nlohmann json, curl, acados
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(CURL REQUIRED)
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

# if on WSL then add preprocessor definition
if(DEFINED ENV{WSLENV})
  add_compile_definitions(WSL)
endif()

# generate custom message and service types
set(msg_files
  "msg/Controls.msg"
  "msg/State.msg"
)
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES std_msgs geometry_msgs
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
  CURL::libcurl
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
find_library(IHM2_FKIN6_MPC_LIB acados_ocp_solver_ihm2_fkin6 PATHS "${CMAKE_CURRENT_SOURCE_DIR}/generated/ihm2_fkin6_mpc_gen_code")
find_library(IHM2_FKIN6_SIM_LIB acados_sim_solver_ihm2_fkin6 PATHS "${CMAKE_CURRENT_SOURCE_DIR}/generated/ihm2_fkin6_mpc_gen_code")
message(STATUS "IHM2_FKIN6_MPC_LIB: ${IHM2_FKIN6_MPC_LIB}, IHM_FKIN6_SIM_LIB: ${IHM2_FKIN6_SIM_LIB}")
ament_auto_add_executable(mpc_control_node src/mpc_control_node.cpp)
target_link_libraries(mpc_control_node
  ${cpp_typesupport_target} # for custom message and service types defined in this package
  ${ACADOS_LIB} # acados library
  ${IHM2_FKIN6_MPC_LIB} # acados ocp solver generated code
  ${IHM2_FKIN6_SIM_LIB} # acados sim solver generated code
  ihm2_common # ihm2 common library
)
target_include_directories(mpc_control_node PUBLIC
  "${ACADOS_PATH}/include"
  "${ACADOS_PATH}/include/acados"
  "${ACADOS_PATH}/include/acados_c"
  "${ACADOS_PATH}/include/blasfeo/include"
  "${ACADOS_PATH}/include/hpipm/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/generated/ihm2_fkin6_mpc_gen_code"
)

# install everything in the ihm2 package
ament_auto_package(INSTALL_TO_SHARE)
