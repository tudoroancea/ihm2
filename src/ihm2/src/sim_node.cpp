// Copyright (c) 2023. Tudor Oancea
#include "acados/utils/math.h"
#include "acados_c/sim_interface.h"
#include "acados_sim_solver_ihm2_kin4.h"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "ihm2/common/marker_color.hpp"
#include "ihm2/msg/controls.hpp"
#include "ihm2/srv/string.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "std_msgs/msg/header.hpp"
#include "std_srvs/srv/empty.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

using namespace std;

geometry_msgs::msg::Quaternion rpy_to_quaternion(double roll, double pitch, double yaw) {
    tf2::Quaternion q;
    q.setRPY(roll, pitch, yaw);
    return tf2::toMsg(q);
}

visualization_msgs::msg::Marker get_car_marker(double X, double Y, double phi) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "world";
    marker.ns = "car";
    marker.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
    marker.action = visualization_msgs::msg::Marker::MODIFY;
    marker.mesh_resource = "https://github.com/tudoroancea/ihm2/releases/download/lego-lrt4/lego-lrt4.stl";
    marker.pose.position.x = X;
    marker.pose.position.y = Y;
    marker.pose.position.z = 0.0;
    marker.pose.orientation = rpy_to_quaternion(0.0, 0.0, M_PI_2 + phi);
    marker.scale.x = 0.3;
    marker.scale.y = 0.3;
    marker.scale.z = 0.3;
    marker.color = marker_colors("white");
    return marker;
}

visualization_msgs::msg::Marker get_cone_marker(uint64_t id, double X, double Y, std::string color, bool small, bool mesh = true) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "world";
    marker.ns = "cones";
    marker.id = id;
    marker.action = visualization_msgs::msg::Marker::MODIFY;
    if (mesh) {
        marker.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
        marker.mesh_resource = "https://github.com/tudoroancea/ihm2/releases/download/lego-lrt4/cone.stl";
        marker.scale.x = 0.001;
        marker.scale.y = 0.001;
        marker.scale.z = 0.001;
        marker.pose.orientation = rpy_to_quaternion(0.0, M_PI / 2, 0.0);
        if (small) {
            marker.scale.x *= 325 / 228;
            marker.scale.y *= 325 / 228;
            marker.scale.z *= 325 / 228;
        }
    } else {
        marker.type = visualization_msgs::msg::Marker::ARROW;
        marker.scale.x = 0.0;
        marker.scale.y = small ? 0.228 : 0.285;
        marker.scale.z = small ? 0.325 : 0.505;
        marker.points.resize(2);
        marker.points[0].x = X;
        marker.points[0].y = Y;
        marker.points[0].z = 0.0;
        marker.points[1].x = X;
        marker.points[1].y = Y;
        marker.points[1].z = small ? 0.325 : 0.505;
    }
    marker.color = marker_colors(color);
    return marker;
}

class SimNode : public rclcpp::Node {
private:
    ihm2::msg::Controls::SharedPtr controls_msg;
    rclcpp::Subscription<ihm2::msg::Controls>::SharedPtr controls_sub;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_srv;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr publish_cones_srv;
    rclcpp::Service<ihm2::srv::String>::SharedPtr load_map_srv;

    ihm2_kin4_sim_solver_capsule* acados_sim_capsule;
    sim_config* acados_sim_config;
    sim_in* acados_sim_in;
    sim_out* acados_sim_out;
    void* acados_sim_dims;

    void controls_callback(const ihm2::msg::Controls::SharedPtr msg) {
        controls_msg = msg;
    }

public:
    SimNode() : Node("sim_node") {
        // parameters:
        // - track_name_or_file: the name of the track or the path to the track file
        // - sim_dt: the simulation time step
        // - T_max: the maximum torque
        // - T_min: the minimum torque
        // - delta_max: the maximum steering angle
        // - delta_min: the minimum steering angle
        // - manual_control: if true, the car can be controlled by the user
        //

        auto track_name_or_file = declare_parameter<std::string>("track_name_or_file", "fsds_competition_2");

        // load acados sim solver
        // check if the sim solver was generated for the track in question
        acados_sim_capsule = ihm2_kin4_acados_sim_solver_create_capsule();
        int status = ihm2_kin4_acados_sim_create(acados_sim_capsule);
        if (status) {
            printf("acados_create() returned status %d. Exiting.\n", status);
            rclcpp::shutdown();
            exit(1);
        }
        acados_sim_config = ihm2_kin4_acados_get_sim_config(acados_sim_capsule);
        acados_sim_in = ihm2_kin4_acados_get_sim_in(acados_sim_capsule);
        acados_sim_out = ihm2_kin4_acados_get_sim_out(acados_sim_capsule);
        acados_sim_dims = ihm2_kin4_acados_get_sim_dims(acados_sim_capsule);
        // publishers
        // services
        // create a timer for the simulation loop (one simulation step and publishing the car mesh)
    }

    ~SimNode() {
        int status = ihm2_kin4_acados_sim_free(acados_sim_capsule);
        if (status) {
            printf("ihm2_kin4_acados_sim_free() returned status %d. \n", status);
        }

        ihm2_kin4_acados_sim_solver_free_capsule(acados_sim_capsule);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SimNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
