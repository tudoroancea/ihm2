// Copyright (c) 2023. Tudor Oancea
#include "acados/utils/math.h"
#include "acados_c/sim_interface.h"
#include "acados_sim_solver_ihm2_kin4.h"
#include "ihm2/msg/controls.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/color_rgba.hpp"

using namespace std;
using ColorRGBA = std_msgs::msg::ColorRGBA;

ColorRGBA marker_colors(string color) {
    ColorRGBA color_msg;
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


class SimNode : public rclcpp::Node {
private:
    ihm2::msg::Controls::SharedPtr controls_msg;
    rclcpp::Subscription<ihm2::msg::Controls>::SharedPtr controls_sub;
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
