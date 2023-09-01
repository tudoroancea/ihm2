// Copyright (c) 2023. Tudor Oancea
#include "acados/utils/math.h"
#include "acados_c/sim_interface.h"
#include "acados_sim_solver_ihm2_kin4.h"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "eigen3/Eigen/Eigen"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "ihm2/common/cone_color.hpp"
#include "ihm2/common/marker_color.hpp"
#include "ihm2/common/track_database.hpp"
#include "ihm2/external/icecream.hpp"
#include "ihm2/msg/controls.hpp"
#include "ihm2/srv/string.hpp"
#include "nlohmann/json.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/header.hpp"
#include "std_srvs/srv/empty.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include <fstream>
#include <unordered_map>

#define NX IHM2_KIN4_NX
#define NU IHM2_KIN4_NU

using namespace std;

double clip(double n, double lower, double upper) {
    return std::max(lower, std::min(n, upper));
}

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
    marker.pose.position.z = 0.0225;
    marker.pose.orientation = rpy_to_quaternion(0.0, 0.0, M_PI_2 + phi);
    marker.scale.x = 0.03;
    marker.scale.y = 0.03;
    marker.scale.z = 0.03;
    marker.color = marker_colors("white");
    return marker;
}

visualization_msgs::msg::Marker get_cone_marker(uint64_t id, double X, double Y, std::string color, bool small, bool mesh = true) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "world";
    marker.ns = color + "_cones";
    marker.id = id;
    marker.action = visualization_msgs::msg::Marker::MODIFY;
    if (mesh) {
        marker.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
        marker.mesh_resource = "https://github.com/tudoroancea/ihm2/releases/download/lego-lrt4/cone.stl";
        marker.scale.x = 0.001;
        marker.scale.y = 0.001;
        marker.scale.z = 0.001;
        marker.pose.position.x = X;
        marker.pose.position.y = Y;
        marker.pose.orientation = rpy_to_quaternion(-M_PI_2, 0.0, 0.0);
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
    rclcpp::Subscription<ihm2::msg::Controls>::SharedPtr controls_sub;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr alternative_controls_sub;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_srv;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr publish_cones_srv;
    rclcpp::Service<ihm2::srv::String>::SharedPtr load_map_srv;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr viz_pub;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr vel_pub;
    rclcpp::Publisher<ihm2::msg::Controls>::SharedPtr controls_pub;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;
    // rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub;
    // rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr last_lap_time_pub;
    // rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr best_lap_time_pub;

    rclcpp::TimerBase::SharedPtr sim_timer;
    // rclcpp::Time sim_start_time;
    ihm2::msg::Controls controls_msg, controls_target_msg;
    visualization_msgs::msg::MarkerArray cones_marker_array;
    geometry_msgs::msg::PoseStamped pose_msg;
    geometry_msgs::msg::TwistStamped vel_msg;
    geometry_msgs::msg::TransformStamped transform;

    double* x;
    double* u;
    ihm2_kin4_sim_solver_capsule* acados_sim_capsule;
    sim_config* acados_sim_config;
    sim_in* acados_sim_in;
    sim_out* acados_sim_out;
    void* acados_sim_dims;

    void controls_callback(const ihm2::msg::Controls::SharedPtr msg) {
        if (!this->get_parameter("manual_control").as_bool()) {
            u[0] = clip(
                    msg->throttle,
                    this->get_parameter("T_min").as_double(),
                    this->get_parameter("T_max").as_double());
            u[1] = clip(
                    u[1],
                    this->get_parameter("delta_min").as_double(),
                    this->get_parameter("delta_max").as_double());
        }
    }

    void alternative_controls_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        if (this->get_parameter("manual_control").as_bool()) {
            u[0] = clip(
                    msg->linear.x,
                    this->get_parameter("T_min").as_double(),
                    this->get_parameter("T_max").as_double());
            u[1] = clip(
                    msg->angular.z,
                    this->get_parameter("delta_min").as_double(),
                    this->get_parameter("delta_max").as_double());
        }
    }

    void publish_cones_srv_cb([[maybe_unused]] const std_srvs::srv::Empty::Request::SharedPtr request, [[maybe_unused]] std_srvs::srv::Empty::Response::SharedPtr response) {
        this->viz_pub->publish(cones_marker_array);
    }

    void reset_srv_cb([[maybe_unused]] const std_srvs::srv::Empty::Request::SharedPtr request, [[maybe_unused]] std_srvs::srv::Empty::Response::SharedPtr response) {
        // reset the simulation
        // - reset the car pose
        // - reset the lap time
        // - reset the simulation time
        // - reset the acados sim solver
    }

    void load_map_srv_cb([[maybe_unused]] const ihm2::srv::String::Request::SharedPtr request, [[maybe_unused]] ihm2::srv::String::Response::SharedPtr response) {
        // load the map
        // - load the center line
        // - load the cones
        // - publish the cones once
    }

    void sim_timer_cb() {
        // simulate one step
        // - set the controls
        // - solve the simulation
        // - publish the car pose
        // - publish the car velocity
        // - publish the car mesh

        // set inputs
        sim_in_set(acados_sim_config, acados_sim_dims,
                   acados_sim_in, "x", x);
        sim_in_set(acados_sim_config, acados_sim_dims,
                   acados_sim_in, "u", u);

        // solve
        int status = ihm2_kin4_acados_sim_solve(acados_sim_capsule);
        if (status != ACADOS_SUCCESS) {
            printf("acados_solve() failed with status %d.\n", status);
        }

        // get outputs
        sim_out_get(acados_sim_config, acados_sim_dims,
                    acados_sim_out, "x", x);

        // override the pose and velocity and controls with the simulation output
        pose_msg.header.stamp = this->now();
        pose_msg.header.frame_id = "world";
        pose_msg.pose.position.x = x[0];
        pose_msg.pose.position.y = x[1];
        pose_msg.pose.orientation = rpy_to_quaternion(0.0, 0.0, x[2]);
        pose_pub->publish(pose_msg);

        transform.header.stamp = this->now();
        transform.header.frame_id = "world";
        transform.child_frame_id = "car";
        transform.transform.translation.x = x[0];
        transform.transform.translation.y = x[1];
        transform.transform.rotation = rpy_to_quaternion(0.0, 0.0, x[2]);
        tf_broadcaster->sendTransform(transform);

        vel_msg.header.stamp = this->now();
        vel_msg.header.frame_id = "car";
        vel_msg.twist.linear.x = x[3];
        vel_pub->publish(vel_msg);

        controls_msg.header.stamp = this->now();
        controls_msg.header.frame_id = "car";
        controls_msg.throttle = x[NX - 2];
        controls_msg.steering = x[NX - 1];
        controls_pub->publish(controls_msg);

        visualization_msgs::msg::MarkerArray markers_msg;
        markers_msg.markers.push_back(get_car_marker(x[0], x[1], x[2]));
        viz_pub->publish(markers_msg);
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
        this->declare_parameter<std::string>("track_name_or_file", "fsds_competition_2");
        this->declare_parameter<double>("T_max", 2000.0);
        this->declare_parameter<double>("T_min", -2000.0);
        this->declare_parameter<double>("delta_max", 0.5);
        this->declare_parameter<double>("delta_min", -0.5);
        this->declare_parameter<bool>("manual_control", true);

        // initialize x and u with zeros
        x = (double*) malloc(sizeof(double) * NX);
        u = (double*) malloc(sizeof(double) * NU);
        for (int i = 0; i < NX; i++) {
            x[i] = 0.0;
        }
        for (int i = 0; i < NU; i++) {
            u[i] = 0.0;
        }
        x[2] = M_PI_2;

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
        this->viz_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("/ihm2/viz", 10);
        this->pose_pub = this->create_publisher<geometry_msgs::msg::PoseStamped>("/ihm2/pose", 10);
        this->vel_pub = this->create_publisher<geometry_msgs::msg::TwistStamped>("/ihm2/velocity", 10);
        this->controls_pub = this->create_publisher<ihm2::msg::Controls>("/ihm2/controls", 10);
        this->tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(this);

        // subscribers
        this->controls_sub = this->create_subscription<ihm2::msg::Controls>(
                "/ihm2/controls_target",
                10,
                std::bind(
                        &SimNode::controls_callback,
                        this,
                        std::placeholders::_1));
        this->alternative_controls_sub = this->create_subscription<geometry_msgs::msg::Twist>(
                "/ihm2/alternative_controls_target",
                10,
                std::bind(
                        &SimNode::alternative_controls_callback,
                        this,
                        std::placeholders::_1));

        // services
        this->reset_srv = this->create_service<std_srvs::srv::Empty>(
                "/ihm2/reset",
                std::bind(
                        &SimNode::reset_srv_cb,
                        this,
                        std::placeholders::_1,
                        std::placeholders::_2));
        this->publish_cones_srv = this->create_service<std_srvs::srv::Empty>(
                "/ihm2/publish_cones",
                std::bind(
                        &SimNode::publish_cones_srv_cb,
                        this,
                        std::placeholders::_1,
                        std::placeholders::_2));
        this->load_map_srv = this->create_service<ihm2::srv::String>(
                "/ihm2/load_map",
                std::bind(
                        &SimNode::load_map_srv_cb,
                        this,
                        std::placeholders::_1,
                        std::placeholders::_2));

        // load cones from track file and create the markers for the cones
        std::unordered_map<ConeColor, Eigen::MatrixX2d> cones_map = load_cones(this->get_parameter("track_name_or_file").as_string());
        cones_marker_array.markers.clear();
        for (auto& [color, cones] : cones_map) {
            for (int i = 0; i < cones.rows(); i++) {
                cones_marker_array.markers.push_back(
                        get_cone_marker(
                                i,
                                cones(i, 0),
                                cones(i, 1),
                                (color == ConeColor::BLUE || color == ConeColor::YELLOW) ? to_string(color) : "orange",
                                color != ConeColor::BIG_ORANGE,
                                true));
            }
        }

        // find simulation time step from the JSON config file generated by acados
        double sim_dt(0.01);
#ifdef SIM_JSON_PATH
        std::string sim_json_path = SIM_JSON_PATH;
        std::ifstream sim_json_file(sim_json_path);
        if (!sim_json_file.is_open()) {
            throw std::runtime_error("Could not open " + sim_json_path);
        }
        nlohmann::json sim_json;
        sim_json_file >> sim_json;
        sim_dt = sim_json["solver_options"]["Tsim"];
#else
        throw std::runtime_error("SIM_JSON_PATH not defined");
#endif
        // create a timer for the simulation loop (one simulation step and publishing the car mesh)
        this->sim_timer = this->create_wall_timer(
                std::chrono::duration<double>(sim_dt),
                std::bind(
                        &SimNode::sim_timer_cb,
                        this));
    }

    ~SimNode() {
        int status = ihm2_kin4_acados_sim_free(acados_sim_capsule);
        if (status) {
            printf("ihm2_kin4_acados_sim_free() returned status %d. \n", status);
        }
        ihm2_kin4_acados_sim_solver_free_capsule(acados_sim_capsule);
        free(x);
        free(u);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SimNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
