// Copyright (c) 2023. Tudor Oancea
#include "acados/utils/math.h"
#include "acados_c/sim_interface.h"
#include "acados_sim_solver_ihm2_dyn6.h"
#include "acados_sim_solver_ihm2_kin6.h"
#include "ament_index_cpp/get_package_share_directory.hpp"
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
#include "ihm2/common/math.hpp"
#include "ihm2/common/tracks.hpp"
#include "ihm2/external/icecream.hpp"
#include "ihm2/msg/cones_observations.hpp"
#include "ihm2/msg/controls.hpp"
#include "ihm2/srv/string.hpp"
#include "nlohmann/json.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
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
#include <cmath>
#include <fstream>
#include <unordered_map>

using namespace std;

std::vector<visualization_msgs::msg::Marker> get_car_markers(double X, double Y, double phi, double delta, std::string car_mesh = "gotthard.stl") {
    std::vector<visualization_msgs::msg::Marker> markers(1);
    markers[0].header.frame_id = "world";
    markers[0].ns = "car";
    markers[0].type = visualization_msgs::msg::Marker::MESH_RESOURCE;
    markers[0].action = visualization_msgs::msg::Marker::MODIFY;
    markers[0].mesh_resource = "file://" + ament_index_cpp::get_package_share_directory("ihm2") + "/meshes/" + car_mesh;
    markers[0].pose.position.x = X;
    markers[0].pose.position.y = Y;
    markers[0].pose.orientation = rpy_to_quaternion(0.0, 0.0, phi);
    markers[0].scale.x = 1.0;
    markers[0].scale.y = 1.0;
    markers[0].scale.z = 1.0;
    markers[0].color = marker_colors("white");
    if (car_mesh == "gotthard.stl") {
        // add the 4 wheels at points (0.819, 0.6), (0.819, -0.6), (-0.711, 0.6), (-0.711, -0.6)
        // and add a yaw of delta to the first two
        markers.resize(5);
        for (size_t i(1); i < 5; ++i) {
            markers[i].header.frame_id = "world";
            markers[i].ns = "car";
            markers[i].id = i;
            markers[i].type = visualization_msgs::msg::Marker::MESH_RESOURCE;
            markers[i].mesh_resource = "file://" + ament_index_cpp::get_package_share_directory("ihm2") + "/meshes/gotthard_wheel.stl";
            markers[i].action = visualization_msgs::msg::Marker::MODIFY;
            double wheel_x(i < 3 ? 0.819 : -0.711), wheel_y(i % 2 == 0 ? 0.6 : -0.6), wheel_phi(i < 3 ? delta : 0.0);
            markers[i].pose.position.x = X + wheel_x * std::cos(phi) - wheel_y * std::sin(phi);
            markers[i].pose.position.y = Y + wheel_x * std::sin(phi) + wheel_y * std::cos(phi);
            markers[i].pose.position.z = 0.232;
            markers[i].pose.orientation = rpy_to_quaternion(0.0, 0.0, phi + wheel_phi);
            markers[i].scale.x = 1.0;
            markers[i].scale.y = 1.0;
            markers[i].scale.z = 1.0;
            markers[i].color = marker_colors("black");
        }
    }
    return markers;
}

visualization_msgs::msg::Marker get_cone_marker(uint64_t id, double X, double Y, std::string color, bool small, bool mesh = true) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "world";
    marker.ns = color + "_cones";
    marker.id = id;
    marker.action = visualization_msgs::msg::Marker::MODIFY;
    if (mesh) {
        marker.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
        marker.mesh_resource = "file://" + ament_index_cpp::get_package_share_directory("ihm2") + "/meshes/cone.stl";
        marker.scale.x = 1.0;
        marker.scale.y = 1.0;
        marker.scale.z = 1.0;
        marker.pose.position.x = X;
        marker.pose.position.y = Y;
        marker.pose.orientation = rpy_to_quaternion(0.0, 0.0, 0.0);
        if (!small) {
            marker.scale.z *= (285.0 / 228.0);
            marker.scale.x *= (285.0 / 228.0);
            marker.scale.y *= (505.0 / 325.0);
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

class FatalNodeError : public std::runtime_error {
public:
    FatalNodeError(const std::string& what) : std::runtime_error(what) {}
};

class NodeError : public std::runtime_error {
public:
    NodeError(const std::string& what) : std::runtime_error(what) {}
};

class SimNode : public rclcpp::Node {
private:
    // publishers
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr viz_pub;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr vel_pub;
    rclcpp::Publisher<ihm2::msg::Controls>::SharedPtr controls_pub;
    rclcpp::Publisher<ihm2::msg::ConesObservations>::SharedPtr cones_observations_pub;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

    // subscribers
    rclcpp::Subscription<ihm2::msg::Controls>::SharedPtr controls_sub;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr alternative_controls_sub;

    // services
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_srv;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr publish_cones_srv;

    // simulation variables
    double* x;
    double* u;
    size_t nx, nu;
    rclcpp::TimerBase::SharedPtr sim_timer;
    rclcpp::TimerBase::SharedPtr cones_observations_timer;
    visualization_msgs::msg::MarkerArray cones_marker_array;
    geometry_msgs::msg::PoseStamped pose_msg;
    geometry_msgs::msg::TwistStamped vel_msg;
    geometry_msgs::msg::TransformStamped transform;
    Eigen::MatrixX2d all_cones;

    // acados sim solver variables
    void* kin6_sim_capsule;
    sim_config* kin6_sim_config;
    sim_in* kin6_sim_in;
    sim_out* kin6_sim_out;
    void* kin6_sim_dims;
    void* dyn6_sim_capsule;
    sim_config* dyn6_sim_config;
    sim_in* dyn6_sim_in;
    sim_out* dyn6_sim_out;
    void* dyn6_sim_dims;

    void controls_callback(const ihm2::msg::Controls::SharedPtr msg) {
        if (!this->get_parameter("manual_control").as_bool()) {
            double T_max = this->get_parameter("T_max").as_double(), delta_max = this->get_parameter("delta_max").as_double();
            u[0] = clip(msg->throttle, -T_max, T_max);
            u[1] = clip(msg->steering, -delta_max, delta_max);
        }
    }

    void alternative_controls_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        if (this->get_parameter("manual_control").as_bool()) {
            double T_max = this->get_parameter("T_max").as_double(), delta_max = this->get_parameter("delta_max").as_double();
            u[0] = clip(msg->linear.x, -T_max, T_max);
            u[1] = clip(msg->angular.z, -delta_max, delta_max);
        }
    }

    void publish_cones_srv_cb([[maybe_unused]] const std_srvs::srv::Empty::Request::SharedPtr request, [[maybe_unused]] std_srvs::srv::Empty::Response::SharedPtr response) {
        this->viz_pub->publish(cones_marker_array);
    }

    void reset_srv_cb([[maybe_unused]] const std_srvs::srv::Empty::Request::SharedPtr request, [[maybe_unused]] std_srvs::srv::Empty::Response::SharedPtr response) {
        for (size_t i = 0; i < nx; i++) {
            x[i] = 0.0;
        }
        x[2] = M_PI_2;
        for (size_t i = 0; i < nu; i++) {
            u[i] = 0.0;
        }
        // create string containing new values of x and u
        std::stringstream ss;
        ss << std::fixed << std::setprecision(3);
        ss << "Reset simulation to x=[";
        for (size_t i = 0; i < nx; i++) {
            ss << x[i];
            if (i < nx - 1) {
                ss << ", ";
            }
        }
        ss << "] and u=[";
        for (size_t i = 0; i < nu; i++) {
            ss << u[i];
            if (i < nu - 1) {
                ss << ", ";
            }
        }
        ss << "]";
        RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
    }

    void sim_timer_cb() {
        auto start = this->now();
        // depending on the last velocity v=sqrt(v_x^2+v_y^2), decide which model to use and set its inputs
        bool use_kin6(std::hypot(x[3], x[4]) < this->get_parameter("v_dyn").as_double());
        try {
            if (use_kin6) {
                sim_in_set(kin6_sim_config,
                           kin6_sim_dims,
                           kin6_sim_in,
                           "x",
                           this->x);
                sim_in_set(kin6_sim_config,
                           kin6_sim_dims,
                           kin6_sim_in,
                           "u",
                           this->u);
            } else {
                sim_in_set(dyn6_sim_config,
                           dyn6_sim_dims,
                           dyn6_sim_in,
                           "x",
                           this->x);
                sim_in_set(dyn6_sim_config,
                           dyn6_sim_dims,
                           dyn6_sim_in,
                           "u",
                           this->u);
            }

        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Caught exception: %s", e.what());
        }

        // call sim solver
        int status = (use_kin6) ? ihm2_kin6_acados_sim_solve((ihm2_kin6_sim_solver_capsule*) kin6_sim_capsule) : ihm2_dyn6_acados_sim_solve((ihm2_dyn6_sim_solver_capsule*) dyn6_sim_capsule);
        if (status != ACADOS_SUCCESS) {
            throw FatalNodeError("acados_solve() failed with status " + std::to_string(status) + " for solver " + (use_kin6 ? "kin6" : "dyn6") + ".");
        }

        // get sim solver outputs
        if (use_kin6) {
            sim_out_get(kin6_sim_config,
                        kin6_sim_dims,
                        kin6_sim_out,
                        "x",
                        this->x);
        } else {
            sim_out_get(
                    dyn6_sim_config,
                    dyn6_sim_dims,
                    dyn6_sim_out,
                    "x",
                    this->x);
        }

        // prohibit the car from going backwards
        if (x[3] < 0.0 or (x[nx - 2] <= 0.1 and x[3] < 0.01)) {
            x[3] = 0.0;
            x[4] = 0.0;
            x[5] = 0.0;
        }

        auto end = this->now();

        // override the pose and velocity and controls with the simulation output
        this->pose_msg.header.stamp = this->now();
        this->pose_msg.header.frame_id = "world";
        this->pose_msg.pose.position.x = x[0];
        this->pose_msg.pose.position.y = x[1];
        this->pose_msg.pose.orientation = rpy_to_quaternion(0.0, 0.0, x[2]);
        this->pose_pub->publish(this->pose_msg);

        this->transform.header.stamp = this->now();
        this->transform.header.frame_id = "world";
        this->transform.child_frame_id = "car";
        this->transform.transform.translation.x = x[0];
        this->transform.transform.translation.y = x[1];
        this->transform.transform.rotation = this->pose_msg.pose.orientation;
        this->tf_broadcaster->sendTransform(this->transform);

        this->vel_msg.header.stamp = this->now();
        this->vel_msg.header.frame_id = "car";
        this->vel_msg.twist.linear.x = x[3];
        this->vel_msg.twist.linear.y = x[4];
        this->vel_msg.twist.angular.z = x[5];
        this->vel_pub->publish(this->vel_msg);

        ihm2::msg::Controls controls_msg;
        controls_msg.header.stamp = this->now();
        controls_msg.header.frame_id = "car";
        controls_msg.throttle = x[nx - 2];
        controls_msg.steering = x[nx - 1];
        this->controls_pub->publish(controls_msg);

        visualization_msgs::msg::MarkerArray markers_msg;
        std::vector<visualization_msgs::msg::Marker> car_markers = get_car_markers(x[0], x[1], x[2], x[nx - 1], this->get_parameter("car_mesh").as_string());
        for (const auto& marker : car_markers) {
            markers_msg.markers.push_back(marker);
        }
        this->viz_pub->publish(markers_msg);

        // publish diagnostics
        diagnostic_msgs::msg::DiagnosticArray diag_msg;
        diag_msg.header.stamp = this->now();
        diag_msg.status.resize(1);
        diag_msg.status[0].name = "sim";
        diag_msg.status[0].level = diagnostic_msgs::msg::DiagnosticStatus::OK;
        diag_msg.status[0].message = "OK";

        diagnostic_msgs::msg::KeyValue sim_runtime_kv;
        sim_runtime_kv.key = "sim runtime (ms)";
        sim_runtime_kv.value = std::to_string(1000 * (end - start).seconds());
        diag_msg.status[0].values.push_back(sim_runtime_kv);

        diagnostic_msgs::msg::KeyValue track_name_kv;
        track_name_kv.key = "track name";
        track_name_kv.value = this->get_parameter("track_name_or_file").as_string();
        diag_msg.status[0].values.push_back(track_name_kv);

        diagnostic_msgs::msg::KeyValue model_kv;
        model_kv.key = "model";
        model_kv.value = use_kin6 ? "kin6" : "dyn6";
        diag_msg.status[0].values.push_back(model_kv);

        this->diag_pub->publish(diag_msg);
    }


    void publish_cones_observations_cb() {
        // get current position and yaw
        double X(x[0]), Y(x[1]), phi(x[2]);
        Eigen::Vector2d pos(X, Y);
        // get bearing and range limits
        std::vector<double> range_limits(this->get_parameter("range_limits").as_double_array()), bearing_limits(this->get_parameter("bearing_limits").as_double_array());
        // compute thee postions of the cones relative to the car
        // Eigen::MatrixX2d cartesian = this->all_cones.rowwise() - pos;
        // Eigen::Rotation2Dd rot(phi);
        // // apply rotation to the cones
        // Eigen::MatrixX2d rotated = rot * cartesian.transpose();
    }
    void create_cones_markers(const std::string& track_name_or_file) {
        std::unordered_map<ConeColor, Eigen::MatrixX2d> cones_map = load_cones(track_name_or_file);
        this->all_cones = Eigen::MatrixX2d::Zero(0, 2);

        // set deleteall to all the markers in the cones_marker_array and publish it
        for (auto& marker : cones_marker_array.markers) {
            marker.action = visualization_msgs::msg::Marker::DELETEALL;
        }
        this->viz_pub->publish(cones_marker_array);
        // create new cones
        cones_marker_array.markers.clear();
        for (auto& [color, cones] : cones_map) {
            this->all_cones.conservativeResize(this->all_cones.rows() + cones.rows(), 2);
            this->all_cones.bottomRows(cones.rows()) = cones;
            for (int i = 0; i < cones.rows(); i++) {
                cones_marker_array.markers.push_back(
                        get_cone_marker(
                                i,
                                cones(i, 0),
                                cones(i, 1),
                                is_orange(color) ? "orange" : cone_color_to_string(color),
                                color != ConeColor::BIG_ORANGE,
                                this->get_parameter("use_meshes").as_bool()));
            }
        }
        RCLCPP_INFO(this->get_logger(), "Loaded %lu cones from %s", cones_marker_array.markers.size(), track_name_or_file.c_str());
    }

public:
    SimNode() : Node("sim_node") {
        // parameters:
        // - track_name_or_file: the name of the track or the path to the track file
        // - T_max: the maximum torque
        // - T_min: the minimum torque
        // - delta_max: the maximum steering angle
        // - delta_min: the minimum steering angle
        // - manual_control: if true, the car can be controlled by the user
        // - use_meshes: if true, the car and cones are represented by meshes, otherwise by arrows
        // - v_dyn: from what speed the dynamic model should be used
        this->declare_parameter<std::string>("track_name_or_file", "fsds_competition_2");
        this->declare_parameter<double>("T_max", 1107.0);
        this->declare_parameter<double>("delta_max", 0.5);
        this->declare_parameter<bool>("manual_control", true);
        this->declare_parameter<bool>("use_meshes", true);
        this->declare_parameter<double>("v_dyn", 3.0);
        this->declare_parameter<double>("cones_observations_freq", 10.0);
        this->declare_parameter<std::vector<double>>("range_limits", {0.0, 15.0});
        this->declare_parameter<std::vector<double>>("bearing_limits", {-deg2rad(50.0), deg2rad(50.0)});
        this->declare_parameter<std::string>("car_mesh", "gotthard.stl");

        // initialize x and u with zeros
        nx = IHM2_DYN6_NX;
        nu = IHM2_DYN6_NU;
        x = (double*) malloc(sizeof(double) * nx);
        u = (double*) malloc(sizeof(double) * nu);
        this->reset_srv_cb(nullptr, nullptr);
        RCLCPP_INFO(this->get_logger(), "Initialized x and u with sizes %zu and %zu", nx, nu);

        // load acados sim solvers (for kin6 and dyn6 models)
        kin6_sim_capsule = ihm2_kin6_acados_sim_solver_create_capsule();
        dyn6_sim_capsule = ihm2_dyn6_acados_sim_solver_create_capsule();
        int status = ihm2_kin6_acados_sim_create((ihm2_kin6_sim_solver_capsule*) kin6_sim_capsule);
        if (status) {
            throw FatalNodeError("ihm2_kin6_acados_sim_create() returned status " + std::to_string(status));
        }
        status = ihm2_dyn6_acados_sim_create((ihm2_dyn6_sim_solver_capsule*) dyn6_sim_capsule);
        if (status) {
            throw FatalNodeError("ihm2_dyn6_acados_sim_create() returned status " + std::to_string(status));
        }
        kin6_sim_config = ihm2_kin6_acados_get_sim_config((ihm2_kin6_sim_solver_capsule*) kin6_sim_capsule);
        kin6_sim_in = ihm2_kin6_acados_get_sim_in((ihm2_kin6_sim_solver_capsule*) kin6_sim_capsule);
        kin6_sim_out = ihm2_kin6_acados_get_sim_out((ihm2_kin6_sim_solver_capsule*) kin6_sim_capsule);
        kin6_sim_dims = ihm2_kin6_acados_get_sim_dims((ihm2_kin6_sim_solver_capsule*) kin6_sim_capsule);
        dyn6_sim_config = ihm2_dyn6_acados_get_sim_config((ihm2_dyn6_sim_solver_capsule*) dyn6_sim_capsule);
        dyn6_sim_in = ihm2_dyn6_acados_get_sim_in((ihm2_dyn6_sim_solver_capsule*) dyn6_sim_capsule);
        dyn6_sim_out = ihm2_dyn6_acados_get_sim_out((ihm2_dyn6_sim_solver_capsule*) dyn6_sim_capsule);
        dyn6_sim_dims = ihm2_dyn6_acados_get_sim_dims((ihm2_dyn6_sim_solver_capsule*) dyn6_sim_capsule);

        // publishers
        this->viz_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("/ihm2/viz/sim", 10);
        this->pose_pub = this->create_publisher<geometry_msgs::msg::PoseStamped>("/ihm2/pose", 10);
        this->vel_pub = this->create_publisher<geometry_msgs::msg::TwistStamped>("/ihm2/vel", 10);
        this->diag_pub = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/ihm2/diag/sim", 10);
        this->controls_pub = this->create_publisher<ihm2::msg::Controls>("/ihm2/current_controls", 10);
        this->cones_observations_pub = this->create_publisher<ihm2::msg::ConesObservations>("/ihm2/cones_observations", 10);
        this->tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(this);

        // subscribers
        this->controls_sub = this->create_subscription<ihm2::msg::Controls>(
                "/ihm2/target_controls",
                10,
                std::bind(
                        &SimNode::controls_callback,
                        this,
                        std::placeholders::_1));
        this->alternative_controls_sub = this->create_subscription<geometry_msgs::msg::Twist>(
                "/ihm2/alternative_target_controls",
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
                "/ihm2/publish_cones_markers",
                std::bind(
                        &SimNode::publish_cones_srv_cb,
                        this,
                        std::placeholders::_1,
                        std::placeholders::_2));

        // load cones from track file and create the markers for the cones
        this->create_cones_markers(this->get_parameter("track_name_or_file").as_string());

        // call once the the publish cones service
        this->publish_cones_srv_cb(nullptr, nullptr);

        // create a timer for the simulation loop (one simulation step and publishing the car mesh)
        this->sim_timer = this->create_wall_timer(
                std::chrono::duration<double>(1.0 / 100.0),
                std::bind(
                        &SimNode::sim_timer_cb,
                        this));
        this->cones_observations_timer = this->create_wall_timer(
                std::chrono::duration<double>(1.0 / this->get_parameter("cones_observations_freq").as_double()),
                std::bind(
                        &SimNode::publish_cones_observations_cb,
                        this));
    }

    ~SimNode() {
        int status = ihm2_kin6_acados_sim_free((ihm2_kin6_sim_solver_capsule*) kin6_sim_capsule);
        if (status) {
            RCLCPP_WARN(this->get_logger(), "ihm2_kin6_acados_sim_free() returned status %d.", status);
        }
        status = ihm2_dyn6_acados_sim_free((ihm2_dyn6_sim_solver_capsule*) dyn6_sim_capsule);
        if (status) {
            RCLCPP_WARN(this->get_logger(), "ihm2_dyn6_acados_sim_free() returned status %d.", status);
        }
        ihm2_kin6_acados_sim_solver_free_capsule((ihm2_kin6_sim_solver_capsule*) kin6_sim_capsule);
        ihm2_dyn6_acados_sim_solver_free_capsule((ihm2_dyn6_sim_solver_capsule*) dyn6_sim_capsule);
        free(x);
        free(u);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SimNode>();
    try {
        rclcpp::spin(node);
    } catch (const FatalNodeError& e) {
        RCLCPP_FATAL(node->get_logger(), "Caught exception: %s", e.what());
    } catch (...) {
        RCLCPP_ERROR(node->get_logger(), "Caught unknown exception");
    }
    rclcpp::shutdown();
    return 0;
}
