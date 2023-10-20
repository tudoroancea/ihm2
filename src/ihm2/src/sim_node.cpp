// Copyright (c) 2023. Tudor Oancea
#include "acados/utils/math.h"
#include "acados_c/sim_interface.h"
#include "acados_sim_solver_ihm2_dyn6.h"
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
#include "ihm2/common/math.hpp"
#include "ihm2/common/tracks.hpp"
#include "ihm2/external/icecream.hpp"
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
#include <fstream>
#include <unordered_map>

using namespace std;

visualization_msgs::msg::Marker get_car_marker(double X, double Y, double phi, [[maybe_unused]] bool mesh = true) {
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

class ControlNode : public rclcpp::Node {
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
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub;
    // rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr last_lap_time_pub;
    // rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr best_lap_time_pub;
    OnSetParametersCallbackHandle::SharedPtr callback_handle;

    rclcpp::TimerBase::SharedPtr sim_timer;
    // rclcpp::Time sim_start_time;
    visualization_msgs::msg::MarkerArray cones_marker_array;
    geometry_msgs::msg::PoseStamped pose_msg;
    geometry_msgs::msg::TwistStamped vel_msg;
    geometry_msgs::msg::TransformStamped transform;

    double* x;
    double* u;
    size_t nx, nu;
    std::string model;

    void* acados_sim_capsule;
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
        for (size_t i = 0; i < nx; i++) {
            x[i] = 0.0;
        }
        x[2] = M_PI_2;
        for (size_t i = 0; i < nu; i++) {
            u[i] = 0.0;
        }
        RCLCPP_INFO(this->get_logger(), "Reset x and u");
    }

    void load_map_srv_cb([[maybe_unused]] const ihm2::srv::String::Request::SharedPtr request, [[maybe_unused]] ihm2::srv::String::Response::SharedPtr response) {
        // load the map
        // - load the center line
        // - load the cones
        // - publish the cones once
    }

    void sim_timer_cb() {
        RCLCPP_INFO(this->get_logger(), "Simulating one step with control inputs u_T = %f, u_delta = %f", u[0], u[1]);

        // set sim solver inputs
        sim_in_set(acados_sim_config,
                   acados_sim_dims,
                   acados_sim_in,
                   "x",
                   this->x);
        sim_in_set(acados_sim_config,
                   acados_sim_dims,
                   acados_sim_in,
                   "u",
                   this->u);

        // call sim solver
        auto start = this->now();
        int status = (model == "kin4") ? ihm2_kin4_acados_sim_solve((ihm2_kin4_sim_solver_capsule*) acados_sim_capsule) : ihm2_dyn6_acados_sim_solve((ihm2_dyn6_sim_solver_capsule*) acados_sim_capsule);
        if (status != ACADOS_SUCCESS) {
            throw FatalNodeError("acados_solve() failed with status " + std::to_string(status) + ".");
        }
        auto end = this->now();

        // get sim solver outputs
        sim_out_get(acados_sim_config,
                    acados_sim_dims,
                    acados_sim_out,
                    "x",
                    this->x);

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
        this->transform.transform.rotation = rpy_to_quaternion(0.0, 0.0, x[2]);
        this->tf_broadcaster->sendTransform(this->transform);

        this->vel_msg.header.stamp = this->now();
        this->vel_msg.header.frame_id = "car";
        this->vel_msg.twist.linear.x = x[3];
        if (nx > 6) {
            this->vel_msg.twist.linear.y = x[4];
            this->vel_msg.twist.angular.z = x[5];
        }
        this->vel_pub->publish(this->vel_msg);

        ihm2::msg::Controls controls_msg;
        controls_msg.header.stamp = this->now();
        controls_msg.header.frame_id = "car";
        controls_msg.throttle = x[nx - 2];
        controls_msg.steering = x[nx - 1];
        this->controls_pub->publish(controls_msg);

        visualization_msgs::msg::MarkerArray markers_msg;
        markers_msg.markers.push_back(get_car_marker(x[0], x[1], x[2], this->get_parameter("use_meshes").as_bool()));
        this->viz_pub->publish(markers_msg);

        // publish diagnostics
        double solver_runtime;
        sim_out_get(acados_sim_config,
                    acados_sim_dims,
                    acados_sim_out,
                    "time_tot",
                    &solver_runtime);
        diagnostic_msgs::msg::DiagnosticArray diag_msg;
        diag_msg.header.stamp = this->now();
        diag_msg.status.resize(1);
        diag_msg.status[0].name = "sim";
        diag_msg.status[0].hardware_id = "sim";
        diag_msg.status[0].level = diagnostic_msgs::msg::DiagnosticStatus::OK;
        diag_msg.status[0].message = "OK";

        diagnostic_msgs::msg::KeyValue solver_runtime_kv;
        solver_runtime_kv.key = "solver runtime (ms)";
        solver_runtime_kv.value = std::to_string(1000 * solver_runtime);
        diag_msg.status[0].values.push_back(solver_runtime_kv);

        diagnostic_msgs::msg::KeyValue sim_runtime_kv;
        sim_runtime_kv.key = "sim runtime (ms)";
        sim_runtime_kv.value = std::to_string(1000 * (end - start).seconds());
        diag_msg.status[0].values.push_back(sim_runtime_kv);

        this->diag_pub->publish(diag_msg);
    }

    void create_cones_markers(const std::string& track_name_or_file) {
        std::unordered_map<ConeColor, Eigen::MatrixX2d> cones_map = load_cones(track_name_or_file);
        // set deleteall to all the markers in the cones_marker_array and publish it
        for (auto& marker : cones_marker_array.markers) {
            marker.action = visualization_msgs::msg::Marker::DELETEALL;
        }
        this->viz_pub->publish(cones_marker_array);
        // create new cones
        cones_marker_array.markers.clear();
        for (auto& [color, cones] : cones_map) {
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

    rcl_interfaces::msg::SetParametersResult set_param_cb(const std::vector<rclcpp::Parameter>& parameters) {
        rcl_interfaces::msg::SetParametersResult result;
        for (const auto& parameter : parameters) {
            if (parameter.get_name() == "track_name_or_file") {
                this->create_cones_markers(parameter.as_string());
                this->viz_pub->publish(this->cones_marker_array);
            } else if (parameter.get_name() == "model") {
                result.successful = false;
                result.reason = "Cannot change model at runtime";
                return result;
            }
        }
        result.successful = true;
        return result;
    }

public:
    ControlNode() : Node("sim_node") {
        // parameters:
        // - track_name_or_file: the name of the track or the path to the track file
        // - T_max: the maximum torque
        // - T_min: the minimum torque
        // - delta_max: the maximum steering angle
        // - delta_min: the minimum steering angle
        // - manual_control: if true, the car can be controlled by the user
        // - use_meshes: if true, the car and cones are represented by meshes, otherwise by arrows
        // - model: the model to use for the simulation (kin4 or dyn6)
        this->declare_parameter<std::string>("track_name_or_file", "fsds_competition_2");
        this->declare_parameter<double>("T_max", 1600.0);
        this->declare_parameter<double>("T_min", -1600.0);
        this->declare_parameter<double>("delta_max", 0.5);
        this->declare_parameter<double>("delta_min", -0.5);
        this->declare_parameter<bool>("manual_control", true);
        this->declare_parameter<bool>("use_meshes", true);
        model = this->declare_parameter<std::string>("model", "kin4");
        // check that model is either kin4 or dyn6
        if (model != "kin4" && model != "dyn6") {
            throw FatalNodeError("model must be either kin4 or dyn6");
        }

        this->callback_handle = this->add_on_set_parameters_callback(
                std::bind(&ControlNode::set_param_cb, this, std::placeholders::_1));

        // initialize x and u with zeros
        nx = (model == "kin4") ? IHM2_KIN4_NX : IHM2_DYN6_NX;
        nu = (model == "kin4") ? IHM2_KIN4_NU : IHM2_DYN6_NU;
        x = (double*) malloc(sizeof(double) * nx);
        u = (double*) malloc(sizeof(double) * nu);
        this->reset_srv_cb(nullptr, nullptr);
        RCLCPP_INFO(this->get_logger(), "Initialized x and u with sizes %zu and %zu", nx, nu);

        // load acados sim solver
        // check if the sim solver was generated for the track in question
        acados_sim_capsule = (model == "kin4") ? (void*) ihm2_kin4_acados_sim_solver_create_capsule() : (void*) ihm2_dyn6_acados_sim_solver_create_capsule();
        int status = (model == "kin4") ? ihm2_kin4_acados_sim_create((ihm2_kin4_sim_solver_capsule*) acados_sim_capsule) : ihm2_dyn6_acados_sim_create((ihm2_dyn6_sim_solver_capsule*) acados_sim_capsule);
        if (status) {
            throw FatalNodeError("acados_create() returned status " + std::to_string(status));
        }
        acados_sim_config = (model == "kin4") ? ihm2_kin4_acados_get_sim_config((ihm2_kin4_sim_solver_capsule*) acados_sim_capsule) : ihm2_dyn6_acados_get_sim_config((ihm2_dyn6_sim_solver_capsule*) acados_sim_capsule);
        acados_sim_in = (model == "kin4") ? ihm2_kin4_acados_get_sim_in((ihm2_kin4_sim_solver_capsule*) acados_sim_capsule) : ihm2_dyn6_acados_get_sim_in((ihm2_dyn6_sim_solver_capsule*) acados_sim_capsule);
        acados_sim_out = (model == "kin4") ? ihm2_kin4_acados_get_sim_out((ihm2_kin4_sim_solver_capsule*) acados_sim_capsule) : ihm2_dyn6_acados_get_sim_out((ihm2_dyn6_sim_solver_capsule*) acados_sim_capsule);
        acados_sim_dims = (model == "kin4") ? ihm2_kin4_acados_get_sim_dims((ihm2_kin4_sim_solver_capsule*) acados_sim_capsule) : ihm2_dyn6_acados_get_sim_dims((ihm2_dyn6_sim_solver_capsule*) acados_sim_capsule);

        // publishers
        this->viz_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("/ihm2/viz/sim", 10);
        this->pose_pub = this->create_publisher<geometry_msgs::msg::PoseStamped>("/ihm2/pose", 10);
        this->vel_pub = this->create_publisher<geometry_msgs::msg::TwistStamped>("/ihm2/velocity", 10);
        this->diag_pub = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/ihm2/diag/sim", 10);
        this->controls_pub = this->create_publisher<ihm2::msg::Controls>("/ihm2/current_controls", 10);
        this->tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(this);

        // subscribers
        this->controls_sub = this->create_subscription<ihm2::msg::Controls>(
                "/ihm2/target_controls",
                10,
                std::bind(
                        &ControlNode::controls_callback,
                        this,
                        std::placeholders::_1));
        this->alternative_controls_sub = this->create_subscription<geometry_msgs::msg::Twist>(
                "/ihm2/alternative_target_controls",
                10,
                std::bind(
                        &ControlNode::alternative_controls_callback,
                        this,
                        std::placeholders::_1));

        // services
        this->reset_srv = this->create_service<std_srvs::srv::Empty>(
                "/ihm2/reset",
                std::bind(
                        &ControlNode::reset_srv_cb,
                        this,
                        std::placeholders::_1,
                        std::placeholders::_2));
        this->publish_cones_srv = this->create_service<std_srvs::srv::Empty>(
                "/ihm2/publish_cones",
                std::bind(
                        &ControlNode::publish_cones_srv_cb,
                        this,
                        std::placeholders::_1,
                        std::placeholders::_2));
        this->load_map_srv = this->create_service<ihm2::srv::String>(
                "/ihm2/load_map",
                std::bind(
                        &ControlNode::load_map_srv_cb,
                        this,
                        std::placeholders::_1,
                        std::placeholders::_2));

        // load cones from track file and create the markers for the cones
        this->create_cones_markers(this->get_parameter("track_name_or_file").as_string());

        // find simulation time step from the JSON config file generated by acados
        double sim_dt(0.01);
#ifdef SIM_JSON_PATH
        std::string sim_json_path = SIM_JSON_PATH;
        std::ifstream sim_json_file(sim_json_path);
        if (!sim_json_file.is_open()) {
            throw FatalNodeError("Could not open " + sim_json_path);
        }
        nlohmann::json sim_json;
        sim_json_file >> sim_json;
        sim_dt = sim_json["solver_options"]["Tsim"];
#else
        throw FatalNodeError("SIM_JSON_PATH not defined");
#endif
        // create a timer for the simulation loop (one simulation step and publishing the car mesh)
        this->sim_timer = this->create_wall_timer(
                std::chrono::duration<double>(sim_dt),
                std::bind(
                        &ControlNode::sim_timer_cb,
                        this));
    }

    ~ControlNode() {
        int status = (model == "kin4") ? ihm2_kin4_acados_sim_free((ihm2_kin4_sim_solver_capsule*) acados_sim_capsule) : ihm2_dyn6_acados_sim_free((ihm2_dyn6_sim_solver_capsule*) acados_sim_capsule);
        if (status) {
            RCLCPP_WARN(this->get_logger(), "ihm2_%s_acados_sim_free() returned status %d.", model.c_str(), status);
        }
        (model == "kin4") ? ihm2_kin4_acados_sim_solver_free_capsule((ihm2_kin4_sim_solver_capsule*) acados_sim_capsule) : ihm2_dyn6_acados_sim_solver_free_capsule((ihm2_dyn6_sim_solver_capsule*) acados_sim_capsule);
        free(x);
        free(u);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ControlNode>();
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
