// Copyright (c) 2023. Tudor Oancea
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
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/time_synchronizer.h"
#include "nlohmann/json.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/header.hpp"
#include "std_srvs/srv/empty.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include <fstream>
#include <memory>
#include <unordered_map>


using namespace std;

class FatalNodeError : public std::runtime_error {
public:
    FatalNodeError(const std::string& what) : std::runtime_error(what) {}
};

class NodeError : public std::runtime_error {
public:
    NodeError(const std::string& what) : std::runtime_error(what) {}
};

class StanleyControlNode : public rclcpp::Node {
private:
    typedef message_filters::sync_policies::ApproximateTime<geometry_msgs::msg::PoseStamped, geometry_msgs::msg::TwistStamped, ihm2::msg::Controls> ApproximatePolicy;
    typedef message_filters::Synchronizer<ApproximatePolicy> ApproximateSynchronizer;

    message_filters::Subscriber<ihm2::msg::Controls> controls_sub;
    message_filters::Subscriber<geometry_msgs::msg::PoseStamped> pose_sub;
    message_filters::Subscriber<geometry_msgs::msg::TwistStamped> vel_sub;
    std::shared_ptr<ApproximateSynchronizer> synchronizer;

    // rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr viz_pub;
    rclcpp::Publisher<ihm2::msg::Controls>::SharedPtr controls_pub;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub;

    ihm2::msg::Controls controls_msg;

    std::unique_ptr<Track> track;
    double s_guess = 0.0;

    void controls_callback(
            const geometry_msgs::msg::PoseStamped::ConstSharedPtr& pose_msg,
            const geometry_msgs::msg::TwistStamped::ConstSharedPtr& vel_msg,
            [[maybe_unused]] const ihm2::msg::Controls::ConstSharedPtr& current_controls_msg) {

        // RCLCPP_INFO(this->get_logger(), "pose: (%f, %f, %f)", pose_msg->pose.position.x, pose_msg->pose.position.y, tf2::getYaw(pose_msg->pose.orientation));
        // RCLCPP_INFO(this->get_logger(), "vel: (%f, %f, %f)", vel_msg->twist.linear.x, vel_msg->twist.linear.y, vel_msg->twist.angular.z);

        auto start = this->now();
        double X(pose_msg->pose.position.x), Y(pose_msg->pose.position.y), phi(tf2::getYaw(pose_msg->pose.orientation)), v_x(vel_msg->twist.linear.x);
        // project the current position on the track
        double* s = new double;
        double* Xref = new double;
        double* Yref = new double;
        double* phi_ref = new double;
        this->track->project(Eigen::Vector2d(X, Y), this->s_guess, 2.0, s, Xref, Yref, phi_ref);
        this->s_guess = std::fmod(*s + v_x * 0.01, this->track->length());
        // longitudinal control
        double epsilon = this->get_parameter("v_x_ref").as_double() - v_x, epsilon_integral = 0.0;
        double T = clip(
                this->get_parameter("k_P").as_double() * epsilon + this->get_parameter("k_I").as_double() * epsilon_integral,
                -this->get_parameter("T_max").as_double(),
                this->get_parameter("T_max").as_double());
        // lateral control
        double rho = wrap_to_pi(*phi_ref);
        double psi = wrap_to_pi(rho - phi);
        double theta = std::atan2(*Yref - Y, *Xref - X);
        double e = std::hypot(*Xref - X, *Yref - Y);
        double k_psi = this->get_parameter("k_psi").as_double(), k_e = this->get_parameter("k_e").as_double(), k_s = this->get_parameter("k_s").as_double();
        double delta = clip(
                k_psi * psi + std::atan(k_e * e / (k_s + v_x)) * (theta - rho > 0 ? 1.0 : -1.0) * (std::abs(theta - rho) > M_PI ? -1.0 : 1.0),
                -this->get_parameter("delta_max").as_double(),
                this->get_parameter("delta_max").as_double());

        // publish controls
        this->controls_msg.header.stamp = this->now();
        this->controls_msg.throttle = T;
        this->controls_msg.steering = delta;
        this->controls_pub->publish(this->controls_msg);

        auto end = this->now();

        // publish diagnostics
        diagnostic_msgs::msg::DiagnosticArray diag_msg;
        diag_msg.header.stamp = this->now();
        diag_msg.status.resize(1);
        diag_msg.status[0].name = "control";
        diag_msg.status[0].level = diagnostic_msgs::msg::DiagnosticStatus::OK;
        diag_msg.status[0].message = "OK";

        diagnostic_msgs::msg::KeyValue runtime_kv;
        runtime_kv.key = "runtime (ms)";
        runtime_kv.value = std::to_string(1000 * (end - start).seconds());
        diag_msg.status[0].values.push_back(runtime_kv);

        diagnostic_msgs::msg::KeyValue long_error_kv;
        long_error_kv.key = "epsilon (m/s)";
        long_error_kv.value = std::to_string(epsilon);
        diag_msg.status[0].values.push_back(long_error_kv);

        diagnostic_msgs::msg::KeyValue lateral_error_kv;
        lateral_error_kv.key = "e (m)";
        lateral_error_kv.value = std::to_string(e);
        diag_msg.status[0].values.push_back(lateral_error_kv);

        diagnostic_msgs::msg::KeyValue heading_error_kv;
        heading_error_kv.key = "psi (°)";
        heading_error_kv.value = std::to_string(rad2deg(psi));
        diag_msg.status[0].values.push_back(heading_error_kv);

        this->diag_pub->publish(diag_msg);
    }

public:
    StanleyControlNode() : Node("stanley_control_node") {
        // parameters:
        // - track_name: name of the track to load (must be the same as the one used to generate the ocp solver)
        this->declare_parameter<std::string>("track_name_or_file", "fsds_competition_2");
        this->declare_parameter<double>("v_x_ref", 5.0);
        this->declare_parameter<double>("k_P", 90.0);
        this->declare_parameter<double>("k_I", 0.0);
        this->declare_parameter<double>("k_psi", 1.7);
        this->declare_parameter<double>("k_e", 1.5);
        this->declare_parameter<double>("k_s", 3.0);
        this->declare_parameter<double>("phi_ref_preview_distance", 0.0);
        this->declare_parameter<double>("T_max", 100.0);
        this->declare_parameter<double>("delta_max", 0.5);


        // publishers
        // this->viz_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("/ihm2/viz/control", 10);
        this->controls_pub = this->create_publisher<ihm2::msg::Controls>("/ihm2/target_controls", 10);
        this->diag_pub = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/ihm2/diag/stanley_control", 10);

        // subscribers
        this->pose_sub.subscribe(this, "/ihm2/pose");
        this->vel_sub.subscribe(this, "/ihm2/vel");
        this->controls_sub.subscribe(this, "/ihm2/current_controls");

        // approximate time synchronizer
        this->synchronizer = std::make_shared<ApproximateSynchronizer>(
                ApproximatePolicy(10),
                this->pose_sub,
                this->vel_sub,
                this->controls_sub);
        this->synchronizer->getPolicy()->setMaxIntervalDuration(rclcpp::Duration(100, 0));
        this->synchronizer->registerCallback(
                std::bind(
                        &StanleyControlNode::controls_callback,
                        this,
                        std::placeholders::_1,
                        std::placeholders::_2,
                        std::placeholders::_3));

#ifdef TRACKS_PATH
        // create track instance from file
        std::string csv_track_file(TRACKS_PATH);
        csv_track_file += "/" + this->get_parameter("track_name_or_file").as_string() + ".csv";
        RCLCPP_INFO(this->get_logger(), "Loading track from %s", csv_track_file.c_str());
        this->track = std::make_unique<Track>(csv_track_file);
        RCLCPP_INFO(this->get_logger(), "v_x_ref = %f", this->get_parameter("v_x_ref").as_double());
#else
        raise FatalNodeError("TRACKS_PATH not defined");
#endif
    }

    ~StanleyControlNode() {}
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<StanleyControlNode>();
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