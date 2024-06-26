// Copyright (c) 2023. Tudor Oancea
#include "acados/utils/math.h"
#include "acados_c/sim_interface.h"
#include "acados_sim_solver_ihm2_dyn6.h"
#include "acados_sim_solver_ihm2_kin6.h"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "curl/curl.h"
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
#include "ihm2/msg/state.hpp"
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
#include <rclcpp/publisher.hpp>
#include <unordered_map>

using namespace std;

// used for the lap time computation
bool ccw(Eigen::Vector2d a, Eigen::Vector2d b, Eigen::Vector2d c) {
    return (c(1) - a(1)) * (b(0) - a(0)) > (b(1) - a(1)) * (c(0) - a(0));
}
bool intersect(Eigen::Vector2d a, Eigen::Vector2d b, Eigen::Vector2d c, Eigen::Vector2d d) {
    return ccw(a, c, d) != ccw(b, c, d) && ccw(a, b, c) != ccw(a, b, d);
}

template <typename Derived>
Eigen::PermutationMatrix<Eigen::Dynamic> argsort(const Eigen::DenseBase<Derived>& vec) {
    Eigen::PermutationMatrix<Eigen::Dynamic> perm(vec.size());
    std::iota(perm.indices().data(), perm.indices().data() + perm.indices().size(), 0);
    std::sort(perm.indices().data(), perm.indices().data() + perm.indices().size(),
              [&](int i, int j) { return vec(i) < vec(j); });
    return perm;
}

// internet connectivity status
bool is_internet_connected() {
    CURL* curl = curl_easy_init();
    if (!curl) {
        curl_global_cleanup();
        return false;  // Failed to initialize curl
    }

    curl_easy_setopt(curl, CURLOPT_URL, "http://www.google.com");
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);  // Use a HEAD request to reduce data transfer

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    return res == CURLE_OK;
}

// error classes
class FatalNodeError : public std::runtime_error {
public:
    FatalNodeError(const std::string& what) : std::runtime_error(what) {}
};

class NodeError : public std::runtime_error {
public:
    NodeError(const std::string& what) : std::runtime_error(what) {}
};

// actual simulation node
class SimNode : public rclcpp::Node {
private:
    // publishers
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;
    rclcpp::Publisher<ihm2::msg::State>::SharedPtr state_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr viz_pub;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub;

    // subscribers
    rclcpp::Subscription<ihm2::msg::Controls>::SharedPtr controls_sub;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr alternative_controls_sub;

    // services
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_srv;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr publish_cones_srv;

    // simulation variables
    rclcpp::TimerBase::SharedPtr sim_timer;
    double *x, *u;
    size_t nx, nu;
    ihm2::msg::State state_msg;
    geometry_msgs::msg::TransformStamped transform;

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

    // cones
    visualization_msgs::msg::MarkerArray cones_marker_array;
    std::unordered_map<ConeColor, Eigen::MatrixX2d> cones_map;

    // lap timing
    Eigen::Vector2d last_pos, start_line_pos_1, start_line_pos_2;
    double last_lap_time = 0.0, best_lap_time = 0.0;
    rclcpp::Time last_lap_time_stamp = rclcpp::Time(0, 0);

    // architecture and internet connectivity status (for visualization)
#ifdef WSL
    static constexpr bool is_wsl = true;
#else
    static constexpr bool is_wsl = false;
#endif

    bool has_internet;


    void controls_callback(const ihm2::msg::Controls::SharedPtr msg) {
        if (!this->get_parameter("manual_control").as_bool()) {
            double T_max = this->get_parameter("T_max").as_double(), delta_max = this->get_parameter("delta_max").as_double(), ddelta_max(0.01 * 2 * delta_max / 1.0);
            u[0] = clip(msg->throttle, -T_max, T_max);
            u[1] = clip(clip(msg->steering, u[1] - ddelta_max, u[1] + ddelta_max), -delta_max, delta_max);
        }
    }

    void alternative_controls_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        if (this->get_parameter("manual_control").as_bool()) {
            double T_max = this->get_parameter("T_max").as_double(), delta_max = this->get_parameter("delta_max").as_double(), ddelta_max(0.01 * 2 * delta_max / 1.0);
            u[0] = clip(msg->linear.x, -T_max, T_max);
            u[1] = clip(clip(msg->angular.z, u[1] - ddelta_max, u[1] + ddelta_max), -delta_max, delta_max);
        }
    }

    void publish_cones_srv_cb([[maybe_unused]] const std_srvs::srv::Empty::Request::SharedPtr request, [[maybe_unused]] std_srvs::srv::Empty::Response::SharedPtr response) {
        this->viz_pub->publish(cones_marker_array);
    }

    void reset_srv_cb([[maybe_unused]] const std_srvs::srv::Empty::Request::SharedPtr request, [[maybe_unused]] std_srvs::srv::Empty::Response::SharedPtr response) {
        // reset x and u
        for (size_t i = 0; i < nx; i++) {
            x[i] = 0.0;
        }
        x[2] = M_PI_2;
        for (size_t i = 0; i < nu; i++) {
            u[i] = 0.0;
        }
        // reset lap timing
        this->last_lap_time_stamp = rclcpp::Time(0, 0);
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

        // check if we have completed a lap
        Eigen::Vector2d pos(x[0], x[1]);
        if (x[3] > 0.0 and intersect(this->start_line_pos_1, this->start_line_pos_2, this->last_pos, pos)) {
            if (this->last_lap_time_stamp.nanoseconds() > 0) {
                double lap_time((end - this->last_lap_time_stamp).seconds());
                if (this->best_lap_time == 0.0 or lap_time < this->best_lap_time) {
                    this->best_lap_time = lap_time;
                }
                this->last_lap_time = lap_time;
                RCLCPP_INFO(this->get_logger(), "Lap completed time: %.3f s (best: %.3f s)", lap_time, this->best_lap_time);
            }
            this->last_lap_time_stamp = end;
        }
        this->last_pos = pos;
        // override the pose and velocity and controls with the simulation output
        this->state_msg.header.stamp = this->now();
        this->state_msg.header.frame_id = "world";
        this->state_msg.pose.position.x = x[0];
        this->state_msg.pose.position.y = x[1];
        this->state_msg.pose.orientation = rpy_to_quaternion(0.0, 0.0, x[2]);
        this->state_msg.twist.linear.x = x[3];
        this->state_msg.twist.linear.y = x[4];
        this->state_msg.twist.angular.z = x[5];
        this->state_msg.controls.throttle = x[nx - 2];
        this->state_msg.controls.steering = x[nx - 1];
        this->state_pub->publish(this->state_msg);

        this->transform.header.stamp = this->now();
        this->transform.header.frame_id = "world";
        this->transform.child_frame_id = "car";
        this->transform.transform.translation.x = x[0];
        this->transform.transform.translation.y = x[1];
        this->transform.transform.rotation = this->state_msg.pose.orientation;
        this->tf_broadcaster->sendTransform(this->transform);

        visualization_msgs::msg::MarkerArray markers_msg;
        std::vector<visualization_msgs::msg::Marker> car_markers = get_car_markers();
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

        diagnostic_msgs::msg::KeyValue last_lap_time_kv;
        last_lap_time_kv.key = "last lap time (s)";
        last_lap_time_kv.value = std::to_string(this->last_lap_time);
        diag_msg.status[0].values.push_back(last_lap_time_kv);

        diagnostic_msgs::msg::KeyValue best_lap_time_kv;
        best_lap_time_kv.key = "best lap time (s)";
        best_lap_time_kv.value = std::to_string(this->best_lap_time);
        diag_msg.status[0].values.push_back(best_lap_time_kv);

        // add wheel speeds in diagnostics
        if (nx >= 10) {
            double omega_FL(x[6]), omega_FR(x[7]), omega_RL(x[8]), omega_RR(x[9]);
            diagnostic_msgs::msg::KeyValue omega_FL_kv;
            omega_FL_kv.key = "omega_FL";
            omega_FL_kv.value = std::to_string(omega_FL);
            diag_msg.status[0].values.push_back(omega_FL_kv);
            diagnostic_msgs::msg::KeyValue omega_FR_kv;
            omega_FR_kv.key = "omega_FR";
            omega_FR_kv.value = std::to_string(omega_FR);
            diag_msg.status[0].values.push_back(omega_FR_kv);
            diagnostic_msgs::msg::KeyValue omega_RL_kv;
            omega_RL_kv.key = "omega_RL";
            omega_RL_kv.value = std::to_string(omega_RL);
            diag_msg.status[0].values.push_back(omega_RL_kv);
            diagnostic_msgs::msg::KeyValue omega_RR_kv;
            omega_RR_kv.key = "omega_RR";
            omega_RR_kv.value = std::to_string(omega_RR);
            diag_msg.status[0].values.push_back(omega_RR_kv);
        }

        this->diag_pub->publish(diag_msg);
    }

    void filter_cones(
            Eigen::MatrixX2d cones, double X, double Y, double phi,
            Eigen::VectorXd& rho, Eigen::VectorXd& theta) {
        // get bearing and range limits
        std::vector<double> range_limits(this->get_parameter("range_limits").as_double_array()), bearing_limits(this->get_parameter("bearing_limits").as_double_array());
        // compute the postions of the cones relative to the car
        Eigen::Vector2d pos(X, Y);
        Eigen::MatrixX2d cartesian = cones.rowwise() - pos.transpose();
        Eigen::Matrix2d rot;
        rot << std::cos(phi), -std::sin(phi), std::sin(phi), std::cos(phi);
        cartesian = cartesian * rot.transpose();
        Eigen::MatrixX2d polar(cartesian.rows(), 2);
        for (Eigen::Index i(0); i < cartesian.rows(); ++i) {
            polar(i, 0) = cartesian.row(i).norm();
            polar(i, 1) = std::atan2(cartesian(i, 1), cartesian(i, 0));
        }
        // only keep the cones that are in the range and bearing limits
        Eigen::Array<bool, Eigen::Dynamic, 1> mask = (range_limits[0] <= polar.col(0).array()) && (polar.col(0).array() <= range_limits[1]) && (bearing_limits[0] <= polar.col(1).array()) && (polar.col(1).array() <= bearing_limits[1]);
        size_t n_cones(mask.count());
        rho.conservativeResize(n_cones);
        theta.conservativeResize(n_cones);
        for (Eigen::Index i(0), j(0); i < cones.rows(); ++i) {
            if (mask(i)) {
                rho(j) = polar(i, 0);
                theta(j) = polar(i, 1);
                ++j;
            }
        }
    }

    void create_cones_markers(const std::string& track_name_or_file) {
        cones_map = load_cones(track_name_or_file);

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
                        this->get_cone_marker(
                                i,
                                cones(i, 0),
                                cones(i, 1),
                                is_orange(color) ? "orange" : cone_color_to_string(color),
                                color != ConeColor::BIG_ORANGE));
            }
        }
        RCLCPP_INFO(this->get_logger(), "Loaded %lu cones from %s", cones_marker_array.markers.size(), track_name_or_file.c_str());
        // if there are big orange cones, find the ones that have the smallest y coordinate and set them as start line
        if (cones_map.find(ConeColor::BIG_ORANGE) != cones_map.end()) {
            Eigen::MatrixX2d big_orange_cones(cones_map[ConeColor::BIG_ORANGE]);
            if (big_orange_cones.rows() >= 2) {
                // Find the indices that would sort the second column
                Eigen::PermutationMatrix<Eigen::Dynamic> indices = argsort(big_orange_cones.col(1));
                // Extract the corresponding rows
                start_line_pos_1 = big_orange_cones.row(indices.indices()(0));
                start_line_pos_2 = big_orange_cones.row(indices.indices()(1));
                RCLCPP_INFO(this->get_logger(), "Found start line at (%.3f, %.3f) and (%.3f, %.3f)", start_line_pos_1(0), start_line_pos_1(1), start_line_pos_2(0), start_line_pos_2(1));
            }
        }
    }

    visualization_msgs::msg::Marker get_cone_marker(uint64_t id, double X, double Y, std::string color, bool small) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "world";
        marker.ns = color + "_cones";
        marker.id = id;
        marker.action = visualization_msgs::msg::Marker::MODIFY;
        if (!this->is_wsl || this->has_internet) {
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

    std::vector<visualization_msgs::msg::Marker> get_car_markers() {
        double X = x[0], Y = x[1], phi = x[2], delta = x[nx - 1];
        std::vector<visualization_msgs::msg::Marker> markers;
        if (!this->is_wsl || this->has_internet) {
            // we either are not on WSL (i.e. we are on Linux of macOS) or we are and we have internet connection,
            // so we use mesh markers for the car
            // note: we used lazey evaluation to avoid calling is_internet_connected() if we are not on WSL
            std::string car_mesh = this->get_parameter("car_mesh").as_string();
            markers.resize(car_mesh == "lego-lrt4.stl" ? 1 : 5);
            markers[0].header.frame_id = "world";
            markers[0].ns = "car";
            markers[0].type = visualization_msgs::msg::Marker::MESH_RESOURCE;
            markers[0].action = visualization_msgs::msg::Marker::MODIFY;
            markers[0].mesh_resource = this->is_wsl ? this->get_remote_mesh_path(car_mesh) : this->get_local_mesh_path(car_mesh);
            markers[0].pose.position.x = X;
            markers[0].pose.position.y = Y;
            markers[0].pose.orientation = rpy_to_quaternion(0.0, 0.0, phi);
            markers[0].scale.x = 1.0;
            markers[0].scale.y = 1.0;
            markers[0].scale.z = 1.0;
            markers[0].color = marker_colors("white");
            if (car_mesh == "gotthard.stl") {
                for (size_t i(1); i < 5; ++i) {
                    markers[i].header.frame_id = "world";
                    markers[i].ns = "car";
                    markers[i].id = i;
                    markers[i].type = visualization_msgs::msg::Marker::MESH_RESOURCE;
                    markers[i].mesh_resource = this->is_wsl ? this->get_remote_mesh_path("gotthard_wheel.stl") : this->get_local_mesh_path("gotthard_wheel.stl");
                    markers[i].action = visualization_msgs::msg::Marker::MODIFY;
                    double wheel_x(i < 3 ? 0.819 : -0.711),
                            wheel_y(i % 2 == 0 ? 0.6 : -0.6),
                            wheel_phi(i < 3 ? delta : 0.0);
                    markers[i].pose.position.x = X + wheel_x * std::cos(phi) - wheel_y * std::sin(phi);
                    markers[i].pose.position.y = Y + wheel_x * std::sin(phi) + wheel_y * std::cos(phi);
                    markers[i].pose.position.z = 0.232;
                    markers[i].pose.orientation = rpy_to_quaternion(0.0, 0.0, phi + wheel_phi);
                    markers[i].scale.x = 1.0;
                    markers[i].scale.y = 1.0;
                    markers[i].scale.z = 1.0;
                    markers[i].color = marker_colors("black");
                }
            } else if (car_mesh == "ariane.stl") {
                for (size_t i(1); i < 5; ++i) {
                    markers[i].header.frame_id = "world";
                    markers[i].ns = "car";
                    markers[i].id = i;
                    markers[i].type = visualization_msgs::msg::Marker::MESH_RESOURCE;
                    markers[i].mesh_resource = this->is_wsl ? this->get_remote_mesh_path("ariane_wheel.stl") : this->get_local_mesh_path("ariane_wheel.stl");
                    markers[i].action = visualization_msgs::msg::Marker::MODIFY;
                    double wheel_x(i < 3 ? 0.7853 : -0.7853),
                            wheel_y(i % 2 == 0 ? 0.6291 : -0.6291),
                            wheel_phi(i < 3 ? delta : 0.0);
                    // the STL represents right wheels, so for left wheels (i.e. i=1,3) we add a yaw of pi
                    wheel_phi += (i % 2 == 1 ? 0.0 : M_PI);
                    markers[i].pose.position.x = X + wheel_x * std::cos(phi) - wheel_y * std::sin(phi);
                    markers[i].pose.position.y = Y + wheel_x * std::sin(phi) + wheel_y * std::cos(phi);
                    markers[i].pose.position.z = 0.20809;
                    markers[i].pose.orientation = rpy_to_quaternion(0.0, 0.0, phi + wheel_phi);
                    markers[i].scale.x = 1.0;
                    markers[i].scale.y = 1.0;
                    markers[i].scale.z = 1.0;
                    markers[i].color = marker_colors("black");
                }
            }
        } else {
            // we are on WSL and we don't have internet connection, so we use cubes and cylinders for the car
            markers.resize(5);
            markers[0].header.frame_id = "world";
            markers[0].ns = "car";
            markers[0].id = 0;
            markers[0].type = visualization_msgs::msg::Marker::CUBE;
            markers[0].action = visualization_msgs::msg::Marker::MODIFY;
            markers[0].pose.position.x = X + 0.5 * std::cos(phi);
            markers[0].pose.position.y = Y + 0.5 * std::sin(phi);
            markers[0].pose.position.z = 0.6 / 2.0;
            markers[0].pose.orientation = rpy_to_quaternion(0.0, 0.0, phi);
            markers[0].scale.x = 3.19;
            markers[0].scale.y = 1.05;
            markers[0].scale.z = 0.6;
            markers[0].color = marker_colors("white");
            for (size_t i(1); i < 5; ++i) {
                markers[i].header.frame_id = "world";
                markers[i].ns = "car";
                markers[i].id = i;
                markers[i].type = visualization_msgs::msg::Marker::CYLINDER;
                markers[i].action = visualization_msgs::msg::Marker::MODIFY;
                double wheel_x(i < 3 ? 0.7853 : -0.7853),
                        wheel_y(i % 2 == 0 ? 0.6291 : -0.6291),
                        wheel_phi(i < 3 ? delta : 0.0);
                markers[i].pose.position.x = X + wheel_x * std::cos(phi) - wheel_y * std::sin(phi);
                markers[i].pose.position.y = Y + wheel_x * std::sin(phi) + wheel_y * std::cos(phi);
                markers[i].pose.position.z = 0.232;
                markers[i].pose.orientation = rpy_to_quaternion(M_PI_2, 0.0, phi + wheel_phi);
                markers[i].scale.x = 0.41618;
                markers[i].scale.y = 0.41618;
                markers[i].scale.z = 0.21;
                markers[i].color = marker_colors("black");
            }
        }
        return markers;
    }

    inline std::string get_local_mesh_path(std::string mesh_file) {
        return "file://" + ament_index_cpp::get_package_share_directory("ihm2") + "/meshes/" + mesh_file;
    }
    inline std::string get_remote_mesh_path(std::string mesh_file) {
        return "https://raw.github.com/EPFL-RT-Driverless/resources/main/meshes/" + mesh_file;
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
        this->declare_parameter<double>("T_max", 1107.0);
        this->declare_parameter<double>("delta_max", 0.5);
        this->declare_parameter<double>("v_dyn", 3.0);
        this->declare_parameter<std::string>("track_name_or_file", "fsds_competition_2");
        this->declare_parameter<bool>("manual_control", true);
        this->declare_parameter<bool>("use_meshes", true);
        this->declare_parameter<std::vector<double>>("range_limits", {0.0, 15.0});
        this->declare_parameter<std::vector<double>>("bearing_limits", {-deg2rad(50.0), deg2rad(50.0)});
        this->declare_parameter<std::string>("car_mesh", "ariane.stl");

        // initialize x and u with zeros
        nx = IHM2_DYN6_NX;
        nu = IHM2_DYN6_NU;
        x = (double*) malloc(sizeof(double) * nx);
        u = (double*) malloc(sizeof(double) * nu);
        this->reset_srv_cb(nullptr, nullptr);
        RCLCPP_INFO(this->get_logger(), "Initialized x and u with sizes %zu and %zu", nx, nu);

        // check internet connectivity at launch time
        // (we assume we won't lose it)
        this->has_internet = is_internet_connected();

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
        this->state_pub = this->create_publisher<ihm2::msg::State>("/ihm2/state", 10);
        this->diag_pub = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/ihm2/diag", 10);
        this->tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(this);

        // subscribers
        this->controls_sub = this->create_subscription<ihm2::msg::Controls>(
                "/ihm2/controls",
                10,
                std::bind(
                        &SimNode::controls_callback,
                        this,
                        std::placeholders::_1));
        this->alternative_controls_sub = this->create_subscription<geometry_msgs::msg::Twist>(
                "/ihm2/alternative_controls",
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
