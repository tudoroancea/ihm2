// Copyright (c) 2023. Tudor Oancea
#include "acados/utils/math.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/sim_interface.h"
#include "acados_sim_solver_ihm2_fkin6.h"
#include "acados_solver_ihm2_fkin6.h"
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
#include "sim_interface.h"
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
#include <Eigen/src/Core/util/Constants.h>
#include <cstdint>
#include <fstream>
#include <memory>
#include <unordered_map>
#define EIGEN_INITIALIZE_MATRICES_BY_ZERO

using namespace std;

class FatalNodeError : public std::runtime_error {
public:
    explicit FatalNodeError(const std::string& what) : std::runtime_error(what) {}
};

class NodeError : public std::runtime_error {
public:
    explicit NodeError(const std::string& what) : std::runtime_error(what) {}
};

class MPCControlNode : public rclcpp::Node {
private:
    // subscribers
    rclcpp::Subscription<ihm2::msg::State>::SharedPtr state_sub;

    // publishers
    rclcpp::Publisher<ihm2::msg::Controls>::SharedPtr controls_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr viz_pub;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub;

    // state and control variables
    // double* x;  // x0, x1, ..., xNf stored in a single vector (row-major)
    // double* u;  // u0, u1, ..., uNf-1 stored in a single vector (row-major)
    double x[IHM2_FKIN6_NX * (IHM2_FKIN6_N + 1)], u[IHM2_FKIN6_NU * IHM2_FKIN6_N];
    Eigen::Map<Eigen::Matrix<double, IHM2_FKIN6_NX, IHM2_FKIN6_N + 1, Eigen::ColMajor>> x_eigen;
    Eigen::Map<Eigen::Matrix<double, IHM2_FKIN6_NU, IHM2_FKIN6_N, Eigen::ColMajor>> u_eigen;
    int idxbx0[IHM2_FKIN6_NBX0];
    double lbx0[IHM2_FKIN6_NBX0], ubx0[IHM2_FKIN6_NBX0];
    double y_ref[IHM2_FKIN6_NY];
    Eigen::Vector<double, IHM2_FKIN6_N + 1> Xpred, Ypred;

    // acados ocp solver
    ihm2_fkin6_solver_capsule* acados_ocp_capsule;
    ocp_nlp_config* acados_ocp_config;
    ocp_nlp_dims* acados_ocp_dims;
    ocp_nlp_in* acados_ocp_in;
    ocp_nlp_out* acados_ocp_out;
    ocp_nlp_solver* acados_ocp_solver;
    void* acados_ocp_opts;

    ihm2_fkin6_sim_solver_capsule* acados_sim_capsule;
    sim_config* acados_sim_config;
    void* acados_sim_dims;
    sim_in* acados_sim_in;
    sim_out* acados_sim_out;

    // to run at 20Hz instead of the full 100Hz offered by the simulation
    uint8_t sim_steps = 5, sim_counter = 0;

    // motion planning
    std::unique_ptr<Track> track;
    double s_guess = 0.0;

    // pre-allocated messages
    ihm2::msg::Controls controls_msg;
    visualization_msgs::msg::MarkerArray viz_msg;

    void controls_callback(const ihm2::msg::State::ConstSharedPtr& state_msg) {
        if (this->sim_counter == 0) {
            auto start = this->now();
            // update x and u by shifting the previous values and using uNf-1 = uNf-2 and xNf = xNf-1
            for (size_t i = 0; i < IHM2_FKIN6_NX * IHM2_FKIN6_N; i++) {
                x[i] = x[i + IHM2_FKIN6_NX];
                // IC(this->acados_nlp_config, this->acados_nlp_dims, this->acados_nlp_out, i, "x", x + i);
                // if (i % IHM2_FKIN6_NX == 0) {
                //     IC(i / IHM2_FKIN6_NX);
                //     ocp_nlp_out_set(this->acados_nlp_config, this->acados_nlp_dims, this->acados_nlp_out, i, "x", x + i);
                //     IC();
                // }
            }
            for (size_t i = 0; i < IHM2_FKIN6_NU * (IHM2_FKIN6_N - 1); i++) {
                u[i] = u[i + IHM2_FKIN6_NU];
                // if (i % IHM2_FKIN6_NU == 0) {
                //     IC(i / IHM2_FKIN6_NU);
                //     ocp_nlp_out_set(this->acados_nlp_config, this->acados_nlp_dims, this->acados_nlp_out, i, "u", u + i);
                // }
            }
            // for (size_t i = 0; i < IHM2_FKIN6_NX; i++) {
            //     x[IHM2_FKIN6_NX * IHM2_FKIN6_N + i] = x[IHM2_FKIN6_NX * (IHM2_FKIN6_N - 1) + i];
            // }
            // for (size_t i = 0; i < IHM2_FKIN6_NU; i++) {
            //     u[IHM2_FKIN6_NU * (IHM2_FKIN6_N - 1) + i] = u[IHM2_FKIN6_NU * (IHM2_FKIN6_N - 2) + i];
            // }

            // get values from messages
            double X = state_msg->pose.position.x,
                   Y = state_msg->pose.position.y,
                   phi = wrap_to_pi(tf2::getYaw(state_msg->pose.orientation)),
                   v_x = state_msg->twist.linear.x,
                   v_y = state_msg->twist.linear.y,
                   r = state_msg->twist.angular.z,
                   T = state_msg->controls.throttle,
                   delta = state_msg->controls.steering;

            // project the current position on the track
            double s, Xref, Yref, phi_ref;
            this->track->project(Eigen::Vector2d(X, Y), this->s_guess, 2.0, &s, &Xref, &Yref, &phi_ref);
            this->s_guess = std::fmod(s + v_x * 0.05, this->track->length());

            // compute Frenet states n,psi
            double rho = wrap_to_pi(phi_ref);
            double psi = wrap_to_pi(phi - rho);
            // double theta = std::atan2(Yref - Y, Xref - X);
            double e = std::hypot(Xref - X, Yref - Y);
            // double n = -e * (theta - rho > 0 ? 1.0 : -1.0) * (std::abs(theta - rho) > M_PI ? -1.0 : 1.0);
            double tpr = (Y - Yref) * std::cos(rho) - (X - Xref) * std::sin(rho);
            if (std::abs(tpr) < 1e-6) {
                throw FatalNodeError("tangent x normal is too small");
            }
            double n = e * (tpr > 0.0 ? 1.0 : -1.0);

            lbx0[0] = s;
            ubx0[0] = s;
            lbx0[1] = n;
            ubx0[1] = n;
            lbx0[2] = psi;
            ubx0[2] = psi;
            lbx0[3] = v_x;
            ubx0[3] = v_x;
            lbx0[4] = v_y;
            ubx0[4] = v_y;
            lbx0[5] = r;
            ubx0[5] = r;
            lbx0[6] = T;
            ubx0[6] = T;
            lbx0[7] = delta;
            ubx0[7] = delta;

            // TODO: check output of those functions and print errors if they fail
            ocp_nlp_constraints_model_set(this->acados_ocp_config, this->acados_ocp_dims, this->acados_ocp_in, 0, "idxbx", idxbx0);
            ocp_nlp_constraints_model_set(this->acados_ocp_config, this->acados_ocp_dims, this->acados_ocp_in, 0, "lbx", lbx0);
            ocp_nlp_constraints_model_set(this->acados_ocp_config, this->acados_ocp_dims, this->acados_ocp_in, 0, "ubx", ubx0);

            // set ref
            double s_ref_Nf = this->get_parameter("s_ref_Nf").as_double();
            for (size_t k = 0; k <= IHM2_FKIN6_N; k++) {
                y_ref[0] = s + (s_ref_Nf * k) / IHM2_FKIN6_N;
                ocp_nlp_cost_model_set(this->acados_ocp_config, this->acados_ocp_dims, this->acados_ocp_in, k, "y_ref", y_ref);
            }

            // call solver
            int status = ihm2_fkin6_acados_solve(acados_ocp_capsule);
            if (status == ACADOS_SUCCESS) {
                // RCLCPP_INFO(this->get_logger(), "ihm2_fkin6_acados_solve(): SUCCESS!\n");
            } else if (status == ACADOS_MAXITER) {
                // RCLCPP_WARN(this->get_logger(), "ihm2_fkin6_acados_solve(): MAX ITER!\n");
            } else {
                // RCLCPP_ERROR(this->get_logger(), "ihm2_fkin6_acados_solve() failed with status %d.\n", status);
                throw FatalNodeError("ihm2_fkin6_acados_solve() failed with status " + std::to_string(status));
            }

            auto end = this->now();

            // get solution
            for (int ii = 0; ii < IHM2_FKIN6_N; ii++) {
                ocp_nlp_out_get(this->acados_ocp_config, this->acados_ocp_dims, this->acados_ocp_out, ii, "u", u + ii * IHM2_FKIN6_NU);
                ocp_nlp_out_get(this->acados_ocp_config, this->acados_ocp_dims, this->acados_ocp_out, ii, "x", x + ii * IHM2_FKIN6_NX);
            }
            ocp_nlp_out_get(this->acados_ocp_config, this->acados_ocp_dims, this->acados_ocp_out, IHM2_FKIN6_N, "x", x + IHM2_FKIN6_NX * IHM2_FKIN6_N);

            // publish controls
            controls_msg.header.stamp = this->now();
            controls_msg.throttle = u[0];
            controls_msg.steering = u[1];
            this->controls_pub->publish(controls_msg);

            // publish diagnostics
            diagnostic_msgs::msg::DiagnosticArray diag_msg;
            diag_msg.header.stamp = this->now();
            diag_msg.status.resize(1);
            diag_msg.status[0].name = "mpc_control";
            diag_msg.status[0].level = diagnostic_msgs::msg::DiagnosticStatus::OK;
            diag_msg.status[0].message = "OK";

            diagnostic_msgs::msg::KeyValue runtime_kv;
            runtime_kv.key = "runtime (ms)";
            runtime_kv.value = std::to_string(1000 * (end - start).seconds());
            diag_msg.status[0].values.push_back(runtime_kv);

            diagnostic_msgs::msg::KeyValue s_kv;
            s_kv.key = "s (m)";
            s_kv.value = std::to_string(s);
            diag_msg.status[0].values.push_back(s_kv);

            diagnostic_msgs::msg::KeyValue lateral_error_kv;
            lateral_error_kv.key = "n (m)";
            lateral_error_kv.value = std::to_string(n);
            diag_msg.status[0].values.push_back(lateral_error_kv);

            diagnostic_msgs::msg::KeyValue heading_error_kv;
            heading_error_kv.key = "psi (°)";
            heading_error_kv.value = std::to_string(rad2deg(psi));
            diag_msg.status[0].values.push_back(heading_error_kv);

            this->diag_pub->publish(diag_msg);

            // publish viz
            Eigen::VectorXd s_pred = x_eigen.row(0), n_pred = x_eigen.row(1);
            Eigen::VectorXd X_pred, Y_pred;
            this->track->frenet_to_cartesian(s_pred, n_pred, X_pred, Y_pred);
            for (size_t i = 0; i < IHM2_FKIN6_N + 1; i++) {
                viz_msg.markers[0].points[i].x = X_pred[i];
                viz_msg.markers[0].points[i].y = Y_pred[i];
            }
            this->viz_pub->publish(viz_msg);
        }
        this->sim_counter = (this->sim_counter + 1) % this->sim_steps;
    }

public:
    MPCControlNode() : Node("mpc_control_node"), x_eigen(x), u_eigen(u) {
        // parameters
        this->declare_parameter<std::string>("track_name", "fsds_competition_1");
        this->declare_parameter<double>("s_ref_Nf", 30.0);
        auto Q_diag_data = this->declare_parameter<std::vector<double>>("Q_diag", {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
        auto R_diag_data = this->declare_parameter<std::vector<double>>("R_diag", {1.0, 1.0});
        auto Qf_diag_data = this->declare_parameter<std::vector<double>>("Qf_diag", {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
        Eigen::Map<Eigen::Vector<double, IHM2_FKIN6_NX>> Q_diag(Q_diag_data.data(), Q_diag_data.size()), Qf_diag(Qf_diag_data.data(), Qf_diag_data.size());
        Eigen::Map<Eigen::Vector<double, IHM2_FKIN6_NU>> R_diag(R_diag_data.data(), R_diag_data.size());
        Eigen::Matrix<double, IHM2_FKIN6_NY, IHM2_FKIN6_NY> W;  // = Eigen::Matrix<double, IHM2_FKIN6_NY, IHM2_FKIN6_NY>::Zero(IHM2_FKIN6_NY, IHM2_FKIN6_NY);
        W.setZero();
        W.diagonal() << Q_diag, R_diag;
        Eigen::Matrix<double, IHM2_FKIN6_NYN, IHM2_FKIN6_NYN> Wf = Qf_diag.asDiagonal();


        // load acados ocp solver
        acados_ocp_capsule = ihm2_fkin6_acados_create_capsule();
        int status = ihm2_fkin6_acados_create(acados_ocp_capsule);
        if (status) {
            throw FatalNodeError("ihm2_fkin6_acados_create_with_discretization() returned status " + std::to_string(status));
        }
        acados_ocp_config = ihm2_fkin6_acados_get_nlp_config(acados_ocp_capsule);
        acados_ocp_dims = ihm2_fkin6_acados_get_nlp_dims(acados_ocp_capsule);
        acados_ocp_in = ihm2_fkin6_acados_get_nlp_in(acados_ocp_capsule);
        acados_ocp_out = ihm2_fkin6_acados_get_nlp_out(acados_ocp_capsule);
        acados_ocp_solver = ihm2_fkin6_acados_get_nlp_solver(acados_ocp_capsule);
        acados_ocp_opts = ihm2_fkin6_acados_get_nlp_opts(acados_ocp_capsule);
        RCLCPP_INFO(this->get_logger(), "Successfully loaded acados ocp solver");

        for (int i = 0; i < IHM2_FKIN6_N; i++) {
            status = ocp_nlp_cost_model_set(this->acados_ocp_config, this->acados_ocp_dims, this->acados_ocp_in, i, "W", W.data());
            if (status) {
                throw FatalNodeError("ocp_nlp_cost_model_set() returned status " + std::to_string(status));
            }
        }
        ocp_nlp_cost_model_set(this->acados_ocp_config, this->acados_ocp_dims, this->acados_ocp_in, IHM2_FKIN6_N, "W", Wf.data());

        // load acados sim solver (for the moment only used to properly warm start the ocp solver, see below)
        acados_sim_capsule = ihm2_fkin6_acados_sim_solver_create_capsule();
        status = ihm2_fkin6_acados_sim_create(acados_sim_capsule);
        if (status) {
            throw FatalNodeError("ihm2_fkin6_acados_sim_create() returned status " + std::to_string(status));
        }
        acados_sim_config = ihm2_fkin6_acados_get_sim_config(acados_sim_capsule);
        acados_sim_dims = ihm2_fkin6_acados_get_sim_dims(acados_sim_capsule);
        acados_sim_in = ihm2_fkin6_acados_get_sim_in(acados_sim_capsule);
        acados_sim_out = ihm2_fkin6_acados_get_sim_out(acados_sim_capsule);
        RCLCPP_INFO(this->get_logger(), "Successfully loaded acados sim solver");


        // initialize x and u with zeros
        for (size_t i = 0; i < IHM2_FKIN6_NX * (IHM2_FKIN6_N + 1); i++) {
            x[i] = 0.0;
        }
        for (size_t i = 0; i < IHM2_FKIN6_NU * IHM2_FKIN6_N; i++) {
            u[i] = 0.0;
        }
        for (int i = 0; i < IHM2_FKIN6_NBX0; i++) {
            idxbx0[i] = i;
        }
        for (size_t i = 0; i < IHM2_FKIN6_NY; i++) {
            y_ref[i] = 0.0;
        }
        // now initialize better with a sim solver
        x[0] = -5.0;
        for (size_t i = 0; i < IHM2_FKIN6_N; i++) {
            // set in
            sim_in_set(acados_sim_config, acados_sim_dims, acados_sim_in, "x", x + i * IHM2_FKIN6_NX);
            sim_in_set(acados_sim_config, acados_sim_dims, acados_sim_in, "u", u + i * IHM2_FKIN6_NU);

            // simulate
            status = ihm2_fkin6_acados_sim_solve(acados_sim_capsule);
            if (status) {
                throw FatalNodeError("ihm2_fkin6_acados_sim_solve() returned status " + std::to_string(status));
            }
            // get solution
            sim_out_get(acados_sim_config, acados_sim_dims, acados_sim_out, "x", x + IHM2_FKIN6_NX * (i + 1));
        }

        // publishers
        this->controls_pub = this->create_publisher<ihm2::msg::Controls>("/ihm2/controls", 10);
        this->viz_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("/ihm2/viz/control", 10);
        this->diag_pub = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/ihm2/diag", 10);

        // subscribers
        this->state_sub = this->create_subscription<ihm2::msg::State>(
                "/ihm2/state",
                10,
                std::bind(
                        &MPCControlNode::controls_callback,
                        this,
                        std::placeholders::_1));

        // load data from file corresponding to a certain track
#ifdef TRACKS_PATH
        // create track instance from file
        std::string csv_track_file(TRACKS_PATH);
        csv_track_file += "/" + this->get_parameter("track_name").as_string() + ".csv";
        RCLCPP_INFO(this->get_logger(), "Loading track from %s", csv_track_file.c_str());
        this->track = std::make_unique<Track>(csv_track_file);

        // set OCP parameters (the kappa_ref values)
        for (int i = 0; i < IHM2_FKIN6_N; i++) {
            if (ihm2_fkin6_acados_update_params(acados_ocp_capsule, i, this->track->get_kappa_ref(), IHM2_FKIN6_NP)) {
                throw FatalNodeError("ihm2_fkin6_acados_update_params() failed");
            }
        }
#else
        raise FatalNodeError("TRACKS_PATH not defined");
#endif

        // initialize messages
        controls_msg.header.frame_id = "car";
        controls_msg.header.stamp = this->now();
        viz_msg.markers.resize(2);
        viz_msg.markers[0].header.frame_id = "world";
        viz_msg.markers[0].header.stamp = this->now();
        viz_msg.markers[0].ns = "trajectory_pred";
        viz_msg.markers[0].id = 0;  // maybe not necessary
        viz_msg.markers[0].type = visualization_msgs::msg::Marker::LINE_STRIP;
        viz_msg.markers[0].scale.x = 0.1;
        viz_msg.markers[0].color = marker_colors("purple");
        viz_msg.markers[0].points.resize(IHM2_FKIN6_N + 1);
        // only points to fill out
        viz_msg.markers[1].header.frame_id = "world";
        viz_msg.markers[1].header.stamp = this->now();
        viz_msg.markers[1].ns = "center_line";
        viz_msg.markers[1].id = 0;  // maybe not necessary
        viz_msg.markers[1].type = visualization_msgs::msg::Marker::LINE_STRIP;
        viz_msg.markers[1].scale.x = 0.05;
        viz_msg.markers[1].color = marker_colors("white");
        viz_msg.markers[1].points.resize(this->track->size());
        for (size_t i = 0; i < this->track->size(); i++) {
            viz_msg.markers[1].points[i].x = this->track->get_X_ref()[i];
            viz_msg.markers[1].points[i].y = this->track->get_Y_ref()[i];
        }
        this->viz_pub->publish(viz_msg);
        viz_msg.markers.pop_back();
    }

    ~MPCControlNode() override {
        int status = ihm2_fkin6_acados_free(acados_ocp_capsule);
        if (status) {
            RCLCPP_ERROR(this->get_logger(), "ihm2_fkin6_acados_free() returned status %d. \n", status);
        }
        status = ihm2_fkin6_acados_free_capsule(acados_ocp_capsule);
        if (status) {
            RCLCPP_ERROR(this->get_logger(), "ihm2_fkin6_acados_free_capsule() returned status %d. \n", status);
        }
        status = ihm2_fkin6_acados_sim_free(acados_sim_capsule);
        if (status) {
            RCLCPP_ERROR(this->get_logger(), "ihm2_fkin6_acados_sim_free() returned status %d. \n", status);
        }
        ihm2_fkin6_acados_sim_solver_free_capsule(acados_sim_capsule);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MPCControlNode>();
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
