// Copyright (c) 2023. Tudor Oancea
#include "acados/utils/math.h"
#include "acados_c/external_function_interface.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_solver_ihm2_fkin4.h"
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

class ControlNode : public rclcpp::Node {
private:
    message_filters::Subscriber<ihm2::msg::Controls> controls_sub;
    message_filters::Subscriber<geometry_msgs::msg::PoseStamped> pose_sub;
    message_filters::Subscriber<geometry_msgs::msg::TwistStamped> vel_sub;
    std::shared_ptr<message_filters::TimeSynchronizer<geometry_msgs::msg::PoseStamped, geometry_msgs::msg::TwistStamped, ihm2::msg::Controls>> synchronizer;

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr viz_pub;
    rclcpp::Publisher<ihm2::msg::Controls>::SharedPtr controls_pub;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub;
    OnSetParametersCallbackHandle::SharedPtr callback_handle;

    double* x;  // x0, x1, ..., xNf stored in a single vector (row-major)
    double* u;  // u0, u1, ..., uNf-1 stored in a single vector (row-major)
    size_t nx, nu, Nf;

    ihm2_fkin4_solver_capsule* acados_ocp_capsule;
    ocp_nlp_config* acados_nlp_config;
    ocp_nlp_dims* acados_nlp_dims;
    ocp_nlp_in* acados_nlp_in;
    ocp_nlp_out* acados_nlp_out;
    ocp_nlp_solver* acados_nlp_solver;
    void* acados_nlp_opts;
    int rti_phase = 0;

    uint sim_steps, sim_counter;

    void controls_callback(
            const geometry_msgs::msg::PoseStamped::ConstSharedPtr& pose_msg,
            const geometry_msgs::msg::TwistStamped::ConstSharedPtr& vel_msg,
            const ihm2::msg::Controls::ConstSharedPtr& current_controls_msg) {
        if (this->sim_counter == 0) {
            // update x and u by shifting the previous values and using uNf-1 = 0 and xNf = xNf-1
            for (size_t i = 0; i < nx * Nf; i++) {
                x[i] = x[i + nx];
            }
            for (size_t i = 0; i < nu * (Nf - 1); i++) {
                u[i] = u[i + nu];
            }
            for (size_t i = 0; i < nx; i++) {
                x[nx * Nf + i] = x[nx * (Nf - 1) + i];
            }
            for (size_t i = 0; i < nu; i++) {
                u[nu * (Nf - 1) + i] = 0;
            }
            // get values from messages
            double X = pose_msg->pose.position.x,
                   Y = pose_msg->pose.position.y,
                   phi = tf2::getYaw(pose_msg->pose.orientation),
                   v_x = vel_msg->twist.linear.x,
                   v_y = vel_msg->twist.linear.y,
                   r = vel_msg->twist.angular.z,
                   T = current_controls_msg->throttle,
                   delta = current_controls_msg->steering;
            // now project the position onto the

            (void) rti_phase;
        }
        sim_counter = (sim_counter + 1) % sim_steps;
    }

public:
    ControlNode() : Node("control_node") {
        // parameters:
        // - track_name: name of the track to load (must be the same as the one used to generate the ocp solver)

        // initialize x and u with zeros
        nx = IHM2_FKIN4_NX;
        nu = IHM2_FKIN4_NU;
        Nf = IHM2_FKIN4_N;
        x = (double*) malloc(sizeof(double) * nx * (Nf + 1));
        u = (double*) malloc(sizeof(double) * nu * Nf);
        RCLCPP_INFO(this->get_logger(), "Initialized x and u with sizes %zu and %zu", nx, nu);

        // load acados ocp solver
        acados_ocp_capsule = ihm2_fkin4_acados_create_capsule();
        double* new_time_steps = NULL;
        int status = ihm2_fkin4_acados_create_with_discretization(acados_ocp_capsule, Nf, new_time_steps);
        if (status) {
            throw FatalNodeError("ihm2_fkin4_acados_create_with_discretization() returned status " + std::to_string(status));
        }
        acados_nlp_config = ihm2_fkin4_acados_get_nlp_config(acados_ocp_capsule);
        acados_nlp_dims = ihm2_fkin4_acados_get_nlp_dims(acados_ocp_capsule);
        acados_nlp_in = ihm2_fkin4_acados_get_nlp_in(acados_ocp_capsule);
        acados_nlp_out = ihm2_fkin4_acados_get_nlp_out(acados_ocp_capsule);
        acados_nlp_solver = ihm2_fkin4_acados_get_nlp_solver(acados_ocp_capsule);
        acados_nlp_opts = ihm2_fkin4_acados_get_nlp_opts(acados_ocp_capsule);
        RCLCPP_INFO(this->get_logger(), "Successfully loaded acados ocp solver");

        // publishers
        this->viz_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("/ihm2/viz/control", 10);
        this->controls_pub = this->create_publisher<ihm2::msg::Controls>("/ihm2/target_controls", 10);

        // subscribers
        this->pose_sub.subscribe(this, "/ihm2/pose");
        this->vel_sub.subscribe(this, "/ihm2/velocity");
        this->controls_sub.subscribe(this, "/ihm2/current_controls");
        // approximate time synchronizer
        this->synchronizer = std::make_shared<
                message_filters::TimeSynchronizer<
                        geometry_msgs::msg::PoseStamped,
                        geometry_msgs::msg::TwistStamped,
                        ihm2::msg::Controls>>(
                this->pose_sub,
                this->vel_sub,
                this->controls_sub,
                10);
        this->synchronizer->registerCallback(
                std::bind(
                        &ControlNode::controls_callback,
                        this,
                        std::placeholders::_1,
                        std::placeholders::_2,
                        std::placeholders::_3));

        // find simulation and OCP time steps from the JSON config files generated by acados
        double dt(0.01), sim_dt(0.01);
#ifdef OCP_JSON_PATH
        std::string ocp_json_path = OCP_JSON_PATH;
        std::ifstream ocp_json_file(ocp_json_path);
        if (!ocp_json_file.is_open()) {
            throw FatalNodeError("Could not open " + ocp_json_path);
        }
        nlohmann::json ocp_json;
        ocp_json_file >> ocp_json;
        dt = ocp_json["solver_options"]["Tsim"];
#else
        throw FatalNodeError("OCP_JSON_PATH not defined");
#endif
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
        sim_steps = std::round(sim_dt / dt);
        sim_counter = 0;


        // load data from file corresponding to a certain track
        
    }

    ~ControlNode() {}
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
