// Copyright (c) 2023. Tudor Oancea
#ifndef OCP_SOLVER_INTERFACE_HPP
#define OCP_SOLVER_INTERFACE_HPP
#include "acados/utils/math.h"
#include "acados_c/external_function_interface.h"
#include "acados_c/ocp_nlp_interface.h"
#include <string>
#include <unordered_map>

// ===================== TEMPLATED CODE ========================================
enum class AvailableTrack : char {
    FSDS_COMPETITION_1,
    FSDS_COMPETITION_2,
    FSDS_COMPETITION_3,
};
// ========================= END ===============================================

AvailableTrack available_track_from_string(const std::string& str);
std::string available_track_to_string(const AvailableTrack& track);

class OCPSolverInterface {
private:
    std::unordered_map<AvailableTrack, void*> capsules;
    std::unordered_map<AvailableTrack, ocp_nlp_config*> configs;
    std::unordered_map<AvailableTrack, ocp_nlp_dims*> dims;
    std::unordered_map<AvailableTrack, ocp_nlp_in*> in;
    std::unordered_map<AvailableTrack, ocp_nlp_out*> out;
    std::unordered_map<AvailableTrack, ocp_nlp_solver*> solver;
    std::unordered_map<AvailableTrack, void*> opts;

public:
    OCPSolverInterface();
    ~OCPSolverInterface();
    void solve(const AvailableTrack& track, double* x, double* u, double* time_tot);  // the first values of x are used to set the initial state constraints
};

#endif  // OCP_SOLVER_INTERFACE_HPP