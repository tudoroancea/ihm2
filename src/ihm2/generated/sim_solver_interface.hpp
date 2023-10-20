// Copyright (c) 2023. Tudor Oancea
#ifndef SIM_SOLVER_INERFACE_HPP
#define SIM_SOLVER_INERFACE_HPP
#include "acados/utils/math.h"
#include "acados_c/external_function_interface.h"
#include "acados_c/sim_interface.h"
#include <string>
#include <unordered_map>

enum class AvailableModel : char {
    IHM2_DYN6,
    IHM2_KIN4,
};

AvailableModel available_model_from_string(const std::string& str);
std::string available_model_to_string(const AvailableModel& model);

class SimSolverInterface {
private:
    std::unordered_map<AvailableModel, void*> capsules;
    std::unordered_map<AvailableModel, sim_config*> configs;
    std::unordered_map<AvailableModel, void*> dims;
    std::unordered_map<AvailableModel, sim_in*> in;
    std::unordered_map<AvailableModel, sim_out*> out;

public:
    SimSolverInterface();
    ~SimSolverInterface();
    void solve(const AvailableModel& model, double* x, double* u, double* time_tot);  // the first values of x are used to set the initial state constraints
    size_t nx(const AvailableModel& model);
    size_t nu(const AvailableModel& model);
};

#endif  // SIM_SOLVER_INERFACE_HPP