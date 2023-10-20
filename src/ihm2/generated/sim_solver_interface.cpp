// Copyright (c) 2023. Tudor Oancea
#include "acados/utils/math.h"
#include "acados_c/sim_interface.h"
#include "sim_interface.hpp"
// ===================== TEMPLATED CODE ========================================
#include "acados_sim_solver_ihm2_dyn6.h"
#include "acados_sim_solver_ihm2_kin4.h"

AvailableModel available_model_from_string(const std::string& str) {
    if (str == "ihm2_dyn6") {
        return AvailableModel::IHM2_DYN6;
    } else if (str == "ihm2_kin4") {
        return AvailableModel::IHM2_KIN4;
    } else {
        throw std::runtime_error("unknown model name " + str);
    }
}

std::string available_model_to_string(const AvailableModel& model) {
    switch (model) {
        case AvailableModel::IHM2_DYN6:
            return "ihm2_dyn6";
        case AvailableModel::IHM2_KIN4:
            return "ihm2_kin4";
        default:
            throw std::runtime_error("unknown model name");
    }
}
// ========================= END ===============================================

#define init_solver(model_id)                                                         \
    AvailableModel model = available_model_from_string(#model_id);                    \
    capsule = ##model_id##_acados_sim_solver_create_capsule();                        \
    this->capsules[model] = capsule;                                                  \
    int status = ##model_id##_acados_sim_create(this->capsules[model]);               \
    if (status) {                                                                     \
        printf("%s_acados_sim_create() returned status %d\n", #model_id, status);     \
        exit(1);                                                                      \
    }                                                                                 \
    this->configs[model] = ##model_id##_acados_get_sim_config(this->capsules[model]); \
    this->dims[model] = ##model_id##_acados_get_sim_dims(this->capsules[model]);      \
    this->in[model] = ##model_id##_acados_get_sim_in(this->capsules[model]);          \
    this->out[model] = ##model_id##_acados_get_sim_out(this->capsules[model]);

#define free_solver(model_id)                                                   \
    AvailableModel model = available_model_from_string(#model_id);              \
    ##model_id##_sim_solver_capsule* capsule = this->capsules[model];           \
    int status = ##model_id##_acados_sim_free(capsule);                         \
    if (status) {                                                               \
        printf("%s_acados_sim_free() returned status %d\n", #model_id, status); \
        exit(1);                                                                \
    }                                                                           \
    ##model_id##_acados_sim_solver_free_capsule(capsule);

#define call_solver(model_id)                                                       \
    AvailableModel model = available_model_from_string(#model_id);                  \
    sim_in_set(this->configs[model], this->dims[model], this->in[model], "x", x);   \
    sim_in_set(this->configs[model], this->dims[model], this->in[model], "u", u);   \
    int status = ##model_id##_acados_sim_solve(this->capsules[model]);              \
    if (status != ACADOS_SUCCESS) {                                                 \
        printf("%s_acados_sim_solve() returned status %d\n", #model_id, status);    \
        exit(1);                                                                    \
    }                                                                               \
    sim_out_get(this->configs[model], this->dims[model], this->out[model], "x", x); \
    sim_out_get(this->configs[model], this->dims[model], this->out[model], "time_tot", time_tot);


#define find_nx(model_id_upper) ##model_id_upper##_NX

#define find_nu(model_id_upper) ##model_id_upper##_NU

SimSolverInterface::SimSolverInterface() {
    init_solver(ihm2_dyn6);
    init_solver(ihm2_kin4);
}

SimSolverInterface::~SimSolverInterface() {
    free_solver(ihm2_dyn6);
    free_solver(ihm2_kin4);
}

void SimSolverInterface::solve(const AvailableModel& model, double* x, double* u, double* time_tot) {
    switch (model) {
        case AvailableModel::IHM2_DYN6:
            call_solver(ihm2_dyn6);
            break;
        case AvailableModel::IHM2_KIN4:
            call_solver(ihm2_kin4);
            break;
        default:
            throw std::runtime_error("unknown model name");
    }
}

size_t SimSolverInterface::nx(const AvailableModel& model) {
    switch (model) {
        case AvailableModel::IHM2_DYN6:
            return find_nx(IHM2_DYN6);
        case AvailableModel::IHM2_KIN4:
            return find_nx(IHM2_KIN4);
        default:
            throw std::runtime_error("unknown model name");
    }
}
size_t SimSolverInterface::nu(const AvailableModel& model) {
    switch (model) {
        case AvailableModel::IHM2_DYN6:
            return find_nu(IHM2_DYN6);
        case AvailableModel::IHM2_KIN4:
            return find_nu(IHM2_KIN4);
        default:
            throw std::runtime_error("unknown model name");
    }
}