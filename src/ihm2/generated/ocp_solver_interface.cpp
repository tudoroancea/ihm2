// Copyright (c) 2023. Tudor Oancea
#include "ocp_solver_interface.hpp"
// ===================== TEMPLATED CODE ========================================
#include "fsds_competition_1/acados_solver_ihm2_fkin4_fsds_competition_1.h"
#include "fsds_competition_2/acados_solver_ihm2_fkin4_fsds_competition_2.h"
#include "fsds_competition_3/acados_solver_ihm2_fkin4_fsds_competition_3.h"

AvailableTrack available_track_from_string(const std::string& str) {
    if (str == "fsds_competition_1") {
        return AvailableTrack::FSDS_COMPETITION_1;
    } else if (str == "fsds_competition_2") {
        return AvailableTrack::FSDS_COMPETITION_2;
    } else if (str == "fsds_competition_3") {
        return AvailableTrack::FSDS_COMPETITION_3;
    } else {
        throw std::runtime_error("unknown track name " + str);
    }
}

std::string available_track_to_string(const AvailableTrack& track) {
    switch (track) {
        case AvailableTrack::FSDS_COMPETITION_1:
            return "fsds_competition_1";
        case AvailableTrack::FSDS_COMPETITION_2:
            return "fsds_competition_2";
        case AvailableTrack::FSDS_COMPETITION_3:
            return "fsds_competition_3";
        default:
            throw std::runtime_error("unknown track name");
    }
}
// ========================= END ===============================================

#define init_solver(track_id, track_id_upper)                                                                          \
    AvailableTrack track = available_track_from_string(#track_id);                                                     \
    this->capsules[track] = ihm2_fkin4_##track_id##_acados_create();                                                   \
    double* null_time_steps = NULL;                                                                                    \
    int N = ##track_id_upper##_N;                                                                                      \
    int status = ihm2_fkin4_##track_id##_acados_create_with_discretization(this->capsules[track], N, null_time_steps); \
    if (status) {                                                                                                      \
        printf("ihm2_fkin4_%s_acados_create_with_discretization() returned status %d\n", #track_id, status);           \
        exit(1);                                                                                                       \
    }                                                                                                                  \
    this->configs[track] = ihm2_fkin4_##track_id##_acados_get_nlp_config(this->capsules[track]);                       \
    this->dims[track] = ihm2_fkin4_##track_id##_acados_get_nlp_dims(this->capsules[track]);                            \
    this->in[track] = ihm2_fkin4_##track_id##_acados_get_nlp_in(this->capsules[track]);                                \
    this->out[track] = ihm2_fkin4_##track_id##_acados_get_nlp_out(this->capsules[track]);                              \
    this->solver[track] = ihm2_fkin4_##track_id##_acados_get_nlp_solver(this->capsules[track]);                        \
    this->opts[track] = ihm2_fkin4_##track_id##_acados_get_nlp_opts(this->capsules[track]);

#define free_solver(track_id)                                                                  \
    AvailableTrack track = available_track_from_string(#track_id);                             \
    int status = ihm2_fkin4_##track_id##_acados_free(this->capsules[track]);                   \
    if (status) {                                                                              \
        printf("ihm2_fkin4_%s_acados_free() returned status %d\n", #track_id, status);         \
        exit(1);                                                                               \
    }                                                                                          \
    status = ihm2_fkin4_##track_id##_acados_free_capsule(this->capsules[track]);               \
    if (status) {                                                                              \
        printf("ihm2_fkin4_%s_acados_free_capsule() returned status %d\n", #track_id, status); \
        exit(1);                                                                               \
    }

#define call_solver(track_id)                                                                                              \
    int nbx0 = this->dims[track]->nbx[0]; /* number of values in x0 constraint */                                          \
    int idxbx0* = malloc(nbx0);                                                                                            \
    for (int i = 0; i < nbx0; i++) {                                                                                       \
        idxbx0[i] = i;                                                                                                     \
    }                                                                                                                      \
    ocp_nlp_constraints_model_set(this->configs[track], this->dims[track], this->in[track], 0, "lbx", x);                  \
    ocp_nlp_constraints_model_set(this->configs[track], this->dims[track], this->in[track], 0, "ubx", x);                  \
    int rti_phase = 0;                                                                                                     \
    int Nf = this->config[track]->N;                                                                                       \
    for (int i = 0; i < Nf; i++) {                                                                                         \
        ocp_nlp_out_set(this->configs[track], this->dims[track], this->out[track], i, "x", x + i * this->dims[track]->nx); \
        ocp_nlp_out_set(this->configs[track], this->dims[track], this->out[track], i, "u", u + i * this->dims[track]->nu); \
    }                                                                                                                      \
    ocp_nlp_out_set(this->configs[track], this->dims[track], this->out[track], Nf, "x", x + Nf * this->dims[track]->nx);   \
    ocp_nlp_solver_opts_set(this->configs[track], this->opts[track], "rti_phase", &rti_phase);                             \
    int status = ihm2_fkin4_##track_id##_acados_solve(this->capsules[track]);                                              \
    if (status) {                                                                                                          \
        printf("ihm2_fkin4_%s_acados_solve() returned status %d\n", #track_id, status);                                    \
        exit(1);                                                                                                           \
    }                                                                                                                      \
    for (int i = 0; i < Nf; i++) {                                                                                         \
        ocp_nlp_out_get(this->configs[track], this->dims[track], this->out[track], i, "x", x + i * this->dims[track]->nx); \
        ocp_nlp_out_get(this->configs[track], this->dims[track], this->out[track], i, "u", u + i * this->dims[track]->nu); \
    }                                                                                                                      \
    ocp_nlp_out_get(this->configs[track], this->dims[track], this->out[track], Nf, "x", x + Nf * this->dims[track]->nx);   \
    ocp_nlp_get(this->configs[track], this->solver[track], "time_tot", *time_tot);


// ===================== TEMPLATED CODE ========================================
OCPSolverInterface::OCPSolverInterface() {
    init_solver(fsds_competition_1, FSDS_COMPETITION_1);
    init_solver(fsds_competition_2, FSDS_COMPETITION_2);
    init_solver(fsds_competition_3, FSDS_COMPETITION_3);
}

OCPSolverInterface::~OCPSolverInterface() {
    free_solver(fsds_competition_1);
    free_solver(fsds_competition_2);
    free_solver(fsds_competition_3);
}

void OCPSolverInterface::solve(const AvailableTrack& track, double* x, double* u, double* time_tot) {
    switch (track) {
        case AvailableTrack::FSDS_COMPETITION_1:
            call_solver(fsds_competition_1);
            break;
        case AvailableTrack::FSDS_COMPETITION_2:
            call_solver(fsds_competition_2);
            break;
        case AvailableTrack::FSDS_COMPETITION_3:
            call_solver(fsds_competition_3);
            break;
    }
}
// ======================= END =================================================