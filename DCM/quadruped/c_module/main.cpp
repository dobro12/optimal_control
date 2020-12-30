#include <iostream>
#include <Python.h>
#include <math.h>
#include "mat.cpp"

#define state_dim 12
#define force_dim 12
#define num_leg 4

#ifndef ERR
#define ERR 1
#endif
#define EPS 0.001

using namespace std;

double obj_func(const mat & x_mat, int time_horizon, double time_step, const mat & Q_mat, const mat & Qf_mat, const mat & R_mat, const mat & target_state_list);
void obj_jacobian(mat & jacobian, const mat & x_mat, int time_horizon, double time_step, const mat & Q_mat, const mat & Qf_mat, const mat & R_mat, const mat & target_state_list);
void eq_cons_func(mat & c_eq, const mat & x_mat, int time_horizon, double time_step, const mat & init_state, const mat & foot_step_list, const mat & contact_list, const mat & base_inertia, double base_mass);
void eq_cons_jacobian(mat & jacobian, const mat & x_mat, int time_horizon, double time_step, const mat & init_state, const mat & foot_step_list, const mat & contact_list, const mat & base_inertia, double base_mass);
void ineq_cons_func(mat & c_ineq, const mat & x_mat, int time_horizon, double time_step, const mat & contact_list, double friction_coef);
void ineq_cons_jacobian(mat & jacobian, const mat & x_mat, int time_horizon, double time_step, const mat & contact_list, double friction_coef);

double gravity_data[] = {0, 0, 0, 0, 0, 0, 0, 0, -9.8, 0, 0, 0};
mat gravity(gravity_data, state_dim, 1);

extern "C" {
    void solve(double* x_data, double* lambda_value, double* mu_value, double learning_rate, \
            int time_horizon, double time_step, \
            double* Q_data, double* Qf_data, double* R_data, double* target_trajectory_data, \
            double* init_state_data, double* foot_step_data, double* contact_data, double* base_inertia_data, double base_mass, \
            double friction_coef){
        cout<<"hello!"<<endl;

        // ######## declare variables ######## //
        mat x_mat(x_data, 1, time_horizon*(state_dim + force_dim));
        mat Qf_mat(Qf_data, state_dim, state_dim);
        mat Q_mat(Q_data, state_dim, state_dim);
        mat R_mat(R_data, force_dim, force_dim);
        mat target_state_list(target_trajectory_data, time_horizon, state_dim);
        mat init_state(init_state_data, state_dim, 1);
        mat foot_step_list(foot_step_data, time_horizon, num_leg*3);
        mat contact_list(contact_data, time_horizon, num_leg);
        mat base_inertia(base_inertia_data, 3, 3);
        mat lambda_mat(lambda_value, 1, time_horizon*state_dim);
        mat mu_mat(mu_value, 1, time_horizon*force_dim);

        double f_value = 0.0;
        double pre_f_value = 0.0;
        mat c_eq(1, time_horizon*state_dim);
        mat c_ineq(1, time_horizon*force_dim);
        mat dfdx(1, time_horizon*(state_dim + force_dim));
        mat dc_eqdx(time_horizon*state_dim, time_horizon*(state_dim + force_dim));
        mat dc_ineqdx(time_horizon*force_dim, time_horizon*(state_dim + force_dim));
        // ################################### //

        eq_cons_func(c_eq, x_mat, time_horizon, time_step, init_state, foot_step_list, contact_list, base_inertia, base_mass);
        ineq_cons_func(c_ineq, x_mat, time_horizon, time_step, contact_list, friction_coef);
        for(int i=0;i<500;i++){
            pre_f_value = f_value;

            // update lagrangian multiplier
            lambda_mat += c_eq*learning_rate;
            mu_mat += c_ineq*learning_rate;
            lambda_mat.clip(-1e10, 1e10);
            mu_mat.clip(0.0, 1e10);

            // update x
            for(int j=0;j<10;j++){
                dfdx.zeros();
                dc_eqdx.zeros();
                dc_ineqdx.zeros();
                obj_jacobian(dfdx, x_mat, time_horizon, time_step, Q_mat, Qf_mat, R_mat, target_state_list);
                eq_cons_jacobian(dc_eqdx, x_mat, time_horizon, time_step, init_state, foot_step_list, contact_list, base_inertia, base_mass);
                ineq_cons_jacobian(dc_ineqdx, x_mat, time_horizon, time_step, contact_list, friction_coef);
                x_mat -= (dfdx + lambda_mat.matmul(dc_eqdx) + mu_mat.matmul(dc_ineqdx))*learning_rate;
            }
            f_value = obj_func(x_mat, time_horizon, time_step, Q_mat, Qf_mat, R_mat, target_state_list);
            cout<<"####################################################"<<endl;
            cout<<"objective function : "<<f_value<<endl;
            cout<<"difference of f : "<<abs(f_value - pre_f_value)<<endl;

            eq_cons_func(c_eq, x_mat, time_horizon, time_step, init_state, foot_step_list, contact_list, base_inertia, base_mass);
            ineq_cons_func(c_ineq, x_mat, time_horizon, time_step, contact_list, friction_coef);
            //cout<<f_value + (lambda_mat*c_eq).sum() + (mu_mat*c_ineq).sum()<<endl;
            cout<<"# of violation of c_eq : "<<c_eq.count_violation(-1e-5, 1e-5)<<endl;            
            cout<<"# of violation of c_ineq : "<<c_ineq.count_violation(-1e10, 1e-5)<<endl;            
        }

        memcpy(x_data, x_mat.data, sizeof(double)*(time_horizon*(state_dim + force_dim)));
        return;
    }
}


double obj_func(const mat & x_mat, int time_horizon, double time_step, \
            const mat & Q_mat, const mat & Qf_mat, const mat & R_mat, const mat & target_state_list){
    mat state_list(x_mat.data, time_horizon, state_dim);
    mat force_list(&x_mat.data[time_horizon*state_dim], time_horizon, force_dim);
    mat state_difference = state_list - target_state_list;
    mat sd1(state_difference.data, time_horizon - 1, state_dim);
    mat sd2(&state_difference.data[(time_horizon - 1)*state_dim], 1, state_dim);

    double f = 0.5*(sd1.matmul(Q_mat)*sd1).sum();
    f += 0.5*time_step*(sd2.matmul(Qf_mat)*sd2).sum();
    f += 0.5*time_step*(force_list.matmul(R_mat)*force_list).sum();
    return f;
}

void obj_jacobian(mat & jacobian, const mat & x_mat, int time_horizon, double time_step, \
            const mat & Q_mat, const mat & Qf_mat, const mat & R_mat, const mat & target_state_list){
    mat state_list(x_mat.data, time_horizon, state_dim);
    mat force_list(&x_mat.data[time_horizon*state_dim], time_horizon, force_dim);
    mat state_difference = state_list - target_state_list;
    mat sd1(state_difference.data, time_horizon - 1, state_dim);
    mat sd2(&state_difference.data[(time_horizon - 1)*state_dim], 1, state_dim);

    memcpy(&jacobian.data[0], (sd1.matmul(Q_mat)*time_step).data, sizeof(double)*((time_horizon - 1)*state_dim));
    memcpy(&jacobian.data[(time_horizon - 1)*state_dim], (sd2.matmul(Qf_mat)*time_step).data, sizeof(double)*state_dim);
    memcpy(&jacobian.data[time_horizon*state_dim], (force_list.matmul(R_mat)*time_step).data, sizeof(double)*(time_horizon*state_dim));
}

void eq_cons_func(mat & c_eq, const mat & x_mat, int time_horizon, double time_step, \
            const mat & init_state, const mat & foot_step_list, const mat & contact_list, const mat & base_inertia, double base_mass){
    mat state(state_dim, 1);
    mat next_state(state_dim, 1);
    mat force(force_dim, 1);
    double yaw;
    mat pos_com(3, 1);
    mat yaw_mat(3, 3);
    mat global_inertia(3, 3);
    mat foot_step(3, 1);
    mat temp_mat;

    state = init_state;
    for(int time_idx=0;time_idx<time_horizon;time_idx++){
        memcpy(next_state.data, &x_mat.data[time_idx*state_dim], sizeof(double)*state_dim);
        for(int leg_idx=0;leg_idx<num_leg;leg_idx++){
            for(int i=0;i<3;i++){
                if(contact_list.data[time_idx*num_leg + leg_idx] == 1)
                    force.data[leg_idx*3 + i] = x_mat.data[time_horizon*state_dim + time_idx*force_dim + leg_idx*3 + i];
                else
                    force.data[leg_idx*3 + i] = 0.0;
            }
        }

        yaw = state.data[5];
        memcpy(pos_com.data, state.data, sizeof(double)*3);
        yaw_mat = mat::z_rot(yaw);
        global_inertia = yaw_mat.matmul(base_inertia.matmul(yaw_mat.transpose()));

        mat A_mat = mat::eye(state_dim);
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                if(i == j)
                    A_mat.data[i*state_dim + j + 6] = time_step;
                A_mat.data[(i + 3)*state_dim + j + 9] = time_step*yaw_mat.data[3*j + i];
            }
        }

        mat B_mat(state_dim, force_dim);
        for(int leg_idx=0;leg_idx<num_leg;leg_idx++){
            memcpy(foot_step.data, &foot_step_list.data[time_idx*num_leg*3 + leg_idx*3], sizeof(double)*3);
            foot_step -= pos_com;
            temp_mat = global_inertia.inverse_matmul(mat::cross_product(foot_step.data))*time_step;
            for(int i=0;i<3;i++){
                for(int j=0;j<3;j++){
                    if(i == j)
                        B_mat.data[(i + 6)*force_dim + j + leg_idx*3] = time_step/base_mass;
                    B_mat.data[(i + 9)*state_dim + j + leg_idx*3] = temp_mat.data[i*3 + j];
                }
            }
        }

        temp_mat = A_mat.matmul(state) + B_mat.matmul(force) + gravity - next_state;
        memcpy(&c_eq.data[time_idx*state_dim], temp_mat.data, sizeof(double)*state_dim);
    }

    return;
}

void eq_cons_jacobian(mat & jacobian, const mat & x_mat, int time_horizon, double time_step, \
            const mat & init_state, const mat & foot_step_list, const mat & contact_list, const mat & base_inertia, double base_mass){
    mat state(state_dim, 1);
    mat next_state(state_dim, 1);
    mat force(force_dim, 1);
    double yaw;
    mat pos_com(3, 1);
    mat ang_vel_base(3, 1);
    mat yaw_mat(3, 3);
    mat global_inertia(3, 3);
    mat foot_step(3, 1);
    mat dR_mat;
    mat sum_r_cross_f(3, 1);
    mat leg_force(3, 1);
    mat temp_mat;
    mat temp_mat2;

    state = init_state;
    for(int time_idx=0;time_idx<time_horizon;time_idx++){
        memcpy(next_state.data, &x_mat.data[time_idx*state_dim], sizeof(double)*state_dim);
        for(int leg_idx=0;leg_idx<num_leg;leg_idx++){
            for(int i=0;i<3;i++){
                if(contact_list.data[time_idx*num_leg + leg_idx] == 1)
                    force.data[leg_idx*3 + i] = x_mat.data[time_horizon*state_dim + time_idx*force_dim + leg_idx*3 + i];
                else
                    force.data[leg_idx*3 + i] = 0.0;
            }
        }

        yaw = state.data[5];
        memcpy(pos_com.data, state.data, sizeof(double)*3);
        memcpy(ang_vel_base.data, &state.data[9], sizeof(double)*3);
        yaw_mat = mat::z_rot(yaw);
        global_inertia = yaw_mat.matmul(base_inertia.matmul(yaw_mat.transpose()));

        mat A_mat = mat::eye(state_dim);
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                if(i == j)
                    A_mat.data[i*state_dim + j + 6] = time_step;
                A_mat.data[(i + 3)*state_dim + j + 9] = time_step*yaw_mat.data[3*j + i];
            }
        }

        mat B_mat(state_dim, force_dim);
        for(int leg_idx=0;leg_idx<num_leg;leg_idx++){
            if(contact_list.data[time_idx*num_leg + leg_idx] == 0)
                continue;

            memcpy(foot_step.data, &foot_step_list.data[time_idx*num_leg*3 + leg_idx*3], sizeof(double)*3);
            foot_step -= pos_com;
            temp_mat = global_inertia.inverse_matmul(mat::cross_product(foot_step.data))*time_step;
            for(int i=0;i<3;i++){
                for(int j=0;j<3;j++){
                    if(i == j)
                        B_mat.data[(i + 6)*force_dim + j + leg_idx*3] = time_step/base_mass;
                    B_mat.data[(i + 9)*state_dim + j + leg_idx*3] = temp_mat.data[i*3 + j];
                }
            }
        }

        for(int i=0;i<state_dim;i++){
            jacobian.data[time_horizon*(state_dim + force_dim)*(time_idx*state_dim + i) + time_idx*state_dim + i] = -1.0;
        }
        for(int i=0;i<state_dim;i++){
            for(int j=0;j<force_dim;j++){
                jacobian.data[time_horizon*(state_dim + force_dim)*(time_idx*state_dim + i) + time_horizon*state_dim + time_idx*force_dim + j] = B_mat.data[i*force_dim + j];
            }
        }

        if(time_idx > 0){
            for(int i=0;i<state_dim;i++){
                for(int j=0;j<state_dim;j++){
                    jacobian.data[time_horizon*(state_dim + force_dim)*(time_idx*state_dim + i) + (time_idx - 1)*state_dim + j] = A_mat.data[i*state_dim + j];
                }
            }

            temp_mat = mat(state_dim, state_dim);
            dR_mat = mat::z_rot_dot(yaw);
            temp_mat2 = dR_mat.transpose().matmul(ang_vel_base)*time_step;
            for(int i=0;i<3;i++){
                temp_mat.data[(i + 3)*state_dim + 5] += temp_mat2.data[i];
            }

            sum_r_cross_f.zeros();
            for(int leg_idx=0;leg_idx<num_leg;leg_idx++){
                memcpy(foot_step.data, &foot_step_list.data[time_idx*num_leg*3 + leg_idx*3], sizeof(double)*3);
                foot_step -= pos_com;
                memcpy(leg_force.data, &force.data[leg_idx*3], sizeof(double)*3);
                sum_r_cross_f += foot_step.cross(leg_force)*time_step;
            }
            temp_mat2 = dR_mat.matmul(base_inertia.inverse_matmul(yaw_mat.transpose().matmul(sum_r_cross_f))) \
                        + yaw_mat.matmul(base_inertia.inverse_matmul(dR_mat.transpose().matmul(sum_r_cross_f)));
            for(int i=0;i<3;i++){
                temp_mat.data[(i + 9)*state_dim + 5] += temp_mat2.data[i];
            }

            for(int i=0;i<state_dim;i++){
                for(int j=0;j<state_dim;j++){
                    jacobian.data[time_horizon*(state_dim + force_dim)*(time_idx*state_dim + i) + (time_idx - 1)*state_dim + j] += temp_mat.data[i*state_dim + j];
                }
            }
        }

        state = next_state;
    }
    return;
}

void ineq_cons_func(mat & c_ineq, const mat & x_mat, int time_horizon, double time_step, \
            const mat & contact_list, double friction_coef){
    mat force(force_dim, 1);

    for(int time_idx=0;time_idx<time_horizon;time_idx++){
        for(int leg_idx=0;leg_idx<num_leg;leg_idx++){
            for(int i=0;i<3;i++){
                if(contact_list.data[time_idx*num_leg + leg_idx] == 1)
                    force.data[leg_idx*3 + i] = x_mat.data[time_horizon*state_dim + time_idx*force_dim + leg_idx*3 + i];
                else
                    force.data[leg_idx*3 + i] = 0.0;
            }

            c_ineq.data[time_idx*force_dim + leg_idx*3] = abs(force.data[leg_idx*3]) - friction_coef*force.data[leg_idx*3 + 2];
            c_ineq.data[time_idx*force_dim + leg_idx*3 + 1] = abs(force.data[leg_idx*3 + 1]) - friction_coef*force.data[leg_idx*3 + 2];
            c_ineq.data[time_idx*force_dim + leg_idx*3 + 2] = -force.data[leg_idx*3 + 2];
        }
    }

    return;
}

void ineq_cons_jacobian(mat & jacobian, const mat & x_mat, int time_horizon, double time_step, \
            const mat & contact_list, double friction_coef){
    mat force(force_dim, 1);

    for(int time_idx=0;time_idx<time_horizon;time_idx++){
        for(int leg_idx=0;leg_idx<num_leg;leg_idx++){
            for(int i=0;i<3;i++){
                if(contact_list.data[time_idx*num_leg + leg_idx] == 1)
                    force.data[leg_idx*3 + i] = x_mat.data[time_horizon*state_dim + time_idx*force_dim + leg_idx*3 + i];
                else
                    force.data[leg_idx*3 + i] = 0.0;
            }

            jacobian.data[time_horizon*(state_dim + force_dim)*(time_idx*force_dim + leg_idx*3) + time_horizon*state_dim + time_idx*force_dim + leg_idx*3] = (force.data[leg_idx*3] > 0) ? 1 : -1;
            jacobian.data[time_horizon*(state_dim + force_dim)*(time_idx*force_dim + leg_idx*3) + time_horizon*state_dim + time_idx*force_dim + leg_idx*3 + 2] = -friction_coef;
            jacobian.data[time_horizon*(state_dim + force_dim)*(time_idx*force_dim + leg_idx*3 + 1) + time_horizon*state_dim + time_idx*force_dim + leg_idx*3 + 1] = (force.data[leg_idx*3 + 1] > 0) ? 1 : -1;
            jacobian.data[time_horizon*(state_dim + force_dim)*(time_idx*force_dim + leg_idx*3 + 1) + time_horizon*state_dim + time_idx*force_dim + leg_idx*3 + 2] = -friction_coef;
            jacobian.data[time_horizon*(state_dim + force_dim)*(time_idx*force_dim + leg_idx*3 + 2) + time_horizon*state_dim + time_idx*force_dim + leg_idx*3 + 2] = -1;
        }
    }
    return;
}
