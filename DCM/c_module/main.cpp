#include <iostream>
#include <Python.h>
#include <math.h>
#include "mat.cpp"

#ifndef ERR
#define ERR 1
#endif
#define EPS 0.001

extern "C" {
    void get_action(int X_DIM, int U_DIM, double* init_x, int time_horizon, double* A_data, double* B_data,
                        double* R_data, double* Q_data, double* Qf_data, double* return_var){
        // ######## declare variables ########
        mat A_mat(A_data, X_DIM, X_DIM);
        mat B_mat(B_data, X_DIM, U_DIM);
        mat R_mat(R_data, U_DIM, U_DIM);
        mat Q_mat(Q_data, X_DIM, X_DIM);
        mat Qf_mat(Qf_data, X_DIM, X_DIM);

        /*
        std::cout<<"x dimension : "<<X_DIM<<", u dimension : "<<U_DIM<<", time horizon : "<<time_horizon<<std::endl;
        A_mat.print();
        B_mat.print();
        R_mat.print();
        Q_mat.print();
        Qf_mat.print();
        */

        mat* P_mat_list = new mat[time_horizon];
        mat S_mat = Qf_mat;
        // ###################################

        for(int t_idx=time_horizon-1; t_idx>=0; t_idx--){
            //std::cout<<"[debuging] "<<t_idx<<std::endl;
            P_mat_list[t_idx] = (R_mat + B_mat.transpose().matmul(S_mat.matmul(B_mat))).inverse_matmul(B_mat.transpose().matmul(S_mat.matmul(A_mat)));
            //std::cout<<"[debuging2] "<<t_idx<<std::endl;
            S_mat = Q_mat + A_mat.transpose().matmul(S_mat.matmul(A_mat - B_mat.matmul(P_mat_list[t_idx])));
        }

        for(int i=0;i<time_horizon;i++){
            memcpy(&return_var[i*X_DIM*U_DIM], P_mat_list[i].data, sizeof(double)*X_DIM*U_DIM);
        }

        delete[] P_mat_list;
        return;
    }
}
