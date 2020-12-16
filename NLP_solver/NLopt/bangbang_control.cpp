#include<algorithm>
#include <iostream>
#include <nlopt.h>
#include <math.h>
#include <time.h>
#include "mat.cpp"

#define X_DIM 2
#define U_DIM 1

mat dynamics(const mat & x, const mat & u){
    static double A_data[] = {0, 1, 0, 0};
    static double B_data[] = {0, 1};
    static mat A_mat(A_data, 2, 2);
    static mat B_mat(B_data, 2, 1);
    return A_mat.matmul(x) + B_mat.matmul(u);
}

double objective(unsigned n, const double *x, double *grad, void *my_func_data){
    if (grad) {
        for(int i=0;i<n;i++) grad[i] = 0.0;
        grad[0] = 1.0;
    }
    return x[0];
}

mat get_collocation_constraints(int N, const double* x){
    double t_f = x[0];
    mat u_list(&x[1], N+1, U_DIM);
    mat x_list(&x[1 + (N+1)*U_DIM], N+1, X_DIM);
    double dt = t_f/N;
    mat x_dot_list = dynamics(x_list.transpose(), u_list.transpose()).transpose();

    mat x_left(x_list.data, N, X_DIM);
    mat x_right(&x_list.data[X_DIM], N, X_DIM);
    mat x_dot_left(x_dot_list.data, N, X_DIM);
    mat x_dot_right(&x_dot_list.data[X_DIM], N, X_DIM);
    mat u_left(u_list.data, N, U_DIM);
    mat u_right(&u_list.data[U_DIM], N, U_DIM);

    mat x_c = (x_left + x_right)*0.5 + (x_dot_left - x_dot_right)*(dt*0.125);
    mat u_c = (u_left + u_right)*0.5;
    mat x_dot_c = dynamics(x_c.transpose(), u_c.transpose()).transpose();

    return x_left - x_right + (x_dot_left + x_dot_c*4 + x_dot_right)*(dt/6);
}

mat get_collocation_constraint(int N, const double* u, const double* x, double t_f){
    mat u_list(u, 2, U_DIM);
    mat x_list(x, 2, X_DIM);
    double dt = t_f/N;
    mat x_dot_list = dynamics(x_list.transpose(), u_list.transpose()).transpose();

    mat x_left(x_list.data, 1, X_DIM);
    mat x_right(&x_list.data[X_DIM], 1, X_DIM);
    mat x_dot_left(x_dot_list.data, 1, X_DIM);
    mat x_dot_right(&x_dot_list.data[X_DIM], 1, X_DIM);
    mat u_left(u_list.data, 1, U_DIM);
    mat u_right(&u_list.data[U_DIM], 1, U_DIM);

    mat x_c = (x_left + x_right)*0.5 + (x_dot_left - x_dot_right)*(dt*0.125);
    mat u_c = (u_left + u_right)*0.5;
    mat x_dot_c = dynamics(x_c.transpose(), u_c.transpose()).transpose();

    return x_left - x_right + (x_dot_left + x_dot_c*4 + x_dot_right)*(dt/6);
}

void equality_constraints(unsigned m, double *result, unsigned n, const double* x, double* grad, void* f_data){
    double t_f = x[0];
    int N = (n - 1)/(X_DIM + U_DIM) - 1;
    mat coll_cons;
    mat temp_cons;
    int i = 0, j = 0;
    double eps = 1e-6;
    double* temp_u_list_data = new double[U_DIM*2];
    double* temp_x_list_data = new double[X_DIM*2];

    for(i=0;i<N;i++){
        coll_cons = get_collocation_constraint(N, &x[1 + i*U_DIM], &x[1 + (N + 1)*U_DIM + i*X_DIM], t_f);
        std::copy(coll_cons.data, coll_cons.data + X_DIM, result + X_DIM*i);

        if (grad){
            /*
            for(j=0;j<n;j++){
                for(int k=0;k<X_DIM;k++) grad[n*(i*X_DIM + k) + j] = 0.0;
            }
            */

            temp_cons = (get_collocation_constraint(N, &x[1 + i*U_DIM], &x[1 + (N + 1)*U_DIM + i*X_DIM], x[0] + eps) - coll_cons)*(1/eps);
            for(int kk=0;kk<X_DIM;kk++) grad[n*(i*X_DIM + kk) + 0] = temp_cons.data[kk];

            for(int k=0;k<U_DIM;k++){
                std::copy(x + (1 + i*U_DIM), x + (1 + (i + 2)*U_DIM), temp_u_list_data);
                std::copy(x + (1 + (N + 1)*U_DIM + i*X_DIM), x + (1 + (N + 1)*U_DIM + (i + 2)*X_DIM), temp_x_list_data);
                temp_u_list_data[k] += eps;
                temp_cons = (get_collocation_constraint(N, temp_u_list_data, temp_x_list_data, t_f) - coll_cons)*(1/eps);
                for(int kk=0;kk<X_DIM;kk++) grad[n*(i*X_DIM + kk) + 1 + i*U_DIM + k] = temp_cons.data[kk];

                std::copy(x + (1 + i*U_DIM), x + (1 + (i + 2)*U_DIM), temp_u_list_data);
                std::copy(x + (1 + (N + 1)*U_DIM + i*X_DIM), x + (1 + (N + 1)*U_DIM + (i + 2)*X_DIM), temp_x_list_data);
                temp_u_list_data[k + U_DIM] += eps;
                temp_cons = (get_collocation_constraint(N, temp_u_list_data, temp_x_list_data, t_f) - coll_cons)*(1/eps);
                for(int kk=0;kk<X_DIM;kk++) grad[n*(i*X_DIM + kk) + 1 + (i + 1)*U_DIM + k] = temp_cons.data[kk];
            }

            for(int k=0;k<X_DIM;k++){
                std::copy(x + (1 + i*U_DIM), x + (1 + (i + 2)*U_DIM), temp_u_list_data);
                std::copy(x + (1 + (N + 1)*U_DIM + i*X_DIM), x + (1 + (N + 1)*U_DIM + (i + 2)*X_DIM), temp_x_list_data);
                temp_x_list_data[k] += eps;
                temp_cons = (get_collocation_constraint(N, temp_u_list_data, temp_x_list_data, t_f) - coll_cons)*(1/eps);
                for(int kk=0;kk<X_DIM;kk++) grad[n*(i*X_DIM + kk) + 1 + (N + 1)*U_DIM + i*X_DIM + k] = temp_cons.data[kk];

                std::copy(x + (1 + i*U_DIM), x + (1 + (i + 2)*U_DIM), temp_u_list_data);
                std::copy(x + (1 + (N + 1)*U_DIM + i*X_DIM), x + (1 + (N + 1)*U_DIM + (i + 2)*X_DIM), temp_x_list_data);
                temp_x_list_data[k + X_DIM] += eps;
                temp_cons = (get_collocation_constraint(N, temp_u_list_data, temp_x_list_data, t_f) - coll_cons)*(1/eps);
                for(int kk=0;kk<X_DIM;kk++) grad[n*(i*X_DIM + kk) + 1 + (N + 1)*U_DIM + (i + 1)*X_DIM + k] = temp_cons.data[kk];
            }
        }
    }

    result[X_DIM*N] = x[1 + (N + 1)*U_DIM]; //x_0
    result[X_DIM*N + 1] = x[1 + (N + 1)*U_DIM + 1]; //x_0_dot
    result[X_DIM*N + 2] = x[1 + (N + 1)*U_DIM + N*X_DIM] - 10.0; //x_N
    result[X_DIM*N + 3] = x[1 + (N + 1)*U_DIM + N*X_DIM + 1]; //x_N_dot

    if (grad){
        /*
        for(i=0;i<4;i++){
            for(j=0;j<n;j++){
                grad[n*(2*N + i) + j] = 0.0;
            }
        }
        */
        grad[n*(2*N) + N + 2] = 1.0;
        grad[n*(2*N + 1) + N + 3] = 1.0;
        grad[n*(2*N + 2) + 3*N + 2] = 1.0;
        grad[n*(2*N + 3) + 3*N + 3] = 1.0;
    }

    delete [] temp_u_list_data;
    delete [] temp_x_list_data;
    return;
}


int main(int argc, char* argv[]){
    clock_t tStart = clock();
    int N = 100;
    double* x = new double[1 + (N + 1)*(X_DIM + U_DIM)];
    double* cons_tol = new double[X_DIM*(N + 2)];
    double min_objective_value;
    nlopt_opt opt;

    opt = nlopt_create(NLOPT_LD_SLSQP, 1 + (N + 1)*(X_DIM + U_DIM));
    nlopt_set_min_objective(opt, objective, NULL);
    nlopt_set_lower_bound(opt, 0, 0.0);
    for(int i=0;i<1 + (N + 1)*(X_DIM + U_DIM);i++){
        if(i < N + 1){
            nlopt_set_lower_bound(opt, i+1, -1.0);
            nlopt_set_upper_bound(opt, i+1, 1.0);
        }
        if(i < X_DIM*(N + 2)){
            cons_tol[i] = 1e-6;
        }
        x[i] = 0.0;
    }
    x[0] = 10.0;
    nlopt_add_equality_mconstraint(opt, X_DIM*(N + 2), equality_constraints, NULL, cons_tol);
    nlopt_set_ftol_abs(opt, 1e-3);
    nlopt_set_maxeval(opt, 100);
    
    if (nlopt_optimize(opt, x, &min_objective_value) < 0) {
        printf("nlopt failed!\n");
    }
    else {
        printf("found minimum = %0.10g\n", min_objective_value);
    }

    std::cout<<"Time taken: "<<(double)(clock() - tStart)/CLOCKS_PER_SEC<<std::endl;

    mat x_list(&x[1 + (N + 1)*U_DIM], N+1, X_DIM);
    mat u_list(&x[1], N+1, U_DIM);
    std::cout<<"x value : "<<std::endl;
    x_list.print();
    std::cout<<"u value : "<<std::endl;
    u_list.print();

    std::cout<<"collocation constraints : "<<std::endl;
    get_collocation_constraints(N, x).print();
    
    //std::cout<<"collocation constraint : "<<std::endl;
    //for(int i=0;i<N;i++) get_collocation_constraint(N, &x[1 + i], &x[N + 2 + 2*i], x[0]).print();

    nlopt_destroy(opt);
    delete [] x;
    delete [] cons_tol;
    return 0;
}