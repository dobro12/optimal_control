##### add python path #####
import sys
import os
PATH = os.getcwd()
for dir_idx, dir_name in enumerate(PATH.split('/')):
    if 'optimal_control' in dir_name.lower():
        PATH = '/'.join(PATH.split('/')[:(dir_idx+1)])
        break
if not PATH in sys.path:
    sys.path.append(PATH)
###########################
from env.quadruped.env import Env

from DCM.quadruped.trajectory_planner import TrajectoryPlanner
from DCM.quadruped.foot_step_planner import FootStepPlanner
from DCM.quadruped.gait_scheduler import GaitScheduler

from copy import deepcopy
from ctypes import cdll
import numpy as np
import ctypes
import time

def main():
    env = Env(enable_draw=False, base_fix=False)
    num_leg = env.num_leg
    time_step = env.time_step
    base_inertia = deepcopy(env.model.inertia)
    base_mass = env.model.mass
    friction_coef = env.friction_coef
    max_force = env.model.max_force

    plan_time_step = 0.05
    plan_time_horizon = 10

    gs_args = dict()
    gs_args['num_leg'] = num_leg
    gs_args['gait_period'] = 1.0
    gs_args['is_visualize'] = False
    gait_scheduler = GaitScheduler(gs_args)

    tp_args = dict()
    tp_args['time_step'] = plan_time_step
    tp_args['time_horizon'] = plan_time_horizon
    tp_args['max_accel'] = 0.1
    tp_args['max_ang_accel'] = 0.1
    trajectory_planner = TrajectoryPlanner(tp_args)

    fsp_args = dict()
    fsp_args['num_leg'] = num_leg
    fsp_args['time_horizon'] = plan_time_horizon
    fsp_args['abduct_org_list'] = env.model.abduct_org_list
    foot_step_planner = FootStepPlanner(fsp_args)

    env.reset()
    gait_scheduler.reset()
    trajectory_planner.reset()
    foot_step_planner.reset()

    vel_cmd = np.zeros(3)
    vel_cmd[0] = 0.0
    vel_cmd[1] = 0.0
    ang_vel_cmd = np.zeros(3)
    ang_vel_cmd[2] = 0.0
    height_cmd = 0.3

    init_state = np.concatenate([env.model.com_pos, env.model.base_rpy, env.model.base_vel, env.model.base_ang_vel])
    init_foot_pos_list = deepcopy(env.model.foot_pos_list)

    #target_trajectory_list = trajectory_planner.get_target_trajectory_list(vel_cmd, ang_vel_cmd, height_cmd, init_state)
    target_trajectory_list = np.zeros((plan_time_horizon, 12))
    target_trajectory_list[:,2] = height_cmd
    contact_list, step_info_list = gait_scheduler.get_contact_list(plan_time_step, plan_time_horizon)
    target_foot_step_list = foot_step_planner.get_target_foot_step_list(init_foot_pos_list, step_info_list, target_trajectory_list, vel_cmd, ang_vel_cmd, height_cmd)

    state_dim = 12
    force_dim = 12
    x_dim = state_dim + force_dim
    R_mat = np.eye(force_dim)*0.01
    Q_mat = np.eye(state_dim)
    Qf_mat = np.eye(state_dim)*100.0

    init_force_list = np.zeros((plan_time_horizon, force_dim))
    init_x_list = np.concatenate([target_trajectory_list.ravel(), init_force_list.ravel()])
    init_lambda_list = np.ones(plan_time_horizon*state_dim)
    init_mu_list = np.ones(plan_time_horizon*force_dim)
    learning_rate = 0.001

    # optimization data
    cpp_x_list = ctype_arr_convert(init_x_list)
    cpp_lambda_list = ctype_arr_convert(init_lambda_list)
    cpp_mu_list = ctype_arr_convert(init_mu_list)
    cpp_learning_rate = ctypes.c_double(learning_rate)
    # planning data
    cpp_time_horizon = ctypes.c_int(plan_time_horizon)
    cpp_time_step = ctypes.c_double(plan_time_step)
    # f args
    cpp_Q_mat = ctype_arr_convert(Q_mat)
    cpp_Qf_mat = ctype_arr_convert(Qf_mat)
    cpp_R_mat = ctype_arr_convert(R_mat)
    cpp_target_trajectory_list = ctype_arr_convert(target_trajectory_list)
    # c_eq args
    cpp_init_state = ctype_arr_convert(init_state)
    cpp_foot_step_list = ctype_arr_convert(target_foot_step_list)
    cpp_contact_list = ctype_arr_convert(contact_list)
    cpp_base_inertia = ctype_arr_convert(base_inertia)
    cpp_base_mass = ctypes.c_double(base_mass)
    # c_ineq args
    cpp_friction_coef = ctypes.c_double(friction_coef)

    CPP_LIB = cdll.LoadLibrary('{}/main.so'.format(os.getcwd()))
    CPP_LIB.solve.argtypes = (
        # optimization data
        ctypes.POINTER(ctypes.c_double*(plan_time_horizon*x_dim)), 
        ctypes.POINTER(ctypes.c_double*(plan_time_horizon*state_dim)), 
        ctypes.POINTER(ctypes.c_double*(plan_time_horizon*force_dim)), 
        ctypes.c_double, 
        # planning data
        ctypes.c_int, ctypes.c_double, 
        # f args
        ctypes.POINTER(ctypes.c_double*(state_dim*state_dim)), ctypes.POINTER(ctypes.c_double*(state_dim*state_dim)),
        ctypes.POINTER(ctypes.c_double*(force_dim*force_dim)), ctypes.POINTER(ctypes.c_double*(plan_time_horizon*state_dim)),
        # c_eq args
        ctypes.POINTER(ctypes.c_double*state_dim), ctypes.POINTER(ctypes.c_double*(plan_time_horizon*num_leg*3)),
        ctypes.POINTER(ctypes.c_double*(plan_time_horizon*num_leg)), ctypes.POINTER(ctypes.c_double*(3*3)), ctypes.c_double,
        # c_ineq args
        ctypes.c_double,
        )

    start_t = time.time()                                
    CPP_LIB.solve(
            cpp_x_list, cpp_lambda_list, cpp_mu_list, cpp_learning_rate, \
            cpp_time_horizon, cpp_time_step, \
            cpp_Q_mat, cpp_Qf_mat, cpp_R_mat, cpp_target_trajectory_list, \
            cpp_init_state, cpp_foot_step_list, cpp_contact_list, cpp_base_inertia, cpp_base_mass, \
            cpp_friction_coef,
        )
    print("elapsed time : {:.5f} s".format(time.time() - start_t))

    x_list = np.array(cpp_x_list)
    #print(x_list)


def ctype_arr_convert(arr):
    arr = arr.ravel()
    return (ctypes.c_double * len(arr))(*arr)


if __name__ == "__main__":
    main()