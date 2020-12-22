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

#from agent import Agent
from c_module.agent import Agent
from env.quadruped.env import Env
import utils

import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import pybullet
import time

'''
# 되는 것들:
    target_vel = np.array([0.2, 0.0, 0.0])
    target_ang_vel = np.array([0.0, 0.0, 0.0])
    gait_type = 'jump' #[trot, bound, jump, walk]
    contact_time = 0.3
    swing_time = 0.15

    target_vel = np.array([0.6, 0.0, 0.0])
    target_ang_vel = np.array([0.0, 0.0, 0.0])
    gait_type = 'trot' #[trot, bound, jump, walk]
    contact_time = 0.06
    swing_time = 0.2

    target_vel = np.array([0.3, 0.0, 0.0])
    target_ang_vel = np.array([0.0, 0.0, 0.5])
    gait_type = 'trot' #[trot, bound, jump, walk]
    contact_time = 0.06
    swing_time = 0.2

    target_vel = np.array([0.2, 0.0, 0.0])
    target_ang_vel = np.array([0.0, 0.0, 0.0])
    gait_type = 'walk' #[trot, bound, jump, walk]
    contact_time = 0.06
    swing_time = 0.15
'''


def main():
    env = Env(enable_draw=True, base_fix=False)
    agent = Agent(env)
    log_file_name = "./log.mp4"
    is_logging = False
    #is_logging = True
    duration = 10
    num_leg = 4
    ground_height = 0

    # set parameter
    initial_time = 0.5
    delta_time = 0.02
    time_horizon = 10
    target_vel = np.array([0.2, 0.0, 0.0])
    target_ang_vel = np.array([0.0, 0.0, 0.0])
    acc_time = 0.5
    target_height = 0.28
    initial_length = int(initial_time/delta_time)
    total_length = int(duration/delta_time) + 1

    # gait parameter
    gait_type = 'walk' #[trot, bound, jump, walk]
    contact_time = 0.06
    swing_time = 0.15

    # set initial x
    com_pos = np.array([0.0, 0, target_height])
    rpy = np.zeros(3)
    com_vel = np.zeros(3)
    base_ang_vel = np.zeros(3)
    init_x = np.concatenate([com_pos, rpy, com_vel, base_ang_vel]).reshape((-1, 1))

    # x initialize
    target_x = deepcopy(init_x)
    target_x[6:9, 0] = target_vel
    target_x[9:12, 0] = target_ang_vel
    total_target_x_list = [init_x]*initial_length
    init_vel = init_x[6:9,0]
    init_ang_vel = init_x[9:12,0]
    target_acc = (target_vel - init_vel)/(acc_time + 1e-10)
    target_ang_acc = (target_ang_vel - init_ang_vel)/(acc_time + 1e-10)
    for t_idx in range(total_length):
        target_pos = deepcopy(target_x[0:3,0])
        target_rpy = deepcopy(target_x[3:6,0])
        temp_target_vel = init_vel + target_acc*(t_idx+1)*delta_time if (t_idx+1)*delta_time < acc_time else target_vel
        temp_target_ang_vel = init_ang_vel + target_ang_acc*(t_idx+1)*delta_time if (t_idx+1)*delta_time < acc_time else target_ang_vel
        temp_target_vel = np.matmul(utils.rpy_rot(target_rpy), temp_target_vel).ravel()
        delta_x = np.concatenate([temp_target_vel, temp_target_ang_vel, np.zeros(6)]).reshape((-1, 1))
        target_x = deepcopy(target_x + delta_x*delta_time)
        total_target_x_list.append(target_x)
    total_target_x_list = np.array(total_target_x_list)

    # u initialize
    target_u = np.array([0.0, 0.0, env.model.mass*0.25*9.8]*num_leg).reshape((num_leg*3, 1))
    init_u_list = np.array([target_u for i in range(time_horizon)])
    target_u_list = np.array([target_u for i in range(time_horizon)])

    # delta time list initialize
    delta_time_list = np.array([delta_time]*time_horizon)

    # contact phi list initialize
    contact_length = int(contact_time/delta_time)
    swing_length = int(swing_time/delta_time)
    if gait_type == 'trot':
        temp_contact_phi_list = [[0,1,1,0]]*swing_length+[[1,1,1,1]]*contact_length+[[1,0,0,1]]*swing_length+[[1,1,1,1]]*contact_length
    elif gait_type == 'bound':
        temp_contact_phi_list = [[0,0,1,1]]*swing_length+[[1,1,1,1]]*contact_length+[[1,1,0,0]]*swing_length+[[1,1,1,1]]*contact_length
    elif gait_type == 'jump':
        temp_contact_phi_list = [[0,0,0,0]]*swing_length+[[1,1,1,1]]*contact_length+[[0,0,0,0]]*swing_length+[[1,1,1,1]]*contact_length
    elif gait_type == 'walk':
        temp_contact_phi_list = [[0,1,1,1]]*swing_length+[[1,1,0,1]]*swing_length+[[1,0,1,1]]*swing_length+[[1,1,1,0]]*swing_length
    total_contact_phi_list = [[1,1,1,1]]*initial_length + temp_contact_phi_list*int(total_length/len(temp_contact_phi_list))
    pre_contact_phi_list = np.zeros(num_leg)

    # set swing trajectory parameter
    swing_trajectory_list = [[np.zeros(3), np.zeros(3), 0, 0] for leg_idx in range(num_leg)]

    # initialize foot pos list
    swing_time_list = np.zeros(num_leg)
    swing_start_list = np.zeros(num_leg)
    init_foot_pos_list = [np.zeros(3) for i in range(num_leg)]
    total_foot_pos_list = []
    for t_idx, contact_phi_list in enumerate(total_contact_phi_list):
        target_x = total_target_x_list[t_idx]
        com_pos = target_x[:3,0]
        rpy = target_x[3:6,0]
        for leg_idx in range(num_leg):
            if contact_phi_list[leg_idx] == 0:
                if pre_contact_phi_list[leg_idx] == 1:
                    swing_start_list[leg_idx] = t_idx
                swing_time_list[leg_idx] += delta_time
            if pre_contact_phi_list[leg_idx] == 0 and contact_phi_list[leg_idx] == 1:
                target_vel = total_target_x_list[int(swing_start_list[leg_idx]), 6:9, 0]
                init_foot_pos_list[leg_idx] = com_pos + np.matmul(utils.rpy_rot(rpy), env.model.abduct_org_list[leg_idx] + env.model.thigh_org_list[leg_idx]).ravel() + swing_time_list[leg_idx]*0.5*target_vel
                init_foot_pos_list[leg_idx][2] = ground_height
                swing_time_list[leg_idx] = 0
        total_foot_pos_list.append(deepcopy(init_foot_pos_list))
        pre_contact_phi_list = deepcopy(contact_phi_list)
    pre_contact_phi_list = np.zeros(num_leg)

    state = env.reset()
    # to start at bottom
    for temp_t_idx in range(20):
        state = env.step(np.zeros(num_leg*3), np.ones(num_leg), [])

    if is_logging:
        pybullet_logging_id = env.start_mp4_logging(log_file_name)
    last_t = 0
    last_update_t = 0
    while env.elapsed_t < duration:
        if env.elapsed_t == 0 or env.elapsed_t - last_t >= delta_time:
            last_t = env.elapsed_t

            # get init_x
            com_pos = env.model.com_pos
            rpy = env.model.base_rpy
            com_vel = env.model.base_vel
            base_ang_vel = np.matmul(env.model.base_rot.T, env.model.base_ang_vel)
            init_x = np.concatenate([com_pos, rpy, com_vel, base_ang_vel])
            init_x = init_x.reshape((-1, 1))

            # get target_x_list
            for rpy_idx in [3, 4, 5]:
                if init_x[rpy_idx, 0] - total_target_x_list[0, rpy_idx, 0] > np.pi:
                    total_target_x_list[:, rpy_idx, 0] += 2*np.pi
                elif init_x[rpy_idx, 0] - total_target_x_list[0, rpy_idx, 0] < -np.pi:
                    total_target_x_list[:, rpy_idx, 0] -= 2*np.pi
            target_x_list = total_target_x_list[:time_horizon+1]

            # get contact_phi_list
            contact_phi_list = total_contact_phi_list[:time_horizon+1]

            # get foot pos list
            for leg_idx in range(num_leg):
                if pre_contact_phi_list[leg_idx] == 0 and contact_phi_list[0][leg_idx] == 1:
                    init_foot_pos_list[leg_idx] = deepcopy(env.model.foot_pos_list[leg_idx])
                    init_foot_pos_list[leg_idx][2] = ground_height
                    temp_idx = 0
                    end_flag = 0
                    while end_flag != 2:
                        total_foot_pos_list[temp_idx][leg_idx] = deepcopy(init_foot_pos_list[leg_idx])
                        temp_idx += 1
                        if total_contact_phi_list[temp_idx][leg_idx] == 0 and end_flag == 0:
                            end_flag += 1
                        elif total_contact_phi_list[temp_idx][leg_idx] == 1 and end_flag == 1:
                            end_flag += 1
            foot_pos_list = np.array(total_foot_pos_list[:time_horizon+1])

            # get swing leg trajectory parameter
            target_vel = target_x_list[0][6:9,0]
            base_com_vel = np.matmul(env.model.base_rot.T, com_vel)
            base_target_vel = np.matmul(env.model.base_rot.T, target_vel)
            for leg_idx in range(num_leg):
                if contact_phi_list[0][leg_idx] == 0 and pre_contact_phi_list[leg_idx] == 1:
                    #z_c = -0.2
                    z_c = 0.1 - target_x_list[0][2,0]
                    init_base_pos = deepcopy(env.model.base_foot_pos_list[leg_idx])
                    #### cheating ####
                    init_base_pos[2] += 1e-2
                    ##################
                    Ts = 0
                    temp_idx = 0
                    while True:
                        if total_contact_phi_list[temp_idx][leg_idx] == 1:
                            break
                        temp_idx += 1
                        Ts += delta_time
                    ref_base_pos = env.model.base_com_pos + env.model.abduct_org_list[leg_idx] + env.model.thigh_org_list[leg_idx] + Ts*0.5*base_target_vel + np.sqrt(target_x[2,0]/9.8)*(base_com_vel - base_target_vel)
                    ref_base_pos[2] = z_c
                    start_t = env.elapsed_t
                    end_t = start_t + Ts
                    swing_trajectory_list[leg_idx] = [init_base_pos, ref_base_pos, start_t, end_t]

            # update parameter
            if env.elapsed_t == 0 or env.elapsed_t - last_update_t >= delta_time:
                last_update_t = env.elapsed_t
                total_target_x_list = total_target_x_list[1:]
                total_contact_phi_list = total_contact_phi_list[1:]
                total_foot_pos_list = total_foot_pos_list[1:]
                pre_contact_phi_list = deepcopy(contact_phi_list[0])

            # get action
            action, u_list  = agent.get_action(init_x, init_u_list, delta_time_list, foot_pos_list, contact_phi_list, target_x_list, target_u_list)
            init_u_list = deepcopy(u_list)

        state = env.step(action, contact_phi_list[0], swing_trajectory_list)

    if is_logging:
        env.stop_mp4_logging(pybullet_logging_id)
        utils.lower_frame(log_file_name, skip_rate=2, duration=duration)
    print("[main] end!")

if __name__ == "__main__":
    main()
