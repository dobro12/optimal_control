from scipy.spatial.transform import Rotation
from copy import deepcopy
import numpy as np

class FootStepPlanner:
    def __init__(self, args):
        self.time_horizon = args['time_horizon']
        self.num_leg = args['num_leg']
        self.abduct_org_list = args['abduct_org_list']


    def reset(self):
        self.step_list = np.zeros((self.num_leg, 3)) 

    def get_target_foot_pos_list(self):
        return deepcopy(self.step_list)
    
    def get_target_foot_step_list(self, init_foot_pos_list, step_info_list, target_trajectory_list, \
                                    vel_cmd, ang_vel_cmd, height_cmd):
        curr_foot_pos_list = deepcopy(init_foot_pos_list)
        target_foot_step_list = np.zeros((self.time_horizon, self.num_leg, 3))

        for leg_idx in range(self.num_leg):
            step_info = step_info_list[leg_idx]
            curr_foot_pos_list[leg_idx][2] = 0.0
            if len(step_info[1]) == 0:
                target_foot_step_list[:,leg_idx, :] = curr_foot_pos_list[leg_idx]
                continue

            is_swinging, temp_step_info_list = step_info

            pre_idx = 0
            pre_foot_pos = curr_foot_pos_list[leg_idx]
            for step_idx in range(len(temp_step_info_list)):
                stand_period, idx1, idx2 = temp_step_info_list[step_idx]
                target_foot_step_list[pre_idx:idx1, leg_idx, :] = pre_foot_pos
                pre_idx = idx2
                if step_idx == 0 and is_swinging:
                    pre_foot_pos = self.step_list[leg_idx]
                else:
                    state = target_trajectory_list[idx2 - 1]
                    pos_com = state[:3]
                    rpy_base = state[3:6]
                    vel_base = state[6:9]
                    ang_vel_base = state[9:]

                    r = Rotation.from_euler('z', rpy_base[2], degrees=False)
                    pre_foot_pos = pos_com + np.matmul(r.as_matrix(), self.abduct_org_list[leg_idx])
                    pre_foot_pos += stand_period*0.5*vel_base + 0.03*(vel_base - vel_cmd)
                    pre_foot_pos += 0.5*np.sqrt(height_cmd/9.8)*np.cross(vel_base, ang_vel_cmd)
                    pre_foot_pos[2] = 0.0

                if step_idx == 0:
                    self.step_list[leg_idx, :] = pre_foot_pos

            target_foot_step_list[pre_idx:self.time_horizon, leg_idx, :] = pre_foot_pos
        return target_foot_step_list


if __name__ == "__main__":
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

    from trajectory_planner import TrajectoryPlanner
    from gait_scheduler import GaitScheduler
    from env.quadruped.env import Env

    env = Env(enable_draw=False, base_fix=False)
    num_leg = env.num_leg
    time_step = env.time_step

    plan_time_step = 0.05
    plan_time_horizon = 20

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

    gait_list = np.zeros((num_leg, 2)) # FR, FL, RR, RL
    gait_list[0,0] = 0.5 # offset
    gait_list[0,1] = 0.5 # duration
    gait_list[2,0] = 0.0 # offset
    gait_list[2,1] = 0.5 # duration
    gait_scheduler.gait_list = deepcopy(gait_list)
    gait_scheduler.update()

    vel_cmd = np.zeros(3)
    vel_cmd[0] = 0.2
    vel_cmd[1] = 0.0
    ang_vel_cmd = np.zeros(3)
    ang_vel_cmd[2] = 0.0
    height_cmd = 0.35

    init_state = np.concatenate([env.model.com_pos, env.model.base_rpy, env.model.base_vel, env.model.base_ang_vel])
    init_foot_pos_list = deepcopy(env.model.foot_pos_list)

    target_trajectory_list = trajectory_planner.get_target_trajectory_list(vel_cmd, ang_vel_cmd, height_cmd, init_state)
    contact_list, step_info_list = gait_scheduler.get_contact_list(plan_time_step, plan_time_horizon)
    target_foot_step_list = foot_step_planner.get_target_foot_step_list(init_foot_pos_list, step_info_list, target_trajectory_list, \
                                                                        vel_cmd, ang_vel_cmd, height_cmd)

    print(foot_step_planner.get_target_foot_pos_list())

    gait_scheduler.step(0.05)
    contact_list, step_info_list = gait_scheduler.get_contact_list(trajectory_planner.time_step, trajectory_planner.time_horizon)
    print(contact_list)
    print(step_info_list)
