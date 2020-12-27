from scipy.spatial.transform import Rotation
from copy import deepcopy
import numpy as np
import time

class TrajectoryPlanner:
    def __init__(self, args):
        self.time_step = args['time_step']
        self.time_horizon = args['time_horizon']
        self.max_accel = args['max_accel']
        self.max_ang_accel = args['max_ang_accel']


    def reset(self):
        pass

    def get_target_trajectory_list(self, vel_cmd, ang_vel_cmd, height_cmd, init_state):
        assert len(vel_cmd) == 3
        assert len(ang_vel_cmd) == 3
        assert len(init_state) == 12
        # for 2d motion
        vel_cmd[2] = 0.0
        ang_vel_cmd[:2] = 0.0

        # state : [pos, rpy, vel, ang_vel]
        pos_com = init_state[:3]
        rpy_base = init_state[3:6]
        vel_base = init_state[6:9]
        ang_vel_base = init_state[9:]

        pos_com[2] = height_cmd
        vel_base[2] = 0.0
        rpy_base[:2] = 0.0
        ang_vel_base[:2] = 0.0

        vel_reach_t = np.linalg.norm(vel_cmd - vel_base)/self.max_accel
        ang_vel_reach_t = np.linalg.norm(ang_vel_cmd - ang_vel_base)/self.max_ang_accel
        target_accel = self.max_accel*(vel_cmd - vel_base)/(np.linalg.norm(vel_cmd - vel_base) + 1e-10)
        target_ang_accel = self.max_ang_accel*(ang_vel_cmd - ang_vel_base)/(np.linalg.norm(ang_vel_cmd - ang_vel_base) + 1e-10)

        target_base_vel = deepcopy(vel_base)
        target_ang_vel = deepcopy(ang_vel_base)
        target_pos = deepcopy(pos_com)
        target_rpy = deepcopy(rpy_base)

        target_trajectory_list = []
        for i in range(self.time_horizon):
            elapsed_t = self.time_step*(i + 1)
            if elapsed_t > vel_reach_t:
                target_base_vel = vel_cmd
            else:
                target_base_vel += self.time_step*target_accel
            if elapsed_t > ang_vel_reach_t:
                target_ang_vel = ang_vel_cmd
            else:
                target_ang_vel += self.time_step*target_ang_accel

            target_rpy += target_ang_vel*self.time_step
            r = Rotation.from_rotvec(target_rpy)
            target_vel = np.matmul(r.as_matrix(), target_base_vel)
            target_pos += target_vel*self.time_step

            target_trajectory = np.concatenate([target_pos, target_rpy, target_vel, target_ang_vel])
            target_trajectory_list.append(target_trajectory)

        target_trajectory_list = np.array(target_trajectory_list)
        return target_trajectory_list


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
    from env.quadruped.env import Env

    env = Env(enable_draw=False, base_fix=False)
    env.reset()

    tp_args = dict()
    tp_args['time_step'] = 0.05
    tp_args['time_horizon'] = 20
    tp_args['max_accel'] = 0.1
    tp_args['max_ang_accel'] = 0.1
    trajectory_planner = TrajectoryPlanner(tp_args)

    init_state = np.concatenate([env.model.com_pos, env.model.base_rpy, env.model.base_vel, env.model.base_ang_vel])

    vel_cmd = np.zeros(3)
    vel_cmd[0] = 0.2
    vel_cmd[1] = 0.0
    ang_vel_cmd = np.zeros(3)
    ang_vel_cmd[2] = 0.0
    height_cmd = 0.35

    target_trajectory_list = trajectory_planner.get_target_trajectory_list(vel_cmd, ang_vel_cmd, height_cmd, init_state)
    print(target_trajectory_list)