from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from scipy.optimize import Bounds
from copy import deepcopy
import numpy as np
import time

class NLPSolver:
    def __init__(self, args):
        self.R_mat = args['R_mat'] #np.eye(self.force_dim)*0.01
        self.Q_mat = args['Q_mat'] #np.eye(self.state_dim)
        self.Qf_mat = args['Qf_mat'] #np.eye(self.state_dim)*100.0
        self.time_horizon = args['time_horizon']
        self.time_step = args['time_step']
        self.base_inertia = args['base_inertia']
        self.base_mass = args['base_mass']
        self.friction_coef = args['friction_coef']
        self.max_force = args['max_force']


    def get_solution(self, init_state, target_trajectory_list, target_foot_step_list, contact_list):
        init_x_list = np.concatenate([target_trajectory_list.ravel(), self.init_force_list.ravel()])

        eq_cons = {
            'type':'eq',
            'fun' :eq_cons_func,
            'jac' :eq_cons_jacobian,
            'args':[[init_state, target_foot_step_list, contact_list, self.time_step, self.base_inertia, self.base_mass]],
            }

        ineq_cons = {
            'type':'ineq',
            'fun' :ineq_cons_func,
            'jac' :ineq_cons_jacobian,
            'args':[[contact_list, self.friction_coef]],
            }

        lowers = np.array([-np.inf]*int(len(init_x_list)/2) + [-self.max_force]*int(len(init_x_list)/2))
        uppers = np.array([np.inf]*int(len(init_x_list)/2) + [self.max_force]*int(len(init_x_list)/2))
        bounds = Bounds(lowers, uppers)

        res = minimize(obj_func, init_x_list, args=[target_trajectory_list, self.Q_mat, self.Qf_mat, self.R_mat, self.time_step], \
                    method="SLSQP", jac=obj_jacobian, bounds=bounds, constraints=[eq_cons, ineq_cons], \
                    options={'ftol':1e-5, 'disp':False, 'maxiter':30}) #, 'eps':1e-10})

        state_list = res.x[:12*self.time_horizon].reshape((self.time_horizon, 12))
        force_list = res.x[12*self.time_horizon:].reshape((self.time_horizon, 12))

        self.init_force_list[:-1, :] = deepcopy(force_list[1:, :])
        self.init_force_list[-1, :] = deepcopy(force_list[-1, :])

        return state_list, force_list

    def reset(self):
        self.init_force_list = np.zeros((self.time_horizon, 12))

    def get_approx_eq_cons_jacobian(self, init_state, target_trajectory_list, target_foot_step_list, contact_list):
        init_x_list = np.concatenate([target_trajectory_list.ravel(), self.init_force_list.ravel()])
        args = [init_state, target_foot_step_list, contact_list, self.time_step, self.base_inertia, self.base_mass]

        approx_jacobian = np.zeros((12*self.time_horizon, 24*self.time_horizon))
        eps = 1e-10
        func_value = eq_cons_func(init_x_list , args)
        for idx in range(len(init_x_list)):
            temp_x_list = np.zeros_like(init_x_list)
            temp_x_list[idx] = eps
            approx_jacobian[:, idx] = (eq_cons_func(init_x_list + temp_x_list, args) - func_value)/eps


def get_cross_product_matrix(vector):
    a, b, c = vector
    mat = np.array([[0, -c, b],
                   [c, 0, -a],
                   [-b, a, 0]])
    return mat

def obj_func(x, args):
    target_trajectory_list = args[0]
    Q_mat = args[1] # 12 x 12
    Qf_mat = args[2] # 12 x 12
    R_mat = args[3] # 12 x 12
    delta_t = args[4]

    time_horizon = int(len(x)/24)
    state_list = x[:int(len(x)/2)].reshape((time_horizon, 12))
    force_list = x[int(len(x)/2):].reshape((time_horizon, 12))
    
    difference_list = state_list - target_trajectory_list # N x 12
    f = 0.5*np.sum(np.matmul(difference_list[-1, :], Qf_mat)*difference_list[-1, :])
    f += 0.5*delta_t*np.sum(np.matmul(difference_list[:-1, :], Q_mat)*difference_list[:-1, :])
    f += 0.5*delta_t*np.sum(np.matmul(force_list, R_mat)*force_list)
    return f

def obj_jacobian(x, args):
    target_trajectory_list = args[0]
    Q_mat = args[1] # 12 x 12
    Qf_mat = args[2] # 12 x 12
    R_mat = args[3] # 12 x 12
    delta_t = args[4]

    time_horizon = int(len(x)/24)
    state_list = x[:int(len(x)/2)].reshape((time_horizon, 12))
    force_list = x[int(len(x)/2):].reshape((time_horizon, 12))

    jacobian = np.zeros(len(x))

    difference_list = state_list - target_trajectory_list # N x 12
    jacobian[:12*(time_horizon-1)] = delta_t*np.matmul(difference_list[:-1, :], Q_mat).ravel()
    jacobian[12*(time_horizon-1):12*time_horizon] = np.matmul(difference_list[-1, :], Qf_mat).ravel()
    jacobian[12*time_horizon:] = delta_t*np.matmul(force_list, R_mat).ravel()
    return jacobian

def eq_cons_func(x, args):
    init_state = args[0]
    target_foot_step_list = args[1]
    contact_list = args[2]
    delta_t = args[3]
    base_inertia = args[4]
    base_mass = args[5]
    num_leg = 4

    time_horizon = int(len(x)/24)
    state_list = x[:int(len(x)/2)].reshape((time_horizon, 12))
    force_list = x[int(len(x)/2):].reshape((time_horizon, 12))
    
    eq_cons = []
    state = init_state
    gravity = np.zeros(12)
    gravity[6:9] = delta_t*np.array([0, 0, -9.8])
    for time_idx in range(time_horizon):
        next_state = state_list[time_idx]
        force = (force_list[time_idx].reshape((4,3))*contact_list[time_idx].reshape(4,1)).ravel()

        yaw = state[5]
        pos_com = state[:3]
        yaw_mat = Rotation.from_euler('z', yaw, degrees=False).as_matrix()
        global_inertia = np.matmul(yaw_mat, np.matmul(base_inertia, yaw_mat.T))
        inv_global_inertia = np.linalg.inv(global_inertia)
        foot_step_list = target_foot_step_list[time_idx]

        A_mat = np.eye(12)
        A_mat[0:3, 6:9] = delta_t*np.eye(3)
        A_mat[3:6, 9:12] = delta_t*yaw_mat.T #delta_t*yaw_mat

        B_mat = np.zeros((12, 12))
        for leg_idx in range(num_leg):
            foot_step = foot_step_list[leg_idx] - pos_com
            B_mat[6:9, leg_idx*3:(leg_idx+1)*3] = np.eye(3)*(delta_t/base_mass)
            B_mat[9:12, leg_idx*3:(leg_idx+1)*3] = delta_t*np.matmul(inv_global_inertia, get_cross_product_matrix(foot_step))
        
        eq_cons.append((np.matmul(A_mat, state) + np.matmul(B_mat, force) + gravity - next_state).ravel())
        state = next_state

    eq_cons = np.concatenate(eq_cons)
    return eq_cons

def eq_cons_jacobian(x, args):
    init_state = args[0]
    target_foot_step_list = args[1]
    contact_list = args[2]
    delta_t = args[3]
    base_inertia = args[4]
    base_mass = args[5]
    num_leg = 4

    time_horizon = int(len(x)/24)
    state_list = x[:int(len(x)/2)].reshape((time_horizon, 12))
    force_list = x[int(len(x)/2):].reshape((time_horizon, 12))
    
    jacobian = np.zeros((12*time_horizon, len(x)))
    state = init_state
    for time_idx in range(time_horizon):        
        next_state = state_list[time_idx]
        force = (force_list[time_idx].reshape((4,3))*contact_list[time_idx].reshape(4,1)).ravel()

        yaw = state[5]
        pos_com = state[:3]
        ang_vel_base = state[9:12]
        yaw_mat = Rotation.from_euler('z', yaw, degrees=False).as_matrix()
        global_inertia = np.matmul(yaw_mat, np.matmul(base_inertia, yaw_mat.T))
        inv_global_inertia = np.linalg.inv(global_inertia)
        inv_base_inertia = np.linalg.inv(base_inertia)
        foot_step_list = target_foot_step_list[time_idx]

        A_mat = np.eye(12)
        A_mat[0:3, 6:9] = delta_t*np.eye(3)
        A_mat[3:6, 9:12] = delta_t*yaw_mat.T #delta_t*yaw_mat

        B_mat = np.zeros((12, 12))
        for leg_idx in range(num_leg):
            foot_step = foot_step_list[leg_idx] - pos_com
            B_mat[6:9, leg_idx*3:(leg_idx+1)*3] = np.eye(3)*(delta_t/base_mass)
            B_mat[9:12, leg_idx*3:(leg_idx+1)*3] = delta_t*np.matmul(inv_global_inertia, get_cross_product_matrix(foot_step))
        
        jacobian[time_idx*12:(time_idx+1)*12, time_idx*12:(time_idx+1)*12] = -np.eye(12)
        c1, c2, c3, c4 = contact_list[time_idx]
        contact_mat = np.diag([c1]*3 + [c2]*3 + [c3]*3 + [c4]*3)
        jacobian[time_idx*12:(time_idx+1)*12, (time_horizon + time_idx)*12:(time_horizon + time_idx + 1)*12] = np.matmul(B_mat, contact_mat)
        if time_idx > 0:
            jacobian[time_idx*12:(time_idx+1)*12, (time_idx-1)*12:time_idx*12] = A_mat

            temp_mat = np.zeros((12, 12))
            derivative_R = np.array([[-np.sin(yaw), -np.cos(yaw), 0],[np.cos(yaw), -np.sin(yaw), 0],[0, 0, 0]])
            temp_mat[3:6, 5] += delta_t*np.matmul(derivative_R.T, ang_vel_base)
            sum_r_cross_f = np.zeros(3)
            for leg_idx in range(num_leg):
                foot_step = foot_step_list[leg_idx] - pos_com
                sum_r_cross_f += delta_t*np.cross(foot_step, force[leg_idx*3:(leg_idx+1)*3])
            temp_mat[9:12, 5] += np.matmul(derivative_R, np.matmul(inv_base_inertia, np.matmul(yaw_mat.T, sum_r_cross_f)))
            temp_mat[9:12, 5] += np.matmul(yaw_mat, np.matmul(inv_base_inertia, np.matmul(derivative_R.T, sum_r_cross_f)))
            jacobian[time_idx*12:(time_idx+1)*12, (time_idx-1)*12:time_idx*12] += temp_mat

        state = next_state
    return jacobian

def ineq_cons_func(x, args):
    contact_list = args[0]
    friction_coef = args[1]

    time_horizon = int(len(x)/24)
    #state_list = x[:int(len(x)/2)].reshape((time_horizon, 12))
    force_list = x[int(len(x)/2):].reshape((time_horizon, 12))
    
    ineq_cons = []
    for time_idx in range(time_horizon):
        force = force_list[time_idx].reshape((4,3))*contact_list[time_idx].reshape(4,1)
        ineq_cons.append(-abs(force[:,0]) + friction_coef*force[:,2])
        ineq_cons.append(-abs(force[:,1]) + friction_coef*force[:,2])
        ineq_cons.append(force[:,2])

    ineq_cons = np.concatenate(ineq_cons)
    return ineq_cons

def ineq_cons_jacobian(x, args):
    contact_list = args[0]
    friction_coef = args[1]

    time_horizon = int(len(x)/24)
    #state_list = x[:int(len(x)/2)].reshape((time_horizon, 12))
    force_list = x[int(len(x)/2):].reshape((time_horizon, 12))
    
    jacobian = np.zeros((12*time_horizon, len(x)))
    for time_idx in range(time_horizon):
        force = force_list[time_idx].reshape((4,3))*contact_list[time_idx].reshape(4,1)
        for i in range(4):
            jacobian[12*time_idx + i, 12*(time_horizon + time_idx) + i*3] = 1 if force[i,0] < 0 else -1
            jacobian[12*time_idx + i, 12*(time_horizon + time_idx) + i*3 + 2] = friction_coef
        for i in range(4):
            jacobian[12*time_idx + 4 + i, 12*(time_horizon + time_idx) + i*3 + 1] = 1 if force[i,1] < 0 else -1
            jacobian[12*time_idx + 4 + i, 12*(time_horizon + time_idx) + i*3 + 2] = friction_coef
        for i in range(4):
            jacobian[12*time_idx + 8 + i, 12*(time_horizon + time_idx) + i*3 + 2] = 1
    return jacobian
