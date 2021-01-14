from tensorflow.python.ops.parallel_for.gradients import jacobian
from copy import deepcopy
import tensorflow as tf
import numpy as np
import scipy
import os

class Agent:
    def __init__(self, env, args):
        self.env = env
        self.discount_factor = args['discount_factor']
        self.time_horizon = args['time_horizon'] #5
        self.time_step = args['time_step'] #0.02

        self.state_dim = 4
        self.action_dim = 1

        self.Q_mat = 1.0*np.eye(self.state_dim)
        self.Q_mat = np.matmul(self.Q_mat.T, self.Q_mat)
        self.R_mat = 0.01*np.eye(self.action_dim)
        self.R_mat = np.matmul(self.R_mat.T, self.R_mat) 
        self.target_x = [0, 0, 0, 0]
        self.target_u = [0]
        self.damping_ratio = 1e-8
        self.learning_rate = 1.0
        self.max_iteration = 10


    def get_action(self, state):
        x_list, u_list = self.solve_MPC(state)
        action = u_list[0]
        return action

    def solve_MPC(self, init_x):
        target_x_list = np.array([self.target_x]*self.time_horizon)
        target_u_list = np.array([self.target_u]*self.time_horizon)

        #get x list
        J_value = 0
        x_list = []
        u_list = []
        for t_idx in range(self.time_horizon):
            x = deepcopy(init_x) if t_idx == 0 else deepcopy(x_list[t_idx - 1])
            u = np.zeros(self.action_dim)
            next_x = self.transition(x, u)

            u_list.append(u)
            x_list.append(next_x)

            temp_J = 0.5*(np.matmul((x_list[t_idx] - target_x_list[t_idx]).T, np.matmul(self.Q_mat, x_list[t_idx] - target_x_list[t_idx])) \
                        + np.matmul((u_list[t_idx] - target_u_list[t_idx]).T, np.matmul(self.R_mat, u_list[t_idx] - target_u_list[t_idx])))
            if t_idx == self.time_horizon - 1:
                J_value += temp_J * np.power(self.discount_factor, self.time_horizon - 1)/(1 - self.discount_factor)
            else:
                J_value += temp_J * np.power(self.discount_factor, t_idx)

        #K, d
        K_list = np.zeros((self.time_horizon, self.action_dim, self.state_dim))
        d_list = np.zeros((self.time_horizon, self.action_dim))

        #print("########### MPC start ###########")
        for _ in range(self.max_iteration):
            pre_J_value = J_value

            #backward pass
            Q_mat = self.Q_mat * np.power(self.discount_factor, self.time_horizon - 1)/(1 - self.discount_factor)
            P_mat = Q_mat
            p_vector = np.matmul(Q_mat, x_list[self.time_horizon - 1] - target_x_list[self.time_horizon - 1])
            for t_idx in range(self.time_horizon - 1, -1, -1):
                x = deepcopy(init_x) if t_idx == 0 else deepcopy(x_list[t_idx - 1])
                u = deepcopy(u_list[t_idx])
                A_mat, B_mat = self.get_A_B_mat(x, u)
                Q_mat = self.Q_mat * np.power(self.discount_factor, t_idx - 1) if t_idx != 0 else np.zeros((self.state_dim, self.state_dim))
                R_mat = self.R_mat * np.power(self.discount_factor, t_idx) if t_idx != self.time_horizon - 1 else self.R_mat * np.power(self.discount_factor, self.time_horizon - 1)/(1 - self.discount_factor)
                Qxx = Q_mat + np.matmul(A_mat.T, np.matmul(P_mat, A_mat))
                Quu = R_mat + np.matmul(B_mat.T, np.matmul(P_mat, B_mat))
                Qux = np.matmul(B_mat.T, np.matmul(P_mat, A_mat))
                Qxu = np.matmul(A_mat.T, np.matmul(P_mat, B_mat))
                if t_idx != 0:
                    Qx = np.matmul(Q_mat, x - target_x_list[t_idx - 1]) + np.matmul(A_mat.T, p_vector)
                else:
                    Qx = np.matmul(A_mat.T, p_vector)
                Qu = np.matmul(R_mat, u - target_u_list[t_idx]) + np.matmul(B_mat.T, p_vector)

                temp_mat = -np.linalg.inv(Quu + self.damping_ratio*np.eye(self.action_dim))
                K_mat = np.matmul(temp_mat, Qux)
                d_vector = np.matmul(temp_mat, Qu)

                P_mat = Qxx + np.matmul(K_mat.T, np.matmul(Quu, K_mat) + Qux) + np.matmul(Qxu, K_mat)
                p_vector = Qx + np.matmul(K_mat.T, np.matmul(Quu, d_vector) + Qu) + np.matmul(Qxu, d_vector)
                K_list[t_idx] = K_mat
                d_list[t_idx] = d_vector

            #forward pass
            learning_rate = self.learning_rate
            while True:
                new_x_list = []
                new_u_list = []
                J_value = 0
                for t_idx in range(self.time_horizon):
                    if t_idx == 0:
                        x = deepcopy(init_x)
                        delta_x = np.zeros(self.state_dim)
                    else:
                        x = deepcopy(new_x_list[t_idx - 1])
                        delta_x = new_x_list[t_idx - 1] - x_list[t_idx - 1]
                    u = u_list[t_idx] + np.matmul(K_list[t_idx], delta_x) + learning_rate*d_list[t_idx]
                    next_x = self.transition(x, u)

                    new_u_list.append(u)
                    new_x_list.append(next_x)

                    temp_J = 0.5*(np.matmul((new_x_list[t_idx] - target_x_list[t_idx]).T, np.matmul(self.Q_mat, new_x_list[t_idx] - target_x_list[t_idx])) \
                                + np.matmul((new_u_list[t_idx] - target_u_list[t_idx]).T, np.matmul(self.R_mat, new_u_list[t_idx] - target_u_list[t_idx])))
                    if t_idx == self.time_horizon - 1:
                        J_value += temp_J * np.power(self.discount_factor, self.time_horizon - 1)/(1 - self.discount_factor)
                    else:
                        J_value += temp_J * np.power(self.discount_factor, t_idx)

                #print("\t\t{}".format(pre_J_value - J_value))
                if pre_J_value - J_value >= - 1e-5:
                    break
                learning_rate *= 0.5

            x_list = new_x_list
            u_list = new_u_list

            #print(pre_J_value - J_value, J_value)
            if abs(pre_J_value - J_value) < 1e-3 :
                break

        return x_list, u_list

    def transition(self, state, action):
        x, x_dot, theta, theta_dot = state
        force = action[0]
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.env.unwrapped.polemass_length * theta_dot * theta_dot * sintheta) / self.env.unwrapped.total_mass
        thetaacc = (self.env.unwrapped.gravity * sintheta - costheta* temp) / (self.env.unwrapped.length * (4.0/3.0 - self.env.unwrapped.masspole * costheta * costheta / self.env.unwrapped.total_mass))
        xacc  = temp - self.env.unwrapped.polemass_length * thetaacc * costheta / self.env.unwrapped.total_mass
        if self.env.unwrapped.kinematics_integrator == 'euler':
            x  = x + self.env.unwrapped.tau * x_dot
            x_dot = x_dot + self.env.unwrapped.tau * xacc
            theta = theta + self.env.unwrapped.tau * theta_dot
            theta_dot = theta_dot + self.env.unwrapped.tau * thetaacc
        else: # semi-implicit euler
            x_dot = x_dot + self.env.unwrapped.tau * xacc
            x  = x + self.env.unwrapped.tau * x_dot
            theta_dot = theta_dot + self.env.unwrapped.tau * thetaacc
            theta = theta + self.env.unwrapped.tau * theta_dot
        next_state = np.array([x, x_dot, theta, theta_dot])
        return next_state

    def get_A_B_mat(self, x, u):
        A = np.zeros((self.state_dim, self.state_dim))
        B = np.zeros((self.state_dim, self.action_dim))

        eps = 1e-5
        next_x = self.transition(x, u)

        for i in range(self.state_dim):
            temp_x = deepcopy(x)
            temp_x[i] += eps
            temp_next_x = self.transition(temp_x, u)
            A[:,i] = (temp_next_x - next_x)/eps

        for i in range(self.action_dim):
            temp_u = deepcopy(u)
            temp_u[i] += eps
            temp_next_x = self.transition(x, temp_u)
            B[:,i] = (temp_next_x - next_x)/eps

        return A, B

