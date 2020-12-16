from ctypes import cdll
import numpy as np
import ctypes
import os

FILE_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-1])

class Agent:
    def __init__(self, x_dim, u_dim, time_horizon, A, B, R, Q, Qf):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.time_horizon = time_horizon
        self.A_mat = A
        self.B_mat = B
        self.R_mat = R
        self.Qf_mat = Q
        self.Q_mat = Qf

        ######################################
        ######## get iLQR.so function ########
        c_module = cdll.LoadLibrary('{}/main.so'.format(FILE_PATH))
        c_module.get_action.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double * self.x_dim), ctypes.c_int, 
            ctypes.POINTER(ctypes.c_double * (self.x_dim*self.x_dim)), 
            ctypes.POINTER(ctypes.c_double * (self.x_dim*self.u_dim)), 
            ctypes.POINTER(ctypes.c_double * (self.u_dim*self.u_dim)), 
            ctypes.POINTER(ctypes.c_double * (self.x_dim*self.x_dim)), 
            ctypes.POINTER(ctypes.c_double * (self.x_dim*self.x_dim)), 
            ctypes.POINTER(ctypes.c_double * (self.time_horizon*self.x_dim*self.u_dim)), 
            )
        self.c_module = c_module
        self.ctype_arr_convert = lambda arr : (ctypes.c_double * len(arr))(*arr)
        ######## get iLQR.so function ########
        ######################################

        [self.A_mat, self.B_mat, self.R_mat, self.Q_mat, self.Qf_mat] = \
            self.ctype_arr_convert_all([self.A_mat, self.B_mat, self.R_mat, self.Q_mat, self.Qf_mat])

        self.reset()


    def reset(self):
        pass

    def ctype_arr_convert_all(self, args): 
        for arg_idx in range(len(args)): 
            args[arg_idx] = self.ctype_arr_convert(np.array(args[arg_idx], dtype=np.float64).ravel())
        return args 

    def get_action(self, init_x):
        init_x = np.array(init_x).reshape((self.x_dim, 1))
        P_mat_list = np.zeros((self.time_horizon, self.x_dim, self.u_dim))
        [cpp_init_x, cpp_P_mat_list] = self.ctype_arr_convert_all([init_x, P_mat_list])

        self.c_module.get_action(self.x_dim, self.u_dim, cpp_init_x, self.time_horizon, \
                                        self.A_mat, self.B_mat, \
                                        self.R_mat, self.Q_mat, self.Qf_mat, cpp_P_mat_list)
        P_mat_list = np.array(cpp_P_mat_list).reshape((self.time_horizon, self.u_dim, self.x_dim))
        action = -np.matmul(P_mat_list[0], init_x).ravel()
        return action, P_mat_list
