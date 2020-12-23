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
import env

from scipy.optimize import minimize
from scipy.optimize import Bounds

import matplotlib.pyplot as plt
import numpy as np
import time
import gym

# declare environmental variables
u_dim = 1
x_dim = 4
dt = 0.1

R_mat = np.eye(u_dim)*0.1
Q_mat = np.eye(x_dim)*0.1
Qf_mat = np.eye(x_dim)*100.0

gravity = 9.8
masscart = 1.0
masspole = 0.1
total_mass = masspole + masscart
length = 0.5
polemass_length = masspole*length

def dynamics(state, action):
    x = state[:, 0]
    x_dot = state[:, 1]
    theta = state[:, 2]
    theta_dot = state[:, 3]
    force = action[:, 0]

    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
    theta_acc = (gravity * sintheta - costheta* temp) / (length * (4.0/3.0 - masspole * costheta * costheta / total_mass))
    x_acc  = temp - polemass_length * theta_acc * costheta / total_mass
    
    state_dot = np.array([x_dot, x_acc, theta_dot, theta_acc]).T
    return state_dot

def obj_func(x):
    N = int(len(x)/5 - 1)    
    u_list = np.array(x[:N+1]).reshape((N+1, u_dim))
    x_list = np.array(x[N+1:]).reshape((N+1, x_dim))
        
    f = 0.5*np.matmul(x_list[N, :], np.matmul(Qf_mat, x_list[N, :]))
    f += 0.5*dt*np.sum(np.matmul(u_list[:N, :], R_mat)*u_list[:N, :])
    f += 0.5*dt*np.sum(np.matmul(x_list[:N, :], Q_mat)*x_list[:N, :])
    f = f.ravel()[0]
    return f

def obj_jacobian(x):
    N = int(len(x)/5 - 1)
    u_list = np.array(x[:N+1]).reshape((N+1, u_dim))
    x_list = np.array(x[N+1:]).reshape((N+1, x_dim))
    jacobian = np.zeros(len(x))
    for i in range(N):
        jacobian[i*u_dim : (i + 1)*u_dim] = dt*np.matmul(R_mat, u_list[i]).ravel()
        jacobian[(N + 1)*u_dim + i*x_dim : (N + 1)*u_dim + (i+1)*x_dim] = dt*np.matmul(Q_mat, x_list[i]).ravel()
    jacobian[(x_dim + u_dim)*(N + 1) - x_dim : (x_dim + u_dim)*(N + 1)] = np.matmul(Qf_mat, x_list[N]).ravel()
    return jacobian

def eq_cons_func(x):
    N = int(len(x)/5 - 1)
    u_list = np.array(x[:N+1]).reshape((N+1, u_dim))
    x_list = np.array(x[N+1:]).reshape((N+1, x_dim))
    
    # get x_dot from dynamics
    x_dot_list = dynamics(x_list, u_list)
    
    # transform variables for pice-wise polynomial
    x_left = x_list[:N,:]
    x_right = x_list[1:,:]
    x_dot_left = x_dot_list[:N,:]
    x_dot_right = x_dot_list[1:,:]
    u_left = u_list[:N,:]
    u_right = u_list[1:,:]
    
    # get collocation points
    x_c = 0.5*(x_left + x_right) + dt*0.125*(x_dot_left - x_dot_right)
    u_c = 0.5*(u_left + u_right)
    x_dot_c = dynamics(x_c, u_c)

    # equality constraint
    e_cons = np.ravel(x_left - x_right + dt*(x_dot_left + 4*x_dot_c + x_dot_right)/6)
    return e_cons

eq_cons = {
    'type':'eq',
    'fun' :eq_cons_func,
    }


def main():
    env_name = "dobro-CartPole-v0"
    env = gym.make(env_name)
    x_list = []
    u_list = []
    steps = 500
    N = 10

    state = env.reset()
    sim_t = 0
    cnt = 0

    start_t = time.time()
    for i in range(steps):
        if sim_t >= cnt*dt:
            init_state = list(state)
            init_action_list = np.zeros((N+1,u_dim))
            init_state_list = np.zeros((N+1,x_dim))
            init_state_list[:x_dim] = init_state
            x_init = np.concatenate([init_action_list.ravel(), init_state_list.ravel()])
            lowers = np.array([-np.inf]*((N+1)*u_dim) + init_state + [-np.inf]*(N*x_dim))
            uppers = np.array([np.inf]*((N+1)*u_dim) + init_state + [np.inf]*(N*x_dim))
            bounds = Bounds(lowers, uppers)
            res = minimize(obj_func, x_init, method="SLSQP", jac=obj_jacobian, bounds=bounds, constraints=[eq_cons], \
                        options={'ftol':1e-5, 'disp':False, 'maxiter':20, 'eps':1e-10})
            cnt += 1
        
        weight = (sim_t - (cnt - 1)*dt)/dt
        action = np.array([res.x[0]*(1 - weight) + res.x[1]*weight])

        state, reward, done, info = env.step(action)
        env.render()

        x_list.append(state)
        u_list.append(action)
        sim_t += env.unwrapped.tau

    env.close()
    print("elapsed time : {:.3f}s, simulation time : {:.3f}".format(time.time()-start_t, sim_t))

    x_list = np.array(x_list)
    u_list = np.array(u_list)

    fig_size = 6
    fig, ax_list = plt.subplots(nrows=2, ncols=1, figsize=(fig_size*1.5, fig_size*1.5))
    ax_list[0].plot(x_list[:,0], label="pos")
    ax_list[0].plot(x_list[:,1], label="pos_dot")
    ax_list[0].plot(x_list[:,2], label="theta")
    ax_list[0].plot(x_list[:,3], label="thtta_dot")
    ax_list[0].grid()
    ax_list[0].legend()
    ax_list[0].set_title('x : state')

    ax_list[1].plot(u_list[:,0])
    ax_list[1].grid()
    ax_list[1].set_title('u : input')

    fig.tight_layout()
    plt.savefig('result.png')
    plt.show()    

if __name__ == "__main__":
    main()
