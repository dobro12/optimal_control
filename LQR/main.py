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

from c_module.agent import Agent

import matplotlib.pyplot as plt
import numpy as np
import time
import gym

def main():
    env_name = "dobro-CartPole-v0"
    env = gym.make(env_name)

    x_dim = env.observation_space.shape[0]
    u_dim = env.action_space.shape[0]
    time_horizon = 200

    #####################################
    ##### set A, B, R, Q, Qf matrix #####
    m1 = env.unwrapped.masscart
    m2 = env.unwrapped.masspole
    L = env.unwrapped.length
    g = env.unwrapped.gravity
    dt = env.unwrapped.tau
    temp_A_mat = np.eye(x_dim)
    temp_A_mat[0, 1] = dt
    temp_A_mat[2, 3] = dt
    temp_B_mat = np.array([[0.5*dt**2, 0.0], [dt, 0.0], [0.0, 0.5*dt**2], [0.0, dt]])
    A_mat = np.array([[0, 0, -(m2/(m1 + m2))*(g/(4.0/3.0 - m2/(m1 + m2))), 0], [0, 0, g/(L*(4.0/3.0 - m2/(m1 + m2))), 0]])
    B_mat = np.array([[(1.0/(m1 + m2))*(1 + 3.0*m2/(4.0*m1 + m2))], [-3.0/(L*(4.0*m1 + m2))]])
    A_mat = temp_A_mat + np.matmul(temp_B_mat, A_mat)
    B_mat = np.matmul(temp_B_mat, B_mat)

    R_mat = np.eye(u_dim)*0.01
    Q_mat = np.eye(x_dim)*1.0
    Qf_mat = np.eye(x_dim)*100.0
    #####################################

    #declare LQR solver
    agent = Agent(x_dim, u_dim, time_horizon, A_mat, B_mat, R_mat, Q_mat, Qf_mat)
    x_list = []
    u_list = []

    state = env.reset()
    action, P_mat_list = agent.get_action(state)

    for i in range(time_horizon):
        action = -np.matmul(P_mat_list[i], state).ravel()
        state, reward, done, info = env.step(action)
        env.render()
        time.sleep(dt)

        x_list.append(state)
        u_list.append(action)

    env.close()
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