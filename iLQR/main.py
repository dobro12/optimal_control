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

from agent import Agent

import numpy as np
import random
import pickle
import time
import sys
import gym

def main():
    env_name = "dobro-CartPole-v0"
    env = gym.make(env_name)

    time_horizon = 20
    agent_args = {'discount_factor':0.99,
        'time_horizon':time_horizon,
        'time_step':0.02,
        }
    agent = Agent(env, agent_args)

    max_steps = 1000
    max_ep_len = min(500, env.spec.max_episode_steps)
    episodes = int(max_steps/max_ep_len)
    epochs = int(1e5)

    for epoch in range(epochs):
        ep_step = 0

        while ep_step < max_steps:
            state = env.reset()
            done = False
            score = 0
            step = 0

            while True:
                step += 1
                ep_step += 1

                action = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                env.render()
                #time.sleep(0.01)

                state = next_state
                score += reward

                if done or step >= max_ep_len:
                    break

            print(score)

if __name__ == "__main__":
    main()
