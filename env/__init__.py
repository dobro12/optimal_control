from gym.envs.registration import register

register(
    id='dobro-CartPole-v0',
    entry_point='env.cartpole:CartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)