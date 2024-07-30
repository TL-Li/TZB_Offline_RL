import logging
import numpy as np
import random
from gym import spaces
from gym.utils import seeding
import gym
gym.logger.set_level(40)



class TZBEnv(gym.Env):
    def __init__(self):
        self.state = np.zeros(12) #状态空间
        self.seed()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12, ), dtype=np.float)
        self.action_space = spaces.Box(low=118, high=119, shape=(12, ), dtype=np.float)
        self.rew_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float)
        # print("debug:", self.action_space.shape)       


    def step(self, action):
        next_state = action        
        self.state = next_state
        reward = -1.0
        done = True
        # print("debug", next_state.shape)       
        return next_state, reward, done, {}
    
    def seed(self, seed=None):
        seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        self.state = np.zeros(12)
        return self.state
    
    def render(self):
        return 
