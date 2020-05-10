import os
import sys
import gym
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from .unity_gym import UnityToGymWrapper
from .objective import cost

class CarRacingEnv(gym.Env):
	def __init__(self, max_time=1000):
		root = os.path.dirname(os.path.abspath(__file__))
		sim_file = os.path.abspath(os.path.join(root, "simulator", sys.platform, "CarRacing"))
		unity_env = UnityEnvironment(file_name=sim_file)
		self.env = UnityToGymWrapper(unity_env)
		self.action_space = self.env.action_space
		self.observation_space = self.env.observation_space
		self.cost_model = cost.CostModel()
		self.max_time = max_time

	def reset(self, idle_timeout=None):
		self.time = 0
		self.idle_timeout = idle_timeout if isinstance(idle_timeout, int) else np.Inf
		state = self.env.reset()
		return state

	def get_reward(self, state):
		x, z, y = state[:3]
		vel = state[3:6]
		reward = 0.1*np.linalg.norm(vel) - self.cost_model.get_cost((x,y))
		return reward

	def step(self, action):
		self.time += 1
		state, reward, done, info = self.env.step(action)
		reward = self.get_reward(state)
		idle = state[-1]
		done = done or idle>self.idle_timeout or self.time > self.max_time
		return state, reward, done, info

	def render(self):
		return self.env.render()

	def close(self):
		self.env.close()
