import os
import numpy as np
import matplotlib.pyplot as plt
from .Gym.gym.envs.atari import AtariEnv
from .Gym.gym.envs.robotics import FetchReachEnv 
gym, rbs, vzd = None, None, None
os.system("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/shawn/.mujoco/mujoco200/bin")
os.system("export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so")

try: from .Gym import gym
except ImportError as e: print(e)

from .CarRacing import car_racing

class RawPreprocess():
	def __init__(self, env):
		self.observation_space = env.observation_space

	def __call__(self, state):
		return state

class AtariPreprocess():
	def __init__(self, env):
		self.observation_space = gym.spaces.Box(0, 255, shape=(105, 80, 3))

	def __call__(self, state):
		assert state.shape == (210,160,3)
		state = state[::2,::2] # downsample by factor of 2
		return state

def get_preprocess(env):
	if isinstance(env.unwrapped, AtariEnv): return AtariPreprocess(env)
	return RawPreprocess(env)

class GymEnv(gym.Wrapper):
	def __init__(self, env, **kwargs):
		super().__init__(env)
		self.unwrapped.verbose = 0
		self.preprocess = get_preprocess(env)
		self.observation_space = self.preprocess.observation_space

	def reset(self, **kwargs):
		self.time = 0
		state = self.env.reset()
		return self.preprocess(state)

	def step(self, action, train=False):
		self.time += 1
		state, reward, done, info = super().step(action)
		return self.preprocess(state), reward, done, info

class CustomEnv(gym.Env):
	def __init__(self, env_name, max_steps):
		self.new_spec = gym.envs.registration.EnvSpec(env_name, max_episode_steps=max_steps)

	@property
	def spec(self):
		return self.new_spec

	@spec.setter
	def spec(self, v):
		max_steps = self.new_spec.max_episode_steps
		self.new_spec = v
		self.new_spec.max_episode_steps = max_steps
