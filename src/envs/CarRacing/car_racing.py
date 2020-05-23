import os
import sys
import inspect
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs import logging_util
from .unity_gym import UnityToGymWrapper
from .objective import CostModel
from ..Gym import gym

logging_util.set_log_level(logging_util.ERROR)

class EnvMeta(type):
	def __new__(meta, name, bases, class_dict):
		cls = super().__new__(meta, name, bases, class_dict)
		gym.register(f"{name}-v1", entry_point=cls)
		return cls

class CarRacing(gym.Env, metaclass=EnvMeta):
	def __new__(cls, **kwargs):
		cls.id = getattr(cls, "id", 0)+1
		return super().__new__(cls)

	def __init__(self, max_time=500):
		root = os.path.dirname(os.path.abspath(__file__))
		sim_file = os.path.abspath(os.path.join(root, "simulator", sys.platform, "CarRacing"))
		self.channel = EngineConfigurationChannel()
		unity_env = UnityEnvironment(file_name=sim_file, side_channels=[self.channel], worker_id=self.id + np.random.randint(10000, 20000))
		self.scale_sim = lambda s: self.channel.set_configuration_parameters(width=50*int(1+9*s), height=50*int(1+9*s), quality_level=int(1+3*s), time_scale=int(1+9*(1-s)))
		self.env = UnityToGymWrapper(unity_env)
		self.pos_scale = 1
		self.vtarget = 20
		self.cost_model = CostModel()
		self.action_space = self.env.action_space
		self.observation_space = gym.spaces.Box(-np.inf, np.inf, self.observation().shape)
		self.src = '\t'.join([line for line in open(os.path.abspath(__file__), 'r')][46:66])
		self.max_time = max_time
		self.reset()

	def reset(self, idle_timeout=10, train=True):
		self.time = 0
		self.scale_sim(0)
		self.idle_timeout = idle_timeout if isinstance(idle_timeout, int) else np.Inf
		self.state = self.observation()
		return self.state

	def get_reward(self, state, prevstate=None):
		prevstate = state if prevstate is None else prevstate
		px, pz, py = prevstate[:3]*self.pos_scale
		x, z, y = state[:3]*self.pos_scale
		_, _, vy = state[3:6]
		idle = state[29]
		cost = self.cost_model.get_cost((x,y), transform=True)
		progress = self.cost_model.track.get_progress([px,py,pz], [x,y,z])
		reward = min(progress,0)*np.exp(2*cost) + max(np.tanh(progress),0)/np.exp(cost) + (1-np.power(vy-self.vtarget,2)/self.vtarget**2) + np.tanh(vy)-cost
		return reward

	def step(self, action):
		self.time += 1
		next_state, reward, done, info = self.env.step(action)
		idle = next_state[29]
		done = done or idle>self.idle_timeout or self.time > self.max_time
		next_state = self.observation(next_state)
		reward = self.get_reward(next_state, self.state) - (1-self.time/self.max_time)*int(done)
		self.state = next_state
		return self.state, reward, done, info

	def render(self, mode=None, **kwargs):
		self.scale_sim(1)
		return self.env.render()

	def observation(self, state=None):
		state = self.env.reset() if state is None else state
		target = self.cost_model.track.get_path([state[0], state[2], state[1]], dirn=True)
		target = np.array(target) - state[:3]
		return np.concatenate([state[:3]/self.pos_scale, state[3:], *target/self.pos_scale], -1)

	def close(self):
		if not hasattr(self, "closed"): self.env.close()
		self.closed = True
