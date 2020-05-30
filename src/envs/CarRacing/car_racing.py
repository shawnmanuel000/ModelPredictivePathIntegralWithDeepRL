import os
import sys
import inspect
import numpy as np
import itertools as it
import pyquaternion as pyq
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
		logging_util.set_log_level(logging_util.ERROR)
		unity_env = UnityEnvironment(file_name=sim_file, side_channels=[self.channel], worker_id=self.id + np.random.randint(10000, 20000))
		self.scale_sim = lambda s: self.channel.set_configuration_parameters(width=50*int(1+9*s), height=50*int(1+9*s), quality_level=int(1+3*s), time_scale=int(1+9*(1-s)))
		self.env = UnityToGymWrapper(unity_env)
		self.cost_model = CostModel()
		self.action_space = self.env.action_space
		self.cost_queries = list(it.product(np.linspace(-2,2,5), [0], np.linspace(0,4,5)))
		self.observation_space = gym.spaces.Box(-np.inf, np.inf, self.observation()[0].shape)
		self.src = '\t'.join([line for line in open(os.path.abspath(__file__), 'r')][47:58])
		self.max_time = max_time
		self.reset()

	def reset(self, idle_timeout=10, train=True):
		self.time = 0
		self.scale_sim(0)
		self.idle_timeout = idle_timeout if isinstance(idle_timeout, int) else np.Inf
		self.state, self.spec = self.observation()
		return self.state

	def step(self, action):
		self.time += 1
		next_state, reward, done, info = self.env.step(action)
		idle = next_state[29]
		done = done or idle>self.idle_timeout or self.time > self.max_time
		next_state, next_spec = self.observation(next_state)
		terminal = -(1-self.time/self.max_time)*int(done)
		reward = -self.cost_model.get_cost(next_spec, self.spec) + terminal
		self.state, self.spec = next_state, next_spec
		return self.state, reward, done, info

	def render(self, mode=None, **kwargs):
		self.scale_sim(1)
		return self.env.render()

	@staticmethod
	def dynamics_spec(state):
		pos = state[...,:3]
		vel = state[...,3:6]
		angvel = state[...,6:9]
		rotation = state[...,9:13]
		fl_drive = state[...,13:17] # steer angle, motor torque, brake torque, rpm
		fr_drive = state[...,17:21]
		rl_drive = state[...,21:25]
		rr_drive = state[...,25:29]
		idle = state[...,29:30]
		steer_angle = fl_drive[...,0:1]
		rpm = np.array([x[...,-1] for x in [fl_drive, fr_drive, rl_drive, rr_drive]])
		spec = {"pos":pos, "vel":vel, "angvel":angvel, "rotation":rotation, "steer_angle":steer_angle, "rpm":rpm, "idle":idle}
		return spec

	def track_spec(self, state):
		spec = self.dynamics_spec(state)
		quat = pyq.Quaternion(spec["rotation"])
		points = np.array([(x,z,y) for x,y,z in [quat.rotate(p) for p in self.cost_queries]])
		costs = self.cost_model.get_point_cost(spec["pos"]+points, transform=False)
		spec.update({"costs":costs})
		return spec

	def observation(self, state=None):
		state = self.env.reset() if state is None else state
		spec = self.track_spec(state)
		self.dynamics_keys = ["pos", "vel", "angvel", "steer_angle", "rpm", "idle", "costs"]
		values = list(map(spec.get, self.dynamics_keys))
		self.dynamics_lens = list(map(len, values))
		observation = np.concatenate(values, -1)
		self.dynamics_size = sum(self.dynamics_lens[:4])
		return observation, spec

	@staticmethod
	def observation_spec(observation):
		spec = {}
		spec["pos"] = observation[...,:3]
		spec["vel"] = observation[...,3:6]
		spec["angvel"] = observation[...,6:9]
		spec["steer_angle"] = observation[...,9:10]
		spec["rpm"] = observation[...,10:14]
		spec["idle"] = observation[...,14:15]
		spec["costs"] = observation[...,15:]
		return spec

	def close(self):
		if not hasattr(self, "closed"): self.env.close()
		self.closed = True
