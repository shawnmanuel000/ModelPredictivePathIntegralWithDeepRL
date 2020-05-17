import torch
import numpy as np
from src.utils.rand import RandomAgent
from ..agents.base import one_hot
from ... import get_env

class RealEnv():
	def __init__(self, config):
		self.env = get_env(config.env_name)
		self.discrete = hasattr(self.env.action_space, "n")
		self.reset()

	def reset(self, initstate=True):
		if initstate:
			self.state = self.env.reset()
			self.istate = self.env.unwrapped.state
		else:
			self.env.unwrapped.state = self.istate
		self.curstate = self.state

	def step(self, action, state=None):
		if self.discrete: action = one_hot(torch.Tensor(action)).cpu().numpy()
		action = RandomAgent.to_env_action(self.env.action_space, action)
		self.curstate, reward, _, _ = self.env.step(action)
		if state is not None:
			self.state = self.curstate
			self.istate = self.env.unwrapped.state
		return self.curstate, -reward