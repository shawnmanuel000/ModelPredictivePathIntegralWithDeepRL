import torch
import numpy as np
from collections import deque
from torchvision import transforms
from src.utils.rand import RandomAgent
from src.utils.misc import resize, rgb2gray, IMG_DIM
from .pytorch.mpc import EnvModel
from .pytorch.icm import ICMNetwork
from .rllib.base import RayEnv

FRAME_STACK = 3					# The number of consecutive image states to combine for training a3c on raw images

class RawState():
	def __init__(self, state_size, load="", gpu=True):
		self.state_size = state_size

	def reset(self):
		pass

	def get_state(self, state):
		return state

class ImgStack(RawState):
	def __init__(self, state_size, stack_len=FRAME_STACK, load="", gpu=True):
		super().__init__(state_size)
		self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(), transforms.Resize((IMG_DIM, IMG_DIM)), transforms.ToTensor()])
		self.process = lambda x: self.transform(x.astype(np.uint8)).unsqueeze(0).numpy()
		self.state_size = [*self.process(np.zeros(state_size)).shape[-2:], stack_len]
		self.stack_len = stack_len
		self.reset()

	def reset(self):
		self.stack = deque(maxlen=self.stack_len)

	def get_state(self, state):
		state = np.concatenate([self.process(s) for s in state]) if len(state.shape)>3 else self.process(state)
		while len(self.stack) < self.stack_len: self.stack.append(state)
		self.stack.append(state)
		return np.concatenate(self.stack, axis=1)

class ParallelAgent(RandomAgent):
	def __init__(self, state_size, action_size, agent, config, load="", gpu=True, **kwargs):
		self.icm = ICMNetwork(state_size, action_size, config) if config.icm else None
		self.stack = (ImgStack if len(state_size)==3 else RawState)(state_size, load=load, gpu=gpu)
		self.agent = agent(self.stack.state_size, action_size, config, load=load, gpu=gpu)
		super().__init__(self.stack.state_size, action_size, config)

	def get_env_action(self, env, state, eps=None, sample=True):
		state = self.stack.get_state(state)
		env_action, action = self.agent.get_env_action(env, state, eps, sample)
		return env_action, action, state

	def train(self, state, action, next_state, reward, done):
		self.stats.sum(r_t=np.mean(reward, axis=-1))
		next_state = self.stack.get_state(next_state)
		reward = self.icm.get_reward(state, action, next_state, reward, done) if self.icm is not None else reward
		self.agent.train(state, action, next_state, reward, done)

	def save_model(self, dirname="pytorch", name="checkpoint", net=None):
		if self.network is not None: 
			if self.icm is not None: self.icm.save_model(dirname, f"{self.network.name}/{name}", net)
			return self.network.save_model(dirname, name, net)

	def load_model(self, dirname="pytorch", name="checkpoint", net=None):
		if self.network is not None: 
			if self.icm is not None: self.icm.load_model(dirname, f"{self.network.name}/{name}", net)
			self.network.load_model(dirname, name, net)
		return self

	@property
	def eps(self):
		return self.agent.eps if hasattr(self, "agent") else 0

	@eps.setter
	def eps(self, value):
		if hasattr(self, "agent"): self.agent.eps = value 

	@property
	def network(self):
		return self.agent.network if hasattr(self.agent, "network") else None

	def get_stats(self):
		stats = self.icm.get_stats() if self.icm is not None else {}
		return {**super().get_stats(), **stats, **self.agent.get_stats()}

class MPCAgent(ParallelAgent):
	def __init__(self, state_size, action_size, controller, config, load="", gpu=True, **kwargs):
		agent = lambda state_size, action_size, config, load, gpu: controller(state_size, action_size, EnvModel, config, gpu=gpu)
		super().__init__(state_size, action_size, agent, config)

	def train(self, state, action, next_state, reward, done):
		pass

	def save_model(self, dirname="pytorch", name="checkpoint", net=None):
		pass

	def load_model(self, dirname="pytorch", name="checkpoint", net=None):
		return self

class RayAgent(RandomAgent):
	def __init__(self, state_size, action_size, model, config, gpu=True):
		self.agent = model(state_size, action_size, config)
		super().__init__(state_size, action_size, config)

	def get_env_action(self, env, state, eps=None, sample=True):
		env_action, action = self.agent.get_env_action(env, state, eps, sample)
		return env_action, action, None

	def train(self, *wargs):
		return self.agent.train()

	@property
	def eps(self):
		return self.agent.eps

	@eps.setter
	def eps(self, value):
		if hasattr(self, "agent"): self.agent.eps = value 
		