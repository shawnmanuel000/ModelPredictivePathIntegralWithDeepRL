import tqdm
import torch
import random
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal
from src.utils.rand import RandomAgent, ReplayBuffer
from src.utils.misc import load_module
from ..agents.base import PTNetwork, PTAgent, Conv, one_hot_from_indices
from . import EnvModel

class MPPIController(PTNetwork):
	def __init__(self, state_size, action_size, config, load="", gpu=True, name="mppi"):
		super().__init__(config, gpu=gpu, name=name)
		self.envmodel = EnvModel(state_size, action_size, config, load=load, gpu=gpu)
		self.mu = np.zeros(action_size)
		self.cov = np.diag(np.ones(action_size))*config.MPC.COV
		self.icov = np.linalg.inv(self.cov)
		self.lamda = config.MPC.LAMBDA
		self.horizon = config.MPC.HORIZON
		self.nsamples = config.MPC.NSAMPLES
		self.config = config
		self.action_size = action_size
		self.init_control()

	def get_action(self, state, eps=None, sample=True):
		batch = state.shape[:-1]
		horizon = max(int((1-eps)*self.horizon),1) if eps else self.horizon
		if len(batch) and self.control.shape[0] != batch[0]: self.init_control(batch[0])
		x = torch.Tensor(state).view(*batch, 1,-1).repeat_interleave(self.nsamples, -2)
		controls = np.clip(self.control[:,None,:,:] + self.noise, -1, 1)
		self.states, rewards = self.envmodel.rollout(controls[...,:horizon,:], x, numpy=True)
		costs = -np.sum(rewards, -1) #+ self.lamda * np.copy(self.init_cost)
		beta = np.min(costs, -1, keepdims=True)
		costs_norm = -(costs - beta)/self.lamda
		weights = sp.special.softmax(costs_norm, axis=-1)
		self.control += np.sum(weights[:,:,None,None]*self.noise, len(batch))
		action = self.control[...,0,:]
		self.control = np.roll(self.control, -1, axis=-2)
		self.control[...,-1,:] = 0
		return action

	def init_control(self, batch_size=1):
		self.control = np.random.uniform(-1, 1, size=[1, self.horizon, *self.action_size]).repeat(batch_size, 0)
		self.noise = np.random.multivariate_normal(self.mu, self.cov, size=[1, self.nsamples, self.horizon]).repeat(batch_size, 0)
		self.init_cost = np.sum(self.control[:,None,:,None,:] @ self.icov[None,None,None,:,:] @ self.noise[:,:,:,:,None], axis=(2,3,4))

	def optimize(self, states, actions, next_states, rewards, dones):
		return self.envmodel.optimize(states, actions, next_states, rewards, dones)

	def save_model(self, dirname="pytorch", name="checkpoint", net=None):
		return self.envmodel.save_model(dirname, name, net)
		
	def load_model(self, dirname="pytorch", name="checkpoint", net=None):
		return self.envmodel.load_model(dirname, name, net)

	def get_stats(self):
		return {**super().get_stats(), **self.envmodel.get_stats()}

class MPPIAgent(PTAgent):
	def __init__(self, state_size, action_size, config, gpu=True, load=None):
		super().__init__(state_size, action_size, config, MPPIController, gpu=gpu, load=load)
		self.dataset = load_module("src.data.loaders:OnlineDataset")

	def get_action(self, state, eps=None, sample=True):
		eps = self.eps if eps is None else eps
		action_random = super().get_action(state, eps)
		action_greedy = self.network.get_action(np.array(state), eps)
		action = np.clip((1-eps)*action_greedy + eps*action_random, -1, 1)
		return action

	def partition(self, x):
		if self.config.NUM_STEPS is None:
			return x[None,...]
		num_splits = x.shape[0]//self.config.NUM_STEPS
		if num_splits == 0:
			arr = np.zeros([self.config.NUM_STEPS, *x.shape[1:]])
			arr[-x.shape[0]:] = x
			num_splits = 1
			x = arr
		arr = x[:num_splits*self.config.NUM_STEPS].reshape(num_splits, self.config.NUM_STEPS, *x.shape[1:])
		return arr

	def train(self, state, action, next_state, reward, done):
		self.time = getattr(self, "time", 0) + 1
		if not hasattr(self, "buffers"): self.buffers = [[] for _ in done]
		for buffer, s, a, ns, r, d in zip(self.buffers, state, action, next_state, reward, done):
			buffer.append((s, a, s if d else ns, r, d))
			if not d: continue
			states, actions, next_states, rewards, dones = map(np.array, zip(*buffer))
			states, actions, next_states, rewards, dones = [self.partition(x) for x in (states, actions, next_states, rewards, dones)]
			buffer.clear()
			self.replay_buffer.extend(list(zip(states, actions, next_states, rewards, dones)), shuffle=False)
		if len(self.replay_buffer) > self.config.REPLAY_BATCH_SIZE and self.time % self.config.TRAIN_EVERY == 0:
			losses = []
			samples = list(self.replay_buffer.sample(self.config.REPLAY_BATCH_SIZE, dtype=None)[0])
			dataset = self.dataset(self.config, samples, seq_len=self.config.MPC.HORIZON)
			loader = torch.utils.data.DataLoader(dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
			pbar = tqdm.tqdm(loader)
			for states, actions, next_states, rewards, dones in pbar:
				losses.append(self.network.optimize(states, actions, next_states, rewards, dones))
				pbar.set_postfix_str(f"Loss: {losses[-1]:.4f}")
			self.network.envmodel.network.schedule(np.mean(losses))
			# self.eps = max(self.eps * self.config.EPS_DECAY, self.config.EPS_MIN)
		self.eps = max(1-(self.time%self.config.TRAIN_EVERY)/self.config.TRAIN_EVERY, self.config.EPS_MIN)
		self.stats.mean(len=len(self.replay_buffer))
