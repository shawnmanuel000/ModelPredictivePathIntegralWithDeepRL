import torch
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal
from src.utils.rand import RandomAgent
from ..agents.base import PTACNetwork, PTAgent, Conv, one_hot_from_indices

K = 1000
T = 20
LAMBDA = 1

class MPPIController(RandomAgent):
	def __init__(self, state_size, action_size, envmodel, config, gpu=True):
		self.envmodel = envmodel(state_size, action_size, config, load=config.env_name)
		self.mu = np.zeros(action_size)
		self.cov = np.diag(np.ones(action_size))
		self.icov = np.linalg.inv(self.cov)
		self.lamda = config.MPC.LAMBDA
		self.horizon = config.MPC.HORIZON
		self.nsamples = config.MPC.NSAMPLES
		self.control = np.random.uniform(-1, 1, [self.horizon, *action_size])
		self.noise = np.random.multivariate_normal(self.mu, self.cov, size=(self.nsamples, self.horizon))
		self.step = 0

	def get_action(self, state, eps=None, sample=True):
		self.step += 1
		if self.step%1 == 0:
			costs = np.zeros(shape=[self.nsamples])
			x = torch.Tensor(state).view(1,-1).repeat(self.nsamples, 1)
			self.envmodel.reset(batch_size=self.nsamples, state=x, initstate=False)
			for t in range(self.horizon):
				u = self.control[None,t]
				e = self.noise[:,t]
				v = np.clip(u + e, -1, 1)
				x, q = self.envmodel.step(v, x, numpy=True)
				costs += q + self.lamda * (u[:,None,:] @ self.icov[None,:,:] @ e[:,:,None]).flatten()
			beta = np.min(costs)
			costs_norm = -(costs - beta)/self.lamda
			weights = sp.special.softmax(costs_norm)
			self.control += np.sum(weights[:,None,None]*self.noise, 0)
		action = np.clip(self.control[0], -1, 1)
		self.control = np.roll(self.control, -1, axis=0)
		self.control[-1] = 0
		return action if len(action.shape)==len(state.shape) else np.repeat(action[None,:], state.shape[0], 0)








