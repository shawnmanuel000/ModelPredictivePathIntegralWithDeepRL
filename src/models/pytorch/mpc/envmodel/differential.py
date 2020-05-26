import os
import torch
import numpy as np
from ...agents.base import PTNetwork, one_hot

class TransitionModel(torch.nn.Module):
	def __init__(self, state_size, action_size, config):
		super().__init__()
		self.gru = torch.nn.GRUCell(action_size[-1] + 5*state_size[-1], 5*state_size[-1])
		self.linear = torch.nn.Linear(5*state_size[-1], 4*state_size[-1])
		self.linear = torch.nn.Linear(4*state_size[-1], 3*state_size[-1])
		self.linear = torch.nn.Linear(3*state_size[-1], 2*state_size[-1])
		self.state_size = state_size

	def forward(self, action, state, state_dot):
		inputs = torch.cat([action, state, state_dot, state.pow(2), state.sin(), state.cos()],-1)
		self.hidden = self.gru(inputs, self.hidden)
		state_diff = self.linear(self.hidden)
		next_state = state + state_diff
		return next_state

	def reset(self, device, batch_size=None):
		if batch_size is None: batch_size = self.hidden[0].shape[1] if hasattr(self, "hidden") else 1
		self.hidden = torch.zeros(batch_size, 5*self.state_size[-1], device=device)

class RewardModel(torch.nn.Module):
	def __init__(self, state_size, action_size, config):
		super().__init__()
		self.linear = torch.nn.Linear(action_size[-1] + 2*state_size[-1], 1)

	def forward(self, action, state, next_state):
		inputs = torch.cat([action, state, next_state-state],-1)
		reward = self.linear(inputs)
		return reward

class DifferentialEnv(PTNetwork):
	def __init__(self, state_size, action_size, config, load="", gpu=True, name="dfrntl"):
		super().__init__(config, gpu, name)
		self.state_size = state_size
		self.action_size = action_size
		self.discrete = type(self.action_size) != tuple
		self.reward = RewardModel(state_size, action_size, config)
		self.dynamics = TransitionModel(state_size, action_size, config)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=config.DYN.LEARN_RATE)
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=config.DYN.FACTOR, patience=config.DYN.PATIENCE)
		self.to(self.device)
		if load: self.load_model(load)

	def step(self, action, state, numpy=False, grad=False):
		with torch.enable_grad() if grad else torch.no_grad():
			state, action = map(self.to_tensor, [state, action])
			if self.discrete: action = one_hot(action)
			if self.state is None: self.state = state
			state_dot = state-self.state
			self.state = self.dynamics(action, state, state_dot)
			reward = self.reward(action, state, self.state).squeeze(-1)
		return [x.cpu().numpy() if numpy else x for x in [self.state, reward]]

	def reset(self, batch_size=None, **kwargs):
		self.dynamics.reset(self.device, batch_size)
		self.state = None

	def rollout(self, actions, states):
		states, actions = map(lambda x: self.to_tensor(x).transpose(0,1), [states, actions])
		next_states = []
		rewards = []
		self.reset(batch_size=states.shape[1])
		for action, state in zip(actions, states):
			next_state, reward = self.step(action, state, grad=True)
			next_states.append(next_state)
			rewards.append(reward)
		next_states, rewards = map(lambda x: torch.stack(x,1), [next_states, rewards])
		return next_states, rewards

	def get_loss(self, states, actions, next_states, rewards, dones):
		s, a, ns, r = map(self.to_tensor, (states, actions, next_states, rewards))
		next_states_hat, rewards_hat = self.rollout(a, s)
		dyn_loss = (next_states_hat - ns).pow(2).sum(-1).mean()
		rew_loss = (rewards_hat - r).pow(2).mean()
		self.stats.mean(dyn_loss=dyn_loss, rew_loss=rew_loss)
		return dyn_loss + rew_loss

	def optimize(self, states, actions, next_states, rewards, dones):
		loss = self.get_loss(states, actions, next_states, rewards, dones)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss

	def schedule(self, test_loss):
		self.scheduler.step(test_loss)

	def save_model(self, dirname="pytorch", name="best", net=None):
		filepath, _ = self.get_checkpoint_path(dirname, name, net)
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		torch.save(self.state_dict(), filepath)
		
	def load_model(self, dirname="pytorch", name="best", net=None):
		filepath, _ = self.get_checkpoint_path(dirname, name, net)
		if os.path.exists(filepath):
			self.load_state_dict(torch.load(filepath, map_location=self.device))
			print(f"Loaded DFRNTL model at {filepath}")
		return self
