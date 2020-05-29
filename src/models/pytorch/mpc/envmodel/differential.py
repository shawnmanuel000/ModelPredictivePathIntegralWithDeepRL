import os
import torch
import numpy as np
from src.utils.misc import load_module
from ...agents.base import PTNetwork, one_hot

class TransitionModel(torch.nn.Module):
	def __init__(self, state_size, action_size, config):
		super().__init__()
		self.gru = torch.nn.GRUCell(action_size[-1] + 2*state_size[-1], config.DYN.TRANSITION_HIDDEN)
		self.linear1 = torch.nn.Linear(config.DYN.TRANSITION_HIDDEN, config.DYN.TRANSITION_HIDDEN)
		self.linear2 = torch.nn.Linear(config.DYN.TRANSITION_HIDDEN, config.DYN.TRANSITION_HIDDEN)
		self.linear3 = torch.nn.Linear(config.DYN.TRANSITION_HIDDEN, state_size[-1])
		self.config = config

	def forward(self, action, state, state_dot):
		inputs = torch.cat([action, state, state_dot],-1)
		self.hidden = self.gru(inputs, self.hidden)
		linear1 = self.linear1(self.hidden).tanh() + self.hidden
		linear2 = self.linear2(linear1).tanh() + linear1
		state_dot = self.linear3(linear2)
		next_state = state + state_dot
		return next_state, state_dot

	def reset(self, device, batch_size=None):
		if batch_size is None: batch_size = self.hidden[0].shape[1] if hasattr(self, "hidden") else 1
		self.hidden = torch.zeros(batch_size, self.config.DYN.TRANSITION_HIDDEN, device=device)

class RewardModel(torch.nn.Module):
	def __init__(self, state_size, action_size, config):
		super().__init__()
		self.linear = torch.nn.Linear(2*state_size[-1], 1)
		self.cost = load_module(config.REWARD_MODEL)() if config.get("REWARD_MODEL") else None
		self.dyn_spec = load_module(config.DYNAMICS_SPEC) if config.get("DYNAMICS_SPEC") else None

	def forward(self, next_state, state_dot):
		if self.cost and self.dyn_spec:
			next_state, state_dot = [x.cpu().numpy() for x in [next_state, state_dot]]
			ns_spec, s_spec = map(self.dyn_spec.observation_spec, [next_state, next_state-state_dot])
			reward = -torch.FloatTensor(self.cost.get_cost(ns_spec, s_spec)).unsqueeze(-1)
		else:
			inputs = torch.cat([next_state, state_dot],-1)
			reward = self.linear(inputs)
		return reward

class DifferentialEnv(PTNetwork):
	def __init__(self, state_size, action_size, config, load="", gpu=True, name="dfrntl"):
		super().__init__(config, gpu, name)
		self.state_size = state_size
		self.action_size = action_size
		self.discrete = type(self.action_size) != tuple
		self.dyn_index = config.get("dynamics_size", state_size[-1])
		self.reward = RewardModel([self.dyn_index], action_size, config)
		self.dynamics = TransitionModel([self.dyn_index], action_size, config)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=config.DYN.LEARN_RATE)
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=config.DYN.FACTOR, patience=config.DYN.PATIENCE)
		self.to(self.device)
		if load: self.load_model(load)

	def step(self, action, state=None, numpy=False, grad=False):
		action, state = map(self.to_tensor, [action, state])
		state = (self.state if state is None else state)[:,:self.dyn_index]
		with torch.enable_grad() if grad else torch.no_grad():
			state, action = map(self.to_tensor, [state, action])
			if self.discrete: action = one_hot(action)
			if self.state is None: self.state = state
			self.state, self.state_dot = self.dynamics(action, state, self.state_dot)
			reward = self.reward(self.state.detach(), self.state_dot.detach()).squeeze(-1)
		return [x.cpu().numpy() if numpy else x for x in [self.state, reward]]

	def reset(self, batch_size=None, state=None, **kwargs):
		self.dynamics.reset(self.device, batch_size)
		self.state = self.to_tensor(state)[:,:self.dyn_index] if state is not None else None
		self.state_dot = torch.zeros_like(self.state) if state is not None else None

	def rollout(self, actions, state):
		self.reset(batch_size=len(state), state=state)
		actions = self.to_tensor(actions).transpose(0,1)
		next_states = []
		states_dot = []
		rewards = []
		for action in actions:
			next_state, reward = self.step(action, self.state, grad=True)
			next_states.append(next_state)
			states_dot.append(self.state_dot)
			rewards.append(reward)
		next_states, states_dot, rewards = map(lambda x: torch.stack(x,1), [next_states, states_dot, rewards])
		return (next_states, states_dot), rewards

	def get_loss(self, states, actions, next_states, rewards, dones):
		s, a, ns, r = map(self.to_tensor, (states, actions, next_states, rewards))
		s, ns = [x[:,:,:self.dyn_index] for x in [s, ns]]
		(next_states, states_dot), rewards = self.rollout(a, s[:,0])
		dyn_loss = (next_states - ns).pow(2).mean(-1).mean()
		dot_loss = (states_dot - (ns-s)).pow(2).sum(-1).mean()
		rew_loss = (rewards - r).pow(2).mean()
		self.stats.mean(dyn_loss=dyn_loss, dot_loss=dot_loss, rew_loss=rew_loss)
		return self.config.DYN.BETA_DYN*dyn_loss + self.config.DYN.BETA_DOT*dot_loss + self.config.DYN.BETA_REW*rew_loss

	def optimize(self, states, actions, next_states, rewards, dones):
		loss = self.get_loss(states, actions, next_states, rewards, dones)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss

	def schedule(self, test_loss):
		self.scheduler.step(test_loss)

	def get_stats(self):
		return {**super().get_stats(), "lr": self.optimizer.param_groups[0]["lr"] if self.optimizer else None}

	def save_model(self, dirname="pytorch", name="checkpoint", net=None):
		filepath, _ = self.get_checkpoint_path(dirname, name, net)
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		torch.save(self.state_dict(), filepath)
		
	def load_model(self, dirname="pytorch", name="checkpoint", net=None):
		filepath, _ = self.get_checkpoint_path(dirname, name, net)
		if os.path.exists(filepath):
			self.load_state_dict(torch.load(filepath, map_location=self.device))
			print(f"Loaded DFRNTL model at {filepath}")
		return self