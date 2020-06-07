import os
import torch
import numpy as np
from src.utils.misc import load_module
from ...agents.base import PTNetwork, one_hot

class TransitionModel(torch.nn.Module):
	def __init__(self, state_size, action_size, config):
		super().__init__()
		self.config = config
		self.gru = torch.nn.GRUCell(action_size[-1] + 2*state_size[-1], config.DYN.TRANSITION_HIDDEN)
		self.linear1 = torch.nn.Linear(config.DYN.TRANSITION_HIDDEN, config.DYN.TRANSITION_HIDDEN)
		self.drop1 = torch.nn.Dropout(p=0.5)
		self.linear2 = torch.nn.Linear(config.DYN.TRANSITION_HIDDEN, config.DYN.TRANSITION_HIDDEN)
		self.drop2 = torch.nn.Dropout(p=0.5)
		self.state_ddot = torch.nn.Linear(config.DYN.TRANSITION_HIDDEN, state_size[-1])
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, action, state, state_dot):
		input_dim = action.shape[:-1]
		hidden = self.hidden.view(np.prod(input_dim),-1)
		inputs = torch.cat([action, state, state_dot],-1)
		hidden = self.gru(inputs.view(np.prod(input_dim),-1), hidden).view(*input_dim,-1)
		linear1 = self.linear1(hidden).relu() + hidden
		linear1 = self.drop1(linear1)
		linear2 = self.linear2(linear1).relu() + linear1
		linear2 = self.drop2(linear2)
		state_ddot = self.state_ddot(linear2)
		state_dot = state_dot + state_ddot
		next_state = state + state_dot
		self.hidden = hidden
		return next_state, state_dot, state_ddot

	def reset(self, device, batch_size=None, train=False):
		self.train() if train else self.eval()
		if batch_size is None: batch_size = self.hidden[0].shape[1:2] if hasattr(self, "hidden") else [1]
		self.hidden = torch.zeros(*batch_size, self.config.DYN.TRANSITION_HIDDEN, device=device)

class RewardModel(torch.nn.Module):
	def __init__(self, state_size, action_size, config):
		super().__init__()
		self.cost = load_module(config.REWARD_MODEL)() if config.get("REWARD_MODEL") else None
		self.dyn_spec = load_module(config.DYNAMICS_SPEC) if config.get("DYNAMICS_SPEC") else None
		self.linear1 = torch.nn.Linear(action_size[-1] + 2*state_size[-1], config.DYN.REWARD_HIDDEN)
		self.drop1 = torch.nn.Dropout(p=0.5)
		self.linear2 = torch.nn.Linear(config.DYN.REWARD_HIDDEN, config.DYN.REWARD_HIDDEN)
		self.drop2 = torch.nn.Dropout(p=0.5)
		self.linear3 = torch.nn.Linear(config.DYN.REWARD_HIDDEN, config.DYN.REWARD_HIDDEN)
		self.linear4 = torch.nn.Linear(config.DYN.REWARD_HIDDEN, 1)
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, action, state, next_state, grad=False):
		if self.cost and self.dyn_spec and not grad:
			next_state, state = [x.cpu().numpy() for x in [next_state, state]]
			ns_spec, s_spec = map(self.dyn_spec.observation_spec, [next_state, state])
			reward = -torch.FloatTensor(self.cost.get_cost(ns_spec, s_spec)).unsqueeze(-1)
		else:
			inputs = torch.cat([action, state, next_state],-1)
			layer1 = self.linear1(inputs).relu()
			layer1 = self.drop1(layer1)
			layer2 = self.linear2(layer1).tanh() + layer1
			layer2 = self.drop2(layer2)
			layer3 = self.linear3(layer2).tanh() + layer1
			reward = self.linear4(layer3)
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
		self.optimizer = torch.optim.Adam(self.parameters(), lr=config.DYN.LEARN_RATE, weight_decay=config.DYN.REG_LAMBDA)
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=config.DYN.FACTOR, patience=config.DYN.PATIENCE)
		self.to(self.device)
		if load: self.load_model(load)

	def step(self, action, state=None, numpy=False, grad=False):
		action, state = map(self.to_tensor, [action, state])
		state = (self.state if state is None else state)[...,:self.dyn_index]
		with torch.enable_grad() if grad else torch.no_grad():
			if self.discrete: action = one_hot(action)
			if self.state is None: self.state = state
			state_dot = self.state_dot
			self.state, self.state_dot, self.state_ddot = self.dynamics(action, state, state_dot)
			reward = self.reward(action.detach(), state.detach(), self.state.detach(), grad=grad)
		return [x.cpu().numpy() if numpy else x for x in [self.state, reward.to(self.device)]]

	def reset(self, batch_size=None, state=None, train=False, **kwargs):
		self.train() if train else self.eval()
		self.dynamics.reset(self.device, batch_size, train=train)
		self.state = self.to_tensor(state)[...,:self.dyn_index] if state is not None else None
		self.state_dot = torch.zeros_like(self.state) if state is not None else None

	def rollout(self, actions, state, timedim=-2, numpy=False, grad=False):
		self.reset(batch_size=state.shape[:-len(self.state_size)], state=state, train=grad)
		actions = self.to_tensor(actions)
		next_states = []
		states_dot = []
		states_ddot = []
		rewards = []
		for action in actions.split(1, dim=timedim):
			next_state, reward = self.step(action.squeeze(timedim), grad=grad)
			next_states.append(next_state)
			states_dot.append(self.state_dot)
			states_ddot.append(self.state_ddot)
			rewards.append(reward)
		next_states, states_dot, states_ddot, rewards = map(lambda x: torch.stack(x,timedim), [next_states, states_dot, states_ddot, rewards])
		if numpy: next_states, states_dot, states_ddot, rewards = map(lambda x: x.cpu().numpy(), [next_states, states_dot, states_ddot, rewards])
		return (next_states, states_dot, states_ddot), rewards.squeeze(-1)

	def get_loss(self, states, actions, next_states, rewards, dones):
		s, a, ns, r = map(self.to_tensor, (states, actions, next_states, rewards))
		s, ns = [x[...,:self.dyn_index] for x in [s, ns]]
		ns_dot = (ns-s)
		s_dot = torch.cat([ns_dot[:,0:1,:], ns_dot[:,:-1,:]], -2)
		next_states, states_dot, states_ddot = self.rollout(a, s[...,0,:], grad=True)[0]
		rewards = self.reward(a, s, ns, grad=True).squeeze(-1)
		dyn_loss = (next_states - ns).pow(2).sum(-1).mean()
		dot_loss = (states_dot - ns_dot).pow(2).sum(-1).mean()
		ddot_loss = (states_ddot - (ns_dot - s_dot)).pow(2).sum(-1).mean()
		rew_loss = (rewards - r).pow(2).mean()
		self.stats.mean(dyn_loss=dyn_loss, dot_loss=dot_loss, ddot_loss=ddot_loss, rew_loss=rew_loss)
		return self.config.DYN.BETA_DYN*dyn_loss + self.config.DYN.BETA_DOT*dot_loss + self.config.DYN.BETA_DDOT*ddot_loss + rew_loss

	def optimize(self, states, actions, next_states, rewards, dones):
		loss = self.get_loss(states, actions, next_states, rewards, dones)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.item()

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
			try:
				self.load_state_dict(torch.load(filepath, map_location=self.device))
				print(f"Loaded DFRNTL model at {filepath}")
			except:
				print(f"Error loading DFRNTL model at {filepath}")
		return self