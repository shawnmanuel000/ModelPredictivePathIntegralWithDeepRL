import os
import torch
import numpy as np
from ...agents.base import PTNetwork, one_hot

class TransitionModel(torch.nn.Module):
	def __init__(self, state_size, action_size, config):
		super().__init__()
		self.gru = torch.nn.GRUCell(action_size[-1], 2*state_size[-1])
		self.linear = torch.nn.Linear(2*state_size[-1], state_size[-1])

	def forward(self, action, state):
		basis = torch.cat([state, state.pow(2)],-1)
		hidden = self.gru(action, basis)
		state_diff = self.linear(hidden)
		next_state = state + state_diff
		return next_state

class RewardModel(torch.nn.Module):
	def __init__(self, state_size, action_size, config):
		super().__init__()
		self.linear = torch.nn.Linear(action_size[-1] + 2*state_size[-1], 1)

	def forward(self, action, state, next_state):
		inputs = torch.cat([state, next_state-state],-1)
		reward = self.linear(inputs)
		return reward

class DifferentialEnv(PTNetwork):
	def __init__(self, state_size, action_size, config, load="", gpu=True, name="dfntl"):
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

	def step(self, action, state):
		if self.discrete: actions = one_hot(actions)
		next_state = self.dynamics(action, state)
		reward = self.reward(action, state, next_state)
		return next_state, reward

	def reset(self, batch_size=None, state=None, **kwargs):
		if batch_size is None: batch_size = self.hidden[0].shape[1] if hasattr(self, "hidden") else 1
		if state is None: state = np.zeros(self.state_size)
		self.hidden = [self.to_tensor(state).view(1, 1, -1).expand(1, batch_size, 1) for _ in range(1)]

	def rollout(self, actions, states):
		states, actions = map(self.to_tensor, [states, actions])
		next_states = []
		rewards = []
		for action, state in zip(actions, states):
			next_state, reward = self.step(action, state)
			next_states.append(next_state)
			rewards.append(reward)
		next_states, rewards = map(torch.stack, [next_states, rewards])
		return next_states, rewards

	def get_loss(self, states, actions, next_states, rewards, dones):
		s, a, ns, r = map(self.to_tensor, (states, actions, next_states, rewards))
		next_states_hat, rewards_hat = self.rollout(a, s)
		dyn_loss = (next_states_hat - ns).pow(2).sum(-1)
		rew_loss = (rewards_hat - r).pow(2).sum(-1)
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
			print(f"Loaded DFNTL model at {filepath}")
		return self

class MDRNNCell(torch.nn.Module):
	def __init__(self, state_size, action_size, config, load="", gpu=True):
		super().__init__()
		self.state_size = state_size
		self.action_size = action_size
		self.n_gauss = config.DYN.NGAUSS
		self.discrete = type(self.action_size) == list
		self.lstm = torch.nn.LSTMCell(action_size[-1] + state_size, state_size)
		self.gmm = torch.nn.Linear(state_size, (2*state_size+1)*self.n_gauss + 2)
		self.device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
		self.to(self.device)
		if load: self.load_model(load)

	def forward(self, actions, states, hiddens):
		with torch.no_grad():
			actions, states = [x.to(self.device) for x in (torch.from_numpy(actions), states)]
			lstm_inputs = torch.cat([actions, states], dim=-1)
			lstm_hidden = self.lstm(lstm_inputs, hiddens)
			return lstm_hidden

	def step(self, hiddens):
		with torch.no_grad():
			gmm_out = self.gmm(hiddens)
			stride = self.n_gauss*self.state_size
			mus = gmm_out[:,:stride]
			sigs = gmm_out[:,stride:2*stride].exp()
			pi = gmm_out[:,2*stride:2*stride+self.n_gauss].softmax(dim=-1)
			rs = gmm_out[:,2*stride+self.n_gauss]
			ds = gmm_out[:,2*stride+self.n_gauss+1].sigmoid()
			mus = mus.view(-1, self.n_gauss, self.state_size)
			sigs = sigs.view(-1, self.n_gauss, self.state_size)
			dist = torch.distributions.categorical.Categorical(pi)
			indices = dist.sample()
			mus = mus[:,indices,:].squeeze(1)
			sigs = sigs[:,indices,:].squeeze(1)
			next_states = mus + torch.randn_like(sigs).mul(sigs)
			return next_states, rs

	def reset(self, batch_size=1):
		return [torch.zeros(batch_size, self.state_size).to(self.device) for _ in range(2)]

	def load_model(self, dirname="pytorch", name="best"):
		filepath = get_checkpoint_path(dirname, name)
		if os.path.exists(filepath):
			self.load_state_dict({k.replace("_l0",""):v for k,v in torch.load(filepath, map_location=self.device).items()})
			print(f"Loaded MDRNNCell model at {filepath}")
		return self

def get_checkpoint_path(self, dirname="pytorch", name="checkpoint", net=None):
	net_path = os.path.join("./logging/saved_models", self.name if net is None else net, dirname)
	filepath = os.path.join(net_path, f"{name}.pth")
	return filepath, net_path