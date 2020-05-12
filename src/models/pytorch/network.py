import os
import torch
import random
import numpy as np

class Conv(torch.nn.Module):
	def __init__(self, state_size, output_size):
		super().__init__()
		self.conv1 = torch.nn.Conv2d(state_size[-1], 32, kernel_size=4, stride=2)
		self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2)
		self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=2)
		self.linear1 = torch.nn.Linear(self.get_conv_output(state_size), output_size)
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, state):
		out_dims = state.size()[:-3]
		state = state.view(-1, *state.size()[-3:])
		state = self.conv1(state).tanh()
		state = self.conv2(state).tanh() 
		state = self.conv3(state).tanh() 
		state = self.conv4(state).tanh() 
		state = state.view(state.size(0), -1)
		state = self.linear1(state).tanh()
		state = state.view(*out_dims, -1)
		return state

	def get_conv_output(self, state_size):
		inputs = torch.randn(1, state_size[-1], *state_size[:-1])
		output = self.conv4(self.conv3(self.conv2(self.conv1(inputs))))
		return np.prod(output.size())

class Linear(torch.nn.Module):
	def __init__(self, input_size, output_size, nlayers=4):
		super().__init__()
		sizes = np.linspace(input_size, output_size, nlayers).astype(np.int32)
		self.layers = torch.nn.ModuleList([torch.nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
		self.output = torch.nn.Linear(sizes[-1], output_size)
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, x):
		for layer in self.layers:
			x = layer(x).relu()
		output = self.output(x)
		return output

class NoisyLinear(torch.nn.Linear):
	def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
		super().__init__(in_features, out_features, bias=bias)
		self.sigma_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
		self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
		if bias:
			self.sigma_bias = torch.nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))
			self.register_buffer("epsilon_bias", torch.zeros(out_features))
		self.reset_parameters()

	def reset_parameters(self):
		std = math.sqrt(3 / self.in_features)
		torch.nn.init.uniform_(self.weight, -std, std)
		torch.nn.init.uniform_(self.bias, -std, std)

	def forward(self, input):
		torch.randn(self.epsilon_weight.size(), out=self.epsilon_weight)
		bias = self.bias
		if bias is not None:
			torch.randn(self.epsilon_bias.size(), out=self.epsilon_bias)
			bias = bias + self.sigma_bias * torch.autograd.Variable(self.epsilon_bias)
		weight = self.weight + self.sigma_weight * torch.autograd.Variable(self.epsilon_weight)
		return torch.nn.functional.linear(input, weight, bias)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
	""" Draw a sample from the Gumbel-Softmax distribution"""
	y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
	return torch.nn.functional.softmax(y / temperature, dim=-1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gsoftmax(logits, temperature=1.0, hard=True):
	"""Sample from the Gumbel-Softmax distribution and optionally discretize.
	Args:
	logits: [batch_size, n_class] unnormalized log-probs
	temperature: non-negative scalar
	hard: if True, take argmax, but differentiate w.r.t. soft sample y
	Returns:
	[batch_size, n_class] sample from the Gumbel-Softmax distribution.
	If hard=True, then the returned sample will be one-hot, otherwise it will
	be a probabilitiy distribution that sums to 1 across classes
	"""
	y = gumbel_softmax_sample(logits, temperature)
	if hard:
		y_hard = one_hot(y)
		y = (y_hard - y).detach() + y
	return y

def one_hot(logits):
	return (logits == logits.max(-1, keepdim=True)[0]).float().to(logits.device)

def one_hot_from_indices(indices, depth, keepdims=False):
	y_onehot = torch.zeros([*indices.shape, depth]).to(indices.device)
	y_onehot.scatter_(-1, indices.unsqueeze(-1).long(), 1)
	return y_onehot.float() if keepdims else y_onehot.squeeze(-2).float()
