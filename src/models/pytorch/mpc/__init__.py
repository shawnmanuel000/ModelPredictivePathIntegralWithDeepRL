from src.utils.config import Config
from src.utils.envs import get_space_size
from src.utils.misc import load_module
from .envmodel import MDRNNEnv, DifferentialEnv, RealEnv

all_envmodels = {
	"real":RealEnv,
	"mdrnn": MDRNNEnv,
	"dfrntl": DifferentialEnv
}

envmodel_config = Config(
	REG_LAMBDA = 1e-6,             	# Penalty multiplier to apply for the size of the network weights
	FACTOR = 0.8,
	PATIENCE = 5,
	LEARN_RATE = 0.0001,
)

dynamics_configs = {
	"mdrnn": envmodel_config.clone(
		HIDDEN_SIZE = 32,
		NGAUSS = 1,
	),
	"dfrntl": envmodel_config.clone(
		TRANSITION_HIDDEN = 512,
		REWARD_HIDDEN = 256,
		BETA_DYN = 1,
		BETA_DOT = 0,
		BETA_DDOT = 0,
	)
}

def set_dynamics_size(config, make_env):
	env = make_env()
	state_size = get_space_size(env.observation_space)
	action_size = get_space_size(env.action_space)
	config.dynamics_size = getattr(env.unwrapped, "dynamics_size", state_size[-1])
	config.state_size = state_size
	config.action_size = action_size
	env.close()
	return config

def get_envmodel(state_size, action_size, config, load="", gpu=True):
	envmodel = config.get("envmodel", config.get("ENV_MODEL"))
	dyn_config = dynamics_configs.get(envmodel, envmodel_config)
	config.update(DYN=dyn_config)
	return all_envmodels[envmodel](state_size, action_size, config, load=load, gpu=gpu)

class EnvModel():
	def __init__(self, state_size, action_size, config, load="", gpu=True):
		self.network = get_envmodel(state_size, action_size, config, load=load, gpu=gpu)
		self.state_size = state_size
		self.action_size = action_size

	def step(self, action, state=None, **kwargs):
		next_state, reward = self.network.step(action, state, **kwargs)
		return next_state, reward

	def rollout(self, actions, state=None, **kwargs):
		next_states, rewards = self.network.rollout(actions, state, **kwargs)
		return next_states, rewards

	def reset(self, **kwargs):
		self.network.reset(**kwargs)

	def optimize(self, states, actions, next_states, rewards, dones):
		return self.network.optimize(states, actions, next_states, rewards, dones)

	def get_stats(self):
		return self.network.get_stats()

	def save_model(self, dirname="pytorch", name="checkpoint", net=None):
		return self.network.save_model(dirname, name, net)
		
	def load_model(self, dirname="pytorch", name="checkpoint", net=None):
		return self.network.load_model(dirname, name, net)