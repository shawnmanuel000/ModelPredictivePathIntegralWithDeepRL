from src.utils.config import Config
from .envmodel import MDRNNEnv, RealEnv
from .mppi import MPPIController

all_envmodels = {
	"real":RealEnv,
	"mdrnn": MDRNNEnv
}

dynamics_configs = {
	"real": {},
	"mdrnn": Config(
		HIDDEN_SIZE = 32,
		NGAUSS = 1,
		FACTOR = 0.5,
		PATIENCE = 10,
		LEARN_RATE = 0.001
	)
}

def get_envmodel(state_size, action_size, config):
	dyn_config = dynamics_configs[config.envmodel]
	config.update(DYN=dyn_config)
	return all_envmodels[config.envmodel](state_size, action_size, config)

class EnvModel():
	def __init__(self, state_size, action_size, config, load="", gpu=True):
		self.network = get_envmodel(state_size, action_size, config)
		self.state_size = state_size
		self.action_size = action_size

	def step(self, action, state):
		next_state, reward = self.network.step(action, state)
		return next_state, reward

	def reset(self, **kwargs):
		self.network.reset(**kwargs)