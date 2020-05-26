from src.utils.config import Config
from .envmodel import MDRNNEnv, DifferentialEnv, RealEnv
from .mppi import MPPIController

all_envmodels = {
	"real":RealEnv,
	"mdrnn": MDRNNEnv,
	"dfrntl": DifferentialEnv
}

envmodel_config = Config(
	FACTOR = 0.5,
	PATIENCE = 5,
	LEARN_RATE = 0.0005,
	MPC = Config(
		NSAMPLES=500, 
		HORIZON=20, 
		LAMBDA=0.5
	)
)

dynamics_configs = {
	"real": envmodel_config.clone(),
	"mdrnn": envmodel_config.clone(
		HIDDEN_SIZE = 32,
		NGAUSS = 1,
	)
}

def get_envmodel(state_size, action_size, config, load="", gpu=True):
	dyn_config = dynamics_configs.get(config.envmodel, envmodel_config)
	config.update(DYN=dyn_config)
	return all_envmodels[config.envmodel](state_size, action_size, config, load=load, gpu=gpu)

class EnvModel():
	def __init__(self, state_size, action_size, config, load="", gpu=True):
		self.network = get_envmodel(state_size, action_size, config, load=load, gpu=gpu)
		self.state_size = state_size
		self.action_size = action_size

	def step(self, action, state, **kwargs):
		next_state, reward = self.network.step(action, state, **kwargs)
		return next_state, reward

	def reset(self, **kwargs):
		self.network.reset(**kwargs)