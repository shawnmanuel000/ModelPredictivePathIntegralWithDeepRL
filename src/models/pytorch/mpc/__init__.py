from src.utils.config import Config
from src.utils.envs import get_space_size
from src.utils.misc import load_module
from .envmodel import MDRNNEnv, DifferentialEnv, RealEnv
from .mppi import MPPIController

all_envmodels = {
	"real":RealEnv,
	"mdrnn": MDRNNEnv,
	"dfrntl": DifferentialEnv
}

envmodel_config = Config(
	REG_LAMBDA = 1e-6,             	# Penalty multiplier to apply for the size of the network weights
	FACTOR = 0.5,
	PATIENCE = 5,
	LEARN_RATE = 0.001,
)

dynamics_configs = {
	"mdrnn": envmodel_config.clone(
		HIDDEN_SIZE = 32,
		NGAUSS = 1,
	),
	"dfrntl": envmodel_config.clone(
		TRANSITION_HIDDEN = 256,
		REWARD_HIDDEN = 256,
		BETA_DYN = 0.1,
		BETA_DOT = 1,
		BETA_DDOT = 1,
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

	def reset(self, **kwargs):
		self.network.reset(**kwargs)