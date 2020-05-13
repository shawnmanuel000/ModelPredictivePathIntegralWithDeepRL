import numpy as np
from src.utils.rand import RandomAgent
from src.envs import get_env, all_envs, env_grps
from .pytorch import PPOAgent, SACAgent, DDQNAgent, DDPGAgent, MPOAgent
from .rllib import rPPOAgent, rSACAgent, rDDPGAgent, rDDQNAgent

all_models = {
	"pt": {
		"ddpg":DDPGAgent, 
		"ppo":PPOAgent, 
		"sac":SACAgent, 
		"ddqn":DDQNAgent, 
		"rand":RandomAgent,
		"mpo":MPOAgent
	},
}

if not None in [rPPOAgent, rSACAgent, rDDPGAgent, rDDQNAgent]:
	all_models.update({"rl": {
		"ppo":rPPOAgent, 
		"sac":rSACAgent,
		"ddpg":rDDPGAgent,
		"ddqn":rDDQNAgent
	}})

class Config(object):
	def __init__(self, **kwargs):
		self.update(**kwargs)

	def props(self):   
		return {k:v for k,v in self.__dict__.items() if k[:1] != '_'}

	def update(self, **kwargs):
		for k,v in kwargs.items():
			if hasattr(self, k) and isinstance(k, Config):
				getattr(self, k).update(**v.props())
			else:
				setattr(self, k, v)
		return self

	def get(self, key, default=None):
		return getattr(self, key, default)

	def clone(self, **kwargs):
		return self.__class__(**self.props()).update(**kwargs)

	def print(self, level=1):
		return "".join([f"\n{'   '*level}{k} = {v.print(level+1) if isinstance(v,Config) else v}" for k,v in self.__dict__.items()])

net_config = Config(
	REG_LAMBDA = 1e-6,             	# Penalty multiplier to apply for the size of the network weights
	LEARN_RATE = 0.0001,           	# Sets how much we want to update the network weights at each training step
	DISCOUNT_RATE = 0.99,			# The discount rate to use in the Bellman Equation
	ADVANTAGE_DECAY = 0.95,			# The discount factor for the cumulative GAE calculation
	INPUT_LAYER = 512,				# The number of output nodes from the first layer to Actor and Critic networks
	ACTOR_HIDDEN = 256,				# The number of nodes in the hidden layers of the Actor network
	CRITIC_HIDDEN = 1024,			# The number of nodes in the hidden layers of the Critic networks

	EPS_MAX = 1.0,                 	# The starting proportion of random to greedy actions to take
	EPS_MIN = 0.1,               	# The lower limit proportion of random to greedy actions to take
	EPS_DECAY = 0.995,             	# The rate at which eps decays from EPS_MAX to EPS_MIN
	NUM_STEPS = 500,				# The number of steps to collect experience in sequence for each GAE calculation
	MAX_BUFFER_SIZE = 100000,    	# Sets the maximum length of the replay buffer
	REPLAY_BATCH_SIZE = 32,        	# How many experience tuples to sample from the buffer for each train step
	TARGET_UPDATE_RATE = 0.0004,   	# How frequently we want to copy the local network to the target network (for double DQNs)
)

model_configs = {
	"ppo": net_config.clone(
		BATCH_SIZE = 32,				# Number of samples to train on for each train step
		PPO_EPOCHS = 2,					# Number of iterations to sample batches for training
		ENTROPY_WEIGHT = 0.005,			# The weight for the entropy term of the Actor loss
		CLIP_PARAM = 0.05,				# The limit of the ratio of new action probabilities to old probabilities
	),
}

env_model_configs = {
	env_grps["gym_cct"]: {
		"ddpg": net_config.clone(
		),
		"ddqn": net_config.clone(
		),
		"sac": net_config.clone(
		),
	},
	env_grps["gym_b2d"]: {
		"ddpg": net_config.clone(
		),
		"ddqn": net_config.clone(
		),
		"sac": net_config.clone(
		),
	}
}

train_config = Config(
	TRIAL_AT = 1000,					# Number of steps between each evaluation rollout
	SAVE_AT = 10, 						# Number of evaluation rollouts between each save weights
)

env_configs = {
	None: train_config,
	env_grps["gym"]: train_config,
}

def get_config(env_name, model_name, framework="pt", render=False):
	assert env_name in all_envs, "Env name not found"
	env_list = [x for x in env_grps.values() if env_name in x][0]
	env_config = env_configs.get(env_list, train_config)
	model = all_models[framework][model_name]
	model_config = env_model_configs.get(env_list, model_configs).get(model_name, model_configs.get(model_name, net_config))
	make_env = lambda: get_env(env_name, render)
	return make_env, model, env_config.update(**model_config.props(), env_name=env_name)
