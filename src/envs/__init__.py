import os
import sys
import numpy as np
from .wrappers import GymEnv, gym
from collections import OrderedDict

gym_types = ["classic_control", "box2d", "mujoco", "robotics"]
env_grps = OrderedDict(
	gym_cct = tuple([env_spec.id for env_spec in gym.envs.registry.all() if type(env_spec.entry_point)==str and any(x in env_spec.entry_point for x in [gym_types[0]])]),
	gym_b2d = tuple([env_spec.id for env_spec in gym.envs.registry.all() if type(env_spec.entry_point)==str and any(x in env_spec.entry_point for x in [gym_types[1]])]),
	# gym_mjc = tuple([env_spec.id for env_spec in gym.envs.registry.all() if type(env_spec.entry_point)==str and any(x in env_spec.entry_point for x in gym_types[2:])]),
	gym = tuple([env_spec.id for env_spec in gym.envs.registry.all() if type(env_spec.entry_point)==str and any(x in env_spec.entry_point for x in gym_types[:2])]),
	unt = tuple([env_spec.id for env_spec in gym.envs.registry.all() if type(env_spec.entry_point)!=str]),
)

def get_group(env_name):
	for group, envs in reversed(env_grps.items()):
		if env_name in envs: return group
	return None

def get_names(groups):
	names = []
	for group in groups:
		names.extend(env_grps.get(group, []))
	return names

def make_env(cls, env_name):
	return lambda **kwargs: cls(env_name, **kwargs)

all_envs = get_names(["gym", "unt"])

def get_env(env_name, render=False):
	return GymEnv(gym.make(env_name))
