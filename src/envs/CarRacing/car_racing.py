import os
import sys
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from gym_wrapper import UnityToGymWrapper
sys.path.append(os.path.abspath("./src/envs/CarRacing/objective"))
from cost import CostModel
np.set_printoptions(precision=3, sign=' ', floatmode="fixed", suppress=True, linewidth=100)

root = os.path.dirname(os.path.abspath(__file__))
sim_file = os.path.abspath(os.path.join(root, "simulator", sys.platform, "CarRacing"))

unity_env = UnityEnvironment(file_name=sim_file)
env = UnityToGymWrapper(unity_env)

action_size = env.action_space.shape
state_size = env.observation_space.shape
cmodel = CostModel()

for ep in range(1):
	state = env.reset()
	done = False
	step = 0
	while not done and step<1000:
		action = np.random.uniform([-1, -1, -1], [1, 1, 1], size=action_size)
		state, reward, done, _ = env.step(action)
		reward = cmodel.get_cost(state[:3])
		print(f"Step: {step:5d}, R: {reward}, A: {action}, Pos: {state[:3]}, Idle: {state[-1]}")
		env.render()
		step += 1

env.close()