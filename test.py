import os
import sys
import numpy as np
from types import SimpleNamespace
from src.envs import get_env
from src.models.input import InputController
from src.envs.CarRacing.car_racing import CarRacing
from src.models.pytorch.mpc import MPPIController, RealEnv
from src.utils.envs import get_space_size

np.set_printoptions(precision=3, sign=' ', floatmode="fixed", suppress=True, linewidth=100)

def test_car_sim():
	env = CarRacing(max_time=500)
	action_size = env.action_space.shape
	state_size = env.observation_space.shape
	agent = InputController(state_size, action_size)
	for _ in range(1):
		state = env.reset(train=False)
		done = False
		step = 0
		while not done:
			# action = np.random.uniform([-1, -1, -1], [1, 1, 1], size=action_size)
			action = agent.get_action(state)
			state, reward, done, _ = env.step(action)
			print(f"Step: {step:5d}, R: {reward}, A: {action}, Pos: {state[:3]}, Vel: {state[3:6]}, Idle: {state[-1]}")
			env.render()
			step += 1
	env.close()

def test_mppi():
	config = SimpleNamespace(env_name="CartPole-v0", MPC=SimpleNamespace(NSAMPLES=50, HORIZON=40, LAMBDA=0.5))
	env = get_env(config.env_name)
	state_size = get_space_size(env.observation_space)
	action_size = get_space_size(env.action_space)
	agent = MPPIController(state_size, action_size, RealEnv, config)
	envmodel = agent.envmodel
	state = envmodel.state
	for s in range(200):
		action = agent.get_action(state)
		envmodel.reset(initstate=False)
		state, cost = envmodel.step(action, state)
		print(f"Step: {s:5d}, Action: {action}, Cost: {cost}")
		envmodel.env.render()


if __name__ == "__main__":
	test_mppi()