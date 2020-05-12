import os
import sys
import numpy as np
from src.models.input import InputController
from src.envs.CarRacing.car_racing import CarRacing

np.set_printoptions(precision=3, sign=' ', floatmode="fixed", suppress=True, linewidth=100)

def test():
	env = CarRacing(max_time=1000)
	action_size = env.action_space.shape
	state_size = env.observation_space.shape
	agent = InputController(state_size, action_size)
	for _ in range(1):
		state = env.reset()
		done = False
		step = 0
		while not done:
			# action = np.random.uniform([-1, -1, -1], [1, 1, 1], size=action_size)
			action = agent.get_action(state)
			state, reward, done, _ = env.step(action)
			print(f"Step: {step:5d}, R: {reward}, A: {action}, Pos: {state[:3]}, Idle: {state[-1]}")
			env.render()
			step += 1
	env.close()

if __name__ == "__main__":
	test()