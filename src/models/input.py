import numpy as np
import keyboard as kbd

class InputController():
	def __init__(self, state_size, action_size, config=None):
		self.state_size = state_size
		self.action_size = action_size

	def get_action(self, state):
		action = np.zeros(self.action_size)
		if kbd.is_pressed(kbd.KEY_UP):
			action[1] = 1
		if kbd.is_pressed("left"):
			action[0] = -1
		elif kbd.is_pressed("right"):
			action[0] = 1
		if kbd.is_pressed(kbd.KEY_DOWN):
			action[2] = 1
		return action