import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from src.envs import get_env
from src.models import RandomAgent, get_config
from src.models.input import InputController
from src.envs.CarRacing.car_racing import CarRacing
from src.models.pytorch.mpc import MPPIController, EnvModel, MDRNNEnv, RealEnv, envmodel_config, set_dynamics_size
from src.utils.envs import get_space_size
from src.utils.config import Config

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
			action = agent.get_action(state)
			state, reward, done, _ = env.step(action)
			print(f"Step: {step:5d}, R: {reward}, A: {action}, Pos: {state[:3]}, Vel: {state[3:6]}, Idle: {state[14]}")
			env.render()
			step += 1
	env.close()

class PathAnimator():
	def __init__(self, track):
		self.track = track
		plt.ion()
		plt.figure()
		self.ax = plt.axes(projection='3d')
		self.X, self.Y, self.Z = track.X, track.Y, track.Z

	def animate_path(self, trajectories, chosen=None):
		self.ax.cla()
		point = trajectories[0,0]
		X, Y, Z = map(lambda x: x, [self.X, self.Y, self.Z])
		self.ax.plot(X,Y,Z, color="#DDDDDD")
		self.ax.set_zlim3d(-100, 100)
		self.ax.set_xlim3d(point[0]-10, point[0]+10)
		self.ax.set_ylim3d(point[2]-10, point[2]+10)
		for path in trajectories:
			xs, zs, ys = path[:,0], path[:,1], path[:,2]
			self.ax.plot(xs, ys, zs, linewidth=0.2)
		if chosen is not None:
			x,y,z = chosen[0,:,0], chosen[0,:,1], chosen[0,:,2]
			self.ax.plot(x, y, z, color="black", linewidth=1)
		plt.draw()
		plt.pause(0.0000001)

def visualize_envmodel():
	make_env, model, config = get_config("CarRacing-v1", "mppi")
	config.MPC.update(NSAMPLES=100, HORIZON=20, LAMBDA=0.5, CONTROL_FREQ=1)
	env = make_env()
	state_size = get_space_size(env.observation_space)
	action_size = get_space_size(env.action_space)
	agent = MPPIController(state_size, action_size, EnvModel, config)
	envmodel = EnvModel(state_size, action_size, config, load=config.env_name)
	state = env.reset()
	envmodel.reset(batch_size=1, state=[state])
	animator = PathAnimator(env.unwrapped.cost_model.track)
	env.render()
	for s in range(500):
		action = agent.get_action(state)
		trajectories = np.stack(agent.states, 1)
		(path, states_dot), rewards = envmodel.network.rollout([agent.control], [state])
		envmodel.reset(batch_size=1, state=[state])
		spec, path_spec = map(CarRacing.observation_spec, (trajectories, path.detach().cpu().numpy()))
		animator.animate_path(spec["pos"], path_spec["pos"])
		env_action = RandomAgent.to_env_action(env.action_space, action)
		state, reward, done, _ = env.step(env_action)
		ns, r = envmodel.step([action], numpy=True)
		print(f"Step: {s:5d}, Action: {action}, Reward: {reward:5.2f} ({r[0]:5.2f}), Pos: {state[:3]} ({ns[0][:3]})")
		if done: break

def test_envmodel():
	config = envmodel_config.clone(env_name="Pendulum-v0", envmodel="dfrntl", MPC=Config(NSAMPLES=500, HORIZON=50, LAMBDA=0.5))
	env = get_env(config.env_name)
	state_size = get_space_size(env.observation_space)
	action_size = get_space_size(env.action_space)
	agent = MPPIController(state_size, action_size, EnvModel, config)
	state = env.reset()
	for s in range(400):
		action = agent.get_action(state)
		env_action = RandomAgent.to_env_action(env.action_space, action)
		state, reward, done, _ = env.step(env_action)
		agent.envmodel.reset(batch_size=1)
		ns, r = agent.envmodel.step([action], [state], numpy=True)
		print(f"Step: {s:5d}, Action: {action}, Reward: {reward:5.2f} ({r[0]:5.2f}), State: {state} ({ns[0]})")
		env.render()

class TestMPPIController(RandomAgent):
	def __init__(self, state_size, action_size, envmodel, config, gpu=True):
		self.mu = np.zeros(action_size)
		self.cov = np.diag(np.ones(action_size))
		self.icov = np.linalg.inv(self.cov)
		self.lamda = config.MPC.LAMBDA
		self.horizon = config.MPC.HORIZON
		self.nsamples = config.MPC.NSAMPLES
		self.envmodel = envmodel(state_size, action_size, config)
		self.control = np.random.uniform(-1, 1, [self.horizon, *action_size])
		self.noise = np.random.multivariate_normal(self.mu, self.cov, size=(self.nsamples, self.horizon))
		self.step = 0

	def get_action(self, state, eps=None, sample=True):
		self.step += 1
		if self.step%1 == 0:
			costs = np.zeros(shape=[self.nsamples])
			for k in range(self.nsamples):
				x = state
				self.envmodel.reset(initstate=False)
				for t in range(self.horizon):
					u = self.control[t]
					e = self.noise[k,t]
					v = np.clip(u + e, -1, 1)
					x, q = self.envmodel.step(v, x)
					costs[k] += q + self.lamda * u.T @ self.icov @ e
			beta = np.min(costs)
			costs_norm = -(costs - beta)/self.lamda
			weights = sp.special.softmax(costs_norm)
			self.control += np.sum(weights[:,None,None]*self.noise, 0)
		action = np.clip(self.control[0], -1, 1)
		self.control = np.roll(self.control, -1, axis=0)
		self.control[-1] = 0
		return action if len(action.shape)==len(state.shape) else np.repeat(action[None,:], state.shape[0], 0)

def test_mppi():
	config = Config(env_name="Pendulum-v0", envmodel="real", MPC=Config(NSAMPLES=50, HORIZON=40, LAMBDA=0.5))
	env = get_env(config.env_name)
	state_size = get_space_size(env.observation_space)
	action_size = get_space_size(env.action_space)
	agent = TestMPPIController(state_size, action_size, EnvModel, config)
	envmodel = agent.envmodel
	state = envmodel.network.state
	for s in range(100):
		action = agent.get_action(state)
		envmodel.reset(initstate=False)
		state, cost = envmodel.step(action, None)
		print(f"Step: {s:5d}, Action: {action}, Cost: {cost}")
		envmodel.network.env.render()


if __name__ == "__main__":
	visualize_envmodel()
	# test_car_sim()
	# test_mppi()
	# test_envmodel()