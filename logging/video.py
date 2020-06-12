import os
import sys
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib as mpl
sys.path.append(os.path.abspath("./"))
from src.envs import get_env
from src.models import RandomAgent, get_config
from src.models.input import InputController
from src.envs.CarRacing.car_racing import CarRacing
from src.models.pytorch.mpc import EnvModel, MDRNNEnv, RealEnv, envmodel_config, set_dynamics_size
from src.models.pytorch import MPPIAgent
from src.utils.config import Config
from src.utils.misc import make_video, resize
np.set_printoptions(precision=2, sign=' ', floatmode="fixed", suppress=True, linewidth=100)

class PathAnimator():
	def __init__(self, track, interactive=True):
		if interactive: plt.ion()
		self.interactive = interactive
		self.track = track
		self.fig = plt.figure(figsize=(8, 8), dpi=80)
		plt.style.use('dark_background')
		self.fig.tight_layout()
		self.fig.subplots_adjust(hspace=0.2, wspace=0.2)
		self.fig.suptitle("Sampled MPPI Trajectories")
		self.ax = plt.axes(projection='3d')
		self.X, self.Y, self.Z = track.X, track.Y, track.Z

	def animate_path(self, trajectories, chosen=None, view=60, info=None):
		self.ax.cla()
		self.plot(trajectories, chosen, view, info)
		if self.interactive:
			plt.draw()
			plt.pause(0.0000001)
			image = None
		else:
			cvs = FigureCanvasAgg(self.fig)
			cvs.draw()
			w, h = map(int, self.fig.get_size_inches()*self.fig.get_dpi())
			image = np.copy(np.frombuffer(cvs.tostring_rgb(), dtype="uint8").reshape(h,w,3))
			image[:,:80] = 0 
			image[:,-80:] = 0
			image[:80] = 0
			image[-80:] = 0
			plt.close()
		return image

	def plot(self, trajectories, chosen, view, info=None):
		point = trajectories[0,0]
		X, Y, Z = map(lambda x: x, [self.X, self.Y, self.Z])
		self.ax.plot(X,Y,Z, color="#999999")
		self.ax.set_zlim3d(-100, 100)
		self.ax.set_xlim3d(point[0]-view, point[0]+view)
		self.ax.set_ylim3d(point[2]-view, point[2]+view)
		if info: self.ax.set_title(',  '.join([f"{k}: {v}" for k,v in info.items()]))
		for path in trajectories:
			xs, zs, ys = path[:,0], path[:,1], path[:,2]
			self.ax.plot(xs, ys, zs, linewidth=0.2)
		if chosen is not None:
			x,z,y = chosen[0,:,0], chosen[0,:,1], chosen[0,:,2]
			self.ax.plot(x, y, z, color="blue", linewidth=1)
			self.ax.scatter(x[0],y[0],z[0], s=2, color="white")
		self.ax.w_xaxis.pane.set_color('#222222')
		self.ax.w_yaxis.pane.set_color('#222222')
		self.ax.w_zaxis.pane.set_color('#222222')
		self.ax.grid(b=True, which='major', color='#555555', linestyle='-')
		self.ax.xaxis._axinfo["grid"]['linewidth'] = 0.1
		self.ax.yaxis._axinfo["grid"]['linewidth'] = 0.1
		self.ax.zaxis._axinfo["grid"]['linewidth'] = 0.1

def visualize_envmodel(save=True):
	make_env, model, config = get_config("CarRacing-v1", "mppi")
	config.MPC.update(NSAMPLES=100, HORIZON=20, LAMBDA=0.5, CONTROL_FREQ=1)
	agent = model(config.state_size, config.action_size, config, load=config.env_name)
	envmodel = EnvModel(config.state_size, config.action_size, config, load=config.env_name)
	env = make_env()
	state = env.reset()
	envmodel.reset(batch_size=[1], state=[state])
	animator = PathAnimator(env.unwrapped.cost_model.track, interactive=not save)
	renders = []
	total_reward = 0
	for s in range(500):
		state = np.array([state])
		action = agent.get_action(state, eps=0)
		trajectories = agent.network.states[0]
		(path, states_dot, states_ddot), rewards = envmodel.rollout(np.clip(agent.network.control,-1,1), state, numpy=True)
		spec, path_spec = map(CarRacing.observation_spec, [trajectories[0], path])
		env_action = RandomAgent.to_env_action(env.action_space, action)
		state, reward, done, _ = env.step(env_action[0])
		envmodel.reset(batch_size=[1], state=[state])
		ns, r = envmodel.step(action, numpy=True)
		total_reward += reward
		rendered = env.render(mode="rgb_array")
		rendered = write_info(rendered, f"Time: {s}, Reward: {total_reward:5.2f}, Vel(r,p,y): {state[[5,3,4]]}")
		graph = animator.animate_path(spec["pos"], path_spec["pos"])
		print(f"Step: {s:5d}, Action: {action}, Reward: {reward:5.2f} ({r[0,0]:5.2f}), Pos: {state[:3]} ({ns[0][:3]}), Vel: {state[3:6]} ({ns[0][3:6]})")
		if save and rendered is not None: 
			renders.append(np.concatenate([rendered, resize(graph, dim=rendered.shape[:2])], axis=1) if graph is not None else rendered)
		if done: break
	print(f"Reward: {total_reward}")
	env.close()
	if renders: make_video(renders, filename=os.path.abspath(os.path.join('logging/videos', config.env_name, f"mppi.avi")))

def write_info(rendered, text):
	if rendered is None: return None
	rendered[-40:] = 0
	image = Image.fromarray(rendered, 'RGB')
	try: font = ImageFont.truetype("/Library/Fonts/Arial.ttf", size=20)
	except: font = ImageFont.truetype("/usr/share/fonts/truetype/open-sans/OpenSans-Light.ttf", size=20)
	draw = ImageDraw.Draw(image)
	point = (25, rendered.shape[1]-35)
	draw.text(point, text, fill="rgb(255,255,255)", font=font)
	rendered = np.array(image.getdata(), dtype=np.uint8).reshape(*rendered.shape)
	return rendered

def visualize_rl(model_name, save=True):
	make_env, model, config = get_config("CarRacing-v1", model_name)
	agent = model(config.state_size, config.action_size, config, load=config.env_name)
	env = make_env()
	state = env.reset()
	renders = []
	total_reward = 0
	for s in range(500):
		state = np.array([state])
		env_action, action = agent.get_env_action(env, state, eps=0)
		state, reward, done, _ = env.step(env_action[0])
		total_reward += reward
		rendered = env.render(mode="rgb_array")
		rendered = write_info(rendered, f"Time: {s}, Reward: {total_reward:5.2f}, Vel(r,p,y): {state[[5,3,4]]}")
		print(f"Step: {s:5d}, Action: {action}, Reward: {reward:5.2f} Pos: {state[:3]}, Vel: {state[3:6]}")
		if save: renders.append(rendered)
		if done: break
	env.close()
	if renders: make_video(renders, filename=os.path.abspath(os.path.join('logging/videos', config.env_name, f"{model_name}.avi")))

def test_envmodel():
	make_env, model, config = get_config("Pendulum-v0", "mppi")
	config.MPC.update(NSAMPLES=100, HORIZON=20, LAMBDA=0.1, COV=0.5)
	env = make_env()
	agent = model(config.state_size, config.action_size, config, load=config.env_name)
	state = env.reset()
	for s in range(400):
		state = np.array([state])
		action = agent.get_action(state, eps=0)
		env_action = RandomAgent.to_env_action(env.action_space, action)
		state, reward, done, _ = env.step(env_action[0])
		agent.network.envmodel.reset(batch_size=[1], state=[state])
		ns, r = agent.network.envmodel.step(action, [state], numpy=True)
		print(f"Step: {s:5d}, Action: {action}, Reward: {reward:5.2f} ({r[0,0]:5.2f}), State: {state} ({ns[0]})")
		if done: break
		env.render()
	env.close()

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
	config = Config(env_name="Pendulum-v0", envmodel="real", MPC=Config(NSAMPLES=100, HORIZON=20, LAMBDA=0.1))
	env = get_env(config.env_name)
	agent = TestMPPIController(config.state_size, config.action_size, EnvModel, config)
	envmodel = agent.envmodel
	state = envmodel.network.state
	for s in range(100):
		action = agent.get_action(state)
		envmodel.reset(initstate=False)
		state, cost = envmodel.step(action, None)
		print(f"Step: {s:5d}, Action: {action}, Cost: {cost}")
		envmodel.network.env.render()
		if done: break
	env.close()

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

if __name__ == "__main__":
	# for model in ["sac","ddpg","ppo"]:
	# 	visualize_rl(model)
	visualize_envmodel(False)
	# test_envmodel()
	# test_car_sim()
	# test_mppi()