import os
import re
import sys
import cv2
import numpy as np
import importlib.util
import matplotlib.pylab as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import deque
from mpl_toolkits import mplot3d
np.set_printoptions(precision=3)

path = os.path.abspath("src/envs/__init__.py")
spec = importlib.util.spec_from_file_location("envs", path)
envs = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = envs
spec.loader.exec_module(envs)

models = ["rand", "ddpg", "ppo", "sac", "ddqn"]
lighter_cols = ["#EEEEEE", "#44DFFF", "#FF4493", "#BDFF4F", "#FFED44"]
light_cols = ["#CCCCCC", "#00BFFF", "#FF1493", "#ADFF2F", "#FFED00"]
dark_cols = ["#777777", "#0000CD", "#FF0000", "#008000", "#FFA500"]
root = "./logging/logs"

def cat_stats(steps, stats):
	try:
		all_keys = set()
		for i in range(len(stats)):
			stats[i] = eval(stats[i])
			[all_keys.add(key) for key in stats[i].keys()]
		all_stats = {k:[] for k in all_keys}
		for stat in stats:
			for k,v in all_stats.items():
				v.append(stat.get(k, None))
	except:
		all_stats = {}
	return all_stats

def read_log(path):
	steps = []
	rewards = []
	rolling = []
	averages = deque(maxlen=100)
	fields_list = []
	with open(path, "r") as f:
		lines = [line for line in f if re.match(r"^Step:", line)]
	for line in lines:
		match = re.match(r"^Step: *(.*), Reward: *([^ ]*) \[ *([^ ]*)\], Avg: *([^ ]*) \(([^\)]*)\) <(.*)> \((\{.*\})\)", line.strip('\n'))
		step, reward, std, avg, eps, time, stats = match.groups()
		fields = [int(step), float(reward), float(std), float(avg), float(eps), time, stats]
		averages.append(fields[1])
		rolling.append(np.mean(averages, axis=0))
		if len(averages)==0: averages.extend([averages[-1]]*(len(lines)//5))
		fields_list.append(fields)
	steps, rewards, stds, avgs, epss, times, stats = map(list, zip(*fields_list))
	return steps, rewards, rolling, stds, avgs, epss, times, cat_stats(steps, stats)

def graph_logs(env_name, show=False):
	plt.figure()
	for framework in ["pt", "rl"]:
		rl = framework=="rl"
		for m,model in enumerate(models):
			folder = f"{root}/{framework}/{model}/{env_name}/"
			if os.path.exists(folder):
				files = sorted(os.listdir(folder), key=lambda v: str(len(v)) + v)
				steps, rewards, rolling, stds, avgs, epss, times, stats = read_log(os.path.join(folder, files[-1]))
				plt.plot(steps, rewards, ls=":" if rl else "-", color=(lighter_cols if rl else light_cols)[m], linewidth=0.5, zorder=0)
				plt.plot(steps, rolling, ls="--" if rl else "-", color=dark_cols[m], label=f"Avg {model.upper()} ({framework}) <{times[-1]}>", zorder=1)
	try: 
		steps
		plt.xlabel("Step")
		plt.ylabel("Total Reward")
		plt.legend(loc="best", prop={'size': 8})
		plt.grid(linewidth=0.3, linestyle='-')
		plt.title(f"Eval Rewards for {env_name}")
		graph_folder = "Robosuite" if env_name in envs.env_grps["rbs"] else "OpenAI"
		path = f"{root}/graphs/{graph_folder}"
		os.makedirs(path, exist_ok=True)
		print(f"Saving: {path}/{env_name}.pdf")
		plt.savefig(f"{path}/{env_name}.pdf", bbox_inches='tight')
		if show: plt.show()
	except NameError:
		pass

def get_laser_path(env_name, dirname, laser="laser"):
	graph_folder = f"LASER/{env_name}/{laser}/{dirname}"
	path = f"{root}/graphs/{graph_folder}"
	os.makedirs(path, exist_ok=True)
	return path

def graph_laser_use(latent="laser", show=False, index=-1, envs=None):
	base = os.path.join(root, "pt")
	env_names = envs if envs is not None else os.listdir(f"{base}/laser")
	for env_name in env_names:
		plt.figure()
		fkeys = []
		laser_folder = os.path.join(base, "laser", env_name, latent)
		for m,model in enumerate(models):
			for i,folder in enumerate([os.path.join(base, model, env_name), os.path.join(laser_folder, "use", model)]):
				laser = i==1
				ind = index if laser else -1
				if os.path.exists(folder):
					files = sorted(os.listdir(folder), key=lambda v: str(len(v)) + v)
					steps, rewards, rolling, stds, avgs, epss, times, stats = read_log(os.path.join(folder, files[ind]))
					plt.plot(steps, rewards, ls=":" if laser else "-", color=(lighter_cols if laser else light_cols)[m], linewidth=0.5, zorder=0)
					plt.plot(steps, rolling, ls="--" if laser else "-", color=dark_cols[m], label=f"{model.upper()} ({'laser' if laser else 'pt'}) [{files[ind].replace('.txt','')}]", zorder=1)
					if laser: fkeys.append(f"_{str(len(files)+ind)}")

		try: 
			steps
			plt.xlabel("Step")
			plt.ylabel("Total Reward")
			plt.legend(loc="lower right", prop={'size': 5})
			plt.grid(linewidth=0.3, linestyle='-')
			plt.title(f"Eval Rewards for {env_name}")
			path = get_laser_path(env_name, "use", latent)
			name = f"{path}/{env_name}{''.join(fkeys)}_use.pdf"
			print(f"Saving: {name}")
			plt.savefig(name, bbox_inches='tight')
			if show: plt.show()
		except NameError:
			pass

def graph_laser_train(latent="laser", show=False, index=-1, envs=None):
	base = os.path.join(root, "pt", "laser")
	env_names = envs if envs is not None else os.listdir(base)
	for env_name in env_names:
		fig = plt.figure()
		fig.tight_layout()
		fig.subplots_adjust(hspace=0.4, wspace=0.4)
		fig.suptitle(f"Losses for training {env_name}")
		fkeys = []
		grid = 2
		for m,model in enumerate(models[:grid*grid]):
			ax = fig.add_subplot(grid,grid,m+1)
			ax.set_title(model.upper())
			folder = os.path.join(base, env_name, latent, "train", model)
			if os.path.exists(folder):
				files = sorted(os.listdir(folder), key=lambda v: str(len(v)) + v)
				steps, rewards, rolling, stds, avgs, epss, times, stats = read_log(os.path.join(folder, files[index]))
				loss_keys = [k for k in stats.keys() if "loss" in k]
				for key, mar in zip(loss_keys, ["--", ":", "-"]):
					ax.plot(steps, stats[key], linewidth=0.8, zorder=0, label=f"{key} [{files[index].replace('.txt','')}]")
				fkeys.append(f"_{str(len(files)+index)}")
				if m>=(grid*(grid-1)): ax.set_xlabel("Step")
				if m%grid==0: ax.set_ylabel(f"Loss")
				ax.legend(loc="best", prop={'size': 5})
				ax.grid(linewidth=0.3, linestyle='-')
		try: 
			steps
			path = get_laser_path(env_name, "train", latent)
			name = f"{path}/{env_name}{''.join(fkeys)}_train.pdf"
			print(f"Saving: {name}")
			plt.savefig(name, bbox_inches='tight')
			if show: plt.show()
		except NameError:
			pass

def graph_grid(Xs, Ys, xlabels, title, save_config=None):
	fig = plt.figure()
	cvs = FigureCanvasAgg(fig)
	fig.tight_layout()
	fig.subplots_adjust(hspace=0.4, wspace=0.4)
	fig.suptitle(title)
	rows = np.floor(np.sqrt(len(Xs)))
	cols = np.ceil(len(Xs)/rows)
	for i,(x,y) in enumerate(zip(Xs, Ys)):
		ax = fig.add_subplot(rows,cols,i+1)
		ax.plot(x,y)
		ax.set_xlabel(xlabels[i])
		ax.set_ylim((-1, 1))
		# ax.legend(loc="best", prop={'size': 5})
		ax.grid(linewidth=0.3, linestyle='-')
	# save_fig(**save_config) if save_config else plt.show()
	cvs.draw()
	w, h = map(int, fig.get_size_inches()*fig.get_dpi())
	image = np.fromstring(cvs.tostring_rgb(), dtype="uint8").reshape(h,w,3)
	plt.close()
	return image

def save_fig(graph_folder, name):
	path = f"{root}/graphs/{graph_folder}"
	os.makedirs(path, exist_ok=True)
	print(f"Saving: {path}/{name}")
	plt.savefig(f"{path}/{name}", bbox_inches='tight')

def get_env_names():
	env_names = set()
	for framework in ["pt", "rl"]:
		for model in reversed(models):
			folder = f"{root}/{framework}/{model}/"
			if os.path.exists(folder) and os.path.isdir(folder):
				for env_name in os.listdir(folder):
					env_names.add(env_name)
	return sorted(env_names)
	
if __name__ == "__main__":
	index = -1
	# for env_name in get_env_names(): 
	# 	graph_logs(env_name, False)
	for enc in ["", "c"]:
		for latent in ["ae","vae","dynae","dynvae"]:
			graph_laser_use(enc+latent, index=index, envs=["Reacher"])
			graph_laser_train(enc+latent, index=index, envs=["Reacher"])