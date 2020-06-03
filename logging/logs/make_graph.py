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
light_cols = ["#CCCCCC", "#00BFFF", "#FF1493", "#9DFF2F", "#FFED00"]
dark_cols = ["#777777", "#0000CD", "#FF0000", "#008000", "#FFA500"]
root = "./logging/logs"

indices = {
	"CarRacing-v1": {"ppo": 30, "ddpg": 20, "sac": 41}
}

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
		if len(averages)==0: averages.extend([averages[-1]]*(len(lines)//10))
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
				index = indices.get(env_name, {}).get(model, len(files)-1)
				steps, rewards, rolling, stds, avgs, epss, times, stats = read_log(os.path.join(folder, files[index]))
				plt.plot(steps, rewards, "--", color=light_cols[m], label=f"Trial {model.upper()} [logs_{index}]", linewidth=0.7, zorder=0)
				plt.plot(steps, rolling, "-", color=dark_cols[m], label=f"Avg {model.upper()} <{times[-1]}>", zorder=1)
	try: 
		steps
		plt.xlabel("Step")
		plt.ylabel("Total Reward")
		plt.legend(loc="best", prop={'size': 8})
		plt.grid(linewidth=0.3, linestyle='-')
		plt.title(f"Eval Rewards for {env_name}")
		graph_folder = "Unity" if env_name in envs.env_grps["unt"] else "OpenAI"
		path = f"{root}/graphs/{graph_folder}"
		os.makedirs(path, exist_ok=True)
		print(f"Saving: {path}/{env_name}.pdf")
		plt.savefig(f"{path}/{env_name}.pdf", bbox_inches='tight')
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
	for env_name in get_env_names(): 
		graph_logs(env_name, False)
