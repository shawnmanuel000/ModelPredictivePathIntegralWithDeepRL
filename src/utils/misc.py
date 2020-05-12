import os
import cv2
import time
import torch
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, sign=' ', floatmode="fixed", linewidth=2000)

SEED = random.randrange(2**32-1)
IMG_DIM = 64

def seed_all(env, seed=None):
	if seed is None: seed = SEED
	env.seed(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	return seed

def rmdir(path, exts=[".pth", ".txt"]):
	[os.remove(f"{path}/{f}") for f in os.listdir(path) if np.any([f.endswith(x) for x in exts])]
	[rmdir(f"{path}/{d}", exts) for d in os.listdir(path) if os.path.isdir(f"{path}/{d}")]
	os.rmdir(path)

def rgb2gray(image):
	gray = np.dot(image, [0.299, 0.587, 0.114]).astype(np.float32)
	return np.expand_dims(gray, -1)

def resize(image, dim=(IMG_DIM,IMG_DIM)):
	img = cv2.resize(image, dsize=dim, interpolation=cv2.INTER_CUBIC)
	return np.expand_dims(img, -1) if image.shape[-1]==1 else img

def show_image(img, filename="test.png", save=True):
	if save: plt.imsave(filename, img)
	plt.imshow(img, cmap=plt.get_cmap('gray'))
	plt.show()

def make_video(imgs, filename):
	dim = (imgs[0].shape[1], imgs[0].shape[0])
	video = cv2.VideoWriter(filename, 0, 30, dim)
	for img in imgs:
		video.write(img.astype(np.uint8))
	video.release()

def rollout(env, agent, eps=None, render=False, sample=False, time_sleep=None, print_action=False):
	state = env.reset()
	total_reward = None
	done = None
	with torch.no_grad():
		while not np.all(done):
			if render: env.render()
			if time_sleep: time.sleep(time_sleep)
			env_action, a = agent.get_env_action(env, state, eps, sample)[0:2]
			state, reward, ndone, _ = env.step(env_action)
			if print_action: print(reward, env_action, a)
			reward = np.equal(done,False).astype(np.float32)*reward if done is not None else reward
			done = np.array(ndone) if done is None else np.logical_or(done, ndone)
			total_reward = reward if total_reward is None else total_reward + reward
	return total_reward

def dict_update(d, u):
	for k,v in u.items():
		d[k] = dict_update(d.get(k,{}),v) if isinstance(v, collections.Mapping) else v
	return d