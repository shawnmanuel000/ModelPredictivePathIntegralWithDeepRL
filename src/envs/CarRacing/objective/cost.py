import os
import torch
import numpy as np
import itertools as it
from multiprocessing import Pool
try: from track import Track
except: from .track import Track

root = os.path.dirname(os.path.abspath(__file__))
map_dir = os.path.abspath(f"{root}/cost_maps")

class CostModel():
	def __init__(self, cost_name="cost_map2Ddense"):
		self.track = Track()
		self.load_cost_map(cost_name)
		self.min_point = np.array([self.X[0], self.Y[0], 0])
		self.max_point = np.array([self.X[-1], self.Y[-1], 0])
		self.src = '\t'.join([line for line in open(os.path.abspath(__file__), 'r')][19:29])
		self.vtarget = 20

	def get_cost(self, state, prevstate=None):
		prevstate = state if prevstate is None else prevstate
		prevpos = prevstate["pos"][...,[0,2,1]]
		pos = state["pos"][...,[0,2,1]]
		vy = state["vel"][...,-1]
		cost = self.get_point_cost(pos, transform=True)
		progress = self.track.get_progress(prevpos, pos)
		# reward = np.minimum(progress,0) + 2*progress + np.tanh(vy/self.vtarget)-np.power(self.vtarget-vy,2)/self.vtarget**2 - cost**2
		reward = progress + 0.5*np.tanh(vy/self.vtarget)-0.5*np.power(self.vtarget-vy,2)/self.vtarget**2 - 0.5*cost**2
		return -reward

	def get_point_cost(self, pos, transform=True):
		point = np.array(pos)
		shape = list(point.shape)
		minref = self.min_point[:shape[-1]].reshape(*[1]*(len(shape)-1), -1)
		maxref = self.max_point[:shape[-1]].reshape(*[1]*(len(shape)-1), -1)
		point = np.clip(point, minref, maxref)
		index = np.round((point-minref)/self.res).astype(np.int32)
		cost = self.cost_map[index[...,0],index[...,1]]
		return np.tanh(cost/2)**2 if transform else cost

	def load_cost_map(self, cost_name, res=0.1, buffer=50):
		cost_file = os.path.join(map_dir, f"{cost_name}.npz")
		if not os.path.exists(cost_file):
			X, Y = self.track.X, self.track.Y
			x_min, x_max = np.min(X), np.max(X)
			y_min, y_max = np.min(Y), np.max(Y)
			X = np.arange(x_min-buffer, x_max+buffer, res)
			Y = np.arange(y_min-buffer, y_max+buffer, res)
			points = list(it.product(X, Y))
			with Pool(16) as p:
				dists = p.map(self.track.min_dist, points)
			dists = np.array(dists).reshape(len(X), len(Y))
			np.savez(cost_file, X=X, Y=Y, cost=dists.T, res=res, buffer=buffer)
		data = np.load(cost_file, allow_pickle=True)
		self.X = data["X"]
		self.Y = data["Y"]
		self.cost_map = data["cost"].T
		self.res = res
