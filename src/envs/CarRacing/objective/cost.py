import os
import numpy as np
import itertools as it
from multiprocessing import Pool
from .track import Track

root = os.path.dirname(os.path.abspath(__file__))
cost_map_dir = os.path.abspath(f"{root}/cost_maps")

class CostModel():
	def __init__(self, cost_name="cost_map2Ddense"):
		self.track = Track()
		self.load_cost_map(cost_name)
		self.min_point = np.array([self.X[0], self.Y[0], 0])

	def get_cost(self, point):
		point = np.array(point)
		shape = list(point.shape)
		ref = self.min_point[:shape[-1]].reshape(*[1]*(len(shape)-1), -1)
		index = np.round((point-ref)/self.res).astype(np.int32)
		cost = self.cost_map[index[...,0],index[...,1]]
		return 20*np.tanh(cost/20)**2

	def load_cost_map(self, cost_name, res=0.1, buffer=50):
		cost_file = self.get_cost_file(cost_name)
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
		data = np.load(cost_file)
		self.X = data["X"]
		self.Y = data["Y"]
		self.cost_map = data["cost"].T
		self.res = res

	def get_cost_file(self, cost_name):
		return os.path.join(cost_map_dir, f"{cost_name}.npz")
