import os
import numpy as np
import itertools as it
from operator import itemgetter
from multiprocessing import Pool

root = os.path.dirname(os.path.abspath(__file__))
track_file = os.path.abspath(f"{root}/track.txt")
map_dir = os.path.abspath(f"{root}/track_maps")

class Track():
	def __init__(self, track_file=track_file, map_name="index_map3D"):
		self.track = self.load_track(track_file)
		self.X, self.Z, self.Y = zip(*self.track)
		self.load_point_map(map_name)
		self.min_point = np.array([self.Xmap[0], self.Ymap[0], self.Zmap[0]])
		self.max_point = np.array([self.Xmap[-1], self.Ymap[-1], self.Zmap[-1]])
		
	def min_dist(self, point):
		xt, yt = point
		dists = [np.sqrt((xt-x)**2 + (yt-y)**2) for x,z,y in self.track]
		return min(dists)

	def nearest_point(self, point):
		xt, yt, zt = point
		dists = {i: np.sqrt((xt-x)**2 + (yt-y)**2 + (zt-z)**2) for i,(x,z,y) in enumerate(self.track)}
		return min(dists, key=dists.get)

	def load_track(self, track_file):
		with open(track_file, "r") as f:
			track = [eval(line.rstrip()) for line in f]
		return track

	def get_nearest(self, point):
		point = np.array(point)
		shape = list(point.shape)
		minref = self.min_point[:shape[-1]].reshape(*[1]*(len(shape)-1), -1)
		maxref = self.max_point[:shape[-1]].reshape(*[1]*(len(shape)-1), -1)
		point = np.clip(point, minref, maxref)
		index = np.round((point-minref)/self.res).astype(np.int32)
		nearest = self.point_map[index[...,0],index[...,1],index[...,2]]
		return nearest

	def get_path(self, point, length=10, step:int=1):
		nearest = self.get_nearest(point)
		ipath = (nearest+np.arange(0,length*step,step)) % len(self.track)
		path = itemgetter(*ipath)(self.track)
		return path

	def get_progress(self, src, dst):
		start = self.get_nearest(src)
		fin = self.get_nearest(dst)
		progress = fin - start
		if progress>len(self.track)/2:
			progress -= len(self.track)
		return progress

	def load_point_map(self, map_name, res=1, buffer=50):
		point_file = os.path.join(map_dir, f"{map_name}.npz")
		if not os.path.exists(point_file):
			X, Y, Z = self.X, self.Y, self.Z
			x_min, x_max = np.min(X), np.max(X)
			y_min, y_max = np.min(Y), np.max(Y)
			z_min, z_max = np.min(Z), np.max(Z)
			X = np.arange(x_min-buffer, x_max+buffer, res)
			Y = np.arange(y_min-buffer, y_max+buffer, res)
			Z = np.arange(z_min-buffer/10, z_max+buffer/10, res)
			points = list(it.product(X, Y, Z))
			with Pool(16) as p:
				nearests = p.map(self.nearest_point, points)
			nearests = np.array(nearests).reshape(len(X), len(Y), len(Z))
			np.savez(point_file, X=X, Y=Y, Z=Z, nearests=nearests, res=res, buffer=buffer)
		data = np.load(point_file)
		self.Xmap = data["X"]
		self.Ymap = data["Y"]
		self.Zmap = data["Z"]
		self.point_map = data["nearests"]
		self.res = data["res"]

	def __len__(self):
		return len(self.track)

	@staticmethod
	def save_track(track):
		with open(track_file, "w+") as f:
			for t in track:
				f.write(f"[{', '.join([f'{p}' for p in t])}]\n")
