import os
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from multiprocessing import Pool

root = os.path.dirname(os.path.abspath(__file__))
cost_map_dir = os.path.abspath(f"{root}/cost_maps")
track_file = os.path.abspath(f"{root}/track.txt")

def save_track(track):
	with open(track_file, "w+") as f:
		for t in track:
			f.write(f"[{', '.join([f'{p}' for p in t])}]\n")

def load_track():
	track = []
	with open(track_file, "r") as f:
		for line in f:
			track.append(eval(line.rstrip()))
	return track

def plot_track2D(track=None):
	if not track: track = load_track()
	X, Z, Y = zip(*track)
	plt.plot(X,Y)
	plt.show()

def plot_track(track):
	X, Z, Y = zip(*track)
	plt.figure()
	ax = plt.axes(projection='3d')
	# ax.plot3D(X, Y, Z, 'gray')
	ax.scatter(X,Y,Z, s=1)
	ax.set_xlim3d(-200, 200)
	ax.set_ylim3d(-200, 200)
	ax.set_zlim3d(-100, 100)
	plt.show()

def min_dist(point, track):
	xt, yt = point
	dists = [np.sqrt((xt-x)**2 + (yt-y)**2) for x,z,y in track]
	return min(dists)

def cost_fn(point):
	track = load_track()
	return min_dist(point,track)

def calc_cost_map(track, buffer=50, res=0.1, fl=f"{cost_map_dir}/cost_map2Ddense.npz"):
	if not os.path.exists(fl):
		X, Z, Y = zip(*track)
		x_min, x_max = np.min(X), np.max(X)
		y_min, y_max = np.min(Y), np.max(Y)
		X = np.arange(x_min-buffer, x_max+buffer, res)
		Y = np.arange(y_min-buffer, y_max+buffer, res)
		points = list(it.product(X, Y))
		with Pool(16) as p:
			dists = p.map(cost_fn, points)
		dists = np.array(dists).reshape(len(X), len(Y))
		np.savez(fl, X=X, Y=Y, cost=dists.T, res=res, buffer=buffer)
	data = np.load(fl)
	return (data["X"], data["Y"], data["cost"])

def plot_cost_map(rmap):
	transform = lambda x: 20*np.tanh(x/20)**2
	X, Y, ZZ = rmap
	plt.figure()
	XX,YY = np.meshgrid(X, Y)
	ax = plt.axes(projection='3d')
	ax.plot_surface(XX, YY, transform(ZZ), cmap='RdYlGn_r')
	ax.set_zlim3d(0, 50)
	plt.show()

if __name__ == "__main__":
	track = load_track()
	# plot_track2D(track)
	# plot_track(track)
	rmap = calc_cost_map(track)
	plot_cost_map(rmap)
