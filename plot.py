import os
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def save_track(track):
    with open("track.txt", "w+") as f:
        for t in track:
            f.write(f"[{', '.join([f'{p}' for p in t])}]\n")

def load_track():
	track = []
	with open("track.txt", "r") as f:
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

def min_dist(target, track, max_val=20):
	xt,zt,yt = target
	dists = []
	for x,z,y in track:
		dists.append(np.sqrt((xt-x)**2 + (yt-y)**2))
	return np.tanh(2*min(dists)/max_val)

def calc_reward_map(track, buffer=25, fl="./cost_map2.npz"):
	if not os.path.exists(fl):
		X, Z, Y = zip(*track)
		x_min, x_max = np.min(X), np.max(X)
		z_min, z_max = np.min(Z), np.max(Z)
		y_min, y_max = np.min(Y), np.max(Y)
		X = np.arange(x_min-buffer, x_max+buffer, 1)
		Y = np.arange(y_min-buffer, y_max+buffer, 1)
		XX,YY = np.meshgrid(X, Y)
		ZZ = np.zeros_like(XX)
		for i,j in it.product(list(range(ZZ.shape[0])),list(range(ZZ.shape[1]))):
			target = (XX[i,j],0, YY[i,j])
			ZZ[i,j] = min_dist(target, track)
		np.savez(fl, X=X, Y=Y, ZZ=ZZ)
	data = np.load(fl)
	return (data["X"], data["Y"], data["ZZ"])

def plot_reward_map(rmap):
	X, Y, ZZ = rmap
	plt.figure()
	XX,YY = np.meshgrid(X, Y)
	ax = plt.axes(projection='3d')
	ax.plot_surface(XX, YY, ZZ, cmap='RdYlGn_r')
	ax.set_zlim3d(0, 100)
	plt.show()

if __name__ == "__main__":
	track = load_track()
	# plot_track2D(track)
	# plot_track(track)
	rmap = calc_reward_map(track)
	plot_reward_map(rmap)
