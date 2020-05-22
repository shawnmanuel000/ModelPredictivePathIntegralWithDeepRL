import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
sys.path.append(os.path.abspath("./src/envs/CarRacing"))
from cost import CostModel

def plot_track2D(track):
	plt.figure()
	plt.plot(track.X,track.Y)

def plot_track(track):
	plt.figure()
	ax = plt.axes(projection='3d')
	ax.scatter(track.X,track.Y,track.Z, s=1)
	ax.set_xlim3d(-200, 200)
	ax.set_ylim3d(-200, 200)
	ax.set_zlim3d(-100, 100)

def plot_cost_map(cmodel):
	plt.figure()
	ax = plt.axes(projection='3d')
	XX,YY = np.meshgrid(cmodel.X, cmodel.Y)
	grid = np.concatenate([XX[:,:,None],YY[:,:,None]], -1)
	ax.plot_surface(XX, YY, cmodel.get_cost(grid), cmap='RdYlGn_r')
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlim3d(0, 50)

def plot_track_map(track):
	plt.figure()
	ax = plt.axes(projection='3d')
	X, Y, Z = track.X, track.Y, track.Z
	grid = np.array(list(zip(X,Y,Z)))
	ax.scatter(X, Y, track.get_nearest(grid))
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlim3d(0, 500)

def animate_path(track):
	plt.ion()
	plt.figure()
	ax = plt.axes(projection='3d')
	X, Y, Z = track.X, track.Y, track.Z
	grid = np.array(list(zip(X,Y,Z)))
	for point in grid:
		path = np.array(track.get_path(point))
		xs, zs, ys = zip(*path)
		ax.set_zlim3d(-100, 100)
		ax.plot(X,Y,Z, color="#DDDDDD")
		ax.plot(xs, ys, zs, linewidth=2)
		relpath = track.get_path(point, dirn=True)
		rx, rz, ry = map(np.array, zip(*relpath))
		ax.plot(rx, ry, rz, linewidth=2)
		plt.draw()
		plt.pause(0.01)
		ax.cla()

if __name__ == "__main__":
	cost_model = CostModel()
	track = cost_model.track
	# plot_track2D(track)
	# plot_track(track)
	# plot_cost_map(cost_model)
	# plot_track_map(track)
	animate_path(track)
	plt.show()
