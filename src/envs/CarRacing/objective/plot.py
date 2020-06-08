import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
sys.path.append(os.path.abspath("./src/envs/CarRacing"))
from cost import CostModel

root = os.path.dirname(os.path.abspath(__file__))
plot_dir = os.path.join(root, "plots")
os.makedirs(plot_dir, exist_ok=True)

def plot_track2D(track):
	plt.figure()
	plt.plot(track.X,track.Y)
	plt.savefig(f"{plot_dir}/Track2D", bounding_box="tight")

def plot_track(track):
	plt.figure()
	ax = plt.axes(projection='3d')
	ax.scatter(track.X,track.Y,track.Z, s=1)
	ax.set_xlim3d(-200, 200)
	ax.set_ylim3d(-200, 200)
	ax.set_zlim3d(-100, 100)
	plt.savefig(f"{plot_dir}/Track3D", bounding_box="tight")

def plot_cost_map3D(cmodel):
	plt.figure()
	ax = plt.axes(projection='3d')
	XX,YY = np.meshgrid(cmodel.X, cmodel.Y)
	ZZ = 8*np.ones_like(XX)
	grid = np.concatenate([XX[:,:,None],YY[:,:,None],ZZ[:,:,None]], -1)
	ax.plot_surface(XX, YY, np.tanh(cmodel.get_point_cost(grid, transform=False)/20)**2, cmap='RdYlGn_r')
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlim3d(0, 2)

def plot_cost_map(cmodel):
	plt.figure()
	XX,YY,ZZ = np.meshgrid(cmodel.X, cmodel.Y, [8])
	# ZZ = 8*np.ones_like(XX)
	grid = np.concatenate([XX,YY,ZZ], -1)
	cost = cmodel.get_point_cost(grid, transform=True)
	XX, YY, cost = [x[...,0] for x in [XX, YY, cost]]
	plt.pcolormesh(XX, YY, cost, cmap='RdYlGn_r')
	plt.colorbar()
	plt.title("Position to Deviation Cost Map")
	plt.xlabel("X (m)")
	plt.ylabel("Y (m)")
	plt.savefig(f"{plot_dir}/Cost2D.png", dpi=500, bounding_box="tight")

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
	cost_model = CostModel(cost_name="cost_map2Ddense")
	track = cost_model.track
	# plot_track2D(track)
	# plot_track(track)
	plot_cost_map(cost_model)
	# plot_cost_map3D(cost_model)
	# plot_track_map(track)
	# animate_path(track)
	# plt.show()
