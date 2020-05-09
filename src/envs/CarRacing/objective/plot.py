import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from multiprocessing import Pool
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

if __name__ == "__main__":
	cost_model = CostModel()
	track = cost_model.track
	plot_track2D(track)
	plot_track(track)
	plot_cost_map(cost_model)
	plt.show()
