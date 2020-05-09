import os
import numpy as np

root = os.path.dirname(os.path.abspath(__file__))
track_file = os.path.abspath(f"{root}/track.txt")

class Track():
	def __init__(self, track_file=track_file):
		self.track = self.load_track(track_file)
		self.X, self.Z, self.Y = zip(*self.track)
		
	def min_dist(self, point):
		xt, yt = point
		dists = [np.sqrt((xt-x)**2 + (yt-y)**2) for x,z,y in self.track]
		return min(dists)

	def load_track(self, track_file):
		with open(track_file, "r") as f:
			track = [eval(line.rstrip()) for line in f]
		return track

	@staticmethod
	def save_track(track):
		with open(track_file, "w+") as f:
			for t in track:
				f.write(f"[{', '.join([f'{p}' for p in t])}]\n")
