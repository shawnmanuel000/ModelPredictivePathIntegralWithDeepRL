import numpy as np
import itertools as it
import pyquaternion as pyq

def quaternion_mult(q,r):
	return [r[0]*q[0]-r[1]*q[1]-r[2]*q[2]-r[3]*q[3],
			r[0]*q[1]+r[1]*q[0]-r[2]*q[3]+r[3]*q[2],
			r[0]*q[2]+r[1]*q[3]+r[2]*q[0]-r[3]*q[1],
			r[0]*q[3]-r[1]*q[2]+r[2]*q[1]+r[3]*q[0]]

def point_rotation_by_quaternion(point,q):
	r = [0]+point
	r = pyq.Quaternion(r)
	# q_conj = [q[0],-1*q[1],-1*q[2],-1*q[3]]
	return ((q * r) * q.conjugate).vector
	# return quaternion_mult(quaternion_mult(q,r),q_conj)[1:]

q = pyq.Quaternion([0,0,0,0])
q = pyq.Quaternion(axis=[0,1,0], angle=np.pi/2)
p = list(it.product(np.linspace(-2,2,5), [0], np.linspace(-2,2,5)))
p = [q.rotate(x) for x in p]

# print(q, point_rotation_by_quaternion([1,0,0], q))
print(q.rotate(p))