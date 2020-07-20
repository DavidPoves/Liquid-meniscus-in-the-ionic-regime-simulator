import numpy as np


def TaylorCone(s):
	r = ((1-2*s)*1)/(1-2*s*(1-s)*(1-20))
	z = (2*(1-s)*s*20*(1/np.tan(np.radians(49.3)))*1)/(1-2*s*(1-s)*(1-20))
	return r, z


def CosineFunction(r):
	z = 0.5*np.cos(np.pi/2 * r)
	return r, z


def ParabolicFunction(r):
	vertex = [0, 0.5]
	a = -(vertex[1])/(1-vertex[0])**2
	z = a*(r-vertex[0])**2 + vertex[1]
	return r, z


def StraightLine(r):
	z = 0.5*(1-r)
	return r, z
