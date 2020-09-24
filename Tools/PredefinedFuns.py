import numpy as np

"""
Within this file, the predefined functions appearing in the main menu may be defined. If a new one is added, it must be
added to the attributes self.predef_funs_show and self.predef_funs of the PredefinedFunctions class from the MainMenu.py
file. Moreover, these functions must be added to the self.save method from the previous class.
"""


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
