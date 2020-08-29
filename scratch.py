import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

# %% CREATE SYMPY EXPRESSIONS WITH REGULAR PYTHON.


def fun(s):
	px = 0.5
	w = px/1e-8 * np.tan(np.radians(49.3))
	return (2*(1-s)*s*w*(1/np.tan(np.radians(49.3)))*px)/(1-2*s*(1-s)*(1-w))


def fprime(s):
	return sym.diff(fun(s))


s = sym.symbols('s')

fprimeLambdified = sym.lambdify(s, fprime(s), 'numpy')
print(fprimeLambdified(0.6))

# %% CREATE SYMPY EXPRESSIONS FROM STRINGS.


def fun_str():
	return '(2*(1-s)*s*58130362.821348004*(1/1.16260725642696)*0.5)/(1-2*s*(1-s)*(1-58130362.821348004))'


sym_exp = sym.sympify(fun_str())
fprimeLambdified_string = sym.lambdify(s, sym_exp.diff(), 'numpy')
print(fprimeLambdified_string(0.6))
