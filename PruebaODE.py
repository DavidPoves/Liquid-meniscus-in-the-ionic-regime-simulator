import numpy as np
import sympy as sp
from gekko import GEKKO
import matplotlib.pyplot as plt


# %% EXAMPLE 1
# Define constants and variables.
Cd = 0.2  # Drag coefficient [-]
m = 1000  # Mass [kg]
g = 9.80665  # Gravity acceleration [m/s^2]
Fg = m * g

# Define the required symbols and functions.
t_1 = sp.Symbol('t')
v = sp.Function('v')

# Define the differential equation.
eq_1 = sp.Eq(Fg - Cd * v(t_1) - m * sp.Derivative(v(t_1)), 0)

# Solve with initial condition v(0) = 0
solved_eq_1 = sp.dsolve(eq_1, ics={v(0): 0})

# %% EXAMPLE 2
# Define the required symbols and functions.
x = sp.Symbol('x')
sym_fun = sp.Function('sym_fun')
y = sym_fun(x)

# Define the equation to be solved.
eq_2 = sp.Eq(y.diff(x) - y / x, -x * sp.exp(-x))

# Solve the defined equation.
solved_eq_2 = sp.dsolve(eq_2)

# %% EXAMPLE 3
"""
In this example, we will solve the mechanical vibrations problem with an unique solver.
"""
g = 9.80665  # Gravity acceleration [m/s^2]


def mech_vibrations(mass, L, k, gamma, omega, init_conditions):

	# Define the required symbols and functions.
	t = sp.Symbol('t')
	u = sp.Function('u')
	forcing_fun = 10 * sp.cos(omega * t)

	# Define the equation.
	eq = sp.Eq(mass * u(t).diff(t, t) + gamma * u(t).diff(t) + k * u(t), mass * g - k * L + forcing_fun)

	# Define the initial conditions.
	init_conditions = {u(0): init_conditions[0], u(t).diff(t).subs(t, 0): init_conditions[1]}

	# Solve the equation.
	solved_eq = sp.dsolve(eq, ics=init_conditions)
	Lambdified = sp.lambdify(t, solved_eq.rhs, modules=['numpy'])

	return solved_eq, Lambdified


# Solve mechanical vibrations with no damping.
m = 3  # Mass [kg]
Length = 392e-3  # Elongation of the spring due to mass only [m]
spring_const = (m*g)/Length  # Spring constant [kg/s^2]
damp_coeff = 0  # Damping coefficient [-]
freq = np.sqrt(spring_const/m)
ics = (0.2, -0.1)  # Initial conditions, (u(0), u'(0))

solved_eq_3, Lambdified3 = mech_vibrations(m, Length, spring_const, damp_coeff, freq, ics)
# t_arr = np.linspace(0, 5, 100)
# plt.figure()
# plt.plot(t_arr, [Lambdified3(num) for num in t_arr])
# plt.grid()
# plt.show()


# Add a damping to the previous system.
"""
For this example, suppose we have a known damping coefficient, which may come from previous operations, depending on the
statement.
"""
solved_eq_4, Lambdified4 = mech_vibrations(m, Length, spring_const, 90, freq, ics)

# %% EXAMPLE 5.
tau_nxt = 1e-3

# Define the necessary symbols and functions.
r = sp.Symbol('r')
C1 = sp.Symbol('C1')
C2 = sp.Symbol('C2')
y = sp.Function('y')

# Define the equation to be solved.
eq_5 = sp.Eq(r*(1+y(r).diff(r)**2)**(3/2)*tau_nxt - 0.5*(1+y(r).diff(r)**2)*y(r).diff(r) - 0.5*r*y(r).diff(r, r))

# Define the boundary conditions.
bcs = {y(1): 0, y(r).diff(r).subs(r, 0): 0}

# %% EXAMPLE 6. USING GEKKO LIBRARY.
m = GEKKO()
timesteps = 200
m.time = np.linspace(0, 1, timesteps)
arr = np.linspace(0, 0.75, timesteps)

y = m.Var(np.zeros(timesteps), fixed_initial=False)
t = m.Param(value=m.time)
dy = m.Var(value=0)
ddy = m.Var()

m.Equations([
	-t*(1+dy**2)**(3/2)*.75 - 0.5*(1+dy**2)*dy - 0.5*t*ddy == 0,
	dy == y.dt(),
	ddy == dy.dt()
])

pi = np.zeros(timesteps)
pi[-1] = 1
p = m.Param(pi)
m.Minimize(p*y**2)

m.options.IMODE = 6
m.solve(disp=False)

plt.figure()
plt.plot(m.time, y.value, label='y')
plt.xlabel('Radial Position')
plt.legend()
plt.show()
