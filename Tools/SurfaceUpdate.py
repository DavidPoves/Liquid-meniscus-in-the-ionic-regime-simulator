import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt


# Define the ODE. For second order (and higher), the equation must be simplified into first order ODEs. This is done
# in the following function.


def SurfaceUpdate(r, y, residuals, mesh):
	"""
	Definition of the system to be solved to obtain the desired surface update, depending on the residuals computed
	from previous calculations. For this case, we define r to be the independent parameter, and y to be y(r), so that
	y_vect = [y, y']. Thus, y is referenced as y[0] and y' is referenced by y[1]. What this function returns is the
	derivative of the y_vect variable previously mentioned, that is: dydt = [y', y'']. Therefore, to define the system,
	one should previously isolate y'' from the original ODE.
	Args:
		r: Array-like. The independent parameter. This term will be used by the scipy solver, and the length of this
		array will determine the number of nodes the solver will use.
		y: Array-like. The dependent parameter.
		residuals: Array-like. The residuals obtained from the electrostatics and hydraulic calculations.

	Returns:
		Numpy array containing the derivative of the y_vect. Here the system of 1st order ODEs with the appropriate
		shape to be used by the solver is returned as a numpy array.

	"""
	print(r.size)
	residuals_int = np.interp(r, mesh, residuals.astype('float'))
	return np.vstack([y[1], 2*(1+y[1]**2)**(3/2)*residuals_int - (1/(r+1e-20))*(1+y[1]**2)*y[1]])


# Define the boundary conditions.


def bcs(ya, yb):
	"""
	Function where the boundary conditions are defined. In this case, ya refers to the boundary conditions at the
	beginning of the considered domain and yb refers to the bcs at the end of the domain. Recall that:
	y_vect = [y, y']
	Args:
		ya: y_vect at the beginning of the domain.
		yb: y_vect at the end of the domain.

	Returns:

	"""
	return np.array([ya[1], yb[0]])


# Run the test.
if __name__ == '__main__':
	# Define the test residuals.
	residuals = np.linspace(0, 1e0, 200)

	# Define the initial mesh.
	mesh = np.linspace(0, 1, residuals.size)

	# Define the initial guess
	init_guess = np.zeros((2, mesh.size))
	init_guess[0, 0] = 0.5
	init_guess[0, 1] = 0

	# Run the solver.
	sol = scipy.integrate.solve_bvp(lambda r, y: SurfaceUpdate(r, y, residuals, mesh), bcs, mesh,
	                                init_guess, verbose=2)

	# Plot the obtained solution.
	r_plot = np.linspace(0, 1, 200)
	y_plot = sol.sol(r_plot)[0]
	plt.plot(r_plot, y_plot)
