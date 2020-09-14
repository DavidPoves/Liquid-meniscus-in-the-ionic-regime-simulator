import scipy.integrate
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


class BVPInterface(object):
	def __init__(self):
		"""Preallocate all the class attributes within the __init__ method."""

		self.x = sp.Symbol('x')  # Sympy variable of the independent coordinate.
		sym_fun = sp.Function('y')  # Sympy function of the solution.
		self.y = sym_fun(self.x)  # Function of the solution, relating it with the independent parameter self.x
		self.system = np.array([])  # Variable containing the system defined by the user with random initialization.
		self.ode = sp.Expr('')  # Define a random expression to initialize the variable.
		self.isolated_ode = sp.Expr('')  # Last term of the ODE system.
		self.ode_order = 0  # Initialize a variable with the ODE order.
		self.derivative_terms = list()  # List containing all the derivative terms contained in the ODE.
		self.dummy_vars = tuple()  # Create a tuple containing all the symbols.
		self.free_symbols = set()  # Create a set with the free symbols of the ODE.
		self.ya = np.array([])  # Array containing the boundary conditions at the initial point of the domain.
		self.yb = np.array([])  # Array containing the boundary conditions at the end point of the domain.
		self.bcs = dict()  # Dictionary containing the boundary conditions of the problem, as defined in load_bcs docs.
		self.mesh = np.array([])  # Array containing the initial node points where the problem will be solved.
		self.defined_init_guess = False
		self.initial_guess = np.array([])
		self.sympy_system = np.array([])  # Variable storing the system (in sympy language) if defined by the user.
		self.functions_dict = dict()  # Dictionary containing the variables defined by the user.
		self.arrays = dict()  # Dictionary containing variables that are array-like.
		self.dummy_vars_iso = tuple()  # Tuple containing isolated dummy variables, to avoid user-defined variables.

	def get_Derivatives(self, arg):
		"""
		This is a Pre order Transversal function for a Sympy expression used to obtain Derivatives within the given
		expression. This function will get all the elemental arguments of the introduced expression. To get an insight
		on this concept, check Sympy docs:
		https://docs.sympy.org/latest/tutorial/manipulation.html
		Args:
			arg: Sympy argument.

		Returns:

		"""
		try:
			if arg.is_Derivative:
				if isinstance(self.derivative_terms, set):  # If we receive a set instead of a list, like in systems.
					self.derivative_terms = list(self.derivative_terms)
					self.derivative_terms.append(arg)
					self.derivative_terms = set(self.derivative_terms)
				else:
					self.derivative_terms.append(arg)
			for arg_ in arg.args:
				self.get_Derivatives(arg_)
		except AttributeError:
			pass

	def substitute_derivatives(self, expr):
		# Get the terms with derivatives.
		self.derivative_terms = list()
		self.get_Derivatives(arg=expr)

		# Make an unique set of derivative terms, in case there are any repeated term.
		self.derivative_terms = set(self.derivative_terms)

		# Create dummy variables for the function y(x) to substitute derivatives.
		self.dummy_vars = sp.symbols(f'y0:{self.ode_order + 1}')  # Create a set (y0, ..., yn); n = derivative count.

		# Substitute y(x) and its derivatives by their corresponding symbols.
		ode_temp = expr
		for derivative in self.derivative_terms:
			ode_temp = ode_temp.subs({derivative: self.dummy_vars[derivative.derivative_count]})
		ode_temp = ode_temp.subs({self.y: self.dummy_vars[0]})

		return ode_temp

	def substitute_functions(self, expr):
		# Create dummy variables for the functions to be substituted.
		n_funs = len(list(self.functions_dict.values()))
		self.dummy_vars = sp.symbols(f'y1:{n_funs+1}')

		# Substitute the functions by their dummy variables.
		ode_temp = expr
		counter = 0
		for fun in self.functions_dict.values():
			ode_temp = ode_temp.subs({fun: self.dummy_vars[counter]})
			counter += 1

		return ode_temp

	@staticmethod
	def get_ode_order(ode, fun):
		return sp.ode_order(ode, func=fun)

	def define_ode(self, ode):
		"""
		Define the ode as a string, which will be translated to sympy format using its parser. Notice that the sympify
		method is avoided in order not to use the eval() method.

		To define expressions, use x to define the independent parameter and y to declare the solution, that is y(x). To
		define derivatives, several options can be used, according to Sympy documentation. However, the simplest one is
		to write y.diff(x, n), being n the order of the derivative (must be an integer). This may be subjected to change
		in the future into a more intuitive definition. The rest of the expressions must be defined according to sympy
		elementary functions:
		https://docs.sympy.org/latest/modules/functions/elementary.html
		Args:
			ode: string. The equation to be parsed.

		"""
		if not isinstance(ode, str):
			raise TypeError(f'The expression must be an integer, not a {type(ode)}')
		self.ode = sp.parse_expr(ode, local_dict={'x': self.x, 'y': self.y})

		# Identify the symbols of the ODE.
		self.free_symbols = self.ode.free_symbols

		# Identify the ODE order.
		self.ode_order = BVPInterface.get_ode_order(self.ode, self.y)

		# Properly initialize the boundary conditions arrays (ya, yb).
		self.ya = np.zeros((1, self.ode_order))[0]
		self.yb = np.zeros((1, self.ode_order))[0]

	def define_system(self, system, functions, vars_dict=None):
		if vars_dict is None:
			vars_dict = dict()

		# Give value to the ode order variable.
		self.ode_order = 1

		# Define the functions from the functions variable:
		local_dict = dict()
		for fun in functions:
			local_dict[str(fun)] = sp.Function(fun)(self.x)
			self.functions_dict[str(fun)] = sp.Function(fun)(self.x)

		# Deal with the variables' dictionary.
		for key, value in vars_dict.items():
			if isinstance(value, str) and len(value) > 1:  # Deal with an expression, which may contain other variables.
				expr = sp.parse_expr(value, local_dict=self.functions_dict)
				free_symbols = expr.free_symbols
				for symbol in free_symbols:
					if str(symbol) == 'x':  # the independent variable.
						local_dict['x'] = self.x
					else:
						if str(symbol) in vars_dict.keys():
							variable = vars_dict[str(symbol)]
							if isinstance(variable, int) or isinstance(variable, float):
								expr = expr.subs({str(symbol): vars_dict[str(symbol)]})
							local_dict[key] = expr
						else:
							raise ValueError(f'{str(symbol)} has not been defined in the variables dictionary.')
			elif isinstance(value, np.ndarray) or isinstance(value, list):
				self.arrays[key] = value
			else:
				local_dict[key] = value

		# Define the system with the sympy expressions.
		self.sympy_system = np.array([])
		for eq in system:
			expr = sp.parse_expr(eq, local_dict=local_dict)

			# Check if there are derivative terms.
			self.get_Derivatives(expr)

			if len(set(self.derivative_terms)) > 0:  # There are derivative terms.
				ode_temp = self.substitute_derivatives(expr)  # Check and substitute possible derivatives.

			elif len(set(self.derivative_terms)) == 0:  # There are no derivative terms.
				ode_temp = self.substitute_functions(expr)

			self.sympy_system = np.append(self.sympy_system, ode_temp)
		if 'x' in str(ode_temp):
			self.dummy_vars = tuple([self.x]) + self.dummy_vars
		self.dummy_vars_iso = self.dummy_vars
		self.functions_dict['x'] = self.x

	def create_system(self, x, y):
		"""
		Create a system with the required structure to be solvable by the scipy solver. To do so, the sympy lambdify
		function is used.
		Args:
			x: Array like. Node points vector. Input introduced by the solver.
			y: Array like. Solution vector. Introduced by the solver.

		Returns:

		"""
		temp_dict = dict()
		counter = 0
		for var in self.dummy_vars_iso[:-1]:
			if self.x == var:
				temp_dict[str(var)] = x
			else:
				temp_dict[str(var)] = y[counter]
				counter += 1
		if self.sympy_system.size == 0:  # If the user input was an ODE and not a system.
			self.system = np.zeros((self.ode_order, x.size))
			for i in np.linspace(1, self.ode_order-1, 1):
				loc = int(i)
				self.system[loc-1] = y[loc]
			self.system[-1] = sp.lambdify(self.dummy_vars[:-1], self.isolated_ode)(*tuple(list(temp_dict.values())))
			self.system = np.vstack(self.system)
			return self.system
		else:  # If the user input was a system.
			temp_dict[str(self.dummy_vars_iso[-1])] = y[counter]

			if len(list(self.arrays.keys())) > 0:  # Arrays have been loaded.
				for key, value in self.arrays.items():
					temp_dict[key] = value
					# We need to append to the self.dummy_vars tuple, following this procedure.
					lst_temp = list(self.dummy_vars)
					lst_temp.append(sp.Symbol(key))
					self.dummy_vars = tuple(lst_temp)

					# Check possible size issues with the introduced array.
					if len(value) != x.size:
						temp_dict[key] = np.interp(x, self.mesh, value.astype('float'))

			# Make the dummy vars tuple unique.
			self.dummy_vars = tuple(sorted(set(self.dummy_vars), key=self.dummy_vars.index))

			self.system = np.zeros((self.sympy_system.size, x.size))
			counter = 0
			for eq in self.sympy_system:
				vals = tuple(list(temp_dict.values()))
				lambd = sp.lambdify(self.dummy_vars, eq)(*vals)
				self.system[counter] = lambd
				counter += 1
			self.system = np.vstack(self.system)
			return self.system

	def get_equivalent_system(self):
		"""
		Obtain the equivalent system of first order ODES from the introduced ODE. To do so, a change of variable is
		done, like in the following example:
			Let us consider the following simple 2nd order ODE:
				y'' + 2y' + y = 0
			Then, the following vector is defined: y_vect = [y, y']. Next, we define the derivative of the previous
			vector: dy_vect = [y', y'']. Thus, following this scheme, the vector containing the derivatives can be
			defined, following the original form of the ODE, as follows:
			dy_vect = [y_vect[1], -2*y_vect[1] - y_vect[0]]
		Following this procedure, any ODE of nth order can be solved.

		Returns:

		"""
		# Get the order of the received ODE.
		self.ode_order = self.get_ode_order(self.ode, fun=self.y)

		# Get the ODE with the proper substitutions of the derivatives, if any.
		ode_temp = self.substitute_derivatives(self.ode)

		# Check if the independent term x is in the temporary ODE.
		if 'x' in str(ode_temp):
			self.dummy_vars = tuple([self.x]) + self.dummy_vars
		self.dummy_vars_iso = self.dummy_vars

		# Isolate the dummy variable with the highest derivative order.
		self.isolated_ode = sp.solve(ode_temp, self.dummy_vars[-1])[0]

	@staticmethod
	def SurfaceUpdate(r, y, residuals, mesh):
		"""Definition of the system to be solved to obtain the desired surface update, depending on the residuals computed
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
		residuals_int = np.interp(r, mesh, residuals.astype('float'))
		return np.vstack([y[1], 2*(1+y[1]**2)**(3/2)*residuals_int - (1/(r+1e-20))*(1+y[1]**2)*y[1]])

	def load_mesh(self, mesh):
		"""
		Load the initial mesh of points where the problem will be solved.
		Args:
			mesh: Array containing the initial node points.

		Returns:

		"""
		self.mesh = mesh

	def load_bcs(self, bcs):
		"""
		Load the boundary conditions for the problem. The structure to be used is the following:
								bcs = {'a':{'y': 0}, 'b':{'y': 0}}, or
								bcs = {'a':{'y.diff(x, n)': 0}, 'b':{'y': 0}},
		or any other combination, when the user input is an ODE, or:
								bcs = {'a': {'n': bc_val}, 'b': {'n': bc_val}},
		when the user input is a first order ODE system (n is the unknown number, must be an integer).
		The 'a' key indicates that the boundary condition is to be imposed at the initial point of the mesh, and the 'b'
		key corresponds to boundary conditions at the final point. Next, the boundary condition is defined. To do so, a
		dictionary is used. Its key must be 'y' if the boundary condition does not involve derivatives, or
		'y.diff(x, n)', where n is the derivative order; and its value is the value of the boundary condition itself.
		As it can be seen, the definition of the boundary condition involving derivatives is the same one as the SymPy
		one.
		Args:
			bcs: Dictionary containing the boundary conditions, following the structure presented above.

		Returns:

		"""
		self.bcs = bcs

	def load_initial_guess(self, guess=np.array([])):
		# Check if the mesh has been loaded.
		assert self.mesh.size != 0, 'The initial mesh was not loaded or was not loaded properly. Before loading the initial guess this parameter must be defined.'
		if guess.size == 0:  # Means that the user has not loaded any initial guess.
			# Create a default initial guess.
			self.initial_guess = np.zeros((2, self.mesh.size))
		else:
			self.initial_guess = guess
			self.defined_init_guess = True

	def set_bcs(self):
		"""
		Set the boundary conditions after loading them using the load_bcs method.
		Returns:
			Array containing the loaded boundary conditions with the required format for the scipy solver.
		"""
		# Structure of the boundary conditions be like (when user's input is an ODE):
		# bcs = {'a':{'y': bc_val}, 'b':{'y': bc_val}}
		# bcs = {'a':{'y.diff(x, 1)': bc_val}, 'b':{'y': bc_val}}

		# Structure of the boundary conditions be like (when user's input is a system):
		# bcs = {'a': {'n': bc_val}, 'b': {'n': bc_val}}, where n is the unknown number.

		bc_arr = np.array([])
		assign_dict = {'a': self.ya, 'b': self.yb}

		# Get the bcs from the dictionary.
		for key, value in self.bcs.items():
			for key_, value_ in value.items():
				if key_ == 'y':
					bc_ix = 0
				elif key_.isdigit():  # If it is a digit, the user's input was a system of 1st order ODES.
					bc_ix = int(key_) - 1
				else:
					temp_der = sp.parse_expr(key_, local_dict={'x': self.x, 'y': self.y})
					bc_ix = temp_der.derivative_count
				bc_arr = np.append(bc_arr, assign_dict[key][bc_ix] - value_)
		return bc_arr

	def solve(self, **kwargs):
		"""

		Args:
			mesh: Array like. The initial mesh of points defining the domain where the BVP will be solved.
			initial_guess: Array like. Initial guess for the BVP. This may define following these steps.
				- Initialize the array with the following piece of code:
										init_guess = np.zeros((n, mesh.size)), where
				n is the ODE order and the command mesh.size may be substituted by length(mesh) in case the mesh is not
				defined using numpy arrays.
				-
			**kwargs: All the keywords accepted by the scipy.integrate.solve_bvp function. See
			https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html for more info.

		Returns:

		"""
		# Transform the given ODE into a 1st order ODE.
		if self.sympy_system.size == 0:
			self.get_equivalent_system()

		# Load the initial guess if it was not loaded by the user (loads default initial guess of 0s everywhere).
		if not self.defined_init_guess:
			self.load_initial_guess()

		# Solve.
		sol = scipy.integrate.solve_bvp(lambda x, y: self.create_system(x, y), self.bcs_fun, self.mesh,
		                                self.initial_guess, **kwargs)
		return sol

	def bcs_fun(self, ya, yb):
		"""
		Function where the boundary conditions are defined. In this case, ya refers to the boundary conditions at the
		beginning of the considered domain and yb refers to the bcs at the end of the domain. Recall that:
		y_vect = [y, y']
		Args:
			ya: y_vect at the beginning of the domain.
			yb: y_vect at the end of the domain.

		Returns:

		"""
		self.ya = ya
		self.yb = yb
		bc_array = self.set_bcs()
		return bc_array


# Run the test.
if __name__ == '__main__':

	# %% EXAMPLE 1.
	bvp_int = BVPInterface()
	# Define the ODE expression following the guidelines from the documentation.
	bvp_int.define_ode('x*(1+y.diff(x, 1)**2)**(3/2) - 0.5*(1+y.diff(x, 1)**2)*y.diff(x, 1) - 0.5*x*y.diff(x, 2)')

	# Define an initial mesh of points to solve the problem.
	node_points = np.linspace(1e-20, 1, 200)
	bvp_int.load_mesh(node_points)

	# Define an initial guess (optional) and load it.
	init_guess = np.zeros((bvp_int.ode_order, node_points.size))
	init_guess[0, 0] = 0.5
	init_guess[0, 1] = 1
	bvp_int.load_initial_guess(init_guess)

	# Define the boundary conditions.
	bvp_int.load_bcs({'a': {'y.diff(x, 1)': 0}, 'b': {'y': 0}})

	# Solve.
	sol = bvp_int.solve()
	r_plot = np.linspace(0, 1, 200)
	y_plot = sol.sol(r_plot)[0]
	plt.figure()
	plt.plot(r_plot, y_plot)

	# %% EXAMPLE 2: Solving Bratu's problem.
	bvp_int = BVPInterface()
	# Define the ODE following the guidelines.
	bvp_int.define_ode('y.diff(x, 2) + exp(y)')

	# Define an initial mesh of points to solve the problem and load it.
	node_points = np.linspace(0, 1, 5)
	bvp_int.load_mesh(node_points)

	# Define an initial guess.
	init_guess_a = np.zeros((2, node_points.size))
	init_guess_b = np.zeros((2, node_points.size))
	init_guess_b[0] = 3

	# Define the boundary conditions.
	bvp_int.load_bcs({'a': {'y': 0}, 'b': {'y': 0}})

	# Solve the problem.
	bvp_int.load_initial_guess(init_guess_a)
	res_a = bvp_int.solve()
	bvp_int.load_initial_guess(init_guess_b)
	res_b = bvp_int.solve()

	# Plot the solutions.
	x_plot = np.linspace(0, 1, 100)
	y_plot_a = res_a.sol(x_plot)[0]
	y_plot_b = res_b.sol(x_plot)[0]
	plt.figure()
	plt.plot(x_plot, y_plot_a, label='y_a')
	plt.plot(x_plot, y_plot_b, label='y_b')
	plt.legend()
	plt.xlabel("x")
	plt.ylabel("y")
	plt.show()

	# %% EXAMPLE 3: Solving an already defined system.
	# Solving the example problem from: https://pythonhosted.org/scikits.bvp_solver/tutorial.html
	bvp_int = BVPInterface()

	# Define some problem constants.
	T10 = 130
	T2Ahx = 70
	Ahx = 5
	U = 1.0

	# Define a system.
	q_expr = '(T1-T2)*U'
	syst = ['-q', 'q/-2.']  # This is the dy/dt vector.
	bvp_int.define_system(syst, functions=('T1', 'T2'), vars_dict={'U': U, 'q': q_expr})
	bvp_int.load_bcs({'a': {'1': T10}, 'b': {'2': T2Ahx}})
	bvp_int.load_mesh(np.linspace(0, Ahx, 200))
	sol = bvp_int.solve()
	r_plot = np.linspace(0, 5, 200)
	y_plot_1 = sol.sol(r_plot)[0]
	y_plot_2 = sol.sol(r_plot)[1]
	plt.figure()
	plt.plot(r_plot, y_plot_1, label='T1')
	plt.plot(r_plot, y_plot_2, label='T2')
	plt.legend()
	plt.show()

	# %% EXAMPLE 4: Re-do example 1 but with a system definition.
	bvp_int = BVPInterface()

	# Define the system. Define the following functions: y1 = y', y2 = y''
	funs = ['y']
	syst = ['y.diff(x, 1)', '2*(1+y.diff(x, 1)**2)**(3/2) - (1/(x+1e-20))*(1+y.diff(x, 1)**2)*y1']
	bvp_int.define_system(syst, functions=funs)

	# Load the boundary conditions.
	bvp_int.load_bcs({'a': {'y.diff(x, 1)': 0}, 'b': {'y': 0}})
	# bvp_int.load_bcs({'a': {'2': 0}, 'b': {'1': 0}})  # This is another option to define the boundary conditions.

	# Load the mesh of initial points where the problem will be solved.
	bvp_int.load_mesh(np.linspace(0, 1, 200))

	# Define an initial guess (optional) and load it.
	init_guess = np.zeros((len(syst), bvp_int.mesh.size))
	init_guess[0, 0] = 0.5
	init_guess[0, 1] = 1
	bvp_int.load_initial_guess(init_guess)

	# Solve the problem.
	sol = bvp_int.solve()

	# Plot the solution.
	r_plot = np.linspace(0, 1, 200)
	y_plot = sol.sol(r_plot)[0]
	plt.figure()
	plt.plot(r_plot, y_plot)

	# %% EXAMPLE 5: Re-do example 1 with numpy array component.
	bvp_int = BVPInterface()

	# Define the numpy array.
	tau = np.linspace(0, 1e-1, 200)

	# Define the system with the corresponding functions.
	funs = ['y']
	syst = ['y.diff(x, 1)', '2*tau*(1+y.diff(x, 1)**2)**(3/2) - (1/(x+1e-20))*(1+y.diff(x, 1)**2)*y.diff(x, 1)']
	bvp_int.define_system(syst, functions=funs, vars_dict={'tau': tau})

	# Load the boundary conditions.
	bvp_int.load_bcs({'a': {'y.diff(x, 1)': 0}, 'b': {'y': 0}})
	# bvp_int.load_bcs({'a': {'2': 0}, 'b': {'1': 0}})  # This is another option to define the boundary conditions.

	# Load the mesh of initial points where the problem will be solved.
	bvp_int.load_mesh(np.linspace(0, 1, 200))

	# Define an initial guess (optional) and load it.
	init_guess = np.zeros((len(syst), bvp_int.mesh.size))
	init_guess[0, 0] = 0.5
	init_guess[0, 1] = 1
	bvp_int.load_initial_guess(init_guess)

	# Solve the problem.
	sol = bvp_int.solve()

	# Plot the solution.
	r_plot = np.linspace(0, 1, 200)
	y_plot = sol.sol(r_plot)[0]
	plt.figure()
	plt.plot(r_plot, y_plot)
