import fenics as fn
import numpy as np
import sympy as sp

from Tools.BVPInterface import BVPInterface as BVP
from Tools.GMSH_Interface import GMSHInterface


class SurfaceUpdate(object):
	def __init__(self, main_class, electrostatics_results, stokes_results, beta=0.05):
		# Preallocate required parameters.
		self.main_class = main_class
		self.zprimeLambdified = None  # Function of the first derivative of the interface function.
		self.zprime2Lambdified = None  # Function of the second derivative of the interface function.
		self.ind_var = None
		self.n_k = None
		self.del_dot_n = None
		self.r_data = None
		self.r_param = None
		self.ind_data = None
		self.residuals = None
		self.tau_s = None
		self.tau_s_next = None
		self.beta = beta
		self.sol = None

		# Initialize the BVP interface.
		self.bvp_solver = BVP()

		# Load necessary data.
		self.electrostatics_data = electrostatics_results
		self.stokes_data = stokes_results

	@staticmethod
	def get_derivatives(independent_param, fun):
		"""
        Get the two first derivatives of a given function. To do, the Sympy library is used. Sympy is quite useful for
        symbolic mathematical operations, and derivatives of a given function (using their own functions) is as easy as
        calling a simple function.

        Computation of the derivatives have been checked using WolframAlpha.
        Args:
            independent_param: string. The independent parameter with respect to which the derivatives will be computed.
            fun: string. The function to be derived. This must contain only a single independent parameter and any angle
            should be introduced in radians.

        Returns:
            Lambda functions of the first and second derivatives, only as a function of the independent parameter.

        """
		# Define a Sympy symbol.
		""" Defining a Sympy symbol will be useful to tell the function with respect to which parameter should the function
        compute the derivative.
        """
		sym_ind_param = sp.Symbol(independent_param)

		# Transform the string with the function into a Sympy expression.
		""" When doing this transformation, Sympy will automatically recognize the independent variable as a symbol, in
        case there is only a single parameter. If more than one parameter is in the string, it will recognize several
        symbols. For the latter case, the parse_expr function is recommended. For help on this function, type
        help(sp.parse_expr).
        """
		sym_exp = sp.sympify(fun)

		# Compute the first and second derivatives of the introduced function.
		fprime = sym_exp.diff(sym_ind_param)
		fprime2 = fprime.diff(sym_ind_param)

		# Substitute possible conflicts between used numerical parser and sympy symbology.
		fprime = fprime.subs({'PI': sp.pi})
		fprime2 = fprime2.subs({'PI': sp.pi})

		# Transform the derivatives into evaluable functions.
		fprimeLambdified = sp.lambdify(sym_ind_param, fprime, 'numpy')
		fprime2Lambdified = sp.lambdify(sym_ind_param, fprime2, 'numpy')

		return fprimeLambdified, fprime2Lambdified

	def get_function_derivatives(self):
		# Define which is the function of the interface.
		if self.main_class.geometry_info.interface_fun is None:
			fun_surf = self.main_class.geometry_info.interface_fun_z
			r_param = self.main_class.geometry_info.interface_fun_r
			self.r_param = sp.sympify(r_param)  # Create a sympy expression.
			self.ind_data = self.main_class.gui_info.geom_data.base_data
		else:
			fun_surf = self.main_class.geometry_info.interface_fun
			self.r_data = self.main_class.gui_info.geom_data.base_data

		# Get independent variable from equation.
		self.ind_var = GMSHInterface.get_independent_var_from_equation(fun_surf)

		# Obtain the two first derivatives of the function defining the interface.
		""" In case two functions have been defined (for radial and axial coords, respectively), then only the
        derivatives of the function of the axial coordinates must be derived.
        """
		self.zprimeLambdified, self.zprime2Lambdified = SurfaceUpdate.get_derivatives(self.ind_var, fun_surf)

	def get_aux_terms(self):
		self.n_k = np.array([])
		self.del_dot_n = np.array([])

		if self.r_data is not None:  # Case when a z(r) function is defined.
			for num in self.r_data:
				self.n_k = np.append(self.n_k, (1 / np.sqrt(1 + self.zprimeLambdified(num) ** 2)) * np.array(
					[-self.zprimeLambdified(num), 1]))
				del_term = ((1 + self.zprimeLambdified(num) ** 2) * self.zprimeLambdified(num) +
				            num * self.zprime2Lambdified(num)) / (
							           num * (1 + self.zprimeLambdified(num) ** 2) ** (3 / 2))
				self.del_dot_n = np.append(self.del_dot_n, del_term)
			self.del_dot_n = self.del_dot_n[::-1]
			self.n_k = self.n_k.reshape((len(self.r_data), 2))
			self.n_k = self.n_k[::-1]

		else:  # when independent functions for r and z were defined.
			self.ind_data = np.array([])
			s = sp.Symbol('s')
			for r in self.main_class.general_inputs.r_nodes:
				self.ind_data = np.append(self.ind_data, sp.solvers.solve(self.r_param - r, s)[0])
			for num in self.ind_data:
				self.n_k = np.append(self.n_k, (1 / (1 + self.zprimeLambdified(num) ** 2) ** 0.5) * np.array(
					[-self.zprimeLambdified(num), 1]))
				del_term = ((1 + self.zprimeLambdified(num) ** 2) * self.zprimeLambdified(num) +
				            self.r_param.evalf(subs={self.ind_var: num}) * self.zprime2Lambdified(num)) / \
				           (self.r_param.evalf(subs={self.ind_var: num}) * (1 + self.zprimeLambdified(num) ** 2) ** (
							           3 / 2))
				self.del_dot_n = np.append(self.del_dot_n, del_term)
			self.del_dot_n = self.del_dot_n[::-1]
			self.n_k = self.n_k.reshape((len(self.ind_data), 2))
			self.n_k = self.n_k[::-1]

	def compute_residuals(self):
		# Unpack required classes.
		Electrostatics = self.electrostatics_data
		Stokes_sim = self.stokes_data

		# Extract required quantities from the classes.
		r_nodes = self.main_class.general_inputs.r_nodes
		z_nodes = self.main_class.general_inputs.z_nodes

		Lambda = self.main_class.general_inputs.Lambda
		Ca = Stokes_sim.Capillary
		C_R = self.main_class.general_inputs.C_R
		B = self.main_class.general_inputs.B
		T_h = self.main_class.general_inputs.T_h
		I_h = Electrostatics.emitted_current
		E_v_r = Electrostatics.radial_component_vacuum
		E_v_z = Electrostatics.axial_component_vacuum
		E_l_r = Electrostatics.radial_component_liquid
		E_l_z = Electrostatics.axial_component_liquid

		# Create a function space in which the velocity derivatives will be projected.
		Q = fn.FunctionSpace(Electrostatics.mesh, 'DG', 0)

		# Define the required derivatives of the velocity.
		ux = fn.project(Stokes_sim.velocity_field.sub(0).dx(0), Q)  # Derivative of radial component of u wrt the radial
		uz = fn.project(Stokes_sim.velocity_field.sub(1).dx(1), Q)  # Derivative of axial component of u wrt the axial.

		# Compute the residuals following the computations from Ximo's thesis.
		""" Notice that residuals will be evaluated at the nodes that are defined on the interface.
		"""
		counter = 0
		self.residuals = np.array([])
		for r_coord, z_coord in zip(r_nodes, z_nodes):
			a_diff = E_v_r[counter] ** 2 - E_v_z[counter] ** 2 - \
			         self.main_class.liquid_properties.eps_r * (E_l_r[counter] ** 2 - E_l_z[counter] ** 2) + \
			         Stokes_sim.class_caller.p_star([r_coord, z_coord]) - I_h * C_R - \
			         ((self.main_class.liquid_properties.eps_r * Ca * np.sqrt(B)) / (1 + Lambda * (T_h - 1))) * (
					         2 * ux([r_coord, z_coord]))
			b_diff = 2 * E_v_r[counter] * E_v_z[counter] - \
			         2 * self.main_class.liquid_properties.eps_r * E_l_r[counter] * E_l_z[counter] - \
			         ((self.main_class.liquid_properties.eps_r * Ca * np.sqrt(B)) / (1 + Lambda * (T_h - 1))) * \
			         (ux([r_coord, z_coord]) + uz([r_coord, z_coord]))
			c_diff = E_v_z[counter] ** 2 - E_v_r[counter] ** 2 - \
			         self.main_class.liquid_properties.eps_r * (E_l_z[counter] ** 2 - E_l_r[counter] ** 2) + \
			         Stokes_sim.class_caller.p_star([r_coord, z_coord]) - I_h * C_R - \
			         ((self.main_class.liquid_properties.eps_r * Ca * np.sqrt(B)) / (1 + Lambda * (T_h - 1))) * (
					         2 * uz([r_coord, z_coord]))

			# Build the difference tensor.
			diff_tensor = np.array([[a_diff, b_diff],
			                        [b_diff, c_diff]])

			# Compute the residual.
			residual = np.dot(np.dot(diff_tensor, self.n_k[counter, :]), self.n_k[counter, :]) - \
			           0.5 * self.del_dot_n[counter]
			self.residuals = np.append(self.residuals, residual)
			counter += 1

	def get_surf_tension_stress_tensor(self):
		self.tau_s = 0.5 * self.del_dot_n

	def get_next_surf_tension_stress_tensor(self):
		# Compute surface tension stress tensor of the previous iteration.
		self.get_surf_tension_stress_tensor()

		# Compute tension stress tensor for next iteration, which will depend on the value of beta introduced by user.
		self.tau_s_next = np.array([])
		for loc in np.arange(0, len(self.residuals) - 1):
			self.tau_s_next = np.append(self.tau_s_next, 0.5 * self.del_dot_n[loc] + self.beta * self.residuals[loc])

	def solve(self):
		# Get function derivatives.
		self.get_function_derivatives()

		# Get auxiliary terms for the computation of the residuals.
		self.get_aux_terms()

		# Compute the residuals.
		self.compute_residuals()

		# Compute surface tension stress tensor for the next iteration.
		self.get_next_surf_tension_stress_tensor()

		# Define the system to be solved.
		funs = ['y']
		syst = ['y.diff(x, 1)', '2*tau*(1+y.diff(x, 1)**2)**(3/2) - (1/(x+1e-20))*(1+y.diff(x, 1)**2)*y.diff(x, 1)']
		self.bvp_solver.define_system(system=syst, functions=funs, vars_dict={'tau': self.tau_s_next})

		# Define the mesh for the solver of the boundary value problem (bvp). This is an initial mesh.
		mesh_bvp = np.linspace(0, 1, self.tau_s_next.size)
		self.bvp_solver.load_mesh(mesh_bvp)

		# Define the boundary conditions and load them into the solver.
		""" 'a' is the initial point of the mesh containing the nodes (that is, mesh_bvp[0]) and 'b' is the end point
		(that is, mesh_bvp[-1])
		"""
		bcs = {'a': {'y.diff(x, 1)': 0}, 'b': {'y': 0}}
		self.bvp_solver.load_bcs(bcs)

		# Define an initial guess and load it into the solver.
		"""
		The structure of the initial guess is similar to the one used by the solver, that is:
		y_init = [y0, y'0;  <- At initial point of the mesh (ya)
				  y0, y'0]  <- At end point of the mesh (yb)
		"""
		init_guess = np.zeros((2, mesh_bvp.size))
		init_guess[0, 0] = 0.8
		init_guess[0, 1] = 0
		self.bvp_solver.load_initial_guess(init_guess)

		# Run the solver.
		self.sol = self.bvp_solver.solve()
