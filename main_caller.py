from OrganizedMAIN import MainWrapper
import fenics as fn


def main_caller(liquid, required_inputs, electrostatics_bcs, Stokes_bcs, **kwargs):
	wrapper = MainWrapper(liquid, required_inputs, electrostatics_bcs, Stokes_bcs, **kwargs)
	return wrapper


def define_inputs():
	"""
	In this function, the user must define all the inputs that are required to run a simulation.
	Returns:

	"""
	# Define the required inputs.
	req_inputs = {'B': 0.0467,
	              'Lambda': 1.,
	              'E0': 0.99,
	              'C_R': 1e3,
	              'T_h': 1.,
	              'P_r': 0}

	# Define the boundary conditions for the electrostatics.
	"""
	Notice the structure of the boundary conditions:

		1. Boundary name as in the .geo file.
		2. The type of boundary condition (Dirichlet or Neumann).
		3. A list where the first value is the value of the bc and the second
			element is the subdomain to which it belongs to.
	"""
	z0 = 10  # As defined in the .geo file.
	top_potential = -req_inputs['E0'] * z0  # Potential at top wall (electrode).
	ref_potential = 0  # Reference potential (ground).
	bcs_electrostatics = {'Top_Wall': {'Dirichlet': [top_potential, 'vacuum']},
	                      'Inlet': {'Dirichlet': [ref_potential, 'liquid']},
	                      'Tube_Wall_R': {'Dirichlet': [ref_potential, 'liquid']},
	                      'Bottom_Wall': {'Dirichlet': [ref_potential, 'vacuum']},
	                      'Lateral_Wall_R': {'Neumann': 'vacuum'},
	                      'Lateral_Wall_L': {'Neumann': 'vacuum'},
	                      'Tube_Wall_L': {'Neumann': 'vacuum'}}

	# Define the solver options for the electrostatics solver.
	"""
	If the user checks the options given by the code above, one can see all the
	available parameters for a parameter set. These parameter sets are:
		- snes_solver.
		- krylov_solver.
		- lu-solver.
	Each of them contain different parameters which can be modified by the user.
	An example of this is given below.

	It is strongly recommended to use the backtracking (bt) line search option,
	which improves robustness and takes care of NaNs. More info about this
	technique at:
	https://en.wikipedia.org/wiki/Backtracking_line_search
	"""
	solver_settings = {"snes_solver": {"linear_solver": "mumps",
	                                   "maximum_iterations": 200,
	                                   "report": True,
	                                   "error_on_nonconvergence": True,
	                                   'line_search': 'bt',
	                                   'relative_tolerance': 1e-2}}

	# Define the boundary conditions of the Stokes simulation.
	bcs_Stokes = {'Tube_Wall_R': {'Dirichlet': ['v', fn.Constant((0., 0.))]}}

	return req_inputs, bcs_electrostatics, solver_settings, bcs_Stokes


if __name__ == '__main__':

	# Load the inputs from the function. Edit the inputs in that function.
	req_inputs, bcs_electrostatics, solver_settings, bcs_Stokes = define_inputs()

	# Call the MAIN script.
	main = main_caller(liquid='EMIBF4', required_inputs=req_inputs, electrostatics_bcs=bcs_electrostatics,
	                   Stokes_bcs=bcs_Stokes, electrostatics_solver_settings=solver_settings)

	# Plot the solutions of the different simulations.
	main.simulation_results.Electrostatics.plot_results(save_images=False, save_mat=False)
	main.simulation_results.Stokes.plot_results(save_images=False, save_mat=False)
