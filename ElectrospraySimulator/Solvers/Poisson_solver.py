# Copyright (C) 2020- by David Poves Ros
#
# This file is part of the End of Degree Thesis.
#
# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This thesis is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#

# Import the modules.
import dolfin as df
import fenics as fn
import matplotlib.pyplot as plt
import multiphenics as mp
import numpy as np

from ElectrospraySimulator.Geometry_scripts.GMSH_Interface import GMSHInterface
from ElectrospraySimulator.Geometry_scripts.ObtainVectors import ObtainVectors
from ElectrospraySimulator.Tools.generate_restrictions import Restrictions

df.parameters["ghost_mode"] = "shared_facet"  # required by dS

"""
This is the module containing all the required methods to obtain the solution
of the Electrostatics part of the thesis and all the auxiliary methods to be
used in this part.
"""


class Poisson(object):
    def __init__(self, inputs, boundary_conditions, msh_filepath, restrictionspath, checkspath):
        """
        Initialize the Poisson solver with the required inputs. The process to really solve the electrostatics
        problem is simple: Load the initial arguments when this class is initialized, load the mesh by calling the
        method get_mesh from this class. Next, get boundaries and subdomains with get_boundaries and get_subdomains,
        respectively. Finally, generate the restrictions with generate_restrictions method, and finally call the solve
        method with the desired solving parameters (these are explained in the solve method from this class).

        It is strongly recommended to write both check files and write the restrictions files into external files to
        check if subdomains, boundaries and subdomains were generated properly, to avoid non desired errors or results
        with no sense. To do so, always open the .xmdf files with Paraview (or any other program compatible with this
        extension) and ensure that these files show proper definitions of these parameters.
        Args:
            inputs: Dictionary containing all the required inputs for the simulation. These are:
                - Relative_perm: Relative permittivity of the medium wrt vacuum.
                - Phi: Non-dimensional parameter defined in Ximo's thesis.
                - Non_dimensional_temperature.
                - Lambda: Non-dimensional parameter as defined by Ximo in his thesis.
                - Chi: Non-dimensional parameter as defined in Ximo's thesis.
                - B: Ratio of the tip radius (r_star) to the contact line radius.
                - Convection charge.
            boundary_conditions: Dictionary with a structure like the one below:
                boundary_conditions = {'boundary_name': {'Dirichlet/Neumann': [value, 'subdomain_name']}, ...}
            msh_filepath: String containing the path of the mesh.
            restrictionspath: String containing the path of the restrictions file.
            checkspath: String with the path of the folder where the checks file will be stored.
        """

        self.filename = msh_filepath.split('/')[-1]
        self.meshpath = msh_filepath
        self.mesh_folder_path = '/'.join(msh_filepath.split('/')[:-1])
        self.restrictionspath = restrictionspath
        self.checkspath = checkspath
        self.boundary_conditions = boundary_conditions
        self.geo_filepath = self.mesh_folder_path + '/' + self.filename.split('.')[0] + '.geo'

        # Unpack iterable objects.
        self.eps_r = inputs['Relative_perm']
        self.Phi = inputs['Phi']
        self.T_h = inputs['Non_dimensional_temperature']
        self.Lambda = inputs['Lambda']
        self.Chi = inputs['Chi']
        self.B = inputs['B']
        self.j_conv = inputs['Convection charge']

    def check_mesh(self):
        """
        Check if the loaded mesh has the correct type before proceeding with
        the code, where debugging is harder.
        Raises
        ------
        TypeError
            This error will raise when the loaded mesh has not the proper type,
            which is dolfin.cpp.mesh.Mesh.
        Returns
        -------
        None.
        """

        if not isinstance(self.mesh, df.cpp.mesh.Mesh):
            raise TypeError('The type of the loaded mesh is not the proper one. It must have the type dolfin.cpp.mesh.Mesh.')

    @staticmethod
    def isDolfinFunction(inp):
        """
        Check if an input is a dolfin function
        Parameters
        ----------
        inp : any type
            The input to be checked.
        Returns
        -------
        bool
            Returns True if the input is a dolfin function and False otherwise.
        """
        return isinstance(inp, df.function.function.Function)

    @staticmethod
    def isfenicsexpression(inp):
        """
        Check if an input is a FEniCS expression.
        Parameters
        ----------
        inp : any type
            The input to be checked.
        Returns
        -------
        bool
            Returns True if the input is a FEniCS ode. and False otherwise.
        """
        return isinstance(inp, df.function.expression.UserExpression)

    @staticmethod
    def isfenicsconstant(inp):
        """
        Check if an input is a FEniCS constant.
        Parameters
        ----------
        inp : any type
            The input to be checked.
        Returns
        -------
        bool
            Returns True if the input is a FEniCS constant and False otherwise.
        """

        return isinstance(inp, df.function.constant.Constant)

    def get_mesh(self):
        """
        Get the mesh of the problem from the .xml file. This method will use
        the meshpath and the filename variables to get the file.
        Raises
        ------
        ValueError
            This error will raise if the dimension of the loaded object is
            not 2.
        Returns
        -------
        dolfin.cpp.mesh.Mesh
            Initialized mesh object.
        """

        self.mesh = df.Mesh(self.meshpath)

        # Check the mesh type.
        self.check_mesh()

        # Smooth the mesh for smoother results.
        """
        Smoothing the mesh produces a deformation of the meniscus boundary,
        which creates an irregular boundary, and as a consequence, produces
        weird results of the electric field (seems like data with a lot of
        noise).
        """
        # self.mesh.smooth(50)  # 50 are the iterations to smooth the mesh.

        # Obtain the dimension of the created mesh.
        D = self.mesh.topology().dim()

        # Check if the dimension of the mesh is 2.
        if D != 2:
            raise ValueError(f'Only 2D meshes are accepted. The dimension of the loaded mesh is {D}')

        # Initialize the mesh
        self.mesh.init(D-1)

        return self.mesh

    def get_boundaries(self):
        """
        Define the boundaries of the problem from the facet_region generated
        by dolfin.
        Parameters
        ----------
        mesh: cpp.mesh.Mesh
            Mesh object.
        name: string
            Name of the .xml file (with or without extension).
        mesh_folder_path: string
            Name of the path of the folder where the mesh is stored.
        Returns
        -------
        boundaries: cpp.mesh.MeshFunctionSizet
            Object containing all the defined boundaries.
        """

        bound_name = self.filename.split('.')[0] + '_facet_region.xml'
        file = self.mesh_folder_path + '/' + bound_name
        self.boundaries = df.MeshFunction('size_t', self.mesh, file)

        # Obtain the boundaries ids from the .geo file.
        """
        For this part, we do not need to take into account if curves or
        physical surfaces are defined first. This is already taken into
        account in the gmsh_handle.get_physical_curves_and_tags. Since we
        only accept two subdomains, if these are defined first, we know that
        the first curve id will be 3, the next one 4 and so on.
        """
        self.boundaries_ids, _ = GMSHInterface.get_boundaries_ids(self.geo_filepath)
        return self.boundaries, self.boundaries_ids

    def get_subdomains(self):
        """
        Obtain the subdomains of the problem from the physical_region file
        generated by the dolfin converter.
        Raises
        ------
        AttributeError
            If boundaries are defined first in the .geo file and boundaries
            were not defined previously, this error will warn the user about
            that, and give the proper advice.
        Returns
        -------
        dolfin.cpp.mesh.MeshFunctionSizet
            Object containing the subdomains info.
        """

        sub_name = self.filename.split('.')[0] + '_physical_region.xml'
        filepath = self.mesh_folder_path + '/' + sub_name
        self.subdomains = df.MeshFunction('size_t', self.mesh, filepath)
        self.subdomains_ids = GMSHInterface.get_subdomains_ids(self.geo_filepath)

        return self.subdomains, self.subdomains_ids

    def get_measures(self):
        """
        Create the measurements from the generated subdomains and boundaries
        objects. The defined measurements are:
            - dx: Area measure.
            - ds: External boundaries measure.
            - dS: Internal boundaries measure.
        Returns
        -------
        None.
        """
        self.dx = fn.Measure('dx')(subdomain_data=self.subdomains)
        self.ds = fn.Measure('ds')(subdomain_data=self.boundaries)
        self.dS = fn.Measure('dS')(subdomain_data=self.boundaries)

    def generate_restrictions(self):
        """
        Generate the restrictions of the loaded subdomains, a restriction for
        the whole domain and one restriction for the interface. This is
        possible by making use of the multiphenics library:
            https://github.com/mathLab/multiphenics
        Returns
        -------
        multiphenics.mesh.mesh_restriction.MeshRestriction
            Restriction of the subdomain with lower id.
        multiphenics.mesh.mesh_restriction.MeshRestriction
            Restriction of the subdomain with upper id.
        multiphenics.mesh.mesh_restriction.MeshRestriction
            Restriction for the whole domain. Note that this could be done
            simply with FEniCS. However, this would not allow the solver to
            use multiphenics, whose main purpose is to define the Lagrange
            multipliers.
        multiphenics.mesh.mesh_restriction.MeshRestriction
            Interface restriction.
        """
        self.restrictions_dict = dict()
        subdomains_ids_list = []

        for key, value in self.subdomains_ids.items():
            rtc = Restrictions.generate_subdomain_restriction(self.mesh, self.subdomains, [value])
            self.restrictions_dict[key.lower()+'_rtc'] = rtc
            subdomains_ids_list.append(value)

        # Create a whole domain restriction.
        self.restrictions_dict['domain_rtc'] = Restrictions.generate_subdomain_restriction(self.mesh, self.subdomains,
                                                                                           subdomains_ids_list)

        # Create the interface restriction.
        rtc_int = Restrictions.generate_interface_restriction(self.mesh, self.subdomains, set(subdomains_ids_list))
        self.restrictions_dict['interface_rtc'] = rtc_int

        self.restrictions_tuple = tuple(list(self.restrictions_dict.values()))

        aux_str = '_restriction'
        restrictions_keys = list(self.restrictions_dict.keys())
        self.restrictions_names = (restrictions_keys[0].lower() + aux_str, restrictions_keys[1].lower() + aux_str,
                                   'domain_restriction', 'interface_restriction')

        return self.restrictions_dict

    @staticmethod
    def plot(object, **kwargs):
        """
        Plot the given object based on its type.
        Parameters
        ----------
        object: The object to be plotted.
        **kwargs: Same kwargs as the ones defined for fenics.plot().
        Check docs to see all available arguments.
        Returns
        -------
        None.
        """
        kwargs.setdefault('figsize', (12, 7))

        # Check if the given object is a mesh or submesh
        plt.figure(figsize=kwargs.get('figsize'))
        kwargs.pop('figsize')
        pp = fn.plot(object, **kwargs)
        plt.colorbar(pp)
        plt.show()

    def solve(self, **kwargs):
        """
        Solves the variational form of the electrostatics as defined in the
        End of Master thesis from Ximo Gallud Cidoncha:
            A comprehensive numerical procedure for solving the Taylor-Melcher
            leaky dielectric model with charge evaporation.
        Parameters
        ----------
        **kwargs : dict
            Accepted kwargs are:
                - electrostatics_solver_settings: The user may define its own solver parameters. They must be defined
                as follows:
                solver_parameters = {"snes_solver": {"linear_solver": "mumps",
                                      "maximum_iterations": 50,
                                      "report": True,
                                      "error_on_nonconvergence": True,
                                      'line_search': 'bt',
                                      'relative_tolerance': 1e-4}}
                where:
                    - snes_solver is the type of solver to be used. In this
                    case, it is compulsory to use snes, since it's the solver
                    accepted by multiphenics. However, one may try other
                    options if only FEniCS is used. These are: krylov_solver
                    and lu_solver.
                    - linear_solver is the type of linear solver to be used.
                    - maximum_iterations is the maximum number of iterations
                    the solver will try to solve the problem. In case no
                    convergence is achieved, the variable
                    error_on_nonconvergence will raise an error in case this
                    is True. If the user preferes not to raise an error when
                    no convergence, the script will continue with the last
                    results obtained in the iteration process.
                    - line_search is the type of line search technique to be
                    used for solving the problem. It is stronly recommended to
                    use the backtracking (bt) method, since it has been proven
                    to be the most robust one, specially in cases where sqrt
                    are defined, where NaNs may appear due to a bad initial
                    guess or a bad step in the iteration process.
                    - relative_tolerance will tell the solver the parameter to
                    consider convergence on the solution.
                All this options, as well as all the other options available
                can be consulted by calling the method
                Poisson.check_solver_options().
                - initial_potential: Dolfin/FEniCS function which will be used as an initial guess on the iterative
                    process. This must be introduced along with kwarg initial_surface_charge_density. Optional.
                - initial_surface_charge_density: Dolfin/FEniCS function which will be used as an initial guess on the
                    iterative process. This must be introduced along with kwarg initial_potential. Optional.
        Raises
        ------
        TypeError
            This error will raise when the convection charge has not one of the
            following types:
                - Dolfin Function.
                - FEniCS UserExpression.
                - FEniCS Constant.
                - Integer or float number, which will be converted to a FEniCS
                Constant.
        Returns
        -------
        phi : dolfin.function.function.Function
            Dolfin function containing the potential solution.
        surface_charge_density : dolfin.function.function.Function
            Dolfin function conataining the surface charge density solution.
        """

        # --------------------------------------------------------------------
        # EXTRACT THE INPUTS #
        # --------------------------------------------------------------------

        # Check if the type of j_conv is the proper one.
        if not isinstance(self.j_conv, (int, float)) \
            and not Poisson.isDolfinFunction(self.j_conv) \
            and not Poisson.isfenicsexpression(self.j_conv) \
                and not Poisson.isfenicsconstant(self.j_conv):
            conv_type = type(self.j_conv)
            raise TypeError(f'Convection charge must be an integer, float, Dolfin function, FEniCS UserExpression or FEniCS constant, not {conv_type}.')
        else:
            if isinstance(self.j_conv, (int, float)):
                self.j_conv = fn.Constant(float(self.j_conv))

        # Extract the solver parameters.
        solver_parameters = kwargs.get('electrostatics_solver_settings')

        # --------------------------------------------------------------------
        # FUNCTION SPACES #
        # --------------------------------------------------------------------
        # Extract the restrictions to create the function spaces.
        """ This variable will be used by multiphenics when creating function spaces. It will create function spaces
            on the introduced restrictions. 
        """
        restrictions_block = [self.restrictions_dict['domain_rtc'], self.restrictions_dict['interface_rtc']]

        # Base Function Space.
        V = fn.FunctionSpace(self.mesh, 'Lagrange', 2)

        # Block Function Space.
        """ Block Function Spaces are similar to FEniCS function spaces. However, since we are creating function spaces
        based on the block of restrictions, we need to create a 'block of function spaces' for each of the restrictions.
        That block of functions is the list [V, V] from the line of code below this comment. They are assigned in the
        same order in which the block of restrictions has been created, that is:
            - V -> domain_rtc
            - V -> interface_rtc
        """
        W = mp.BlockFunctionSpace([V, V], restrict=restrictions_block)

        # Check the dimensions of the created block function spaces.
        for ix, _ in enumerate(restrictions_block):
            assert W.extract_block_sub_space((ix,)).dim() > 0., f'Subdomain {ix} has dimension 0.'

        # --------------------------------------------------------------------
        # TRIAL/TEST FUNCTIONS #
        # --------------------------------------------------------------------
        # Trial Functions.
        dphisigma = mp.BlockTrialFunction(W)

        # Test functions.
        vl = mp.BlockTestFunction(W)
        (v, l) = mp.block_split(vl)

        phisigma = mp.BlockFunction(W)
        (phi, sigma) = mp.block_split(phisigma)

        # --------------------------------------------------------------------
        # MEASURES #
        # --------------------------------------------------------------------
        self.get_measures()
        self.dS = self.dS(self.boundaries_ids['Interface'])  # Restrict to the interface.

        # Check proper marking of the interface.
        assert fn.assemble(1*self.dS(domain=self.mesh)) > 0., "The length of the interface is zero, wrong marking. Check the files in Paraview."

        # --------------------------------------------------------------------
        # DEFINE THE F TERM #
        # --------------------------------------------------------------------
        n = fn.FacetNormal(self.mesh)
        t = fn.as_vector((n[1], -n[0]))

        # Define auxiliary terms.
        r = fn.SpatialCoordinate(self.mesh)[0]
        K = 1+self.Lambda*(self.T_h - 1)

        E_v_n_aux = fn.dot(-fn.grad(phi("-")), n("-"))

        def expFun():
            sqrterm = E_v_n_aux
            expterm = (self.Phi/self.T_h)*(1-pow(self.B, 0.25)*fn.sqrt(sqrterm))
            return fn.exp(expterm)

        def sigma_fun():
            num = K*E_v_n_aux + self.eps_r*self.j_conv
            den = K + (self.T_h/self.Chi)*expFun()
            return r*num/den

        # Define the relative permittivity.

        class relative_perm(fn.UserExpression):
            def __init__(self, markers, subdomain_ids, relative, **kwargs):
                super().__init__(**kwargs)
                self.markers = markers
                self.subdomain_ids = subdomain_ids
                self.relative = relative

            def eval_cell(self, values, x, cell):
                if self.markers[cell.index] == self.subdomain_ids['Vacuum']:
                    values[0] = 1.
                else:
                    values[0] = self.relative

        rel_perm = relative_perm(self.subdomains, self.subdomains_ids, relative=self.eps_r, degree=0)

        # Define the variational form.
        # vacuum_int = r*fn.inner(fn.grad(phi), fn.grad(v))*self.dx(self.subdomains_ids['Vacuum'])
        # liquid_int = self.eps_r*r*fn.inner(fn.grad(phi), fn.grad(v))*self.dx(self.subdomains_ids['Liquid'])

        F = [r*rel_perm*fn.inner(fn.grad(phi), fn.grad(v))*self.dx - r*sigma("-")*v("-")*self.dS,
             r*sigma_fun()*l("-")*self.dS - r*sigma("-")*l("-")*self.dS]

        J = mp.block_derivative(F, phisigma, dphisigma)

        # --------------------------------------------------------------------
        # BOUNDARY CONDITIONS #
        # --------------------------------------------------------------------
        bcs_block = []
        for i in self.boundary_conditions:
            if 'Dirichlet' in self.boundary_conditions[i]:
                bc_val = self.boundary_conditions[i]['Dirichlet'][0]
                bc = mp.DirichletBC(W.sub(0), bc_val, self.boundaries,
                                    self.boundaries_ids[i])
                # Check the created boundary condition.
                assert len(bc.get_boundary_values()) > 0., f'Wrongly defined boundary {i}'
                bcs_block.append(bc)

        bcs_block = mp.BlockDirichletBC([bcs_block])

        # --------------------------------------------------------------------
        # SOLVE #
        # --------------------------------------------------------------------
        # Define and assign the initial guesses.
        if kwargs.get('initial_potential') is None:
            """
            Check if the user is introducing a potential from a previous
            iteration.
            """
            phiv, phil, sigma_init = self.solve_initial_problem()
            # phi_init = self.solve_initial_problem_v2()
            phi.assign(phiv)
            sigma.assign(sigma_init)
        else:
            phi.assign(kwargs.get('initial_potential'))
            sigma.assign(kwargs.get('initial_surface_charge_density'))

        # Apply the initial guesses to the main function.
        phisigma.apply('from subfunctions')

        # Solve the problem with the solver options (either default or user).
        problem = mp.BlockNonlinearProblem(F, phisigma, bcs_block, J)
        solver = mp.BlockPETScSNESSolver(problem)
        solver_type = [i for i in solver_parameters.keys()][0]
        solver.parameters.update(solver_parameters[solver_type])
        solver.solve()

        # Extract the solutions.
        (phi, _) = phisigma.block_split()
        self.phi = phi
        # --------------------------------------------------------------------

        # Compute the electric field at vacuum and correct the surface charge density.
        self.E_v = self.get_electric_field('Vacuum')
        self.E_v_n = self.get_normal_field(n("-"), self.E_v)
        self.E_t = self.get_tangential_component(t("+"), self.E_v)
        C = self.Phi / self.T_h * (1 - self.B ** 0.25 * fn.sqrt(self.E_v_n))
        self.sigma = (K * self.E_v_n) / (K + self.T_h / self.Chi * fn.exp(-C))

    def solve_initial_problem(self):
        """
        Solve a simple intial problem for a first guess on the iteration
        process of the main problem. This simple problem is defined as:
            div(r*grad(phiv)) = 0, in vacuum subdomain
            phil              = 0, in liquid subdomain
            sigma             = 0, at interface
        Returns
        -------
        phiv : dolfin.function.function.Function
            Solution of the potential at vacuum.
        phil : dolfin.function.function.Function
            Solution of potential at liquid.
        sigma : dolfin.function.function.Function
            Solution of the surface charge density.
        """

        V = fn.FunctionSpace(self.mesh, 'Lagrange', 2)

        # Define the restrictions.
        restrictions_init = []
        for key in self.subdomains_ids.keys():
            key = key.lower() + '_rtc'
            restrictions_init.append(self.restrictions_dict[key])
        restrictions_init.append(self.restrictions_dict['interface_rtc'])

        # Define the block Function Space.
        W = mp.BlockFunctionSpace([V, V, V], restrict=restrictions_init)

        # Define the trial and test functions.
        test = mp.BlockTestFunction(W)
        (v1, v2, l) = mp.block_split(test)

        trial = mp.BlockTrialFunction(W)
        (phiv, phil, sigma) = mp.block_split(trial)

        # Define auxiliary terms.
        r = fn.SpatialCoordinate(self.mesh)[0]

        #                                       phiv                                                         phil                                sigma             #
        aa = [[r*fn.inner(fn.grad(phiv), fn.grad(v1))*self.dx(self.subdomains_ids['Vacuum'])  , 0                                       , 0                        ],  # Test Function v1
              [0                                                                        , phil*v2*self.dx(self.subdomains_ids['Liquid']), 0                        ],  # Test function v2
              [0                                                                        , 0                                       , sigma("+")*l("+")*self.dS]]        # Test function l
        bb = [fn.Constant(0.)*v1*self.dx(self.subdomains_ids['Vacuum']), fn.Constant(0.)*v2*self.dx(self.subdomains_ids['Liquid']), fn.Constant(0.)*l("+")*self.dS]

        # Assemble the previous expressions.
        AA = mp.block_assemble(aa)
        BB = mp.block_assemble(bb)

        # Define the boundary conditions.
        bcs_v = []
        bcs_l = []
        bcs_i = []
        for i in self.boundary_conditions:
            if 'Dirichlet' in self.boundary_conditions[i]:
                sub_id = self.boundary_conditions[i]['Dirichlet'][1]
                if sub_id.lower() == list(self.subdomains_ids.keys())[0].lower():
                    sub_id = 0
                elif sub_id.lower() == list(self.subdomains_ids.keys())[1].lower():
                    sub_id = 1
                else:
                    raise ValueError(f'Subdomain {sub_id} is not defined on the .geo file.')
                bc_val = self.boundary_conditions[i]['Dirichlet'][0]
                bc = mp.DirichletBC(W.sub(sub_id), bc_val, self.boundaries,
                                    self.boundaries_ids[i])
                # Check the created boundary condition.
                assert len(bc.get_boundary_values()) > 0., f'Wrongly defined boundary {i}'
                if sub_id == 0:
                    bcs_v.append(bc)
                elif sub_id == 1:
                    bcs_l.append(bc)
                else:
                    bcs_i.append(bc)
        bcs = mp.BlockDirichletBC([bcs_v, bcs_l, bcs_i])

        # Apply the boundary conditions.
        bcs.apply(AA)
        bcs.apply(BB)

        # Define the solution function and solve.
        sol = mp.BlockFunction(W)
        mp.block_solve(AA, sol.block_vector(), BB)

        # Split the solution.
        (phiv, phil, sigma) = sol.block_split()

        return phiv, phil, sigma

    def solve_initial_problem_v2(self):
        V = fn.FunctionSpace(self.mesh, 'Lagrange', 2)

        # Define test and trial functions.
        v = fn.TestFunction(V)
        u = fn.TrialFunction(V)

        # Define radial coordinates.
        r = fn.SpatialCoordinate(self.mesh)[0]

        # Define the relative permittivity.

        class relative_perm(fn.UserExpression):
            def __init__(self, markers, subdomain_ids, **kwargs):
                super().__init__(**kwargs)
                self.markers = markers
                self.subdomain_ids = subdomain_ids

            def eval_cell(self, values, x, cell):
                if self.markers[cell.index] == self.subdomain_ids['Vacuum']:
                    values[0] = 1.
                else:
                    values[0] = 10.

        rel_perm = relative_perm(self.subdomains, self.subdomains_ids, degree=0)

        # Define the variational form.
        a = r*rel_perm*fn.inner(fn.grad(u), fn.grad(v))*self.dx
        L = fn.Constant(0.)*v*self.dx

        # Define the boundary conditions.
        bc1 = fn.DirichletBC(V, fn.Constant(0.), self.boundaries, self.boundaries_ids['Bottom_Wall'])
        bc4 = fn.DirichletBC(V, fn.Constant(-10.), self.boundaries, self.boundaries_ids['Top_Wall'])
        bcs = [bc1, bc4]

        # Solve the problem.
        init_phi = fn.Function(V)
        fn.solve(a == L, init_phi, bcs)

        return Poisson.block_project(init_phi, self.mesh, self.restrictions_dict['vacuum_rtc'], self.subdomains,
                                     self.subdomains_ids['Vacuum'], space_type='scalar')

    @staticmethod
    def check_solver_options():
        """
        Check all the options available for the implemented solver.
        Returns
        -------
        None.
        """
        # Create some trivial nonlinear solver instance.
        mesh = fn.UnitIntervalMesh(1)
        V = fn.FunctionSpace(mesh, "CG", 1)
        problem = fn.NonlinearVariationalProblem(fn.Function(V)*fn.TestFunction(V)*fn.dx,
                                                 fn.Function(V))
        solver = mp.BlockPETScSNESSolver(problem)

        # Print out all the options for its parameters:
        fn.info(solver.parameters, True)

    def write_check_files(self):
        """
        Write the boundaries and subdomains check files. These files have
        .xdmf extension, which are ready to be opened with Paraview or any
        other compatible application. These files will be stored in the folder
        specified by the user (checkspath variable) on the __init__ method of
        this class.
        Returns
        -------
        None.
        """
        with mp.XDMFFile(self.checkspath + '/' + 'check_boundaries.xdmf') as out:
            out.write(self.boundaries)
        with mp.XDMFFile(self.checkspath + '/' + 'check_subdomains.xdmf') as out:
            out.write(self.subdomains)

    def write_restrictions(self):
        """
        Write the .xml files corresponding to the restrictions created in the
        generate_restrictions method of this class. These files can be used to
        load the restrictions once they are created. This method will also
        create check files, which are useful to check if the restrictions are
        created according to the user input. This check is useful for debugging
        purposes.
        Note that these files will have the name introduced by the user in the
        generate_restrictions method of this class.
        Returns
        -------
        None.
        """

        for rtc, name in zip(self.restrictions_tuple, self.restrictions_names):
            mp.File(self.restrictionspath + '/' + name + '.rtc.xml') << rtc
            mp.XDMFFile(self.checkspath + '/' + name + '.rtc.xdmf').write(rtc)

    @staticmethod
    def block_project(u, mesh, restriction, subdomain_data, project_id,
                      **kwargs):
        """
        Implement the FEniCS' project function for multiphenics. This method
        allows to extract and project the solution of implicit FEniCS functions
        into user-friendly functions, and then extract the solution of a
        specific subdomain (up to know subdomains of dimensions 1 or 2 are
        accepted). Applications of this method may be to extract the gradient
        of a function (when using FEniCS grad function) and get the solution
        in/on a desired subdomain, or do the same with the dot product when
        using FEniCS dot (or inner) function.
        The implemented method is the same one used when obtaining the weak
        form of a PDE. That is, we impose: f = u, where u is the function
        containing the desired data and f is the trial function. Then, we
        multiply by a test function and integrate. In that way, we are
        projecting the solution into a function that can be easily used for
        post-processing.
        Parameters
        ----------
        u: ufl.tensor...
            Tensor containing the desired solution.
        mesh: cpp.mesh.Mesh
            Mesh object.
        restriction: mesh.mesh_restriction.MeshRestriction
            Subdomain restriction.
        subdomain_data: cpp.mesh.MeshFunctionSizet
            Contains the all information of the subdomain.
        project_id: int
            Indentifier of the place (in the subdomain_data) where the
            solution is desired.
        **kwargs:
            Accepted kwargs are space_type, boundary_type sign and restricted.
            The space_type identifies the type of Function Space in which we
            want to project the solution into. It can be scalar (for scalar
            fields) or vectorial (for vectorial fields).
            The boundary_type is only required when subdomains of dimension 1
            are introduced, and it is used to specify the type of facets (they
            can be internal or external).
            The sign is only required when subdomains of dimension 1
            (boundaries) are introduced. It is used to specify the sign to
            evaluate the line integral For example, let us consider two
            subdomains whose ids are 0 and 1, respectively. To indicate that we
            are integrating quantities on subdomain 0, we should introduce '-'
            as sign, because it is the the subdomain with the lower id. Thus,
            for subdomain 1, we should introduce '+' as sign, which is the
            default one.
            The restricted kwarg indicates if the function to be projected is
            already restricted. This should be taken into account because
            UFL cannot restrict an expression twice.
        Raises
        ------
        TypeError
            This error will rise when the specified type of function space is
            unknown.
        NotImplementedError
            This error will raise when the dimension of the introduced
            subdomain is greater than 2.
        Returns
        -------
        sol: dolfin.function.function.Function
            Dolfin Function which can be easily used for post-processing tasks.
        """
        if 'space_type' not in kwargs:
            print("** WARNING: No Function Space type was specified. Assuming scalar space.", flush=True)
        if subdomain_data.dim() == 1 and 'boundary_type' not in kwargs:
            print("** WARNING: No boundary type specified. Assuming internal type.", flush=True)
        kwargs.setdefault('space_type', 'scalar')
        kwargs.setdefault('boundary_type', 'internal')
        kwargs.setdefault('sign', "+")
        kwargs.setdefault('restricted', False)
        kwargs.setdefault('function_space_degree', 2)
        # Create a Block Function Space.
        function_space_degree = kwargs.get('function_space_degree')

        if kwargs['space_type'] == 'scalar':
            aux_space = mp.BlockFunctionSpace([fn.FunctionSpace(mesh, 'CG', function_space_degree)],
                                              restrict=[restriction])
        elif kwargs['space_type'] == 'vectorial':
            aux_space = mp.BlockFunctionSpace([fn.VectorFunctionSpace(mesh, 'CG', function_space_degree)],
                                              restrict=[restriction])
        else:
            raise TypeError(f"Unknown type of Function Space: {kwargs['space_type']}")

        # Define the trial and test functions.
        trial, = mp.BlockTrialFunction(aux_space)
        test, = mp.BlockTestFunction(aux_space)

        # Define the measure of the subdomain and the variational problem.
        if subdomain_data.dim() == 2:
            dom_measure = fn.Measure('dx')(subdomain_data=subdomain_data)
            lhs = [[fn.inner(trial, test)*dom_measure(project_id)]]
            rhs = [fn.inner(u, test)*dom_measure(project_id)]
        elif subdomain_data.dim() == 1:
            if kwargs['boundary_type'] == 'internal':
                dom_measure = fn.Measure('dS')(subdomain_data=subdomain_data)
            elif kwargs['boundary_type'] == 'external':
                dom_measure = fn.Measure('ds')(subdomain_data=subdomain_data)
            # Check if the interface exists.
            assert fn.assemble(1*dom_measure(domain=mesh)) > 0., "The length of the interface is zero, wrong marking."
            lhs = [[fn.inner(trial, test)(kwargs['sign'])*dom_measure(project_id)]]
            if not kwargs.get('restricted'):
                rhs = [fn.inner(u, test)(kwargs['sign'])*dom_measure(project_id)]
            else:
                rhs = [fn.inner(u, test(kwargs['sign']))*dom_measure(project_id)]
        else:
            raise NotImplementedError(f"Domains of dimension {subdomain_data.dim()} are not supported.")

        # Define the variational form and solve.
        LHS = mp.block_assemble(lhs)
        RHS = mp.block_assemble(rhs)
        sol = mp.BlockFunction(aux_space)
        mp.block_solve(LHS, sol.block_vector(), RHS)

        return sol[0]

    def get_electric_field(self, subdomain_id, **kwargs):
        """
        Get the electric field given the potential and project it into the
        specified subdomain using the block_project method.
        Parameters
        ----------
        subdomain_id : string
            Subdomain identification number.
        Returns
        -------
        E : dolfin.function.function.Function
            Dolfin/FEniCS function containing the electric field information.
        """
        if subdomain_id == 'Vacuum' or subdomain_id == 'vacuum':
            subdomain_id_num = self.subdomains_ids[subdomain_id.capitalize()]
            rtc = self.restrictions_dict[subdomain_id.lower() + '_rtc']
            phi_sub = Poisson.block_project(self.phi, self.mesh, rtc, self.subdomains, subdomain_id_num,
                                            space_type='scalar')
            E_tensor = -fn.grad(phi_sub)
            E = Poisson.block_project(E_tensor, self.mesh, rtc, self.subdomains,
                                      subdomain_id_num, space_type='vectorial')
            return E
        elif subdomain_id == 'Liquid' or subdomain_id == 'liquid':
            E_l_r, E_l_z = Poisson.get_liquid_electric_field(kwargs)
            return E_l_r, E_l_z

    @staticmethod
    def get_liquid_electric_field(**kwargs):
        n_arr = ObtainVectors.get_normal_vectors_boundary(kwargs.get('mesh'), kwargs.get('subdomain_data'),
                                                          kwargs.get('boundary_id'))
        n_arr[:, 1] *= (-1)
        t_arr = ObtainVectors.get_tangential_vectors_boundary(n_arr)
        E_l_r = []
        E_l_z = []
        E_l_n_array = kwargs.get('normal_liquid')
        E_t_array = kwargs.get('tangential_liquid')
        for i in range(len(E_l_n_array)):
            l_r = (E_l_n_array[i] - E_t_array[i] / t_arr[i, 1] * n_arr[i, 1]) / (n_arr[i, 0] - t_arr[i, 0] / t_arr[i, 1] * n_arr[i, 1])
            l_z = (E_t_array[i] - l_r * t_arr[i, 0]) / t_arr[i, 1]
            E_l_r.append(l_r)
            E_l_z.append(l_z)
        return E_l_r, E_l_z

    def get_normal_field(self, n, E):
        """
        Get the normal field of the given electric field
        Parameters
        ----------
        n : ufl.geometry.FacetNormal
            Normal vector of the facets of the mesh, defined positively
            outwards.
        E : dolfin.function.function.Function
            Dolfin/FEniCS function of the electric field.
        mesh : dolfin.cpp.mesh.Mesh
            Mesh object.
        boundaries : dolfin.cpp.mesh.MeshFunctionSizet
            Dolfin/FEniCS object containing the information regarding the
            boundaries of the mesh.
        interface_rtc : multiphenics.mesh.mesh_restriction.MeshRestriction
            Restriction of the interface.
        boundary_id : int
            Boundary identification number.
        sign : str, optional
            String containing the sign of the projection. See the block_project
            method documentation for more information. The default is "+".
        Returns
        -------
        E_n : dolfin.function.function.Function
            Dolfin/FEniCS function containing the normal component of the
            electric field.
        """
        E_n = fn.dot(E, n)
        E_n = Poisson.block_project(E_n, self.mesh, self.restrictions_dict['interface_rtc'], self.boundaries,
                                    self.boundaries_ids['Interface'], space_type='scalar',
                                    boundary_type='internal', sign=n.side(), restricted=True)
        return E_n

    def get_tangential_component(self, t, E):
        E_t = fn.dot(E, t)
        E_t = Poisson.block_project(E_t, self.mesh, self.restrictions_dict['interface_rtc'], self.boundaries,
                                    self.boundaries_ids['Interface'], space_type='scalar',
                                    boundary_type='internal', sign=t.side(), restricted=True)
        return E_t

    @staticmethod
    def split_field_components(E, coords):
        """
        Split the components of the given electric field at a given coordinates
        Parameters
        ----------
        E : dolfin.function.function.Function
            Dolfin/FEniCS function containing the information of the electric
            field.
        coords : array like
            Array containing the coordinates (in this case, of the points
            forming the interface). This list/array must have the following
            form: coords = [r_coords, z_coords], where r_coords and z_coords
            are the lists/arrays containing the radial and axial coordinates,
            respectively.
        Returns
        -------
        E_r : numpy.ndarray
            Numpy array containing the radial component of the electric field
            at the given coordinates.
        E_z : numpy.ndarray
            Numpy array containing the axial component of the electric field
            at the given coordinates.
        """
        E_r = np.array([])
        E_z = np.array([])

        r_coords = coords[0]
        z_coords = coords[1]
        zip_coords = zip(r_coords, z_coords)

        for r, z in zip_coords:
            E_eval = E([r, z])
            E_r = np.append(E_r, E_eval[0])
            E_z = np.append(E_z, E_eval[1])
        return E_r, E_z

    def get_nd_current(self, j_ev_nd):
        """
        Get the current emitted through the interface by evaluating an integral, as indicated in Ximo's thesis p.46
        Args:
            j_ev_nd: FEniCS function of the non dimensional evaporated current through interface.

        Returns:
            Non-dimensional emitted current.
        """

        return fn.assemble(j_ev_nd*self.dS(self.boundaries_ids['Interface']))
