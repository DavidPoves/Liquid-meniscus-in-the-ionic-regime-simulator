"""
Author: David Poves Ros.
"""

# Import the modules.
import fenics as fn
import dolfin as df
import multiphenics as mp
import numpy as np
import os

from Tools.PostProcessing import PostProcessing
from Tools.GMSH_Interface import GMSHInterface


df.parameters["ghost_mode"] = "shared_facet"  # required by dS


class Stokes(object):
    def __init__(self, inputs, boundary_conditions, **kwargs):
        """
        Initialize the Stokes_sim solver to solve the hydrodynamics problem from Ximo's thesis. The process to achieve a
        solution is quite simple: Initialize the Stokes_sim class with the required inputs, as explained below, and call the
        solve method from this class.

        When solving Stokes_sim, it is assumed that the electrostatics problem has been solved. Thus, no check files are
        written on this class. If this is not the case (you may have a FEniCS function for the potential) then you may
        call the methods from the Poisson class, which are static (no need of initialization of the main class).
        Args:
            inputs: Dictionary with the inputs required to solve the Stokes_sim simulation. The dictionary keys must be:
                - Weber number.
                - Capillary number.
                - Relative perm: Relative permittivity of the medium with respect to vacuum.
                - B: Ratio of the meniscus tip (r_star) to the contact line radius.
                - Lambda: Non-dimensional parameter as defined in Ximo's thesis.
                - Kc: Non-dimensional parameter as defined in Ximo's thesis.
                - Non dimensional temperature.
                - Sigma: Non-dimensional surface charge density.
                - Phi, Chi: Non-dimensional parameters as defined in Ximo's thesis.
                - Potential.
                - Vacuum electric field: This is returned by the Poisson solver.
                - Vacuum normal component: Normal component of the vacuum electric field at the meniscus.
            boundary_conditions: Dictionary with a structure like the one below:
                boundary_conditions = {'boundary_name': {'Dirichlet/Neumann': [value, 'subdomain_name']}, ...}
            **kwargs: Accepted kwargs are:
                - boundaries: File containing all the information regarding boundaries.
                - boundaries_ids: Dictionary whose keys are the boundaries names, just as defined in GMSH, whose values
                are the ids of these boundaries.
                - subdomains: File containing all the information related with subdomains.
                - mesh: Mesh object of the problem.
                - restrictions: Dictionary whose keys have the following form: 'subdomain-name_rtc' and whose values are
                the restrictions of that specific subdomain. These are files required by multiphenics to define
                functions on specific subdomains/boundaries. These are useful to define the Lagrange multipliers on
                the meniscus boundary. Otherwise, it would much harder to define using raw FEniCS.
                - mesh_path: String containing the path where the path is stored. In case a mesh object is not provided,
                this kwarg must be introduced when initializing the class.
                - restrictions_path: String containing the path where the restrictions are stored. If the user does not
                provide any restrictions files, this kwarg must be an input.
                - filename: String containing the name of the .msh file (extension included).
        """

        # Unpack the inputs.
        self.We = inputs['Weber number']
        self.Ca = inputs['Capillary number']
        self.eps_r = inputs['Relative perm']
        self.B = inputs['B']
        self.Lambda = inputs['Lambda']
        self.Kc = inputs['Kc']
        self.T_h = inputs['Non dimensional temperature']
        self.sigma = inputs['Sigma']
        self.Phi = inputs['Phi']
        self.phi = inputs['Potential']
        self.Chi = inputs['Chi']
        self.E_v = inputs['Vacuum electric field']
        self.E_v_n = inputs['Vacuum normal component']

        # Load the boundary conditions.
        self.boundary_conditions = boundary_conditions

        # Handle kwargs.
        kwargs.setdefault('boundaries', None)
        kwargs.setdefault('boundaries_ids', None)
        kwargs.setdefault('subdomains', None)
        kwargs.setdefault('mesh', None)
        kwargs.setdefault('restrictions', None)
        kwargs.setdefault('mesh_path', '')
        kwargs.setdefault('restrictions_path', None)
        kwargs.setdefault('restrictions_names', None)
        kwargs.setdefault('interface_name', None)
        kwargs.setdefault('filename', '')

        # Set kwargs to the corresponding variables.
        self.boundaries = kwargs.get('boundaries')
        self.subdomains = kwargs.get('subdomains')
        self.mesh = kwargs.get('mesh')
        self.restrictions_dict = kwargs.get('restrictions')
        self.restrictions_path = kwargs.get('restrictions_path')
        self.mesh_folder_path = kwargs.get('mesh_path')
        self.boundaries_ids = kwargs.get('boundaries_ids')
        self.interface_name = kwargs.get('interface_name')
        self.filename = kwargs.get('filename')

        # Initialize other variables.
        self.geo_file = self.mesh_folder_path + '/' + self.filename.split('.')[0] + '.geo'
        self.subdomains_ids = None
        self.j_conv = None
        self.dx = None
        self.ds = None
        self.dS = None
        self.u = None
        self.u_n = None
        self.u_t = None
        self.p_star = None
        self.theta = None

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

    def load_mesh(self):
        """
        Method to load the mesh file given the mesh folder path.
        Returns:

        """
        return fn.Mesh(self.mesh_folder_path)

    def load_restrictions(self):
        """
        Load the restrictions files given its folder.
        Returns:

        """
        self.restrictions_dict = dict()
        for rtc_folder in os.listdir(self.restrictions_path):
            rtc_name = rtc_folder.split('_')[0].lower() + '_rtc'
            self.restrictions_dict[rtc_name] = mp.MeshRestriction(self.mesh, self.restrictions_path + '/' + rtc_folder)

    def get_mesh(self):
        """
        Method to check the different options available when loading a mesh. It can be given directly by its object, or
        the user may introduce the path. If none of them is given, an error will be raised.
        Returns:

        """
        # Do proper checks before loading the mesh.
        if self.mesh_folder_path is None and self.mesh is None:
            raise Exception('If a mesh has not been loaded when loading this class, user must include a path to load the .xml file under the variable mesh_path.')
        elif self.mesh_folder_path is not None and self.mesh is None:
            self.filename = self.mesh_folder_path.split('/')[-1]
            self.geo_file = self.filename.split('.')[0] + '.geo'
            self.mesh_folder_path = '/'.join(self.mesh_folder_path.split('/')[:-1])
            self.mesh = self.load_mesh()
        elif self.mesh is not None:
            self.check_mesh()

    def get_boundaries(self):
        """
        Load the boundaries data of the mesh. More specifically, this method loads the facet_region.xml file generated
        by the dolfin-converter.
        Returns:

        """
        if self.boundaries is None:
            bound_name = self.filename.split('.')[0] + '_facet_region.xml'
            file = self.mesh_folder_path + '/' + bound_name
            self.boundaries = fn.MeshFunction('size_t', self.mesh, file)

        if self.boundaries_ids is None:
            self.boundaries_ids, _ = GMSHInterface.get_boundaries_ids(self.geo_file)

    def get_subdomains(self):
        """
        Load the subdomains data for a given mesh. More specifically, this method loads the physical_region.xml file
        generated by the dolfin-converter.
        Returns:

        """
        if self.subdomains is None:
            sub_name = self.filename.split('.')[0] + '_physical_region.xml'
            file = self.mesh_folder_path + '/' + sub_name
            self.subdomains = df.MeshFunction('size_t', self.mesh, file)

        # Obtain the ids of the subdomains.
        self.subdomains_ids = GMSHInterface.get_subdomains_ids(self.geo_file)

    def get_restrictions(self):
        """
        Get restrictions based on user's input. If the user does not provide a restrictions variable nor a path to them,
        an error will raise. Otherwise, they will be loaded accordingly.
        Returns:

        """
        if self.restrictions_dict is None and self.restrictions_path is None:
            raise Exception('If restrictions are not loaded when initializing this class, the path to them must be included.')
        elif self.restrictions_dict is None and self.restrictions_path is not None:
            self.load_restrictions()
        elif self.restrictions_dict is not None:
            pass

    def get_measures(self):
        """
        Get measures of the domain given the required data.
        The defined measurements are:
            - dx: Area measure.
            - ds: External boundaries measure.
            - dS: Internal boundaries measure.
        Returns:

        """
        self.dx = fn.Measure('dx')(subdomain_data=self.subdomains)  # Area measure.
        self.ds = fn.Measure('ds')(subdomain_data=self.boundaries)  # External boundaries measures.
        self.dS = fn.Measure('dS')(subdomain_data=self.boundaries)  # Internal boundaries measures (interface boundary).

    def solve(self):
        """
        Solve the Stokes_sim problem based on the mathematical procedure presented by Ximo in this thesis.
        Returns:

        """

        # --------------------------------------------------------------------
        # DEFINE THE INPUTS #
        # --------------------------------------------------------------------
        self.get_mesh()
        self.get_boundaries()
        self.get_subdomains()
        self.get_restrictions()

        # Create a block of restrictions.
        """ This variable will be used by multiphenics when creating function spaces. It will create function spaces
        on the introduced restrictions. 
        """
        block_restrictions = [self.restrictions_dict['liquid_rtc'], self.restrictions_dict['liquid_rtc'],
                              self.restrictions_dict['interface_rtc']]

        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # FUNCTION SPACES #
        # --------------------------------------------------------------------
        V = fn.VectorFunctionSpace(self.mesh, "CG", 2)
        Q = fn.FunctionSpace(self.mesh, "DG", 1)
        L = fn.FunctionSpace(self.mesh, "DGT", 0)  # DGT 0.

        # Create a block function space.
        """ Block Function Spaces are similar to FEniCS function spaces. However, since we are creating function spaces
        based on the block of restrictions, we need to create a 'block of function spaces' for each of the restrictions.
        That block of functions is the list [V, Q, L] from the line of code below this comment. They are assigned in the
        same order in which the block of restrictions has been created, that is:
            - V -> liquid_rtc
            - Q -> liquid_rtc
            - L -> interface_rtc
        """
        W = mp.BlockFunctionSpace([V, Q, L], restrict=block_restrictions)
        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # TRIAL/TEST FUNCTIONS #
        # --------------------------------------------------------------------
        """ Trial and test functions are created the multiphenics commands for creating these functions. However, the
        difference wrt the FEniCS functions for this purpose, a trial/test function will be created for each of the
        restrictions (for each function space of the BlockFunctionSpace).
        """
        test = mp.BlockTestFunction(W)
        (v, q, l) = mp.block_split(test)

        trial = mp.BlockTrialFunction(W)
        (u, p, theta) = mp.block_split(trial)

        # Use a value of previous velocity to make the system linear, as explained by Ximo.
        u_prev = fn.Function(V)
        u_prev.assign(fn.Constant((0.1, 0.1)))

        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # MEASURES #
        # --------------------------------------------------------------------
        self.get_measures()
        self.dS = self.dS(self.boundaries_ids['Interface'])  # Restrict to the interface.

        # Check proper marking of the interface.
        assert fn.assemble(1*self.dS(domain=self.mesh)) > 0., "The length of the interface is zero, wrong marking."

        # --------------------------------------------------------------------
        # DEFINE THE VARIATIONAL PROBLEM #
        # --------------------------------------------------------------------
        r = fn.SpatialCoordinate(self.mesh)[0]
        n = fn.FacetNormal(self.mesh)
        tan_vector = fn.as_vector((n[1], -n[0]))
        e_r = fn.Constant((1., 0.))  # Define unit radial vector
        e_z = fn.Constant((0., 1.))  # Define unit axial vector
        aux_term = (self.eps_r*self.Ca*np.sqrt(self.B))/(1+self.Lambda*(self.T_h-1))

        # Define the term a.
        a = r * aux_term * fn.inner((fn.grad(u)+fn.grad(u).T), (fn.grad(v)+fn.grad(v).T))*self.dx(
            self.subdomains_ids['Liquid'])
        a += 2/r*aux_term*fn.dot(u, e_r)*fn.dot(v, e_r)*self.dx(self.subdomains_ids['Liquid'])

        # Define the term d.
        del_operation = fn.dot(fn.grad(u), u_prev)
        d = r*self.eps_r**2*self.We*fn.dot(del_operation, v)*self.dx(self.subdomains_ids['Liquid'])

        # Define the term l1.
        def evaporated_charge():
            return (self.sigma*self.T_h)/(self.eps_r*self.Chi)*fn.exp(
                -self.Phi/self.T_h*(1-self.B**0.25*fn.sqrt(self.E_v_n)))

        l1 = -r*evaporated_charge()*l("+")*self.dS

        # Define the term l2.
        l2 = r*self.sigma*fn.dot(self.E_v, tan_vector("-"))*fn.dot(v("+"), tan_vector("-"))*self.dS

        # Define the term b.
        def b(vector, scalar):
            radial_term = r*fn.dot(vector, e_r)
            axial_term = r*fn.dot(vector, e_z)
            return -(radial_term.dx(0) + axial_term.dx(1))*scalar*self.dx(self.subdomains_ids['Liquid'])

        # Define the term c.
        c1 = -r*fn.dot(v("+"), n("-"))*theta("+")*self.dS
        c2 = -r*fn.dot(u("+"), n("-"))*l("+")*self.dS

        # Define the tensors to be solved.
        # The following order is used.
        #       u            p           theta       #
        aa = [[a+d   ,     b(v,p)       , c1],       # Test function v
              [b(u,q),       0          , 0 ],       # Test function q
              [c2    ,       0          , 0 ]]       # Test function l

        bb = [l2, fn.Constant(0.)*q("+")*self.dS, l1]

        # --------------------------------------------------------------------
        # DEFINE THE BOUNDARY CONDITIONS #
        # --------------------------------------------------------------------
        """ When creating Dirichlet boundary conditions with the multiphenics code, a function space from the Block
        must be selected, depending on which subdomain/boundary should it be applied. To do so, the .sub method is used.
        The input is an integer, which depends on the function space in which you want the BC to be applied. For this
        case, inputs of 0, 1 and 2 are accepted, because we have 3 restrictions. The assignments of these ids to the 
        function space is the one done in the block of restrictions.
        """
        bcs_u = []
        bcs_p = []
        for i in self.boundary_conditions:
            if 'Dirichlet' in self.boundary_conditions[i]:
                bc_val = self.boundary_conditions[i]['Dirichlet'][1]
                if self.boundary_conditions[i]['Dirichlet'][0] == 'v':
                    bc = mp.DirichletBC(W.sub(0), bc_val, self.boundaries,
                                        self.boundaries_ids[i])

                    # Check the created boundary condition.
                    assert len(bc.get_boundary_values()) > 0., f'Wrongly defined boundary {i}'
                    bcs_u.append(bc)
                elif self.boundary_conditions[i]['Dirichlet'][0] == 'p':
                    bc = mp.DirichletBC(W.sub(1), bc_val, self.boundaries,
                                        self.boundaries_ids[i])
                    # Check the created boundary condition.
                    assert len(bc.get_boundary_values()) > 0., f'Wrongly defined boundary {i}'
                    bcs_p.append(bc)

        bcs_block = mp.BlockDirichletBC([bcs_u, bcs_p])

        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # SOLVE #
        # --------------------------------------------------------------------
        # Assemble the system.
        AA = mp.block_assemble(aa)
        BB = mp.block_assemble(bb)

        # Apply the boundary conditions.
        bcs_block.apply(AA)
        bcs_block.apply(BB)

        # Solve.
        uptheta = mp.BlockFunction(W)
        mp.block_solve(AA, uptheta.block_vector(), BB)
        (u, p, theta) = uptheta.block_split()

        self.u = u
        self.p_star = p
        self.theta = theta

        # Compute normal and tangential velocity components.
        u_n = fn.dot(u, n)
        self.u_n = Stokes.block_project(u_n, self.mesh, self.restrictions_dict['interface_rtc'], self.boundaries,
                                        self.boundaries_ids['Interface'], space_type='scalar',
                                        boundary_type='internal', sign='-')

        u_t = fn.dot(u, tan_vector)
        self.u_t = Stokes.block_project(u_t, self.mesh, self.restrictions_dict['interface_rtc'], self.boundaries,
                                        self.boundaries_ids['Interface'], space_type='scalar',
                                        boundary_type='internal', sign='+')

        # Compute the convection charge transport.
        special = (fn.Identity(self.mesh.topology().dim()) - fn.outer(n, n)) * fn.grad(self.sigma)
        self.j_conv = self.Kc * self.B**(3/2) * (fn.dot(self.sigma * n, fn.dot(fn.grad(self.u), n)) -
                                                 fn.dot(self.u, special))
        self.j_conv = Stokes.block_project(self.j_conv, self.mesh, self.restrictions_dict['interface_rtc'],
                                           self.boundaries, self.boundaries_ids['Interface'], space_type='scalar',
                                           boundary_type='internal', sign='-')

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
        # Create a Block Function Space.
        """
        Notice in the next step that we are using a continuous space of
        functions (CG). However, it should have been DCG, but it seems that it
        cannot be plotted.
        """
        if kwargs['space_type'] == 'scalar':
            aux_space = mp.BlockFunctionSpace([fn.FunctionSpace(mesh, 'CG', 1)],
                                              restrict=[restriction])
        elif kwargs['space_type'] == 'vectorial':
            aux_space = mp.BlockFunctionSpace([fn.VectorFunctionSpace(mesh, 'CG', 1)],
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

    @staticmethod
    def extract_velocity_components(u, coords):
        """
        Split the components of the given velocity field at given coordinates

        Parameters
        ----------
        u : dolfin.function.function.Function
            Dolfin/FEniCS function containing the information of the velocity
            field.
        coords : list/array
            List/array contaning the coodinates (in this case, of the points
            forming the interface). This list/array must have the following
            form: coords = [r_coords, z_coords], where r_coords and z_coords
            are the lists/arrays containing the radial and axial coordinates,
            respectively.

        Returns
        -------
        u_r : numpy.ndarray
            Numpy array containing the radial component of the velocity field
            at the given coordinates.
        u_z : numpy.ndarray
            Numpy array containing the axial component of the velocity field
            at the given coordinates.

        """
        u_r = np.array([])
        u_z = np.array([])

        r_coords = coords[0]
        z_coords = coords[1]
        zip_coords = zip(r_coords, z_coords)

        for r, z in zip_coords:
            u_eval = u([r, z])
            u_r = np.append(u_r, u_eval[0])
            u_z = np.append(u_z, u_eval[1])
        return u_r, u_z

    @staticmethod
    def check_evaporation_condition(u_n, j_ev, coords):
        """
        Check the absolute difference between the normal component of the velocity and the evaporated charge. According
        to the problem definition, these two parameters must be equal.
        Args:
            u_n: FEniCS function of the normal component of the velocity.
            j_ev: FEniCS function of the evaporated charge.
            coords: Array containing the radial and axial coordinates of the meniscus, where this must be fulfilled.

        Returns:

        """

        check = abs(u_n - j_ev)/j_ev
        check = PostProcessing.extract_from_function(check, coords)

        return check
