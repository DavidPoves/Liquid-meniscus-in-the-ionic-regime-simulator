"""
Author: David Poves Ros.
"""

# Import the modules.
import fenics as fn
import dolfin as df
import multiphenics as mp
import numpy as np
import re
from Tools.gmsh_handle import gmsh_handle
from Tools.PostProcessing import PostProcessing


df.parameters["ghost_mode"] = "shared_facet"  # required by dS


class Navier_Stokes(object):
    def __init__(self, inputs, boundary_conditions, **kwargs):

        # Unpack the inputs.
        self.We = inputs['Weber number']
        self.Ca = inputs['Capillary number']
        self.eps_r = inputs['Relative perm']
        self.B = inputs['B']
        self.Lambda = inputs['Lambda']
        self.T_h = inputs['Non dimensional temperature']
        self.sigma = inputs['Sigma']
        self.Phi = inputs['Phi']
        self.phi = inputs['Potential']
        self.Chi = inputs['Chi']
        self.r0 = inputs['Contact line radius']

        # Load the boundary conditions.
        self.boundary_conditions = boundary_conditions

        # Handle kwargs.
        kwargs.setdefault('boundaries', None)
        kwargs.setdefault('boundaries_ids', None)
        kwargs.setdefault('subdomains', None)
        kwargs.setdefault('mesh', None)
        kwargs.setdefault('restrictions', None)
        kwargs.setdefault('mesh_path', None)
        kwargs.setdefault('restrictions_path', None)
        kwargs.setdefault('filename', None)
        kwargs.setdefault('restrictions_names', None)
        kwargs.setdefault('interface_name', None)

        # Set kwargs to the corresponding variables.
        self.boundaries = kwargs.get('boundaries')
        self.subdomains = kwargs.get('subdomains')
        self.mesh = kwargs.get('mesh')
        self.restrictions = kwargs.get('restrictions')
        self.restrictionspath = kwargs.get('restrictions_path')
        self.meshpath = kwargs.get('mesh_path')
        self.filename = kwargs.get('filename')
        self.restrictionsnames = kwargs.get('restrictions_names')
        self.boundaries_ids = kwargs.get('boundaries_ids')
        self.interface_name = kwargs.get('interface_name')

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
        return fn.Mesh(self.meshpath + '/' + self.filename.split('.')[0] +
                       '.xml')

    def load_restrictions(self):
        self.restrictions = {}
        D = self.mesh.topology().dim
        for rtc_name in self.restrictionsnames:
            self.restrictions[rtc_name] = mp.MeshRestriction(self.mesh, self.restrictionspath + '/' + rtc_name + ".rtc.xml")

    def get_mesh(self):

        # Do proper checks before loading the mesh.
        if self.meshpath is None and self.mesh is None:
            raise Exception('If a mesh has not been loaded when loading this class, user must include a path to load the .xml file under the variable mesh_path.')
        elif self.meshpath is not None and self.mesh is None and self.filename is None:
            raise NameError('When a mesh is not loaded, the user must specify the path to the .xml file and the name of the file. You did not introduce the name of the file. Introduce it when loading the class with variable filename.')
        elif self.meshpath is not None and self.mesh is None and self.filename is not None:
            self.geo_file = self.filename.split('.')[0] + '.geo'
            self.mesh = self.load_mesh()
        elif self.mesh is not None:
            self.check_mesh()

    def get_boundaries(self):

        if self.boundaries is None:
            bound_name = self.filename.split('.')[0] + '_facet_region.xml'
            file = self.meshpath + '/' + bound_name
            self.boundaries = fn.MeshFunction('size_t', self.mesh, file)

        if self.boundaries_ids is None:
            self.boundaries_ids = gmsh_handle.get_physical_curves_and_tags(self.geo_file, self.meshpath)

            if self.interface_name is not None:
                self.boundaries_ids['Interface'] = self.boundaries_ids.pop(self.interface_name)
            else:
                raise Exception('If the boundaries ids are not introduced, the interface name must be included.')

            # Now, we should clean the keys of the dictionary.
            """
            The process below is carried out so it is easier for the user to
            call the boundaries.
            """
            re_pattern = re.compile(r"(.*?)")  # Check for quoted words.
            old_keys = [key for key in self.boundaries_ids.keys()]
            new_keys = []
            for key in self.boundaries_ids:
                string = re_pattern.findall(key)
                final_str = ''
                for count in string:
                    if count.isalpha() or '_' in count:
                        final_str += count
                new_keys.append(final_str)
            for new_key, old_key in zip(new_keys, old_keys):
                self.boundaries_ids[new_key] = self.boundaries_ids.pop(old_key)

    def get_subdomains(self):

        if self.subdomains is None:
            sub_name = self.filename.split('.')[0] + '_physical_region.xml'
            file = self.meshpath + '/' + sub_name
            self.subdomains = df.MeshFunction('size_t', self.mesh, file)

        # Obtain the ids of the subdomains.
        check_first = gmsh_handle.check_which_physical_first(self.filename,
                                                             self.meshpath)

        # Check if the proper number of subdomains were defined in the .geo.
        gmsh_handle.get_physical_surfaces(self.geo_file, self.meshpath)


        if check_first[1]:
            self.lower_subdomain_id = 1
            self.upper_subdomain_id = 2
        else:
            try:
                self.lower_subdomain_id = max([val for val in self.boundaries_ids.values()]) + 1
                self.upper_subdomain_id = self.lower_subdomain_id + 1
            except AttributeError:
                raise AttributeError('The boundaries ids have not been generated. Generate them first and re-run this method.')

    def get_restrictions(self):
        if self.restrictions is None and self.restrictionspath is None:
            raise Exception('If restrictions are not loaded when initializing this class, the path to them must be included.')
        elif self.restrictions is None and self.restrictionspath is not None and self.restrictionsnames is None:
            raise NameError('The names of the restrictions names were not loaded.')
        elif self.restrictions is None and self.restrictionspath is not None and self.restrictionsnames is not None:
            self.load_restrictions()
        elif self.restrictions is not None:
            pass

    def get_measures(self):
        self.dx = fn.Measure('dx')(subdomain_data=self.subdomains)
        self.ds = fn.Measure('ds')(subdomain_data=self.boundaries)
        self.dS = fn.Measure('dS')(subdomain_data=self.boundaries)

    def solve(self):

        # --------------------------------------------------------------------
        # DEFINE THE INPUTS #
        # --------------------------------------------------------------------
        self.geo_file = self.filename.split('.')[0] + '.geo'
        self.get_mesh()
        self.get_boundaries()
        self.get_subdomains()
        self.get_restrictions()

        lower_rtc = self.restrictions[self.restrictionsnames[0]]
        upper_rtc = self.restrictions[self.restrictionsnames[1]]
        try:
            interface_rtc = self.restrictions['interface_restriction']
        except KeyError:
            try:
                interface_rtc = self.restrictions[self.interface_name]
            except KeyError:
                try:
                    interface_rtc = self.restrictions['Interface']
                except KeyError:
                    raise KeyError("Interface restriction name was not one of the expected ones: '{self.interface_name}', 'Interface' or 'interface_restriction'")
        block_restrictions = [upper_rtc, upper_rtc, interface_rtc]

        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # FUNCTION SPACES #
        # --------------------------------------------------------------------
        V = fn.VectorFunctionSpace(self.mesh, "CG", 2)
        Q = fn.FunctionSpace(self.mesh, "CG", 1)
        L = fn.FunctionSpace(self.mesh, "DGT", 1)
        W = mp.BlockFunctionSpace([V, Q, L], restrict=block_restrictions)
        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # TRIAL/TEST FUNCTIONS #
        # --------------------------------------------------------------------
        test = mp.BlockTestFunction(W)
        (v, q, l) = mp.block_split(test)

        trial = mp.BlockTrialFunction(W)
        (u, p, theta) = mp.block_split(trial)

        u_prev = fn.Function(V)
        u_prev.assign(fn.Constant((0., 0.)))

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
        t = fn.as_vector((n[1], -n[0]))
        e_r = fn.Constant((1., 0.))  # Define unit radial vector
        e_z = fn.Constant((0., 1.))  # Define unit axial vector
        aux_term = (self.eps_r*self.Ca*self.B**0.5)/(1+self.Lambda*(self.T_h-1)
                                                     )

        # Compute the eletric field and its normal component.
        """
        Since we will be using the tangential component of the electric field,
        and it must be the same for the vacuum and liquid fields, we will use
        the vacuum field.
        """
        E_v = -fn.grad(self.phi("-"))
        E_v_n = fn.dot(E_v, n("-"))

        # Define the term a.
        a = r * aux_term * fn.inner((fn.grad(u)+fn.grad(u).T),(fn.grad(v)+fn.grad(v).T))*self.dx(self.upper_subdomain_id)
        a += 2*(aux_term*fn.dot(u, e_r)*fn.dot(v, e_r))/r*self.dx(self.upper_subdomain_id)

        # Define the term d.
        nabla_term_r = r*fn.dot(u_prev, e_r)
        nabla_term_z = fn.dot(u_prev, e_z)
        d = r*self.eps_r**2*self.We*fn.dot((1/r*nabla_term_r.dx(0)+nabla_term_z.dx(1))*u, v)*self.dx(self.upper_subdomain_id)

        # Define the term l1.
        def evaporated_charge():
            return (self.sigma*self.T_h)/(self.eps_r*self.Chi)*fn.exp(-self.Phi/self.T_h*(1-self.B**0.25*fn.sqrt(E_v_n)))

        term_check = evaporated_charge()

        l1 = -r*evaporated_charge()*l("-")*self.dS

        # Define the term l2.
        l2 = r*self.sigma*fn.dot(E_v, t("-"))*fn.dot(v("-"), t("-"))*self.dS

        # Define the term b.
        def b(vector, scalar):
            radial_term = r*fn.dot(vector, e_r)
            axial_term = r*fn.dot(vector, e_z)
            return -(radial_term.dx(0) + axial_term.dx(1))*scalar*self.dx(self.upper_subdomain_id)

        # Define the term c.
        c1 = -r*fn.dot(v("+"),n("+"))*theta("+")*self.dS
        c2 = -r*fn.dot(u("+"),n("+"))*l("+")*self.dS

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
        (u, p, t) = uptheta.block_split()

        return u, p, theta

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
    def check_evaporation_condition(mesh, interface_rtc, boundaries, u, j_ev,
                                    coords, boundary_id=7):

        n = fn.FacetNormal(mesh)
        u_n = fn.dot(u, n)

        u_n = Navier_Stokes.block_project(u_n, mesh, interface_rtc, boundaries,
                                          boundary_id, space_type='scalar',
                                          boundary_type='internal', sign='+')
        j_ev = Navier_Stokes.block_project(j_ev, mesh, interface_rtc, boundaries,
                                          boundary_id, space_type='scalar',
                                          boundary_type='internal', sign='-',
                                          restricted=True)
        check = u_n - j_ev
        check = PostProcessing.extract_from_function(check, coords)

        return check
