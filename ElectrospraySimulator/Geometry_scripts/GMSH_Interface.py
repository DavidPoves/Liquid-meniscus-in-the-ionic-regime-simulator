import re

import numpy as np
from py2gmsh import Mesh, Entity

from ElectrospraySimulator.Tools.CreateMesh import create_mesh, write_mesh
from ElectrospraySimulator.Tools.CreateMesh import str_2_num
from ElectrospraySimulator.Tools.EvaluateString import NumericStringParser
from ElectrospraySimulator.GUI_scripts.MeshMenu import run_app


class GMSHInterface(object):

    def __init__(self):
        self.geoPath = ''  # Initialize the path where the .geo file will be stored.
        self.filename = '' # Initialize the name of the .geo file.
        self.refinement = ''  # This string will be later modified by the user to indicate the refinement of mesh.
        self.msh_filename = ''
        self.geo_filename = ''
        self.mesh_filename = ''
        self.app = None  # Variable containing all the mesh details specified by the user.

        # Define the geometry and mesh object, as well as some of its properties.
        self.my_mesh = Mesh()
        self.max_element_size = None
        self.min_element_size = None
        self.length_from_curvature = None
        self.length_extend = None

        # Define auxiliary variables.
        self.interface_fun = ''
        self.interface_fun_r = ''
        self.interface_fun_z = ''

        # Initialize physical groups and geometry parameters.
        self.p_dict = dict()  # Create a dictionary containing all the points of the geometry.
        self.interface_points = dict()  # Dictionary that will isolate the points conforming the interface.
        self.list_points_interface = list()
        self.refinement_points_tip = None  # List containing the refinement points of the tip.
        self.refinement_points_knee = None  # List containing the refinement points of the knee.
        self.point_num = 1  # Number of points counter.
        self.key = ''  # Variable creating the dicts of the self.p_dict variable.
        self.interface_end_point = None  # Locator of the last point defining the interface
        self.interface_end_point_z = None  # Locator of the z coordinate of the last point defining the interface
        self.inlet = Entity.PhysicalGroup(name='Inlet', mesh=self.my_mesh)
        self.twr = Entity.PhysicalGroup(name='Tube_Wall_R', mesh=self.my_mesh)
        self.twl = Entity.PhysicalGroup(name='Tube_Wall_L', mesh=self.my_mesh)
        self.bw = Entity.PhysicalGroup(name='Bottom_Wall', mesh=self.my_mesh)
        self.lwr = Entity.PhysicalGroup(name='Lateral_Wall_R', mesh=self.my_mesh)
        self.tw = Entity.PhysicalGroup(name='Top_Wall', mesh=self.my_mesh)
        self.lwl = Entity.PhysicalGroup(name='Lateral_Wall_L', mesh=self.my_mesh)
        self.interface = Entity.PhysicalGroup(name='Interface', mesh=self.my_mesh)

        self.vacuum = Entity.PhysicalGroup(name='Vacuum', mesh=self.my_mesh)
        self.liquid = Entity.PhysicalGroup(name='Liquid', mesh=self.my_mesh)

    @staticmethod
    def set_transfinite_line(tag, nodes, progression=1):
        """
        Create the code required to set a GMSH transfinite line (curve).
        Args:
            tag (int): Tag of the curve.
            nodes (int): Number of nodes that will be created on the introduced curve.
            progression (int): Using Progression expression’ instructs the transfinite algorithm to distribute the nodes
            following a geometric progression (Progression 2 meaning for example that each line element in the series
            will be twice as long as the preceding one).

        Returns:

        """
        if not isinstance(tag, int) or not isinstance(nodes, int) or not isinstance(progression, int):
            raise TypeError('Inputs of this function must be integers.')
        else:
            code = f"Transfinite Line {{{tag}}} = {nodes} Using Progression {progression}"
            return code

    @staticmethod
    def set_transfinite_surface(tag):
        """
        This method will create a transfinite surface from a given surface id tag.
        Args:
            tag (int): Tag of the surface.

        Returns:
            GMSH-ready code to generate a transfinite surface.
        """
        if not isinstance(tag, int):
            raise TypeError('Id tag must be an integer.')
        else:
            code = f"Transfinite Surface {{{tag}}}"
            return code

    @staticmethod
    def compute_curve_length(curve):
        """
        Compute the length of a given curve object. The implemented code extracts the points of the curve and do a
        simple operation to compute the distance between points.
        Args:
            curve (py2gmsh.Entity.Curve): py2gmsh curve object from which the distance will be calculated.

        Returns:
            Distance between the two points conforming the curve.
        """
        if not isinstance(curve, Entity.Curve):
            raise TypeError(f'Inputs must be of the type py2gmsh.Entity.Curve, not {type(curve)}.')
        points = curve.points
        point1 = points[0].xyz
        point2 = points[1].xyz
        return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    @staticmethod
    def compute_boundary_length(boundary):
        """
        Compute the length of a given boundary (physical group, for example, self.inlet). This method iterates through
        the curves conforming the boundary and applies the GMSHInterface.compute_curve_length() to compute its length.
        Args:
            boundary (Entity.PhysicalGroup): Physical group object of the boundary.

        Returns:
            length (float): Length (or arc length) of the considered boundary.
        """
        if not isinstance(boundary, Entity.PhysicalGroup):
            return TypeError(f'Input must be of the type py2gmsh.Entity.PhysicalGroup, not {type(boundary)}')
        length = 0
        for curve in list(boundary.curves.values()):
            length += GMSHInterface.compute_curve_length(curve)
        return length

    @staticmethod
    def length_wrapper(curve=None, points=None, boundary=None):
        """
        General wrapper of the length computation methods.
        Args:
            curve (Entity.Curve): If a curve object is introduced, its length will be computed as defined in the
            GMSHInterface.compute_curve_length method.
            points (array like): Array containing two points of the type Entity.Point.
            boundary (Entity.PhysicalGroup): Boundary from which is length/ar length will be computed, based on the
            compute_boundary_length method.

        Returns:
            Distance/length, based on the input type.
        """
        if curve is not None:
            return GMSHInterface.compute_curve_length(curve)
        elif points is not None:
            point1 = points[0].xyz
            point2 = points[1].xyz
            return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
        elif boundary is not None:
            return GMSHInterface.compute_boundary_length(boundary)

    def adaptive_refinement_tip(self):
        """
        Small Algorithm to adaptively refine the tip of the interface, to obtain a much higher quality of the mesh at
        that part. This method is independent on the number of points defining the interface, thus it can be widely
        used.

        This algorithm will calculate the distance between all the points defining the interface, and then it will
        'project' these points onto the Lateral Wall Left and the Tube Wall Left. In this way, we will be forcing GMSH
        to dramatically increment the quality of the mesh at that zone by increasing the density of nodes. This method
        will yield increments of quality of ten times their quality without the algorithm, or even more in less refined
        cases.
        """
        if self.refinement_points_tip is None:  # This if statement will be initialized by the Lateral Wall L refinement
            self.refinement_points_tip = np.array([])
            prev_z = self.interface_end_point_z
            for i in np.arange(1, len(self.list_points_interface)):
                distance = GMSHInterface.length_wrapper(points=[self.list_points_interface[i],
                                                                self.list_points_interface[i - 1]])
                self.refinement_points_tip = np.append(self.refinement_points_tip, prev_z + distance)
                self.p_dict[self.key] = Entity.Point([0, prev_z + distance, 0], mesh=self.my_mesh)
                self.point_num += 1
                self.key = 'p' + str(self.point_num)
                prev_z += distance
        else:  # This statement will be executed by the refinement of the Tube Wall Left boundary.
            for z in self.refinement_points_tip[::-1]:
                self.p_dict[self.key] = Entity.Point([0, 2*self.interface_end_point_z - z, 0], mesh=self.my_mesh)
                self.point_num += 1
                self.key = 'p' + str(self.point_num)

    @staticmethod
    def avoid_built_in_functions(string, replace_by=''):
        """
        Avoid built-in functions from being replaced by substituting them by another character/string. This will be
        useful when parsing the user input for the function of the interface.
        An example of its usefulness is when looking for the independent parameter of an equation. If the independent
        parameter character is contained within one of the built in functions, it will be replaced by the corresponding
        value, which is not desired.
        Args:
            string: the expression to be parsed.
            replace_by: expression by which built-in functions will be replaced.

        Returns:
            Expression without built-in functions.
        """

        # Define the built-in functions (this should be updated if any other function is added to the parser).
        """ Built-in functions must be defined with blank spaces between letters in case there are words, like sin, cos,
        exp, etc.
        """
        funs = ['s i n', 't a n', 'c o s', 'e x p', 'a b s', 't r u n c', 'r o u n d', 's g n', 'P I', 'E']

        for fun in funs:
            string = string.replace(fun, replace_by)

        return string

    @staticmethod
    def get_independent_var_from_equation(string):
        """
        Get the independent parameter of an equation, provided that it has only dependence on a single parameter. To do
        so, a regexp pattern will be used to detect letters within the expression.
        Args:
            string: expression containing a single independent parameter.

        Returns:

        """
        ind_var = ' '.join(re.split("[^a-zA-Z]*", string)).strip()

        # Avoid false positives of the accepted functions from the parser.
        ind_var = GMSHInterface.avoid_built_in_functions(ind_var)

        # Obtain the independent variable.
        aux = []
        for i in ind_var:
            if i.isalpha():
                aux.append(i)
        aux = set(aux)

        if len(aux) > 1:
            raise NotImplementedError('One of the introduced equations has more than one independent parameter.')
        else:
            return list(aux)[0]

    @staticmethod
    def replace_ind_var(string, ind_var, value):
        """
        Replace the independent parameter by an input value, to be later evaluated by a parser.
        Args:
            string: Expression containing the independent parameter.
            ind_var: string. The independent parameter.
            value: int or float. The value by which the independent parameter will be replaced.

        Returns:
            String containing the introduced value instead of the independent parameter.
        """
        # Define a dictionary to replace given functions by an unique non-alphanumeric character.
        avail_chars = ['!', '@', '#', '$', '%', '&', '?', '¿', '¡', 'ç']
        replacements = {}
        nsp_ = NumericStringParser()
        functions = list(nsp_.fn.keys())
        functions.append('PI')
        functions.append('E')

        for i in range(0, len(avail_chars)):
            replacements[functions[i]] = avail_chars[i]

        pattern = '|'.join([fun for fun in functions])
        matches = list(set(re.findall(pattern, string)))

        for match in matches:
            string = string.replace(match, replacements[match])

        # Now, we can safely replace the independent variable.
        string = string.replace(ind_var, value)

        # Put the built in functions into place (they were previously replaced).
        for match in matches:
            string = string.replace(replacements[match], match)

        return string

    @staticmethod
    def angle_handler(string):
        """
        If called, degrees will be replaced by radians, since the numerical parser works with radians.
        Args:
            string: Expression where the degrees are present.

        Returns:
            string with the degrees replaced by radians.
        """
        nsp_local = NumericStringParser()
        trig_functions = ['sin', 'cos', 'tan']

        for fun in trig_functions:
            pattern = fr'[^a-zA-Z]*{fun}\((.*?)\)'
            matches = re.findall(pattern, string)
            for match in matches:
                if re.search('[a-zA-Z]', match) is None:
                    """
                    At this stage of the code, when calling this function in the geometry generator, all the
                    independent variables have been replaced. Therefore, only numbers and built-in functions exist. The
                    previous re pattern will match everything between parentheses, and this re.search method will
                    check if there's any letter between the detected parenthesis. In that way, we are able to nest
                    several trigonometric functions within a single one (for example, tan(2*cos(49.3))).
                    
                    Note: The variable 'pattern' will detect anything between parentheses which are preceded by one of
                    the defined trigonometric functions. That is, it will detect cos(30), but not (2-1), for example.
                    """
                    string = string.replace(match, str(np.radians(nsp_local.eval(match))))
        return string

    @staticmethod
    def extract_points_from_geo(filepath):
        """
        Extract all the points of a given .geo file. It can also extract curves ids and the points that make each of the
        curves. This is possible by the use of regexp patterns.
        Args:
            filepath: string. Path of the .geo file (with the extension).

        Returns:
            points_data: Dictionary whose keys are the points' ids and its values are their coordinates.
            curves_data: Dictionary whose keys are the curves' ids and its values are the points' ids which conform the
            curve.
        """
        # Create a dictionary with the id of the point as the key and the coordinates as the value.
        points_data = dict()
        curves_data = dict()

        # Define regexp patterns.
        pattern_br = r"\{(.*?)\}"  # Catch everything between curly braces.
        pattern_pa = r"\(([^\)]+)\)"  # Catch everything between parenthesis.

        # Read the file.
        with open(filepath, 'r') as f:  # Open the file with read permissions ONLY.
            for line in f:
                if re.match('Point', line):  # Points have to be read
                    point = []
                    content_re_br = re.findall(pattern_br, line)  # Get the coordinates of the point.
                    content_re_pa = re.findall(pattern_pa, line)  # Get the point id.
                    for str_point in content_re_br[0].strip().split(',')[:-1]:  # Iterate through the coords of point.
                        point.append(str_2_num(str_point))  # Append each of the coords of point.
                    point = np.array(point)  # Transform Python array to Numpy array.
                    points_data[content_re_pa[0]] = point
                elif re.match(r"Curve\(([^\)]+)\)", line):  # Curves have to be read.
                    curve_id = re.findall(pattern_pa, line)[0]
                    curves = re.findall(pattern_br, line)[0].strip()
                    curves_data[curve_id] = curves
        f.close()  # Close the file to avoid undesired modifications.

        return points_data, curves_data

    @staticmethod
    def get_boundaries_ids(filepath):
        """
        Get the ids of each of the boundaries of the loaded geometry, as defined in GMSH.
        Args:
            filepath: Path of the .geo file (with the extension).

        Returns:
            boundaries_ids: Dictionary whose keys are the names of the boundaries, as defined in GMSH; and their
            corresponding values are the ids of the boundaries. This id is unique for each curve.
            physical_curves: Dictionary whose keys are the names of the boundaries, as defined in GMSH; and their
            values are the curves that conform a physical curve.
        """
        # Define the necessary regexp patterns.
        parentheses_pattern = r"\(([^\)]+)\)"  # Get everything between parentheses.
        quotes_pattern = r'"([A-Za-z0-9_\./\\-]*)"'  # Get everything between double quotes.
        pattern_br = r"\{(.*?)\}"  # Catch everything between curly braces.

        # Preallocate the storage objects.
        boundaries_ids = dict()
        names = []
        tags = []
        physical_curves = dict()

        # Open the file.
        with open(filepath, 'r') as f:  # Open the file with reading permission ONLY ('r')
            for line in f:
                if re.search('Physical Curve', line):  # Search for the physical curves of the file.
                    p = re.findall(parentheses_pattern, line)  # Get everything between the parentheses.
                    b = re.findall(pattern_br, line)[0]
                    p = p[0].split(',')  # Split the resulting string by the comma to separate name from tag.
                    name = re.findall(quotes_pattern, p[0])  # To eliminate double quotes resulting from previous re.
                    tag = p[1]
                    names.append(name[0])
                    physical_curves[name[0]] = b
                    tags.append(tag.strip())
        f.close()
        for i in range(len(names)):
            boundaries_ids[names[i]] = int(str_2_num(tags[i]))

        return boundaries_ids, physical_curves

    @staticmethod
    def get_subdomains_ids(filepath):
        """
        Get the ids of each of the subdomains defined in GMSH. To do so, regexp patterns are used.
        Args:
            filepath: Path of the .geo file (with the extension).

        Returns:
            Dictionary whose keys are the subdomain names and their corresponding values are the ids of each of these
            subdomains.
        """
        parentheses_pattern = r"\(([^\)]+)\)"  # Get everything between parentheses.
        quotes_pattern = r'"([A-Za-z0-9_\./\\-]*)"'

        # Preallocate the storage objects.
        subdomains_ids = dict()
        names = []
        tags = []

        # Open the file.
        with open(filepath, 'r') as f:  # Open the file with reading permission ONLY ('r')
            for line in f:
                if re.search('Physical Surface', line):  # Search for the physical curves of the file.
                    p = re.findall(parentheses_pattern, line)  # Get everything between the parentheses.
                    p = p[0].split(',')  # Split the resulting string by the comma to separate name from tag.
                    name = re.findall(quotes_pattern, p[0])  # To eliminate double quotes resulting from previous re.
                    tag = p[1]
                    names.append(name[0])
                    tags.append(tag.strip())
        f.close()
        for i in range(len(names)):
            subdomains_ids[names[i]] = int(str_2_num(tags[i]))

        return subdomains_ids

    def reset_geom_params(self):
        """
        Reset the necessary parameters to create a new geometry.
        Returns:

        """
        # Reset the mesh object.
        self.my_mesh = Mesh()

        # Initialize physical groups and geometry parameters.
        self.p_dict = dict()  # Create a dictionary containing all the points of the geometry.
        self.interface_points = dict()  # Dictionary that will isolate the points conforming the interface.
        self.list_points_interface = list()
        self.refinement_points_tip = None  # List containing the refinement points of the tip.
        self.refinement_points_knee = None  # List containing the refinement points of the knee.
        self.point_num = 1  # Number of points counter.
        self.key = ''  # Variable creating the dicts of the self.p_dict variable.
        self.interface_end_point = None  # Locator of the last point defining the interface
        self.interface_end_point_z = None  # Locator of the z coordinate of the last point defining the interface
        self.inlet = Entity.PhysicalGroup(name='Inlet', mesh=self.my_mesh)
        self.twr = Entity.PhysicalGroup(name='Tube_Wall_R', mesh=self.my_mesh)
        self.twl = Entity.PhysicalGroup(name='Tube_Wall_L', mesh=self.my_mesh)
        self.bw = Entity.PhysicalGroup(name='Bottom_Wall', mesh=self.my_mesh)
        self.lwr = Entity.PhysicalGroup(name='Lateral_Wall_R', mesh=self.my_mesh)
        self.tw = Entity.PhysicalGroup(name='Top_Wall', mesh=self.my_mesh)
        self.lwl = Entity.PhysicalGroup(name='Lateral_Wall_L', mesh=self.my_mesh)
        self.interface = Entity.PhysicalGroup(name='Interface', mesh=self.my_mesh)

        self.vacuum = Entity.PhysicalGroup(name='Vacuum', mesh=self.my_mesh)
        self.liquid = Entity.PhysicalGroup(name='Liquid', mesh=self.my_mesh)

    def geometry_generator(self, interface_fun=None, interface_fun_r=None, interface_fun_z=None, factor=10, **kwargs):
        """
        Generate the geometry given some parameters. The generation of the geometry has been created in such a way to
        let the user introduce any shape of the interface by introducing the corresponding function(s). The process of
        creating the interface, works as follows:
            1. Create the points defining the shape of the domain. To establish a common way of defining these points
            independently on how the user introduces the r array or the independent parameter array, all the points will
            be defined counterclockwise. In this way, we can create a general way of defining the domain.\n
            2. Create the curves and assign it a physical curve. Basically, physical curve = FEniCS boundary. By
            defining these physical groups, a file wil be generated by the mesh creator so that FEniCS will be able to
            recognize these boundaries, which will then be identified by the physical group id contained in the .geo
            file.\n
            3. Create curveloops. This step is crucial for the each of the subdomains' definitions. A curveloop is a set
            of curves that enclose a subdomain. It must be a closed loop and it should be defined following a logical
            direction. Notice that a badly defined curveloop will not be notified here. However, one must always check
            the .geo file within GMSH to make sure all the steps above have been successful.\n
        This generator has been designed by taking into account that the geometry is non-dimensionalized by the radius
        of the capillary tube.\n
        Args:
            interface_fun: string. Expression containing the equation y(x) for the interface. In this case, there is an
            unique expression for the interface shape. This expression will be used when none of the other options are
            used (interface_fun_r and interface_fun_z must be None). Otherwise, an error will rise.
            interface_fun_r: Expression containing the equation r(s) which will create the radial coordinates of the
            interface. It should only depend on a single parameter.
            interface_fun_z: Expression containing the equation z(s) which will create the z coordinates of the
            interface. It should only depend on a single parameter.
            factor: Ratio between the top wall z coordinate and bottom wall coord. Optional, default is 10.
            **kwargs:
                    filename: String containing the name of the generated geometry (with the extension).
                    number_points: Number of points that will define the shape of the interface. This kwarg is necessary
                    if the user does not input an r or independent parameter array. Default is 300.
                    angle_unit: 'degrees' of 'radians'. If no angle is present, ignore this kwarg. Default is radians.
                    r: Array containing the radial coords of the points defining the interface. Required if
                    interface_fun is used.
                    independent_param: Array containing the values of the independent parameter. Required if
                    interface_fun_r and interface_fun_z are used.

        Returns:

        """

        # Define default values for the kwargs.
        kwargs.setdefault('filename', 'GeneratedGeometry.geo')
        kwargs.setdefault('number_points', 300)
        kwargs.setdefault('angle_unit', 'radians')

        # Do some checks before proceeding.
        if interface_fun is not None and kwargs.get('r') is None:
            print('** WARNING: No array for the r coordinates was given. Assuming equidistant points based on number of points selected.', flush=True)
            kwargs.setdefault('r', np.linspace(0, 1, kwargs.get('number_points')))
        elif interface_fun is not None and interface_fun_r is not None and interface_fun_z is not None:
            raise NotImplementedError('Incompatible functions were introduced.')
        elif interface_fun is not None and interface_fun_r is not None and interface_fun_z is None:
            raise NotImplementedError('Incompatible functions were introduced.')
        elif interface_fun is not None and interface_fun_r is None and interface_fun_z is not None:
            raise NotImplementedError('Incompatible functions were introduced.')

        if interface_fun_r is not None and interface_fun_z is not None and kwargs.get('independent_param') is None:
            print('** WARNING: No array for the independent parameter was given. Assuming equidistant parameters based on number of points selected.', flush=True)
            kwargs.setdefault('independent_param', np.linspace(0, 1, kwargs.get('number_points')))

        self.filename = kwargs.get('filename')

        # Initialize the string parser class.
        nsp = NumericStringParser()

        # Save the original functions in the self object.
        self.interface_fun = interface_fun
        self.interface_fun_r = interface_fun_r
        self.interface_fun_z = interface_fun_z

        # Detect which is the independent variable in the string.
        if isinstance(self.interface_fun, str) or isinstance(self.interface_fun_r, str) or \
                isinstance(self.interface_fun_z, str):

            if self.interface_fun is not None:  # The user has introduced a function of the form z = f(r).
                ind_var = GMSHInterface.get_independent_var_from_equation(interface_fun)
                if kwargs.get('angle_unit') == 'degrees':
                    self.interface_fun = GMSHInterface.angle_handler(self.interface_fun)

            elif self.interface_fun_z is not None and self.interface_fun_r is not None:
                ind_var_r = GMSHInterface.get_independent_var_from_equation(interface_fun_r)
                ind_var_z = GMSHInterface.get_independent_var_from_equation(interface_fun_z)
                assert ind_var_r == ind_var_z, 'Independent parameters for r and z must be the same.'
                if kwargs.get('angle_unit') == 'degrees':
                    self.interface_fun_r = GMSHInterface.angle_handler(self.interface_fun_r)
                    self.interface_fun_z = GMSHInterface.angle_handler(self.interface_fun_z)
        else:
            pass  # We assume it is an already evaluable function.

        # %% STEP 1: DEFINE THE POINTS OF THE GEOMETRY.
        r_arr = np.array([])
        z_arr = np.array([])
        # MENISCUS POINTS DEFINITION.
        if interface_fun is not None:
            r_arr = np.sort(kwargs.get('r'))[::-1]  # Sort the r array from higher to lower.
            for r_val in r_arr:
                self.key = 'p' + str(self.point_num)
                if isinstance(interface_fun, str):
                    interface_fun = self.interface_fun  # Necessary because degrees may have changed to radians.
                    interface_fun = GMSHInterface.replace_ind_var(interface_fun, ind_var, str(r_val))
                    z_val = nsp.eval(interface_fun)
                else:
                    try:  # In case the user introduces a Scipy solution object, we need to specify which sol to use.
                        z_val = interface_fun(r_val)[0]
                    except TypeError:  # In case the user introduces a standard function.
                        z_val = interface_fun(r_val)
                if r_val != 1:
                    self.p_dict[self.key] = Entity.Point([r_val, z_val, 0], mesh=self.my_mesh)
                    self.interface_points[self.key] = Entity.Point([r_val, z_val, 0], mesh=self.my_mesh)
                    self.point_num += 1
                    z_arr = np.append(z_arr, z_val)
            r_arr = r_arr[1:]

        elif interface_fun_r is not None and interface_fun_z is not None:
            for s in kwargs.get('independent_param'):
                interface_fun_r = self.interface_fun_r
                interface_fun_z = self.interface_fun_z
                self.key = 'p' + str(self.point_num)
                # Replace the independent variables with the corresponding values.
                if isinstance(interface_fun_r, str) and isinstance(interface_fun_z, str):
                    interface_fun_r = GMSHInterface.replace_ind_var(interface_fun_r, ind_var_r, str(s))
                    interface_fun_z = GMSHInterface.replace_ind_var(interface_fun_z, ind_var_z, str(s))
                    r = nsp.eval(interface_fun_r)
                    z = nsp.eval(interface_fun_z)
                else:
                    r = interface_fun_r(s)
                    z = interface_fun_z(s)
                if r != 1:
                    r_arr = np.append(r_arr, r)
                    z_arr = np.append(z_arr, z)
                    self.p_dict[self.key] = Entity.Point([r, z, 0], mesh=self.my_mesh)
                    self.interface_points[self.key] = Entity.Point([r, z, 0], mesh=self.my_mesh)
                    self.point_num += 1

        self.list_points_interface = list(self.interface_points.values())[::-1]
        self.interface_end_point = self.p_dict[self.key]
        self.interface_end_point_z = self.interface_end_point.xyz[1]
        self.key = 'p' + str(self.point_num)

        # LATERAL WALL LEFT POINTS DEFINITION.
        # Create an adaptive point generation based on the density of points of the interface.
        self.adaptive_refinement_tip()

        self.p_dict[self.key] = Entity.Point([0, factor, 0], mesh=self.my_mesh)
        self.point_num += 1
        self.key = 'p' + str(self.point_num)

        # TOP WALL POINTS DEFINITION.
        self.p_dict[self.key] = Entity.Point([factor, factor, 0], mesh=self.my_mesh)
        self.point_num += 1
        self.key = 'p' + str(self.point_num)

        # LATERAL WALL RIGHT POINTS DEFINITION.
        self.p_dict[self.key] = Entity.Point([factor, 0, 0], mesh=self.my_mesh)
        self.point_num += 1
        self.key = 'p' + str(self.point_num)

        # BOTTOM WALL POINTS DEFINITION.
        # Create the knee point.
        self.p_dict[self.key] = Entity.Point([1, 0, 0], mesh=self.my_mesh)
        knee_point = self.p_dict[self.key]
        self.point_num += 1
        self.key = 'p' + str(self.point_num)

        # TUBE WALL RIGHT POINTS DEFINITION.
        self.p_dict[self.key] = Entity.Point([1, -factor, 0], mesh=self.my_mesh)
        self.point_num += 1
        self.key = 'p' + str(self.point_num)

        # INLET POINTS DEFINITION.
        self.p_dict[self.key] = Entity.Point([0, -factor, 0], mesh=self.my_mesh)
        self.point_num += 1
        self.key = 'p' + str(self.point_num)

        # TUBE WALL LEFT POINTS DEFINITION.
        # Create an adaptive point generation based on the density of points of the interface.
        self.adaptive_refinement_tip()

        # Create the curves.
        p_list = list(self.p_dict)
        for i in np.arange(0, len(p_list)-1):
            curve = Entity.Curve([self.p_dict[p_list[i]], self.p_dict[p_list[i+1]]], mesh=self.my_mesh)
            if self.p_dict[p_list[i]].xyz[1] in z_arr and self.p_dict[p_list[i+1]].xyz[1] in z_arr:
                self.interface.addEntity(curve)
            elif self.p_dict[p_list[i]].xyz[0] == 0 and self.p_dict[p_list[i+1]].xyz[0] == 0 and self.p_dict[p_list[i]].xyz[1] >= self.interface_end_point_z:
                self.lwl.addEntity(curve)
            elif self.p_dict[p_list[i]].xyz[1] == 0 and self.p_dict[p_list[i+1]].xyz[1] == 0:
                self.bw.addEntity(curve)
            elif self.p_dict[p_list[i]].xyz[1] == factor and self.p_dict[p_list[i+1]].xyz[1] == factor:
                self.tw.addEntity(curve)
            elif self.p_dict[p_list[i]].xyz[0] == factor and self.p_dict[p_list[i+1]].xyz[0] == factor:
                self.lwr.addEntity(curve)
            elif self.p_dict[p_list[i]].xyz[0] == 1 and self.p_dict[p_list[i+1]].xyz[0] == 1:
                self.twr.addEntity(curve)
            elif self.p_dict[p_list[i]].xyz[1] == -factor and self.p_dict[p_list[i+1]].xyz[1] == -factor:
                self.inlet.addEntity(curve)
            elif self.p_dict[p_list[i]].xyz[0] == 0 and self.p_dict[p_list[i+1]].xyz[0] == 0 and self.p_dict[p_list[i]].xyz[1] <= self.interface_end_point_z:
                self.twl.addEntity(curve)

        # Lastly, join remaining points.
        add_int_curve = Entity.Curve([knee_point, self.p_dict[p_list[0]]], mesh=self.my_mesh)
        self.interface.addEntity(add_int_curve)
        twl_l = Entity.Curve([self.p_dict[p_list[-1]], self.interface_end_point], mesh=self.my_mesh)
        self.twl.addEntity(twl_l)

        # Create the subdomains physical groups.
        vacuum_curveloop = Entity.CurveLoop(list(self.lwl.curves.values()) +
                                            list(self.tw.curves.values()) +
                                            list(self.lwr.curves.values()) +
                                            list(self.bw.curves.values()) +
                                            [add_int_curve] +
                                            list(self.interface.curves.values())[:-1], mesh=self.my_mesh)
        vacuum_surf = Entity.PlaneSurface([vacuum_curveloop], mesh=self.my_mesh)
        self.vacuum.addEntity(vacuum_surf)

        liquid_curveloop = Entity.CurveLoop(list(self.twr.curves.values()) +
                                            list(self.inlet.curves.values()) +
                                            list(self.twl.curves.values()) +
                                            list(self.interface.curves.values())[:-1][::-1] +
                                            [add_int_curve], mesh=self.my_mesh)
        liquid_surf = Entity.PlaneSurface([liquid_curveloop], mesh=self.my_mesh)
        self.liquid.addEntity(liquid_surf)

    def mesh_generation_noGUI(self, filename):
        """
        Generate the mesh from a .geo file without calling the Graphical User Interface (GUI). This method is useful
        when the mesh details have already been defined or specified in some way. In particular, this method will be
        used when generating the geometries from the solutions of the iterative process.
        Returns:
            The path of the mesh (.msh) file.
        """

        self.geo_filename = create_mesh(self.my_mesh, self.app, filename)
        self.mesh_filename = write_mesh(self.geo_filename)

        return self.mesh_filename

    def mesh_generation_GUI(self):
        """
        Generate the mesh file from the .geo file by calling a Graphical User Interface (GUI), in which the user is
        able to define some other parameters of the mesh.
        Returns:
            The path of the mesh (.msh) file.
        """

        # Initialize the GUI to get user's inputs.
        self.app = run_app(self.my_mesh)

        # Create the mesh.
        self.geo_filename = create_mesh(self.my_mesh, self.app, self.filename)
        self.mesh_filename = write_mesh(self.geo_filename)
        return self.mesh_filename

# %% TEST THE CLASS.


if __name__ == '__main__':
    class_call = GMSHInterface()
    s_arr = np.linspace(0, 0.5, 200)
    p_x = 1
    w = 20
    beta = np.radians(49.3)
    r_array = np.linspace(0, 1, 800)


    def r_fun(s):
        return ((1-2*s)*1)/(1-2*s*(1-s)*(1-20))


    def z_fun(s):
        return (2*(1-s)*s*20*(1/np.tan(beta))*1)/(1-2*s*(1-s)*(1-20))


    def cos_fun(r):
        tip_height = 1
        return tip_height*np.cos(np.pi/2*r)


    def parabolic_fun(r):
        return 1 - r**2


    class_call.geometry_generator(interface_fun_r=r_fun, interface_fun_z=z_fun,
                                  independent_param=s_arr)
    class_call.mesh_generation_GUI()
