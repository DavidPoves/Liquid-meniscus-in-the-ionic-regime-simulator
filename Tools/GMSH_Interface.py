from Tools.MeshMenu import run_app
from Tools.CreateMesh import create_mesh, str_2_num

import numpy as np
import re

from py2gmsh import Mesh, Entity

from Tools.EvaluateString import NumericStringParser


class GMSHInterface(object):

    def __init__(self):
        self.geoPath = ''  # Initialize the path where the .geo file will be stored.
        self.filename = '' # Initialize the name of the .geo file.
        self.refinement = ''  # This string will be later modified by the user to indicate the refinement of mesh.
        self.msh_filename = ''

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
        This method will create a transfinite surface from a given surface with id tag.
        Args:
            tag (int): Tag of the surface.

        Returns:

        """
        if not isinstance(tag, int):
            raise TypeError('Inputs of this function must be integers.')
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
    def get_independent_var_from_equation(string):
        ind_var = ' '.join(re.split("[^a-zA-Z]*", string)).strip()

        # Avoid false positives of the accepted functions from the parser.
        ind_var = ind_var.replace('s i n', '')
        ind_var = ind_var.replace('t a n', '')
        ind_var = ind_var.replace('c o s', '')
        ind_var = ind_var.replace('e x p', '')
        ind_var = ind_var.replace('a b s', '')
        ind_var = ind_var.replace('t r u n c', '')
        ind_var = ind_var.replace('r o u n d', '')
        ind_var = ind_var.replace('s g n', '')
        ind_var = ind_var.replace('P I', '')
        ind_var = ind_var.replace('E', '')

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
    def angle_handler(string):
        trig_functions = ['sin', 'cos', 'tan']

        # Detect if any trigonometric function is present.
        for fun in trig_functions:
            str_list = string.split(fun)
            if len(str_list) == 1:  # No trigonometric function is present on the expression.
                pass
            else:
                for i in str_list[1:]:
                    aux = str_2_num(re.search(r'\((.*?)\)', i).group(1))  # This is the angle in degrees.
                    string = string.replace(str(aux), str(np.radians(aux)))
        return string

    def geometry_generator(self, interface_fun=None, interface_fun_r=None, interface_fun_z=None, factor=10, **kwargs):

        # Do some checks before proceeding.
        kwargs.setdefault('filename', 'GeneratedGeometry')
        kwargs.setdefault('number_points', 300)
        kwargs.setdefault('angle_unit', 'radians')
        if interface_fun is not None and kwargs.get('r') is None:
            print('** WARNING: No array for the r coordinates was given. Assuming equidistant points based on number of points selected.', flush=True)
            kwargs.setdefault('r', np.linspace(0, 1, kwargs.get('number_points')))
        elif interface_fun is not None and interface_fun_r is not None and interface_fun_z is not None:
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
        if self.interface_fun is not None:
            ind_var = GMSHInterface.get_independent_var_from_equation(interface_fun)
            if kwargs.get('angle_unit') == 'degrees':
                self.interface_fun = GMSHInterface.angle_handler(self.interface_fun)
        elif self.interface_fun_z is not None and self.interface_fun_r is not None:
            ind_var_r = GMSHInterface.get_independent_var_from_equation(interface_fun_r)
            ind_var_z = GMSHInterface.get_independent_var_from_equation(interface_fun_z)
            if kwargs.get('angle_unit') == 'degrees':
                self.interface_fun_r = GMSHInterface.angle_handler(self.interface_fun_r)
                self.interface_fun_z = GMSHInterface.angle_handler(self.interface_fun_z)

        # %% STEP 1: DEFINE THE POINTS OF THE GEOMETRY.
        # MENISCUS POINTS DEFINITION.
        if interface_fun is not None:
            r_arr = np.sort(kwargs.get('r'))[::-1]
            z_arr = np.array([])
            for r_val in r_arr:
                interface_fun = self.interface_fun
                self.key = 'p' + str(self.point_num)
                interface_fun = interface_fun.replace(ind_var, str(r_val))  # Now, we can replace the the independent variable with the corresponding number.
                z_val = nsp.eval(interface_fun)
                if r_val != 1:
                    self.p_dict[self.key] = Entity.Point([r_val, z_val, 0], mesh=self.my_mesh)
                    self.point_num += 1
                    z_arr = np.append(z_arr, z_val)
            r_arr = r_arr[1:]
        elif interface_fun_r is not None and interface_fun_z is not None:
            r_arr = np.array([])
            z_arr = np.array([])
            for s in kwargs.get('independent_param'):
                interface_fun_r = self.interface_fun_r
                interface_fun_z = self.interface_fun_z
                self.key = 'p' + str(self.point_num)
                # Replace the independent variables with the corresponding values.
                interface_fun_r = interface_fun_r.replace(ind_var_r, str(s))
                interface_fun_z = interface_fun_z.replace(ind_var_z, str(s))
                r = nsp.eval(interface_fun_r)
                z = nsp.eval(interface_fun_z)
                if r != 1:
                    r_arr = np.append(r_arr, r)
                    z_arr = np.append(z_arr, z)
                    self.p_dict[self.key] = Entity.Point([r, z, 0], mesh=self.my_mesh)
                    self.point_num += 1

        self.interface_points = self.p_dict  # Save the points of the interface in another dict.
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

    def mesh_generation(self):

        # Initialize the GUI to get user's inputs.
        app = run_app(self.my_mesh)

        # Create the mesh.
        create_mesh(self.my_mesh, app, self.filename)

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
        return (2*(1-s)*s*20*(1/tan(49.3))*1)/(1-2*s*(1-s)*(1-20))


    def cos_fun(r):
        tip_height = 1
        return tip_height*np.cos(np.pi/2*r)


    def parabolic_fun(r):
        return 1 - r**2


    class_call.geometry_generator(interface_fun_r=r_fun, interface_fun_z=z_fun,
                                  independent_param=s_arr)
    class_call.mesh_generation()
