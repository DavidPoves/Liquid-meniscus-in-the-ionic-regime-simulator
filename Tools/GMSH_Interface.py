import tkinter as tk
from tkinter import filedialog

import numpy as np
from py2gmsh import Mesh, Entity
import gmsh_api.gmsh as gmsh
import os


class GMSHInterface(object):

    def __init__(self):
        self.geoPath = ''  # Initialize the path where the .geo file will be stored.
        self.filename = '' # Initialize the name of the .geo file.
        self.refinement = 'Normal'  # This bool will be later modified by the user to indicate the refinement of mesh.

        # Define the geometry and mesh object.
        self.my_mesh = Mesh()

        # Initialize physical groups.
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
    def interface_style_detect():
        """
        This method will be able to detect if there is a special style mode activated, in case the user is running the
        program on MacOS. This will useful to detect if the user is using Apple's Dark Mode.
        """
        system = os.name
        if system == 'posix':
            has_interface = os.popen("defaults find AppleInterfaceStyle").read().split()
        if has_interface[1] == 0:
            return False
        else:
            return True

    @staticmethod
    def apply_dark_mode():
        """
        Apply the MacOS dark mode to the GUI in case the user is running this style by default.
        """
        interface_system = os.popen("defaults read -g AppleInterfaceStyle").read().split('\n')[0]
        if interface_system == 'Dark':  # This condition will enable the dark mode for the GUI.
            os.system('defaults write -g NSRequiresAquaSystemAppearance -bool No')

    @staticmethod
    def return_default_interface_style():
        """
        Once the GUI is closed, return the settings to the default ones to avoid any problem that could arise due to
        the imposition of the dark mode.
        """
        os.system('defaults delete -g NSRequiresAquaSystemAppearance')

    def path_selection(self):
        bool_dark = GMSHInterface.interface_style_detect
        if bool_dark:
            GMSHInterface.apply_dark_mode()
        root = tk.Tk()
        root.withdraw()
        self.geoPath = filedialog.askdirectory(title='Select folder to save the .geo file...',
                                               initialdir=os.getcwd())

        # Return to default parameters of dark mode in case they were activated.
        GMSHInterface.return_default_interface_style()

    @staticmethod
    def set_transfinite_line(tag, size, progression=1):
        code = f"Transfinite Line {{{tag}}} = {size} Using Progression {progression}"
        return code

    @staticmethod
    def set_transfinite_surface(tag):
        code = f"Transfinite Surface {{{tag}}}"
        return code

    def geometry_generator(self, interface_fun=None, interface_fun_r=None,
                           interface_fun_z=None, number_points=800, factor=10,
                           refinement='Normal', **kwargs):

        self.refinement = refinement

        # Do some checks before proceeding.
        kwargs.setdefault('filename', 'GeneratedGeometry')
        if interface_fun is not None and kwargs.get('r') is None:
            print('** WARNING: No array for the r coordinates was given. Assuming equidistant points based on number of points selected.', flush=True)
            kwargs.setdefault('r', np.linspace(0, 1, number_points))
        elif interface_fun is not None and kwargs.get('r') is not None:
            number_points = len(kwargs.get('r'))
        if not isinstance(refinement, str):
            raise TypeError(f'Refinement keyword must be a string, not a {type(refinement)}.')
        if interface_fun is not None and not hasattr(interface_fun, '__call__'):
            raise TypeError('Type of function not correct. It must be of the format z=f(r).')
        elif interface_fun_r is not None and not hasattr(interface_fun_r, '__call__'):
            raise TypeError('Type of function not correct. It must be of the format r=f(s).')
        elif interface_fun_z is not None and not hasattr(interface_fun_z, '__call__'):
            raise TypeError('Type of function not correct. It must be of the format z=f(s).')
        elif interface_fun is not None and interface_fun_r is not None and interface_fun_z is not None:
            raise NotImplementedError('Incompatible functions were introduced.')

        if interface_fun_r is not None and interface_fun_z is not None and kwargs.get('independent_param') is None:
            print('** WARNING: No array for the independent parameter was given. Assuming equidistant parameters based on number of points selected.', flush=True)
            kwargs.setdefault('independent_param', np.linspace(0, 1, number_points))
        elif interface_fun_r is not None and interface_fun_z is not None and kwargs.get('independent_param') is not None:
            number_points = len(kwargs.get('independent_param'))

        # %% STEP 1: DEFINE THE POINTS OF THE GEOMETRY.

        # Create a dictionary containing all the points of the geometry.
        p_dict = dict()

        # MENISCUS POINTS DEFINITION.
        point_num = 1

        if interface_fun is not None:
            r_arr = np.sort(kwargs.get('r'))[::-1]
            z_arr = np.array([])
            for r_val in r_arr:
                key = 'p' + str(point_num)
                z_val = interface_fun(r_val)
                if r_val != 1:
                    p_dict[key] = Entity.Point([r_val, z_val, 0], mesh=self.my_mesh)
                    point_num += 1
                    z_arr = np.append(z_arr, z_val)
            r_arr = r_arr[1:]
        elif interface_fun_r is not None and interface_fun_z is not None:
            r_arr = np.array([])
            z_arr = np.array([])
            for s in kwargs.get('independent_param'):
                key = 'p' + str(point_num)
                r = interface_fun_r(s)
                z = interface_fun_z(s)
                if r != 1:
                    r_arr = np.append(r_arr, r)
                    z_arr = np.append(z_arr, z)
                    p_dict[key] = Entity.Point([r, z, 0], mesh=self.my_mesh)
                    point_num += 1

        meniscus_tip = p_dict[key]
        meniscus_tip_z = meniscus_tip.xyz[1]
        key = 'p' + str(point_num)

        # LATERAL WALL LEFT POINTS DEFINITION.
        if self.refinement == 'Fine':
            factor_refinement = 0.1
            number_points = 10
            multiplier = 1
            for _ in np.arange(0, number_points):
                p_dict[key] = Entity.Point([0, meniscus_tip_z +
                                            multiplier*factor_refinement, 0],
                                           mesh=self.my_mesh)
                point_num += 1
                key = 'p' + str(point_num)
                multiplier += 1

        p_dict[key] = Entity.Point([0, factor, 0], mesh=self.my_mesh)
        point_num += 1
        key = 'p' + str(point_num)

        # TOP WALL POINTS DEFINITION.
        p_dict[key] = Entity.Point([factor, factor, 0], mesh=self.my_mesh)
        point_num += 1
        key = 'p' + str(point_num)

        # LATERAL WALL RIGHT POINTS DEFINITION.
        p_dict[key] = Entity.Point([factor, 0, 0], mesh=self.my_mesh)
        point_num += 1
        key = 'p' + str(point_num)

        # BOTTOM WALL POINTS DEFINITION.
        if self.refinement == 'Fine':
            multiplier = 1
            total_distance = 1 + number_points * factor_refinement
            for _ in np.arange(0, number_points):
                p_dict[key] = Entity.Point([total_distance - multiplier * factor_refinement,
                                            0, 0],
                                           mesh=self.my_mesh)
                point_num += 1
                key = 'p' + str(point_num)
                multiplier += 1
        p_dict[key] = Entity.Point([1, 0, 0], mesh=self.my_mesh)
        knee_point = p_dict[key]
        point_num += 1
        key = 'p' + str(point_num)

        # TUBE WALL RIGHT POINTS DEFINITION.
        if self.refinement == 'Fine':
            multiplier = 1
            for _ in np.arange(0, number_points):
                p_dict[key] = Entity.Point([1, -multiplier * factor_refinement,
                                            0],
                                           mesh=self.my_mesh)
                point_num += 1
                key = 'p' + str(point_num)
                multiplier += 1
        p_dict[key] = Entity.Point([1, -factor, 0], mesh=self.my_mesh)
        point_num += 1
        key = 'p' + str(point_num)

        # INLET POINTS DEFINITION.
        p_dict[key] = Entity.Point([0, -factor, 0], mesh=self.my_mesh)
        point_num += 1
        key = 'p' + str(point_num)

        # TUBE WALL LEFT POINTS DEFINITION.
        if self.refinement == 'Fine':
            multiplier = 1
            total_distance = meniscus_tip_z - number_points * factor_refinement
            for _ in np.arange(0, number_points):
                p_dict[key] = Entity.Point([0, total_distance +  multiplier * factor_refinement, 0], mesh=self.my_mesh)
                point_num += 1
                key = 'p' + str(point_num)
                multiplier += 1

        # Create the curves.
        p_list = list(p_dict)
        for i in np.arange(0, len(p_list)-1):
            curve = Entity.Curve([p_dict[p_list[i]], p_dict[p_list[i+1]]], mesh=self.my_mesh)
            if p_dict[p_list[i]].xyz[1] in z_arr and p_dict[p_list[i+1]].xyz[1] in z_arr:
                self.interface.addEntity(curve)
            elif p_dict[p_list[i]].xyz[0] == 0 and p_dict[p_list[i+1]].xyz[0] == 0 and p_dict[p_list[i]].xyz[1] >= meniscus_tip_z:
                self.lwl.addEntity(curve)
            elif p_dict[p_list[i]].xyz[1] == 0 and p_dict[p_list[i+1]].xyz[1] == 0:
                self.bw.addEntity(curve)
            elif p_dict[p_list[i]].xyz[1] == factor and p_dict[p_list[i+1]].xyz[1] == factor:
                self.tw.addEntity(curve)
            elif p_dict[p_list[i]].xyz[0] == factor and p_dict[p_list[i+1]].xyz[0] == factor:
                self.lwr.addEntity(curve)
            elif p_dict[p_list[i]].xyz[0] == 1 and p_dict[p_list[i+1]].xyz[0] == 1:
                self.twr.addEntity(curve)
            elif p_dict[p_list[i]].xyz[1] == -factor and p_dict[p_list[i+1]].xyz[1] == -factor:
                self.inlet.addEntity(curve)
            elif p_dict[p_list[i]].xyz[0] == 0 and p_dict[p_list[i+1]].xyz[0] == 0 and p_dict[p_list[i]].xyz[1] <= meniscus_tip_z:
                self.twl.addEntity(curve)

        # Lastly, join remaining points.
        add_int_curve = Entity.Curve([knee_point, p_dict[p_list[0]]], mesh=self.my_mesh)
        self.interface.addEntity(add_int_curve)
        twl_l = Entity.Curve([p_dict[p_list[-1]], meniscus_tip], mesh=self.my_mesh)
        self.twl.addEntity(twl_l)

        # Create the subdomains physical groups.
        vacuum_curveloop = Entity.CurveLoop(list(self.lwl.curves.values()) +
                                            list(self.tw.curves.values()) +
                                            list(self.lwr.curves.values()) +
                                            list(self.bw.curves.values()) +
                                            [add_int_curve] +
                                            list(self.interface.curves.values())[:-1]
                                            ,
                                            mesh=self.my_mesh)
        vacuum_surf = Entity.PlaneSurface([vacuum_curveloop], mesh=self.my_mesh)
        self.vacuum.addEntity(vacuum_surf)

        liquid_curveloop = Entity.CurveLoop(list(self.twr.curves.values()) +
                                            list(self.inlet.curves.values()) +
                                            list(self.twl.curves.values()) +
                                            list(self.interface.curves.values())[:-1][::-1] +
                                            [add_int_curve], mesh=self.my_mesh)
        liquid_surf = Entity.PlaneSurface([liquid_curveloop], mesh=self.my_mesh)
        self.liquid.addEntity(liquid_surf)

        # print(interface.curves)

        # Create the .geo file.
        # Generate the .geo file.
        self.my_mesh.Coherence = True
        self.path_selection()
        self.filename = self.geoPath + '/' + kwargs.get('filename') + '.geo'  # Include the full path.
        self.my_mesh.writeGeo(self.filename)

    def mesh_generation(self):

        # Initialize the gmsh api.
        gmsh.initialize()

        # Save the mesh with v2 to use it with dolfin.
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.)

        # Create the Transfinite Elements.
        trans_curves = np.array([])
        trans_surfs = np.array([])

        # 1. Bottom Wall.
        for tag in self.bw.curves.keys():
            if self.refinement == 'Coarse':
                size = 50
            elif self.refinement == 'Normal':
                size = 100
            elif self.refinement == 'Fine':
                size = 150
            trans_curves = np.append(trans_curves, GMSHInterface.set_transfinite_line(tag, size, progression=1))

        # 2. Lateral Wall Right.
        for tag in self.lwr.curves.keys():
            if self.refinement == 'Coarse':
                size = 25
            elif self.refinement == 'Normal':
                size = 50
            elif self.refinement == 'Fine':
                size = 100
            trans_curves = np.append(trans_curves, GMSHInterface.set_transfinite_line(tag, size, progression=1))

        # 3. Top Wall.
        for tag in self.tw.curves.keys():
            if self.refinement == 'Coarse':
                size = 25
            elif self.refinement == 'Normal':
                size = 50
            elif self.refinement == 'Fine':
                size = 100
            trans_curves = np.append(trans_curves, GMSHInterface.set_transfinite_line(tag, size, progression=1))

        # 4. Lateral Wall Left.
        for tag in list(self.lwl.curves.keys()):
            if self.refinement == 'Coarse':
                size = 100
            elif self.refinement == 'Normal':
                size = 150
            elif self.refinement == 'Fine':
                size = 250
            trans_curves = np.append(trans_curves, GMSHInterface.set_transfinite_line(tag, size, progression=1))

        # 5. Meniscus.
        for tag in list(self.interface.curves.keys()):
            size = 1
            trans_curves = np.append(trans_curves, GMSHInterface.set_transfinite_line(tag, size, progression=1))

        # 6. Tube Wall Left.
        for tag in list(self.twl.curves.keys()):
            if self.refinement == 'Coarse':
                size = 100
            elif self.refinement == 'Normal':
                size = 150
            elif self.refinement == 'Fine':
                size = 250
            trans_curves = np.append(trans_curves, GMSHInterface.set_transfinite_line(tag, size, progression=1))

        # 7. Inlet.
        for tag in list(self.inlet.curves.keys()):
            if self.refinement == 'Coarse':
                size = 25
            elif self.refinement == 'Normal':
                size = 50
            elif self.refinement == 'Fine':
                size = 100
            trans_curves = np.append(trans_curves, GMSHInterface.set_transfinite_line(tag, size, progression=1))

        # 8. Tube Wall Right.
        for tag in list(self.twr.curves.keys()):
            if self.refinement == 'Coarse':
                size = 100
            elif self.refinement == 'Normal':
                size = 150
            elif self.refinement == 'Fine':
                size = 250
            trans_curves = np.append(trans_curves, GMSHInterface.set_transfinite_line(tag, size, progression=1))

        # Generate the transfinite surfaces.
        # 1. Vacuum.
        trans_surfs = np.append(trans_surfs, GMSHInterface.set_transfinite_surface(list(self.vacuum.surfaces.keys())[0]))

        # 2. Liquid.
        trans_surfs = np.append(trans_surfs, GMSHInterface.set_transfinite_surface(list(self.liquid.surfaces.keys())[0]))

        # Write the generated codes on the .geo file.
        with open(self.filename, 'a') as my_file:
            for code in trans_curves:
                my_file.write(code + ';\n')
            for code in trans_surfs:
                my_file.write(code + ';\n')
            my_file.close()

        # Re-open the file.
        gmsh.open(self.filename)

        # Generate the 2D mesh.
        gmsh.model.mesh.generate(2)  # 2 indicating 2 dimensions.
        msh_filename = self.filename.split('/')[-1].split('.')[0] + '.msh'
        gmsh.write(self.geoPath + '/' + msh_filename)

        # Finalize the gmsh processes.
        gmsh.finalize()


# %% TEST THE CLASS.
class_call = GMSHInterface()
s_arr = np.linspace(0, 0.5, 800)
p_x = 1
w = 20
beta = np.radians(49.3)
r_array = np.linspace(0, 1, 800)


def r_fun(s):
    return ((1-2*s)*p_x)/(1-2*s*(1-s)*(1-w))


def z_fun(s):
    return (2*(1-s)*s*w*(1/np.tan(beta))*p_x)/(1-2*s*(1-s)*(1-w))


def cos_fun(r):
    tip_height = 1
    return tip_height*np.cos(np.pi/2*r)


def parabolic_fun(r):
    return 1 - r**2


class_call.geometry_generator(interface_fun_r=r_fun, interface_fun_z=z_fun,
                              independent_param=s_arr, refinement='Fine',
                              filename='FinalGeometry')
class_call.mesh_generation()
