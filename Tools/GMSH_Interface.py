from py2gmsh import (Mesh, Entity)
import numpy as np
import sys
from PyQt5.QtWidgets import QFileDialog, QApplication


class GMSHInterface(object):

    def __init__(self):
        self.geoPath = ''  # Initialize the path where the .geo file will be initialized.
        return

    def path_selection(self):
        QApplication(sys.argv)  # Create an application before calling the directory
        QApplication.setStyle('fusion')
        dialog = QFileDialog()
        self.geoPath = dialog.getExistingDirectory(dialog, caption='Choose Path to save .geo file...')

    def geometry_generator(self, interface_fun=None, interface_fun_r=None,
                           interface_fun_z=None, number_points=800, factor=10,
                           refinement='Normal', **kwargs):

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

        # Define the geometry and mesh object.
        my_mesh = Mesh()

        # %% STEP 1: DEFINE THE POINTS OF THE GEOMETRY.

        # Create a dictionary containing all the points of the geometry.
        p_dict = dict()

        inlet = Entity.PhysicalGroup(name='Inlet', mesh=my_mesh)
        twr = Entity.PhysicalGroup(name='Tube_Wall_R', mesh=my_mesh)
        twl = Entity.PhysicalGroup(name='Tube_Wall_L', mesh=my_mesh)
        bw = Entity.PhysicalGroup(name='Bottom_Wall', mesh=my_mesh)
        lwr = Entity.PhysicalGroup(name='Lateral_Wall_R', mesh=my_mesh)
        tw = Entity.PhysicalGroup(name='Top_Wall', mesh=my_mesh)
        lwl = Entity.PhysicalGroup(name='Lateral_Wall_L', mesh=my_mesh)
        interface = Entity.PhysicalGroup(name='Interface', mesh=my_mesh)

        vacuum = Entity.PhysicalGroup(name='Vacuum', mesh=my_mesh)
        liquid = Entity.PhysicalGroup(name='Liquid', mesh=my_mesh)

        # MENISCUS POINTS DEFINITION.
        point_num = 1

        if interface_fun is not None:
            r_arr = np.sort(kwargs.get('r'))[::-1]
            z_arr = np.array([])
            for r_val in r_arr:
                key = 'p' + str(point_num)
                z_val = interface_fun(r_val)
                if r_val != 1:
                    p_dict[key] = Entity.Point([r_val, z_val, 0], mesh=my_mesh)
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
                    p_dict[key] = Entity.Point([r, z, 0], mesh=my_mesh)
                    point_num += 1

        meniscus_tip = p_dict[key]
        meniscus_tip_z = meniscus_tip.xyz[1]
        key = 'p' + str(point_num)

        # LATERAL WALL RIGHT POINTS DEFINITION.
        if refinement == 'Fine':
            factor_refinement = 0.1
            number_points = 10
            multiplier = 1
            for _ in np.arange(0, number_points):
                p_dict[key] = Entity.Point([0, meniscus_tip_z +
                                            multiplier*factor_refinement, 0],
                                           mesh=my_mesh)
                point_num += 1
                key = 'p' + str(point_num)
                multiplier += 1

        p_dict[key] = Entity.Point([0, factor, 0], mesh=my_mesh)
        point_num += 1
        key = 'p' + str(point_num)

        # TOP WALL POINTS DEFINITION.
        p_dict[key] = Entity.Point([factor, factor, 0], mesh=my_mesh)
        point_num += 1
        key = 'p' + str(point_num)

        # LATERAL WALL RIGHT POINTS DEFINITION.
        p_dict[key] = Entity.Point([factor, 0, 0], mesh=my_mesh)
        point_num += 1
        key = 'p' + str(point_num)

        # BOTTOM WALL POINTS DEFINITION.
        if refinement == 'Fine':
            multiplier = 1
            for _ in np.arange(0, number_points):
                p_dict[key] = Entity.Point([1 + multiplier*factor_refinement,
                                            0, 0],
                                           mesh=my_mesh)
                point_num += 1
                key = 'p' + str(point_num)
                multiplier += 1
        p_dict[key] = Entity.Point([1, 0, 0], mesh=my_mesh)
        knee_point = p_dict[key]
        point_num += 1
        key = 'p' + str(point_num)

        # TUBE WALL RIGHT POINTS DEFINITION.
        if refinement == 'Fine':
            multiplier = 1
            for _ in np.arange(0, number_points):
                p_dict[key] = Entity.Point([1, -multiplier*factor_refinement,
                                            0],
                                           mesh=my_mesh)
                point_num += 1
                key = 'p' + str(point_num)
                multiplier += 1
        p_dict[key] = Entity.Point([1, -factor, 0], mesh=my_mesh)
        point_num += 1
        key = 'p' + str(point_num)

        # INLET POINTS DEFINITION.
        p_dict[key] = Entity.Point([0, -factor, 0], mesh=my_mesh)
        point_num += 1
        key = 'p' + str(point_num)

        # TUBE WALL LEFT POINTS DEFINITION.
        if refinement == 'Fine':
            multiplier = 1
            for _ in np.arange(0, number_points):
                p_dict[key] = Entity.Point([0, meniscus_tip_z -
                                            multiplier*factor_refinement, 0],
                                           mesh=my_mesh)
                point_num += 1
                key = 'p' + str(point_num)
                multiplier += 1

        # Create the curves.
        p_list = list(p_dict)
        for i in np.arange(0, len(p_list)-1):
            curve = Entity.Curve([p_dict[p_list[i]], p_dict[p_list[i+1]]], mesh=my_mesh)
            if p_dict[p_list[i]].xyz[1] in z_arr and p_dict[p_list[i+1]].xyz[1] in z_arr:
                interface.addEntity(curve)
            elif p_dict[p_list[i]].xyz[0] == 0 and p_dict[p_list[i+1]].xyz[0] == 0 and p_dict[p_list[i]].xyz[1] >= meniscus_tip_z:
                lwl.addEntity(curve)
            elif p_dict[p_list[i]].xyz[1] == 0 and p_dict[p_list[i+1]].xyz[1] == 0:
                bw.addEntity(curve)
            elif p_dict[p_list[i]].xyz[1] == factor and p_dict[p_list[i+1]].xyz[1] == factor:
                tw.addEntity(curve)
            elif p_dict[p_list[i]].xyz[0] == factor and p_dict[p_list[i+1]].xyz[0] == factor:
                lwr.addEntity(curve)
            elif p_dict[p_list[i]].xyz[0] == 1 and p_dict[p_list[i+1]].xyz[0] == 1:
                twr.addEntity(curve)
            elif p_dict[p_list[i]].xyz[1] == -factor and p_dict[p_list[i+1]].xyz[1] == -factor:
                inlet.addEntity(curve)
            elif p_dict[p_list[i]].xyz[0] == 0 and p_dict[p_list[i+1]].xyz[0] == 0 and p_dict[p_list[i]].xyz[1] <= meniscus_tip_z:
                twl.addEntity(curve)

        # Lastly, join remaining points.
        add_int_curve = Entity.Curve([knee_point, p_dict[p_list[0]]], mesh=my_mesh)
        interface.addEntity(add_int_curve)
        twl_l = Entity.Curve([p_dict[p_list[-1]], meniscus_tip], mesh=my_mesh)
        twl.addEntity(twl_l)

        # Create the subdomains physical groups.
        vacuum_curveloop = Entity.CurveLoop(list(lwl.curves.values()) +
                                            list(tw.curves.values()) +
                                            list(lwr.curves.values()) +
                                            list(bw.curves.values()) +
                                            [add_int_curve] +
                                            list(interface.curves.values())[:-1]
                                            ,
                                            mesh=my_mesh)
        vacuum_surf = Entity.PlaneSurface([vacuum_curveloop], mesh=my_mesh)
        vacuum.addEntity(vacuum_surf)

        liquid_curveloop = Entity.CurveLoop(list(twr.curves.values()) +
                                            list(inlet.curves.values()) +
                                            list(twl.curves.values()) +
                                            list(interface.curves.values())[:-1][::-1] +
                                            [add_int_curve], mesh=my_mesh)
        liquid_surf = Entity.PlaneSurface([liquid_curveloop], mesh=my_mesh)
        liquid.addEntity(liquid_surf)

        # print(interface.curves)

        # Create the .geo file.
        # Generate the .geo file.
        my_mesh.Coherence = True
        self.path_selection()
        filename = self.geoPath + '/' + kwargs.get('filename') + '.geo'
        my_mesh.writeGeo(filename)


# %% TEST THE CLASS.
gmsh = GMSHInterface()
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


gmsh.geometry_generator(interface_fun_r=r_fun, interface_fun_z=z_fun,
                        independent_param=s_arr, refinement='Fine',
                        filename='FinalGeometry')
