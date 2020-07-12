# Import the modules.
from py2gmsh import (Mesh, Entity)
import os
import pygmsh
import numpy as np
import gmsh_api.gmsh as gmsh  # pip install gmsh_api
import platform
import subprocess
import warnings
import sys
import importlib
import re


"""
This module will generate the geometry of the problem by using the user's
input. In order to do so, just two inputs are required: the height of the
interphase and the radius (L) of the whole tank. With these values, a domain
will be generated in CARTESIAN COORDINATES, since GMSH DOES NOT accept any
other reference frame, and neither FEniCS does so, meaning that an internal
change of varible will be required. This code will follow the exact same
procedure as in the 'Mesh Generation process section'.
INSTRUCTIONS.
---------
1. If none of the modules have been installed previously, let the program
install them for you.
2. Replace the file Entity.py from py2gmsh by the one available in the
download folder (future update: replace it automatically).
"""


class gmsh_handle(object):

    def __init__(self):
        return

    @staticmethod
    def open_file(mesh_folder, z=None):
        """
        Menu to select a file or create a new one. To be executed only at the
        beginning of the program.
        Returns
        -------
        name : STR
            Name of the file with the extension '.msh'.
        """
        while True:
            file_list = []  # List containing the name of detected files.
            for file in os.listdir(mesh_folder):
                if file.endswith(".geo") or file.endswith(".msh"):
                    file_list.append(file)  # Save files matching the extensions.
            if len(file_list) == 0:
                create = input(f'No .geo or .msh file was detected. Create new geometry? [y/n]: ')
                if create == 'y' or create == 'yes':
                    name, b0, h, factor = gmsh_handle.parameters_definition()
                else:  # Stop the execution of the program with assert.
                    assert create == 'y', 'File not chosen, stoping execution...'
                gmsh_handle.Geometry_Generator(b0, h, z, factor, name,
                                               mesh_folder)
                print('Geometry generated correctly. ', end='')
                print(f'Generating the mesh for {name} ...')
                gmsh_handle.Mesh_Generator(name, mesh_folder)
                gmsh_handle.openGMSH(name, mesh_folder)
                name += '.msh'
            else:
                loc = 0
                print(f'All files with extensions .geo and .msh:\n')
                for item in file_list:
                    loc += 1
                    if item.split('.')[-1] == 'msh':
                        print(f'{loc}: Open {item} \n')
                    elif item.split('.')[-1] == 'geo':
                        print(f'{loc}: Generate mesh for {item}\n')
                    else:
                        pass
                    if item == file_list[-1]:
                        loc += 1
                        gen = loc
                        print(f'{loc} Generate new geometry \n')
                loc += 1
                print(f'{loc}: Exit the program')
                choice = int(input('Select an option: '))
                if choice == loc:
                    assert choice != loc, 'Stoping execution...'
                else:
                    while choice <= 0 or choice > len(file_list) and choice != gen:
                        warnings.warn('Your file choice is incorrect, please try again.')
                        choice = int(input('Select a file: '))
                    if choice == gen:
                        name, b0, h, factor = gmsh_handle.parameters_definition()
                        gmsh_handle.Geometry_Generator(b0, h, z, factor, name,
                                                       mesh_folder)
                        name += '.geo'
                    else:
                        name = file_list[choice-1]
                    # Detect the extension.
                    extension = name.split('.')[-1]
                    if extension == 'geo':
                        print(f'Generating the mesh for {name}...\n')
                        gmsh_handle.Mesh_Generator(name.split('.')[0])
                        gmsh_handle.openGMSH(name.split('.')[0], mesh_folder)
                    else:
                        gmsh_handle.openGMSH(name.split('.')[0])
            proceed_flag = input('Proceed with this mesh? [y/n]: ')
            if proceed_flag == 'y':
                return name.split('.')[0] + '.msh', b0, h, factor
            else:
                continue

    @staticmethod
    def parameters_definition():
        """
        Define the main geometric parameters.

        Returns
        -------
        name: string
            Name of the file to be generated for the geometry.
        b0: float
            Radius of the tube [m].
        h: float
            Initial height of the meniscus [m].
        factor: int/float
            Ratio between the radius of the tube and the top wall.

        """
        name = input('Introduce a name for the new .geo file (without the extension): ')
        b0 = float(input('Introduce Tube radius: '))
        h = float(input('Introduce initial meniscus height: '))
        factor = float(input('Introduce ratio between tank radius and tube radius: '))

        return name, b0, h, factor

    @staticmethod
    def get_key_from_value(d, value):
        for key, value_ in d.items():
            if value_ == value:
                return key

    @staticmethod
    def Geometry_Generator(b0, h, z, factor, name, mesh_folder, N=800):
        """
        Generate the .geo file given the main geometry parameters. This is the
        function to be modified in order to change the domain. Notice that the
        points should be sorted in order, since the lines are automatically
        generated using the order established by the user at the time of
        defining the geometry points.
        IT IS STRONGLY ADVISED TO NOT MODIFY THE PARTS WHERE NO INDICATIONS
        ARE WRITTEN.

        Parameters
        ----------
        b0: float
            Radius of the tube [m].
        h: float
            Initial height of the meniscus [m].
        z: function
            Function describing the shape of the meniscus (z(r)).
        factor: float/int
            Ratio between the radius of the tube and the top wall.
        name: string
            Name of the file to be generated for the geometry.
        N: integer, optional
            Number of points at the meniscus. The default is 800.

        Returns
        -------
        None.

        """
        L = b0 * factor  # Radius of tank [m]

        # Check that all modules are installed.
        gmsh_handle.check_modules()

        # Define the geometry and mesh object.
        my_mesh = Mesh()

        # %% STEP 1: GENERATE THE MAIN GEOMETRY.

        # Create the dictionary containing all the points.
        p_dict = dict()

        ######################################################################
        #                    ONLY PART TO BE TOUCHED                         #
        ######################################################################
        # Generate the points to define the vacuum subdomain.
        p_dict['p1'] = Entity.Point([b0, 0, 0], mesh=my_mesh)
        knee_point = p_dict['p1']
        p_dict['p2'] = Entity.Point([2*b0, 0., 0.], mesh=my_mesh)
        check_point = p_dict['p2']
        p_dict['p3'] = Entity.Point([3*b0, 0., 0.], mesh=my_mesh)
        p_dict['p4'] = Entity.Point([4*b0, 0., 0.], mesh=my_mesh)
        p_dict['p5'] = Entity.Point([5*b0, 0., 0.], mesh=my_mesh)
        ref_point_1 = p_dict['p5']
        p_dict['p6'] = Entity.Point([L, 0, 0], mesh=my_mesh)
        check_point_5 = p_dict['p6']
        p_dict['p7'] = Entity.Point([L, L, 0], mesh=my_mesh)
        check_point_6 = p_dict['p7']
        p_dict['p8'] = Entity.Point([0, L, 0], mesh=my_mesh)
        check_point_7 = p_dict['p8']
        p_dict['p9'] = Entity.Point([0, h + 4*b0, 0], mesh=my_mesh)
        ref_point_2 = p_dict['p9']
        p_dict['p10'] = Entity.Point([0, h + 3*b0, 0], mesh=my_mesh)
        check_point_2 = p_dict['p10']
        p_dict['p11'] = Entity.Point([0, h + 2*b0, 0], mesh=my_mesh)
        check_point_8 = p_dict['p11']
        p_dict['p12'] = Entity.Point([0, h+b0, 0], mesh=my_mesh)
        check_point_3 = p_dict['p12']
        ######################################################################
        #                    ONLY PART TO BE TOUCHED                         #
        ######################################################################

        # Automatically generate the points of the meniscus boundary.
        p = len(p_dict)  # Get the last point number (p__, last key of p_dict)

        r_meniscus = np.linspace(0, b0, N)

        for r in r_meniscus:  # CAN BE MODIFIED
            if r != b0:  # To not overwrite the knee point.
                p += 1
                key = f'p{p}'
                if r == 0:
                    key_tip = key
                else:
                    pass
                p_dict[key] = Entity.Point([r, z(r, h, b0), 0], mesh=my_mesh)
            else:
                pass

        # Create a list containing all the keys of p_dict.
        p_key_list = list(p_dict)
        # Create a dictionary containing all the lines.
        l_dict = dict()
        line_num = 1
        for loc in np.arange(0, len(p_key_list)-1):
            key = f'l{line_num}'
            l_dict[key] = Entity.Curve([p_dict[p_key_list[loc]],
                                       p_dict[p_key_list[loc+1]]])
            line_num += 1
        # Get the name of the last line.
        last_prev_line_num = len(l_dict)
        last_line_key = f'l{last_prev_line_num+1}'

        # Generate the last line to connect boundary -> CAN BE MODIFIED.
        l_dict[last_line_key] = Entity.Curve([p_dict[p_key_list[loc+1]],
                                              knee_point])

        # Do the same process for the liquid subdomain.
        last_vac_p_key = p_key_list[-1]  # Get last vacuum point key.
        last_vac_p = int(last_vac_p_key.split('p')[-1])  # Get last point num.
        liq_first_p = 'p' + str(last_vac_p+1)
        next_liq_p = liq_first_p

        ######################################################################
        #                    ONLY PART TO BE TOUCHED                         #
        ######################################################################
        # Generate the points to define the liquid boundaries.
        p_dict[next_liq_p] = Entity.Point([b0, -b0, 0.], mesh=my_mesh)
        check_point_4 = p_dict[next_liq_p]
        next_liq_p = 'p' + str(int(next_liq_p.split('p')[-1])+1)
        p_dict[next_liq_p] = Entity.Point([b0, -2*b0, 0.], mesh=my_mesh)
        next_liq_p = 'p' + str(int(next_liq_p.split('p')[-1])+1)
        p_dict[next_liq_p] = Entity.Point([b0, -3*b0, 0.], mesh=my_mesh)
        next_liq_p = 'p' + str(int(next_liq_p.split('p')[-1])+1)
        p_dict[next_liq_p] = Entity.Point([b0, -4*b0, 0.], mesh=my_mesh)
        next_liq_p = 'p' + str(int(next_liq_p.split('p')[-1])+1)
        p_dict[next_liq_p] = Entity.Point([b0, -6*b0, 0.], mesh=my_mesh)
        right_1 = p_dict[next_liq_p]
        next_liq_p = 'p' + str(int(next_liq_p.split('p')[-1])+1)
        p_dict[next_liq_p] = Entity.Point([b0, -b0*10, 0], mesh=my_mesh)
        inlet_1 = p_dict[next_liq_p]
        next_liq_p = 'p' + str(int(next_liq_p.split('p')[-1])+1)
        p_dict[next_liq_p] = Entity.Point([0, -b0*10, 0], mesh=my_mesh)
        inlet_2 = p_dict[next_liq_p]
        next_liq_p = 'p' + str(int(next_liq_p.split('p')[-1])+1)
        p_dict[next_liq_p] = Entity.Point([0, -6*b0, 0], mesh=my_mesh)
        next_liq_p = 'p' + str(int(next_liq_p.split('p')[-1])+1)
        p_dict[next_liq_p] = Entity.Point([0, -4*b0, 0], mesh=my_mesh)
        next_liq_p = 'p' + str(int(next_liq_p.split('p')[-1])+1)
        p_dict[next_liq_p] = Entity.Point([0, -3*b0, 0], mesh=my_mesh)
        next_liq_p = 'p' + str(int(next_liq_p.split('p')[-1])+1)
        p_dict[next_liq_p] = Entity.Point([0, -2*b0, 0], mesh=my_mesh)
        next_liq_p = 'p' + str(int(next_liq_p.split('p')[-1])+1)
        p_dict[next_liq_p] = Entity.Point([0, -b0, 0], mesh=my_mesh)
        next_liq_p = 'p' + str(int(next_liq_p.split('p')[-1])+1)
        c = (b0/2) - h**2 / (2*b0)
        p_dict[next_liq_p] = Entity.Point([c, 0, 0], mesh=my_mesh)
        circle_center = p_dict[next_liq_p]
        p_dict[next_liq_p] = Entity.Point([0, 0, 0], mesh=my_mesh)
        center_point = p_dict[next_liq_p]
        ######################################################################
        #                    ONLY PART TO BE TOUCHED                         #
        ######################################################################

        # Redefine the list containing all the point keys.
        p_key_list = list(p_dict)

        # Create the liquid lines.
        liq_first_line = len(list(l_dict))+1
        # Connect vacuum and liquid (1) -> CAN BE MODIFIED.
        l_dict[f'l{liq_first_line}'] = Entity.Curve([knee_point,
                                                     p_dict[liq_first_p]])
        # Create liquid boundaries.
        l_n = liq_first_line
        for loc in np.arange(last_vac_p, int(next_liq_p.split('p')[-1])-1):
            l_n += 1
            l_key = f'l{l_n}'
            l_dict[l_key] = Entity.Curve([p_dict[p_key_list[loc]],
                                         p_dict[p_key_list[loc+1]]])

        # Connect liquid and vacuum (2) -> CAN BE MODIFIED.
        l_dict[f'l{l_n+1}'] = Entity.Curve([p_dict[p_key_list[-1]],
                                            p_dict[key_tip]])
        # Generate the first (internal) refinement curve -> CAN BE MODIFIED.
        last_line_num = len(l_dict)
        ref_curve_key_1 = f'l{last_line_num+1}'
        l_dict[ref_curve_key_1] = Entity.Circle(check_point_3,
                                                circle_center, check_point)
        # Generate the second (external) refinement curve -> CAN BE MODIFIED.
        last_line_num += 1
        ref_curve_key_2 = f'l{last_line_num+1}'
        l_dict[ref_curve_key_2] = Entity.Circle(ref_point_2,
                                                circle_center,
                                                ref_point_1)
        l_list = []
        for line in list(l_dict):
            l_list.append(l_dict[line])

        # Create the list containing the line keys.
        l_key_list = list(l_dict)

        my_mesh.addEntities(l_list)
        my_mesh.Coherence = True  # Add coherence to the geometry.

        # %% STEP 2: ADD PLANE SURFACES. CAN BE MODIFIED.
        """
        A total number of 4 plane surfaces should be made. Therefore, 4
        curveloops should be made to define the latter.
        """

        # Create curveloops:
        ll1_list = []  # Create first curveloop.
        # Check the line that has the points of interest as starting or end.
        for key in list(l_dict):
            try:
                points = l_dict[key].points
                if ref_point_1 == points[0]:
                    first = key
                elif ref_point_2 == points[1]:
                    last = key
                else:
                    pass
            except AttributeError:  # Circle object has no attr points.
                pass
        """
        Once we know the initial line, and because they have been sorted in
        order, it is easy to define the first curveloop.
        """
        for n in np.arange(l_key_list.index(first), l_key_list.index(last)+1):
            ll1_list.append(l_dict[l_key_list[n]])
        # Add the external refinement curve.
        ll1_list.append(l_dict[ref_curve_key_2])

        ll1 = Entity.CurveLoop(ll1_list)

        # Create second curveloop.
        ll2_list = []
        """
        Now we need to go from the second refinement point to the tip.
        """
        for key in list(l_dict):
            try:
                points = l_dict[key].points
                if ref_point_2 == points[0] and check_point_2 == points[1]:
                    first = key
                elif l_dict[ref_curve_key_1].start == points[1] and check_point_8 == points[0]:
                    last = key
                else:
                    pass
            except AttributeError:  # Circle object has no attr points.
                pass
        start = l_key_list.index(first)
        end = l_key_list.index(last)
        for n in np.arange(start-1, end+1):
            ll2_list.append(l_dict[l_key_list[n]])

        ll2_list.append(l_dict[ref_curve_key_1])

        # Now, we go from knee point to refinement point 1.
        for key in list(l_dict):
            try:
                points = l_dict[key].points
                if knee_point == points[0] and check_point == points[1]:
                    first = key
                elif ref_point_1 == points[1]:
                    last = key
                else:
                    pass
            except AttributeError:  # Circle object has no attr points.
                pass

        for n in np.arange(l_key_list.index(first), l_key_list.index(last)+1):
            ll2_list.append(l_dict[l_key_list[n]])

        # Add the external refinement curve.
        ll2_list.append(l_dict[ref_curve_key_2])

        ll2 = Entity.CurveLoop(ll2_list)

        # Create third curveloop.
        ll3_list = []
        """
        For this curveloop, we will start from the tip point, continue along
        the meniscus and close the loop with the internal refinement curve.
        """
        meniscus_boundary = []
        for key in list(l_dict):
            try:
                points = l_dict[key].points
                if p_dict[key_tip] == points[0]:
                    first = key
                elif knee_point == points[1]:
                    last = key
                else:
                    pass
            except AttributeError:  # Circle object has no attr points.
                pass
        for n in np.arange(l_key_list.index(first), l_key_list.index(last)+1):
            ll3_list.append(l_dict[l_key_list[n]])
            meniscus_boundary.append(l_dict[l_key_list[n]])

        for key in list(l_dict):
            try:
                points = l_dict[key].points
                if knee_point == points[0] and l_dict[ref_curve_key_1].end == points[1]:
                    first = key
                else:
                    pass
            except AttributeError:  # Circle object has no attr points.
                pass
        ll3_list.append(l_dict[first])

        # Add the internal refinement curve.
        ll3_list.append(l_dict[ref_curve_key_1])

        for key in list(l_dict):
            try:
                points = l_dict[key].points
                if l_dict[ref_curve_key_1].start == points[0] and p_dict[key_tip] == points[1]:
                    first = key
                else:
                    pass
            except AttributeError:  # Circle object has no attr points.
                pass
        ll3_list.append(l_dict[first])

        ll3 = Entity.CurveLoop(ll3_list)

        # Create the fourth curveloop.
        ll4_list = []
        for line in meniscus_boundary:
            ll4_list.append(line)

        for key in list(l_dict):
            try:
                points = l_dict[key].points
                if knee_point == points[0] and check_point_4 == points[1]:
                    first = key
                elif center_point == points[0] and p_dict[key_tip] == points[1]:
                    last = key
                else:
                    pass
            except AttributeError:  # Circle object has no attr points.
                pass
        tube_boundary = []
        for n in np.arange(l_key_list.index(first), l_key_list.index(last)+1):
            ll4_list.append(l_dict[l_key_list[n]])
            tube_boundary.append(l_dict[l_key_list[n]])
        ll4 = Entity.CurveLoop(ll4_list)
        my_mesh.addEntities([ll1, ll2, ll3, ll4])

        # Create plane surfaces.
        s1 = Entity.PlaneSurface([ll1], mesh=my_mesh)
        s2 = Entity.PlaneSurface([ll2], mesh=my_mesh)
        s3 = Entity.PlaneSurface([ll3], mesh=my_mesh)
        s4 = Entity.PlaneSurface([ll4], mesh=my_mesh)

        # %% ADD PHYSICAL GROUPS.

        # # Create the physical groups and assign them to the mesh.
        gs1 = Entity.PhysicalGroup(name='Vacuum_1')
        gs2 = Entity.PhysicalGroup(name='Vacuum_2')
        gs3 = Entity.PhysicalGroup(name='Vacuum_3')
        gs4 = Entity.PhysicalGroup(name='Vacuum')
        gs5 = Entity.PhysicalGroup(name='Liquid')
        gc1 = Entity.PhysicalGroup(name='Inlet')
        gc2 = Entity.PhysicalGroup(name='Tube_Wall_R')
        gc3 = Entity.PhysicalGroup(name='Meniscus')
        gc4 = Entity.PhysicalGroup(name='Bottom_Wall')
        gc5 = Entity.PhysicalGroup(name='Lateral_Wall_R')
        gc6 = Entity.PhysicalGroup(name='Lateral_Wall_L')
        gc7 = Entity.PhysicalGroup(name='Tube_Wall_L')
        gc8 = Entity.PhysicalGroup(name='Top_Wall')
        my_mesh.addEntities([gs1, gs2, gs3, gs4, gs5, gc1, gc2, gc3, gc4, gc5,
                             gc6, gc7, gc8])

        # Add existing entities to the physical groups.
        # First, we need to identify the lines that define the boundaries.

        # 1. Check the inlet curve.
        for key in list(l_dict):
            try:
                points = l_dict[key].points
                if inlet_1 == points[0] and inlet_2 == points[1]:
                    in_bound = key
                else:
                    pass
            except AttributeError:  # Circle object has no attr points.
                pass
        inlet_boundary = l_dict[in_bound]

        # 2. Define the right tube wall.
        right_tube_wall_boundary = []
        for key in list(l_dict):
            try:
                points = l_dict[key].points
                if knee_point == points[0] and check_point_4 == points[1]:
                    first = key
                elif right_1 == points[0] and inlet_1 == points[1]:
                    last = key
                else:
                    pass
            except AttributeError:  # Circle object has no attr points.
                pass
        for n in np.arange(l_key_list.index(first), l_key_list.index(last)+1):
            right_tube_wall_boundary.append(l_dict[l_key_list[n]])

        # 3. Define the meniscus. Already defined in previous steps.
        # 4. Define the bottom wall.
        bottom_wall_boundary = []
        for key in list(l_dict):
            try:
                points = l_dict[key].points
                if knee_point == points[0] and check_point == points[1]:
                    first = key
                elif ref_point_1 == points[0] and check_point_5 == points[1]:
                    last = key
                else:
                    pass
            except AttributeError:  # Circle object has no attr points.
                pass
        for n in np.arange(l_key_list.index(first), l_key_list.index(last)+1):
            bottom_wall_boundary.append(l_dict[l_key_list[n]])

        # 5. Define the lateral walls.
        lateral_boundaries_vacuum_R = []
        # First, we will define the right wall.
        for key in list(l_dict):
            try:
                points = l_dict[key].points
                if check_point_5 == points[0] and check_point_6 == points[1]:
                    lat_bound = key
                else:
                    pass
            except AttributeError:  # Circle object has no attr points.
                pass
        lateral_boundaries_vacuum_R.append(l_dict[lat_bound])
        # Next, we define the left wall (upper part).
        lateral_boundaries_vacuum_L = []
        for key in list(l_dict):
            try:
                points = l_dict[key].points
                if check_point_7 == points[0]:
                    first = key
                elif check_point_3 == points[0] and p_dict[key_tip] == points[1]:
                    last = key
                else:
                    pass
            except AttributeError:  # Circle object has no attr points.
                pass
        for n in np.arange(l_key_list.index(first), l_key_list.index(last)+1):
            lateral_boundaries_vacuum_L.append(l_dict[l_key_list[n]])

        # Next, we define the left wall (lower part).
        lateral_boundaries_liquid = []
        for key in list(l_dict):
            try:
                points = l_dict[key].points
                if inlet_2 == points[0]:
                    first = key
                elif center_point == points[0] and p_dict[key_tip] == points[1]:
                    last = key
                else:
                    pass
            except AttributeError:  # Circle object has no attr points.
                pass
        for n in np.arange(l_key_list.index(first), l_key_list.index(last)+1):
            lateral_boundaries_liquid.append(l_dict[l_key_list[n]])

        # Finally, define the top wall.
        for key in list(l_dict):
            try:
                points = l_dict[key].points
                if check_point_6 == points[0] and check_point_7 == points[1]:
                    tw_key = key
                else:
                    pass
            except AttributeError:  # Circle object has no attr points.
                pass

        """
        In order to add entities, notice that if only once instance is to be
        added, use addEntity instead of addEntities, or define the entity as a
        list.
        """

        gs1.addEntity(s1)
        gs2.addEntity(s2)
        gs3.addEntity(s3)
        gs4.addEntities([s1, s2, s3])
        gs5.addEntity(s4)
        gc1.addEntity(inlet_boundary)
        gc2.addEntities(right_tube_wall_boundary)
        gc3.addEntities(meniscus_boundary)
        gc4.addEntities(bottom_wall_boundary)
        gc5.addEntities(lateral_boundaries_vacuum_R)
        gc6.addEntities(lateral_boundaries_vacuum_L)
        gc7.addEntities(lateral_boundaries_liquid)
        gc8.addEntity(l_dict[tw_key])

        # %% SAVE THE FILE INTO A .GEO FILE.

        # Generate the .geo file.
        my_mesh.Coherence = True
        filename = name + '.geo'
        my_mesh.writeGeo(mesh_folder + '/' + filename)

    @staticmethod
    def Mesh_Generator(name, mesh_folder):
        """
        Generates the .msh file given the .geo file.
        Parameters
        ----------
        name: str
            Name of the .geo file without the extension.
        Returns
        -------
        None.
        """
        # Create the name of the file.
        filename = name + '.geo'

        # Check that all the modules are installed.
        gmsh_handle.check_modules()

        # Initialize the gmsh api to get the elements.
        gmsh.initialize()
        # Save the mesh with v2 to use it with dolfin.
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.)
        gmsh.open(mesh_folder + '/' + filename)
        geom = pygmsh.built_in.Geometry()

        # Add the model and get its entities.
        entities_lst = gmsh.model.getEntities()  # Get all the entities.

        # Get the tag of the physical curves.
        physical_tags, tag_list = gmsh_handle.get_physical_curves_and_tags(filename,
                                                                           mesh_folder, curves='"Meniscus"',
                                                                           return_tags=True)
        _, tag_list_lateral_V = gmsh_handle.get_physical_curves_and_tags(filename,
                                                                               mesh_folder, curves='"Lateral_Wall_L"',
                                                                               return_tags=True)
        _, tag_list_lateral_L = gmsh_handle.get_physical_curves_and_tags(filename,
                                                                           mesh_folder, curves='"Tube_Wall_L"',
                                                                           return_tags=True)
        _, tag_list_lateral_L_2 = gmsh_handle.get_physical_curves_and_tags(filename,
                                                                           mesh_folder, curves='"Tube_Wall_R"',
                                                                           return_tags=True)
        _, tag_list_bottom = gmsh_handle.get_physical_curves_and_tags(filename,
                                                                           mesh_folder, curves='"Bottom_Wall"',
                                                                           return_tags=True)
        tag_list_lateral_V = tag_list_lateral_V[0]
        tag_list_lateral_L = tag_list_lateral_L[0]
        tag_list_lateral_L_2 = tag_list_lateral_L_2[0]
        tag_list_bottom = tag_list_bottom[0]

        # Create the Transfinite Elements.
        TCL = np.array([])
        TSL = np.array([])
        for dim, tag in entities_lst:
            skip = False
            if dim == 1:
                curves_2_refine = len(tag_list)
                for num in range(curves_2_refine):
                    if tag in tag_list[num]:
                        TCL = np.append(TCL, geom.set_transfinite_lines(tag, 3,
                                                                        progression=1))
                        skip = True
                if not skip:
                    if tag in tag_list_lateral_L or tag in tag_list_lateral_V or tag in tag_list_lateral_L_2:
                        TCL = np.append(TCL, geom.set_transfinite_lines(tag, 80,
                                                                        progression=1))
                    elif tag in tag_list_bottom:
                        TCL = np.append(TCL, geom.set_transfinite_lines(tag, 40,
                                                                        progression=1))
                    else:
                        TCL = np.append(TCL, geom.set_transfinite_lines(tag, 20,
                                                                        progression=1))
            elif dim == 2:
                # Create the Transfinite Surfaces.
                TSL = np.append(TSL, geom.set_transfinite_surface(tag))

        # Append the created files into the .geo file.
        with open('MESH_&_RESTRICTIONS/' + filename, 'a') as my_file:
            for code in TCL:
                my_file.write(code+';\n')
            for code in TSL:
                my_file.write(code+';\n')
            my_file.close()

        # Re-open the file.
        gmsh.open(mesh_folder + '/' + filename)

        # Generate the 2D mesh.
        gmsh.model.mesh.generate(2)  # 2 indicating 2 dimensions.
        filename = name + '.msh'
        gmsh.write(mesh_folder + '/' + filename)

        # Finalize the gmsh processes.
        gmsh.finalize()

    @staticmethod
    def openGMSH(name, mesh_folder, extension='.msh'):
        """
        Method to open the a compatible file in GMSH automatically. Works in
        both MacOS and Windows.
        Parameters
        ----------
        name : STR
            Name of the file without the extension.
        file : STR, optional
            Extension of the file to be opened. The default is '.msh'.
        Returns
        -------
        None.
        """
        filename = name + extension

        # Detect the os.
        os_ = platform.system()
        if os_ == 'Windows':
            os.system('start ' + mesh_folder + '/' + filename)
        else:
            dir_ = mesh_folder + '/' + filename
            subprocess.call(['open', dir_])

    @staticmethod
    def check_which_physical_first(filename, mesh_folder_path):
        # Load and open the file with reading permission.
        file = mesh_folder_path + '/' + filename
        f = open(file, 'r')

        # Preallocate bools.
        check_curves = True
        check_surfaces = True
        curves_first = False
        surfaces_first = False

        # Iterate over the lines of the .geo file.
        for line in f:
            if check_curves:
                if re.findall('Physical Curve', line):
                    check_curves = False
                    check_surfaces = False
                    curves_first = True
            if check_surfaces:
                if re.findall('Physical Surface', line):
                    check_curves = False
                    check_surfaces = False
                    surfaces_first = True
        return curves_first, surfaces_first


    @staticmethod
    def get_physical_curves_and_tags(filename, mesh_folder_path,
                                     curves='', return_tags=False):
        """
        Obtain the physical curves and their ids from a .geo file
        Parameters
        ----------
        filename: string
            Name of the .geo file (with the extension).
        mesh_folder: string
            Name of the path of the folder where the mesh is stored.
        Returns
        -------
        physical_curves: dict
            Dictionary containing the physical curves (boundaries).
        """
        file = mesh_folder_path + '/' + filename
        f = open(file, 'r')
        physical_curves = dict()
        # Define a pattern to catch all between parenthesis.
        regex_pattern_pa = re.compile(r'\(([^\)]+)\)')
        regex_pattern_br = re.compile(r'\{([^\)]+)\}')
        tags_arr = []
        check_first = gmsh_handle.check_which_physical_first(filename,
                                                             mesh_folder_path)
        if check_first[0]:
            counter = 1
        else:
            counter = 3
        found_curve = False
        for line in f:
            if re.findall('Physical Curve', line):
                mo = regex_pattern_pa.findall(line)
                mc = regex_pattern_br.findall(line)
                found_curve = True
                for string in mc:
                    tags = [int(num.replace(' ', '')) for num in string.split(',')]
                    tags_arr.append(tags)
                for string in mo:
                    key = string.split(',')[0]
                    value_str = string.split(',')[-1].replace(" ", '')
                    physical_curves[key] = counter
                    counter += 1

        f.close()
        if not found_curve:  # Check if physical curves were found.
            raise ValueError('No physical curves were found. They must be defined then in GMSH to create the boundaries.')
        starting_curve_id = min(physical_curves.values())
        if curves != '' and isinstance(curves, str):
            curve_id = physical_curves.get(curves)
            tags_arr = tags_arr[curve_id - starting_curve_id]
            if return_tags:
                return physical_curves, [tags_arr]
            else:
                return physical_curves
        elif curves != '' and (isinstance(curves, list) or isinstance(curves, np.ndarray)):
            tags_arr_return = []
            for curve_name in curves:
                if isinstance(curve_name, str):
                    curve_id = physical_curves.get(curve_name)
                    tags_arr_return.append(tags_arr[curve_id - starting_curve_id])
                    if return_tags:
                        return physical_curves, tags_arr_return
                    else:
                        return physical_curves
        else:
            if return_tags:
                return physical_curves, tags_arr
            else:
                return physical_curves

    @staticmethod
    def get_physical_surfaces(filename, mesh_folder):
        """
        Obtain the physical surfaces and their ids from a .geo file
        Parameters
        ----------
        filename: string
            Name of the .geo file (with the extension).
        mesh_folder: string
            Name of the path of the folder where the mesh is stored.
        Returns
        -------
        physical_surfaces: dict
            Dictionary containing the physical surfaces (subdomains).
        """
        file = mesh_folder + '/' + filename
        f = open(file, 'r')
        physical_surfaces = dict()
        # Define a pattern to catch all between parenthesis.
        regex_pattern = re.compile(r'\(([^\)]+)\)')
        found_surface = False
        no_surfaces = 0
        for line in f:
            if re.findall('Physical Surface', line):
                no_surfaces += 1
                mo = regex_pattern.findall(line)
                found_surface = True
                # for string in mo:
                #     key = string.split(',')[0]
                #     value_str = string.split(',')[-1].replace(" ", '')
                #     value = int(value_str)
                #     physical_surfaces[key] = value
        f.close()

        # Check if no physical surface was found.
        if not found_surface:
            raise ValueError('No physical surface was found. They must be defined in GMSH to create the subdomains.')

        # Check if there are more than 2 subdomains.
        if no_surfaces > 2:
            raise ValueError(f'{no_surfaces} were found. Only 2 can be loaded.')

        return physical_surfaces

    @staticmethod
    def check_modules():
        """
        Required modules that are not by default in anaconda:
            - py2gmsh
            - pygmsh
            - gmsh_api
            - gmsh
        """
        modules_list = ['py2gmsh', 'pygmsh', 'gmsh_api']
        for module in modules_list:
            if module not in sys.modules:
                install = input(f"Module {module} is not installed. Install it? [y/n]: ")
                if install == 'y' or install == 'yes':
                    subprocess.check_call([sys.executable, "-m", "pip", "install", module])
                    importlib.import_module(module.lower())
                else:
                    raise ModuleNotFoundError(f': No module named {module}')


# %% DEBUGGING PURPOSES.
"""
This section has been done for debugging the geometry generation isolated from
the main script.
"""
if __name__ == '__main__':
    def z(r, h0, b0):
        A = h0
        B = np.pi/(2*b0)
        return A*np.cos(B*r)
    mesh_folder = '/Users/davidpoves/TFG/TFG-CODE/MESH_&_RESTRICTIONS'
    gmsh_handle.Geometry_Generator(1e-5, 1e-5, z, 10, 'Tank', mesh_folder)
    gmsh_handle.Mesh_Generator('Tank', mesh_folder)
    pc, tags = gmsh_handle.get_physical_curves_and_tags('Tank.geo', mesh_folder,
                                                        curves=['"Meniscus"', '"Inlet"'])
    ps = gmsh_handle.get_physical_surfaces('Tank.geo', mesh_folder)
