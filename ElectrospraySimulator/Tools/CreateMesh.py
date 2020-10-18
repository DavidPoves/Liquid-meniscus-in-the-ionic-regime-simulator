import os
import subprocess
import tkinter as tk
from tkinter import filedialog

import gmsh_api.gmsh as gmsh


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
    else:
        return


def apply_dark_mode():
    """
    Apply the MacOS dark mode to the GUI in case the user is running this style by default.
    """
    interface_system = os.popen("defaults read -g AppleInterfaceStyle").read().split('\n')[0]
    if interface_system == 'Dark':  # This condition will enable the dark mode for the GUI.
        os.system('defaults write -g NSRequiresAquaSystemAppearance -bool No')


def return_default_interface_style():
    """
    Once the GUI is closed, return the settings to the default ones to avoid any problem that could arise due to
    the imposition of the dark mode.
    """
    os.system('defaults delete -g NSRequiresAquaSystemAppearance')


def save_file():
    """
    Spawn a GUI from TKinter to let the user choose the folder where a certain file will be saved. This method is
    ready to use Apple's Dark Mode if the user is using it.
    """
    # bool_dark = interface_style_detect
    # if bool_dark:
    #     apply_dark_mode()
    root = tk.Tk()
    root.title('Select folder to save the .geo file...')
    root.withdraw()
    extensions = [('Geometric file', '(.geo)'),
                  ('All files', '(.*)')]
    filename = filedialog.asksaveasfile(initialdir=os.getcwd())

    # Return to default parameters of dark mode in case they were activated.
    # return_default_interface_style()

    root.destroy()

    return filename


def str_2_num(string):
    try:
        return float(string)
    except ValueError:
        return int(string)


def translate_inputs(mesh_options):

    # Create a dictionary to automate translation of user's inputs to the main program.
    mesh_alg_dict = {'MeshAdapt': 1, 'Automatic': 2, 'Delaunay': 5,
                     'Frontal-Delaunay': 6, 'bamg': 7, 'delquad': 8}
    remesh_alg_dict = {'Harmonic': 0, 'Conformal': 1}

    mesh_options.min_element_size_num = str_2_num(mesh_options.min_element_size.get())
    mesh_options.max_element_size_num = str_2_num(mesh_options.max_element_size.get())
    mesh_options.length_from_curvature_num = str_2_num(mesh_options.length_from_curvature.get())
    mesh_options.length_extend_num = str_2_num(mesh_options.length_extend.get())
    mesh_options.mesh_alg_num = mesh_alg_dict.get(mesh_options.mesh_alg.get())
    mesh_options.remesh_param_num = remesh_alg_dict.get(mesh_options.remesh_param.get())

    return mesh_options


def open_gmsh(filename):
    # Detect the os.
    if os.name == 'nt':
        os.system('start ' + filename)
    else:
        subprocess.call(['open', filename])


def create_mesh(mesh, mesh_options, filename):

    # Translate the inputs from the GUI.
    mesh_options = translate_inputs(mesh_options)

    # Set Mesh Options.
    mesh.Options.Mesh.CharacteristicLengthMin = mesh_options.min_element_size_num
    mesh.Options.Mesh.CharacteristicLengthMax = mesh_options.max_element_size_num
    mesh.Options.Mesh.CharacteristicLengthFromCurvature = mesh_options.length_from_curvature_num
    mesh.Options.Mesh.CharacteristicLengthFactor = mesh_options.length_extend_num
    mesh.Options.Mesh.Algorithm = mesh_options.mesh_alg_num
    mesh.Options.Mesh.RemeshParametrization = mesh_options.remesh_param_num

    # Set Coherence and write the .geo file.
    mesh.Coherence = True
    if filename == 'temp':
        filename = os.getcwd() + '/temp.geo'
    else:
        try:
            os.remove(os.getcwd() + '/temp.geo')  # Remove Temporary files.
            os.remove(os.getcwd() + '/temp.msh')  # Remove Temporary files.
        except FileNotFoundError:  # In case the user does not preview the mesh.
            pass
        filename = save_file().name
    mesh.writeGeo(filename)

    return filename


def write_mesh(geo_filename, preview=False):
    # Initialize the gmsh api.
    gmsh.initialize()

    # Save the mesh with v2 to use it with dolfin-converter application.
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.)

    # Re-open the file.
    gmsh.open(geo_filename)

    # Generate the 2D mesh.
    gmsh.model.mesh.generate(2)  # 2 indicating 2 dimensions.
    msh_filename = geo_filename.split('.geo')[0] + '.msh'
    gmsh.write(msh_filename)

    # Finalize the gmsh processes.
    gmsh.finalize()

    if preview:
        open_gmsh(msh_filename)

    return msh_filename
