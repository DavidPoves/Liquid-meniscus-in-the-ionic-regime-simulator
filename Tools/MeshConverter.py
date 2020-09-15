import dolfin as df
import os


def msh2xml(filename, root_folder, mesh_folder_path):
	"""
	Convert .msh files into .xml files, which are the type of files accepted by FEniCS. Depending on how the .msh file
	has been defined, it will return up to three files: mesh.xml, physical_region.xml and facet_region.xml files.
	To get these three files, one should define in the .geo file the physical curves and physical surfaces.
	The physical region file is the one containing the information regarding the subdomains, and the facet_region one
	contains all the boundary's data.
	Args:
		filename: Name of the mesh file (with the .msh extension).
		root_folder: Folder where the console is being executed (current working dir) -> obtained with os.getcwd()
		mesh_folder_path: Folder where the mesh (.msh) file is located.

	Returns:
		Name of the .xml file (with the extension).
	"""

	name = filename.split('.')[0]
	ofilename = name + '.xml'
	ifilename_ = filename + ' '
	ofilename_ = ofilename + ' '  # Formatting for next command.
	input_str = "dolfin-convert " + ifilename_ + ofilename_

	"""
	Next, we call the dolfin-converter, which will transform the generated
	.msh file into a readable .xml file, which is the extension accepted by
	FEniCS and multiphenics.
	"""
	os.chdir(mesh_folder_path)
	os.system(input_str)  # Call the dolfin-converter
	os.chdir(root_folder)

	return ofilename