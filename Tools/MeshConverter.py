import dolfin as df
import os


def msh2xml(filename, root_folder, mesh_folder_path):

	name = filename.split('.')[0]
	ofilename = name + '.xml'
	ifilename_ = name + '.msh' + ' '
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