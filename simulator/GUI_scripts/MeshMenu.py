from tkinter import *

from simulator.Tools.CreateMesh import create_mesh, write_mesh
from simulator.GUI_scripts.ToolTip_creator import CreateToolTip

"""
Options for the mesh can be found at:
http://transit.iut2.upmf-grenoble.fr/cgi-bin/info2www?(gmsh)Mesh+options+list
"""


class MeshApp(Frame):
    def __init__(self, master3=None, mesh=None):
        Frame.__init__(self, master3)

        ref_label = Label(master3, text="Refinement:")
        ref_label.grid(row=0, column=0)
        CreateToolTip(ref_label, 'Select the degree of refinement for the mesh.')

        min_label = Label(master3, text="Max. Element Size:")
        min_label.grid(row=1, column=0)
        CreateToolTip(min_label, 'Maximum mesh element size.')

        max_label = Label(master3, text="Min. Element Size:")
        max_label.grid(row=2, column=0)
        CreateToolTip(max_label, 'Minimum mesh element size.')

        curv_label = Label(master3, text="Char. Length from Curvature:")
        curv_label.grid(row=3, column=0)
        CreateToolTip(curv_label, 'Automatically compute mesh element sizes from curvature.')

        len_factor_label = Label(master3, text="Char. Length Factor:")
        len_factor_label.grid(row=4, column=0)
        CreateToolTip(len_factor_label, 'Compute mesh element sizes from values given at geometry points.')

        mesh_alg_label = Label(master3, text="Mesh Algorithm:")
        mesh_alg_label.grid(row=5, column=0)
        CreateToolTip(mesh_alg_label, '2D mesh algorithm.')

        remesh_label = Label(master3, text="Remesh Parametrization:")
        remesh_label.grid(row=6, column=0)
        CreateToolTip(remesh_label, 'Select a Remesh Parametrization algorithm.')

        # DEFINE THE STRING VARIABLES.
        self.refinement = StringVar(master3)
        self.max_element_size = StringVar(master3)
        self.max_element_size.set('0')
        self.min_element_size = StringVar(master3)
        self.min_element_size.set('0')
        self.length_from_curvature = StringVar(master3)
        self.length_from_curvature.set('0')
        self.length_extend = StringVar(master3)
        self.length_extend.set('0')
        self.mesh_alg = StringVar(master3)
        self.mesh_alg.set('Automatic')
        self.remesh_param = StringVar(master3)
        self.remesh_param.set('Harmonic')

        # ADD THE REFINEMENT DROPDOWN MENU.
        self.refinementDict = {'Coarse': 'Coarse', 'Normal': 'Normal', 'Fine': 'Fine', 'Custom': 'Custom'}

        self.refinement.set("Normal")  # default value
        self.refinement_default = self.refinement.get()

        self.refinementdrop = OptionMenu(master3, self.refinement, *self.refinementDict,
                                         command=self.control_fun)
        self.refinementdrop.grid(row=0, column=1, padx=10, pady=10)
        self.refinementdrop.configure(foreground='BLACK', activeforeground='BLACK')

        # ADD THE MAX ELEMENT SIZE INPUT BOX.
        self.MaxDict ={'Coarse': '0.75', 'Normal': '0.5', 'Fine': '0.1', 'Custom': ''}

        # ADD THE MIN ELEMENT SIZE INPUT BOX.
        self.MinDict = {'Coarse': '1e-11', 'Normal': '1e-11', 'Fine': '1e-11', 'Custom': ''}

        # ADD THE CHARACTERISTIC LENGTH FROM CURVATURE INPUT BOX.
        self.CharCurv = {'Coarse': '0.1', 'Normal': '0.075', 'Fine': '0.01', 'Custom': ''}

        # ADD THE CHARACTERISTIC LENGTH FACTOR INPUT BOX.
        self.CharLFactor = {'Coarse': '2', 'Normal': '1.75', 'Fine': '1.25', 'Custom': ''}

        # ADD THE MESH ALGORITHM OPTIONS.
        self.MeshAlgDict = {'MeshAdapt': 'MeshAdapt', 'Automatic': 'Automatic', 'Delaunay': 'Delaunay',
                            'Frontal-Delaunay': 'Frontal-Delaunay', 'bamg': 'bamg',
                            'delquad': 'delquad'}

        self.mesh_alg_drop = OptionMenu(master3, self.mesh_alg, *self.MeshAlgDict,
                                        command=self.control_mesh_alg)
        self.mesh_alg_drop.grid(row=5, column=1, padx=10, pady=10)
        self.mesh_alg_drop.configure(foreground='BLACK', activeforeground='BLACK')

        # ADD THE REMESH PARAMETRIZATION OPTIONS.
        self.RemeshParamDict = {'Harmonic': 'Harmonic', 'Conformal': 'Conformal'}
        self.remesh_param_drop = OptionMenu(master3, self.remesh_param, *self.RemeshParamDict,
                                            command=self.control_remesh_param)
        self.remesh_param_drop.grid(row=6, column=1, padx=10, pady=10)
        self.remesh_param_drop.configure(foreground='BLACK', activeforeground='BLACK')

        # ADD A SAVE AND CLOSE BUTTON.
        save_close_but = Button(master3, text='Save and close', command=master3.quit)
        save_close_but.grid(row=7, column=2)
        save_close_but.configure(foreground='BLACK', activeforeground='BLACK')

        # ADD A PREVIEW BUTTON.
        preview_but = Button(master3, text='Preview Mesh', command=lambda: MeshApp.preview_fun(mesh, self, 'temp',
                                                                                               preview=True))
        preview_but.grid(row=7, column=3)
        preview_but.configure(foreground='BLACK', activeforeground='BLACK')

    def control_fun(self, value):
        self.refinement.set(self.refinementDict[value])
        self.max_element_size.set(self.MaxDict[value])
        self.min_element_size.set(self.MinDict[value])
        self.length_from_curvature.set(self.CharCurv[value])
        self.length_extend.set(self.CharLFactor[value])

        # Show the maximum size input box.
        self.display_boxes()

    def control_mesh_alg(self, value):
        self.mesh_alg.set(self.MeshAlgDict[value])

    def control_remesh_param(self, value):
        self.remesh_param.set(self.RemeshParamDict[value])

    def display_boxes(self):
        if self.refinement.get() != 'Custom':
            Entry(textvariable=self.max_element_size, state=DISABLED, justify='center').grid(row=1, column=1,
                                                                                             padx=10, pady=10)
            Entry(textvariable=self.min_element_size, state=DISABLED, justify='center').grid(row=2, column=1,
                                                                                             padx=10, pady=10)
            Entry(textvariable=self.length_from_curvature, state=DISABLED, justify='center').grid(row=3, column=1,
                                                                                                  padx=10, pady=10)
            Entry(textvariable=self.length_extend, state=DISABLED, justify='center').grid(row=4, column=1,
                                                                                          padx=10, pady=10)
        else:
            Entry(textvariable=self.max_element_size, justify='center').grid(row=1, column=1, padx=10, pady=10)
            Entry(textvariable=self.min_element_size, justify='center').grid(row=2, column=1, padx=10, pady=10)
            Entry(textvariable=self.length_from_curvature, justify='center').grid(row=3, column=1, padx=10, pady=10)
            Entry(textvariable=self.length_extend, justify='center').grid(row=4, column=1, padx=10, pady=10)

    @staticmethod
    def preview_fun(mesh, mesh_options, filename, preview=False):
        geo_filename = create_mesh(mesh, mesh_options, filename)
        write_mesh(geo_filename, preview)


def run_app(mesh=None):
    root = Tk()
    root.title('Setup the Mesh Properties')
    app = MeshApp(root, mesh)
    root.mainloop()
    root.destroy()
    return app

if __name__ == '__main__':
    app = run_app()