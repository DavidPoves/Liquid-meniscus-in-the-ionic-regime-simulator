# Copyright (C) 2016-2020 by the multiphenics authors
#
# This file is part of multiphenics.
#
# multiphenics is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# multiphenics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

import dolfin as df
import multiphenics as mp


class Restrictions(object):
    def __init__(self):
        return None
    # Helper function to generate subdomain restriction based on a gmsh
    # subdomain id.
    @staticmethod
    def generate_subdomain_restriction(mesh, subdomains, subdomains_ids):
        D = mesh.topology().dim()
        # Initialize empty restriction
        restriction = mp.MeshRestriction(mesh, None)
        for d in range(D + 1):
            mesh_function_d = df.MeshFunction("bool", mesh, d)
            mesh_function_d.set_all(False)
            restriction.append(mesh_function_d)
        # Mark restriction mesh functions based on subdomain id
        for c in df.cells(mesh):
            for subdomain_id in subdomains_ids:
                if subdomains[c] == subdomain_id:
                    restriction[D][c] = True
                    for d in range(D):
                        for e in df.entities(c, d):
                            restriction[d][e] = True
        return restriction

    # Helper function to generate interface restriction based on a pair of gmsh
    # subdomain ids.
    @staticmethod
    def generate_interface_restriction(mesh, subdomains, subdomain_ids):
        assert isinstance(subdomain_ids, set)
        assert len(subdomain_ids) == 2, f'Only an interface between 2 subdomains is accepted.'
        D = mesh.topology().dim()
        # Initialize empty restriction
        restriction = mp.MeshRestriction(mesh, None)
        for d in range(D + 1):
            mesh_function_d = df.MeshFunction("bool", mesh, d)
            mesh_function_d.set_all(False)
            restriction.append(mesh_function_d)
        # Mark restriction mesh functions based on subdomain ids (except the
        # mesh function corresponding to dimension D, as it is trivially false)
        for f in df.facets(mesh):
            subdomains_ids_f = set(subdomains[c] for c in df.cells(f))
            # assert len(subdomains_ids_f) in (1, 2)
            if subdomains_ids_f == subdomain_ids:
                restriction[D - 1][f] = True
                for d in range(D - 1):
                    for e in df.entities(f, d):
                        restriction[d][e] = True
        return restriction


if __name__ == '__main__':
    """
    Just for debugging purposes, we make use of this part of the code. However,
    the MAIN code will also contain this part.
    """
    # Read in mesh generated with gmsh
    #   gmsh mesh.geo
    # and converted with dolfin-convert (old-style xml format)
    #   dolfin-convert mesh.geo mesh.xml
    mesh = df.Mesh("mesh.xml")
    subdomains = df.MeshFunction("size_t", mesh, "mesh_physical_region.xml")
    boundaries = df.MeshFunction("size_t", mesh, "mesh_facet_region.xml")

    # Write out new-style xml files
    df.File("mesh.xml") << mesh
    df.File("mesh_physical_region.xml") << subdomains
    df.File("mesh_facet_region.xml") << boundaries

    # Write out for visualization
    df.XDMFFile("mesh.xdmf").write(mesh)
    df.XDMFFile("mesh_physical_region.xdmf").write(subdomains)
    df.XDMFFile("mesh_facet_region.xdmf").write(boundaries)

    # Generate restriction corresponding to interior subdomain (id = 2)
    sphere_restriction = Restrictions.generate_subdomain_restriction(mesh, subdomains, 2)

    # Generate restriction corresp. to interface between the two subdomains
    interface_restriction = Restrictions.generate_interface_restriction(mesh,
                                                                        subdomains, {1, 2})

    # Write out for simulation import (.xml) and visualization (.xdmf)
    df.File("mesh_sphere_restriction.rtc.xml") << sphere_restriction
    df.File("mesh_interface_restriction.rtc.xml") << interface_restriction
    df.XDMFFile("mesh_sphere_restriction.rtc.xdmf").write(sphere_restriction)
    df.XDMFFile("mesh_interface_restriction.rtc.xdmf").write(interface_restriction)
