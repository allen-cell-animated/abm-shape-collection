import unittest

import trimesh
from vtk import vtkCellArray, vtkIdList, vtkPoints, vtkPolyData

from abm_shape_collection.extract_mesh_wireframe import extract_mesh_wireframe


class TestConstructExtractMeshProjections(unittest.TestCase):
    def setUp(self) -> None:
        vertices = [
            (-1, -1, 0),
            (1, -1, 0),
            (-1, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]

        faces = [
            (0, 1, 4),
            (1, 3, 4),
            (3, 2, 4),
            (2, 0, 4),
            (0, 1, 5),
            (1, 3, 5),
            (3, 2, 5),
            (2, 0, 5),
        ]

        self.vertices = vertices
        self.faces = faces

        offset_vertices = [
            (-1.5, -1.0, 0.0),
            (0.5, -1.0, 0.0),
            (-1.5, 1.0, 0.0),
            (0.5, 1.0, 0.0),
            (-0.5, 0.0, 1.0),
            (-0.5, 0.0, -1.0),
        ]

        vtk_mesh = vtkPolyData()
        points = vtkPoints()
        polys = vtkCellArray()

        for index, vertex in enumerate(vertices):
            points.InsertPoint(index, vertex)

        for face in faces:
            id_list = vtkIdList()
            for index in face:
                id_list.InsertNextId(index)
            polys.InsertNextCell(id_list)

        vtk_mesh.SetPoints(points)
        vtk_mesh.SetPolys(polys)

        tri_mesh = trimesh.Trimesh(vertices=offset_vertices, faces=faces)

        self.vtk_mesh = vtk_mesh
        self.tri_mesh = tri_mesh

        wireframe = [
            [(0.0, 0.0, 1.0), (-1.0, -1.0, 0.0)],
            [(0.0, 0.0, 1.0), (-1.0, 1.0, 0.0)],
            [(0.0, 0.0, 1.0), (1.0, -1.0, 0.0)],
            [(0.0, 0.0, 1.0), (1.0, 1.0, 0.0)],
            [(0.0, 0.0, -1.0), (-1.0, -1.0, 0.0)],
            [(0.0, 0.0, -1.0), (-1.0, 1.0, 0.0)],
            [(0.0, 0.0, -1.0), (1.0, -1.0, 0.0)],
            [(0.0, 0.0, -1.0), (1.0, 1.0, 0.0)],
            [(-1.0, 1.0, 0.0), (-1.0, -1.0, 0.0)],
            [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0)],
            [(1.0, -1.0, 0.0), (1.0, 1.0, 0.0)],
            [(1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)],
        ]

        self.wireframe = [sorted(edge) for edge in wireframe]

    def test_extract_extract_mesh_wireframe_no_translation(self):
        wireframe = extract_mesh_wireframe(self.vtk_mesh, offset=None)

        self.assertTrue(len(self.wireframe) == len(wireframe))
        for edge in wireframe:
            self.assertTrue(sorted(edge) in self.wireframe)

    def test_extract_extract_mesh_wireframe_with_translation(self):
        offset = (0.5, 0.0, 0.0)
        wireframe = extract_mesh_wireframe(self.tri_mesh, offset=offset)

        self.assertTrue(len(self.wireframe) == len(wireframe))
        for edge in wireframe:
            self.assertTrue(sorted(edge) in self.wireframe)


if __name__ == "__main__":
    unittest.main()
