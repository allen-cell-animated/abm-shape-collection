import unittest

import trimesh
from vtk import vtkCellArray, vtkIdList, vtkPoints, vtkPolyData

from abm_shape_collection.extract_mesh_projections import (
    PROJECTIONS,
    ProjectionType,
    convert_vtk_to_trimesh,
    extract_mesh_projections,
    get_mesh_extent,
    get_mesh_slice,
)


class TestExtractMeshProjections(unittest.TestCase):
    def setUp(self):
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
            (-1.5, -1, 0),
            (0.5, -1, 0),
            (-1.5, 1, 0),
            (0.5, 1, 0),
            (-0.5, 0, 1),
            (-0.5, 0, -1),
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

        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        tri_mesh_offset = trimesh.Trimesh(vertices=offset_vertices, faces=faces)

        self.vtk_mesh = vtk_mesh
        self.tri_mesh = tri_mesh
        self.tri_mesh_offset = tri_mesh_offset

        top_full_square = [[-1, 1], [-1, -1], [1, -1], [1, 1], [-1, 1]]
        top_half_square = [
            [-0.5, 0.5],
            [-0.5, -0.5],
            [0.5, -0.5],
            [0.5, 0.5],
            [-0.5, 0.5],
        ]

        side_full_diamond = [[0, -1], [1, 0], [0, 1], [-1, 0], [0, -1]]
        side_half_diamond = [
            [0, -1],
            [0.5, -0.5],
            [0.5, 0.5],
            [0, 1],
            [-0.5, 0.5],
            [-0.5, -0.5],
            [0, -1],
        ]

        self.slices = {
            "top": [top_full_square],
            "side1": [side_full_diamond],
            "side2": [side_full_diamond],
        }

        self.extents = {
            "top": {
                -0.5: [top_half_square],
                0: [top_full_square],
                0.5: [top_half_square],
            },
            "side1": {
                -1: [],
                -0.5: [side_half_diamond],
                0: [side_full_diamond],
                0.5: [side_half_diamond],
            },
            "side2": {
                -1: [],
                -0.5: [side_half_diamond],
                0: [side_full_diamond],
                0.5: [side_half_diamond],
            },
        }

    def test_extract_mesh_projections_all_types_no_translation(self):
        projections = extract_mesh_projections(self.vtk_mesh)
        for proj, _, _ in PROJECTIONS:
            self.assertCountEqual(self.slices[proj], projections[f"{proj}_slice"])
            self.assertDictEqual(self.extents[proj], projections[f"{proj}_extent"])
            self.assertFalse(f"{proj}_extent" in proj)
            self.assertFalse(f"{proj}_slice" in proj)

    def test_extract_mesh_projections_slices_no_translation(self):
        projections = extract_mesh_projections(self.vtk_mesh, [ProjectionType.SLICE], offset=None)
        for proj, _, _ in PROJECTIONS:
            self.assertCountEqual(self.slices[proj], projections[f"{proj}_slice"])
            self.assertFalse(f"{proj}_extent" in proj)

    def test_extract_mesh_projections_extents_no_translation(self):
        projections = extract_mesh_projections(self.vtk_mesh, [ProjectionType.EXTENT], offset=None)
        for proj, _, _ in PROJECTIONS:
            self.assertDictEqual(self.extents[proj], projections[f"{proj}_extent"])
            self.assertFalse(f"{proj}_slice" in proj)

    def test_extract_mesh_projections_with_translation(self):
        offset = (0.5, 0, 0)
        projections = extract_mesh_projections(
            self.tri_mesh_offset, [ProjectionType.SLICE, ProjectionType.EXTENT], offset=offset
        )
        for proj, _, _ in PROJECTIONS:
            self.assertCountEqual(self.slices[proj], projections[f"{proj}_slice"])
            self.assertDictEqual(self.extents[proj], projections[f"{proj}_extent"])

    def test_convert_vtk_to_trimesh(self):
        tri_mesh = convert_vtk_to_trimesh(self.vtk_mesh)
        self.assertTrue((self.tri_mesh.vertices == tri_mesh.vertices).all())
        self.assertTrue((self.tri_mesh.faces == tri_mesh.faces).all())

    def test_get_mesh_slice(self):
        top_points = get_mesh_slice(self.tri_mesh, (0, 0, 1))
        side1_points = get_mesh_slice(self.tri_mesh, (1, 0, 0))
        side2_points = get_mesh_slice(self.tri_mesh, (0, 1, 0))

        self.assertCountEqual(self.slices["top"], top_points)
        self.assertCountEqual(self.slices["side1"], side1_points)
        self.assertCountEqual(self.slices["side2"], side2_points)

    def test_get_mesh_extent(self):
        top_points = get_mesh_extent(self.tri_mesh, (0, 0, 1), 2)
        side1_points = get_mesh_extent(self.tri_mesh, (1, 0, 0), 1)
        side2_points = get_mesh_extent(self.tri_mesh, (0, 1, 0), 0)

        self.assertDictEqual(self.extents["top"], top_points)
        self.assertDictEqual(self.extents["side1"], side1_points)
        self.assertDictEqual(self.extents["side2"], side2_points)


if __name__ == "__main__":
    unittest.main()
