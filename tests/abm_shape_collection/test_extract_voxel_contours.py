import unittest

import numpy as np

from abm_shape_collection.extract_voxel_contours import (
    connect_array_edges,
    extract_voxel_contours,
    get_array_edges,
    merge_contour_edges,
)


class TestExtractVoxelContours(unittest.TestCase):
    def test_extract_voxel_contours_top_projection(self):
        box = (6, 8, 10)
        all_voxels = [
            [1, 1, -1],
            [2, 1, -1],
            [2, 2, -1],
            [0, 0, -1],
            [5, 5, -1],
        ]

        expected_short_contour = [
            (5, 6),
            (6, 6),
            (6, 5),
            (5, 5),
            (5, 6),
        ]

        expected_long_contour = [
            (0, 1),
            (3, 1),
            (3, 3),
            (2, 3),
            (2, 2),
            (1, 2),
            (1, 0),
            (0, 0),
            (0, 1),
        ]

        contours = extract_voxel_contours(all_voxels, "top", box)

        self.assertEqual(2, len(contours))
        self.assertTrue(len(expected_short_contour), len(contours[0]))
        self.assertTrue(len(expected_long_contour), len(contours[1]))
        self.assertEqual(contours[0][0], contours[0][-1])
        self.assertEqual(contours[1][0], contours[1][-1])

        num_short_points = len(expected_short_contour) - 1
        expected_short_contour_copy = expected_short_contour[:-1] * 2
        self.assertIsNotNone(
            next(
                (
                    i
                    for i in range(num_short_points)
                    if expected_short_contour_copy[i : i + num_short_points] == contours[0][:-1]
                ),
                None,
            )
        )

        num_long_points = len(expected_long_contour) - 1
        expected_long_contour_copy = expected_long_contour[:-1] * 2
        self.assertIsNotNone(
            next(
                (
                    i
                    for i in range(num_long_points)
                    if expected_long_contour_copy[i : i + num_long_points] == contours[1][:-1]
                ),
                None,
            )
        )

    def test_extract_voxel_contours_side1_projection(self):
        box = (6, 8, 10)
        all_voxels = [
            [1, -1, 1],
            [2, -1, 1],
            [2, -1, 2],
            [0, -1, 0],
            [5, -1, 5],
        ]

        expected_short_contour = [
            (5, 6),
            (6, 6),
            (6, 5),
            (5, 5),
            (5, 6),
        ]

        expected_long_contour = [
            (0, 1),
            (3, 1),
            (3, 3),
            (2, 3),
            (2, 2),
            (1, 2),
            (1, 0),
            (0, 0),
            (0, 1),
        ]

        contours = extract_voxel_contours(all_voxels, "side1", box)

        self.assertEqual(2, len(contours))
        self.assertTrue(len(expected_short_contour), len(contours[0]))
        self.assertTrue(len(expected_long_contour), len(contours[1]))
        self.assertEqual(contours[0][0], contours[0][-1])
        self.assertEqual(contours[1][0], contours[1][-1])

        num_short_points = len(expected_short_contour) - 1
        expected_short_contour_copy = expected_short_contour[:-1] * 2
        self.assertIsNotNone(
            next(
                (
                    i
                    for i in range(num_short_points)
                    if expected_short_contour_copy[i : i + num_short_points] == contours[0][:-1]
                ),
                None,
            )
        )

        num_long_points = len(expected_long_contour) - 1
        expected_long_contour_copy = expected_long_contour[:-1] * 2
        self.assertIsNotNone(
            next(
                (
                    i
                    for i in range(num_long_points)
                    if expected_long_contour_copy[i : i + num_long_points] == contours[1][:-1]
                ),
                None,
            )
        )

    def test_extract_voxel_contours_side2_projection(self):
        box = (6, 8, 10)
        all_voxels = [
            [-1, 1, 1],
            [-1, 2, 1],
            [-1, 2, 2],
            [-1, 0, 0],
            [-1, 5, 5],
        ]

        expected_short_contour = [
            (5, 6),
            (6, 6),
            (6, 5),
            (5, 5),
            (5, 6),
        ]

        expected_long_contour = [
            (0, 1),
            (3, 1),
            (3, 3),
            (2, 3),
            (2, 2),
            (1, 2),
            (1, 0),
            (0, 0),
            (0, 1),
        ]

        contours = extract_voxel_contours(all_voxels, "side2", box)

        self.assertEqual(2, len(contours))
        self.assertTrue(len(expected_short_contour), len(contours[0]))
        self.assertTrue(len(expected_long_contour), len(contours[1]))
        self.assertEqual(contours[0][0], contours[0][-1])
        self.assertEqual(contours[1][0], contours[1][-1])

        num_short_points = len(expected_short_contour) - 1
        expected_short_contour_copy = expected_short_contour[:-1] * 2
        self.assertIsNotNone(
            next(
                (
                    i
                    for i in range(num_short_points)
                    if expected_short_contour_copy[i : i + num_short_points] == contours[0][:-1]
                ),
                None,
            )
        )

        num_long_points = len(expected_long_contour) - 1
        expected_long_contour_copy = expected_long_contour[:-1] * 2
        self.assertIsNotNone(
            next(
                (
                    i
                    for i in range(num_long_points)
                    if expected_long_contour_copy[i : i + num_long_points] == contours[1][:-1]
                ),
                None,
            )
        )

    def test_extract_get_array_edges_single_component(self):
        array = np.array(
            [
                [False, False, False, False, False],
                [False, True, False, False, False],
                [False, True, True, False, False],
                [False, False, False, False, False],
            ]
        )

        expected_edges = [
            [(1, 1), (1, 2)],
            [(1, 2), (2, 2)],
            [(2, 2), (2, 3)],
            [(2, 3), (3, 3)],
            [(3, 3), (3, 2)],
            [(3, 2), (3, 1)],
            [(3, 1), (2, 1)],
            [(2, 1), (1, 1)],
        ]

        edges = get_array_edges(array)

        self.assertEqual(len(expected_edges), len(edges))

        for edge in edges:
            self.assertTrue(
                any([set(expected_edge) == set(edge) for expected_edge in expected_edges])
            )

    def test_extract_get_array_edges_multiple_components(self):
        array = np.array(
            [
                [False, False, False, False, False],
                [False, True, False, False, False],
                [False, True, True, False, False],
                [False, False, False, False, True],
            ]
        )

        expected_edges = [
            [(1, 1), (1, 2)],
            [(1, 2), (2, 2)],
            [(2, 2), (2, 3)],
            [(2, 3), (3, 3)],
            [(3, 3), (3, 2)],
            [(3, 2), (3, 1)],
            [(3, 1), (2, 1)],
            [(2, 1), (1, 1)],
            [(3, 4), (3, 5)],
            [(3, 5), (4, 5)],
            [(4, 5), (4, 4)],
            [(4, 4), (3, 4)],
        ]

        edges = get_array_edges(array)

        self.assertEqual(len(expected_edges), len(edges))

        for edge in edges:
            self.assertTrue(
                any([set(expected_edge) == set(edge) for expected_edge in expected_edges])
            )

    def test_connect_array_edges_single_component(self):
        edges = [
            [(3, 3), (3, 2)],
            [(3, 1), (3, 2)],
            [(3, 1), (2, 1)],
            [(1, 1), (2, 1)],
            [(1, 1), (1, 2)],
            [(2, 2), (1, 2)],
            [(2, 2), (2, 3)],
            [(3, 3), (2, 3)],
        ]

        expected_contour = [
            (1, 1),
            (1, 2),
            (2, 2),
            (2, 3),
            (3, 3),
            (3, 2),
            (3, 1),
            (2, 1),
            (1, 1),
        ]

        contours = connect_array_edges(edges)

        self.assertEqual(1, len(contours))
        self.assertTrue(len(expected_contour), len(contours[0]))
        self.assertEqual(contours[0][0], contours[0][-1])

        num_points = len(expected_contour) - 1
        expected_contour_copy = expected_contour[:-1] * 2
        self.assertIsNotNone(
            next(
                (
                    i
                    for i in range(num_points)
                    if expected_contour_copy[i : i + num_points] == contours[0][:-1]
                ),
                None,
            )
        )

    def test_connect_array_edges_multiple_components(self):
        edges = [
            [(3, 3), (3, 2)],
            [(3, 1), (3, 2)],
            [(3, 1), (2, 1)],
            [(1, 1), (2, 1)],
            [(1, 1), (1, 2)],
            [(2, 2), (1, 2)],
            [(2, 2), (2, 3)],
            [(3, 3), (2, 3)],
            [(4, 5), (4, 4)],
            [(3, 4), (4, 4)],
            [(3, 4), (3, 5)],
            [(4, 5), (3, 5)],
        ]

        expected_short_contour = [
            (3, 4),
            (3, 5),
            (4, 5),
            (4, 4),
            (3, 4),
        ]

        expected_long_contour = [
            (1, 1),
            (1, 2),
            (2, 2),
            (2, 3),
            (3, 3),
            (3, 2),
            (3, 1),
            (2, 1),
            (1, 1),
        ]

        contours = connect_array_edges(edges)

        self.assertEqual(2, len(contours))
        self.assertTrue(len(expected_short_contour), len(contours[0]))
        self.assertTrue(len(expected_long_contour), len(contours[1]))
        self.assertEqual(contours[0][0], contours[0][-1])
        self.assertEqual(contours[1][0], contours[1][-1])

        num_short_points = len(expected_short_contour) - 1
        expected_short_contour_copy = expected_short_contour[:-1] * 2
        self.assertIsNotNone(
            next(
                (
                    i
                    for i in range(num_short_points)
                    if expected_short_contour_copy[i : i + num_short_points] == contours[0][:-1]
                ),
                None,
            )
        )

        num_long_points = len(expected_long_contour) - 1
        expected_long_contour_copy = expected_long_contour[:-1] * 2
        self.assertIsNotNone(
            next(
                (
                    i
                    for i in range(num_long_points)
                    if expected_long_contour_copy[i : i + num_long_points] == contours[1][:-1]
                ),
                None,
            )
        )

    def test_merge_contour_edges_square(self):
        contour = [
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 3),
            (3, 3),
            (3, 2),
            (3, 1),
            (2, 1),
            (1, 1),
        ]

        expected_merged_contour = [
            (1, 1),
            (1, 3),
            (3, 3),
            (3, 1),
        ]

        merged_contour = merge_contour_edges(contour)

        self.assertTrue(len(expected_merged_contour) + 1, len(merged_contour))
        self.assertEqual(merged_contour[0], merged_contour[-1])

        num_points = len(expected_merged_contour)
        expected_merge_copy = expected_merged_contour * 2
        self.assertIsNotNone(
            next(
                (
                    i
                    for i in range(num_points)
                    if expected_merge_copy[i : i + num_points] == merged_contour[:-1]
                ),
                None,
            )
        )

    def test_merge_contour_edges_rectangle(self):
        contour = [
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 5),
            (3, 5),
            (3, 4),
            (3, 3),
            (3, 2),
            (3, 1),
            (2, 1),
            (1, 1),
        ]

        expected_merged_contour = [
            (1, 1),
            (1, 5),
            (3, 5),
            (3, 1),
        ]

        merged_contour = merge_contour_edges(contour)

        self.assertTrue(len(expected_merged_contour) + 1, len(merged_contour))
        self.assertEqual(merged_contour[0], merged_contour[-1])

        num_points = len(expected_merged_contour)
        expected_merge_copy = expected_merged_contour * 2
        self.assertIsNotNone(
            next(
                (
                    i
                    for i in range(num_points)
                    if expected_merge_copy[i : i + num_points] == merged_contour[:-1]
                ),
                None,
            )
        )


if __name__ == "__main__":
    unittest.main()
