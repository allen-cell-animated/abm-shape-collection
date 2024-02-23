import numpy as np


def extract_voxel_contours(
    all_voxels: list[tuple[int, int, int]],
    projection: str,
    box: tuple[int, int, int],
) -> list[list[tuple[int, int]]]:
    voxels = set()
    length, width, height = box

    if projection == "top":
        voxels.update({(x, y) for x, y, _ in all_voxels})
        x_bounds = length
        y_bounds = width
    elif projection == "side1":
        voxels.update({(x, z) for x, _, z in all_voxels})
        x_bounds = length
        y_bounds = height
    else:
        voxels.update({(y, z) for _, y, z in all_voxels})
        x_bounds = width
        y_bounds = height

    array = np.full((x_bounds, y_bounds), False)
    array[tuple(np.transpose(list(voxels)))] = True

    edges = get_array_edges(array)
    contours = connect_array_edges(edges)

    return [merge_contour_edges(contour) for contour in contours]


def get_array_edges(array: np.ndarray) -> list[list[tuple[int, int]]]:
    edges = []
    x, y = np.nonzero(array)

    for i, j in zip(x.tolist(), y.tolist()):
        if j == array.shape[1] - 1 or not array[i, j + 1]:
            edges.append([(i, j + 1), (i + 1, j + 1)])

        if i == array.shape[0] - 1 or not array[i + 1, j]:
            edges.append([(i + 1, j), (i + 1, j + 1)])

        if j == 0 or not array[i, j - 1]:
            edges.append([(i, j), (i + 1, j)])

        if i == 0 or not array[i - 1, j]:
            edges.append([(i, j), (i, j + 1)])

    return edges


def connect_array_edges(edges: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    contours: list[list[tuple[int, int]]] = []

    while edges:
        contour = edges[0]
        contour_length = 0
        edges.remove(contour)

        while contour_length != len(contour):
            contour_length = len(contour)

            forward = list(filter(lambda edge: contour[-1] == edge[0], edges))

            if len(forward) > 0:
                edges.remove(forward[0])
                contour.extend(forward[0][1:])

            backward = list(filter(lambda edge: contour[-1] == edge[-1], edges))

            if len(backward) > 0:
                edges.remove(backward[0])
                contour.extend(list(reversed(backward[0]))[1:])

            if contour_length == len(contour):
                contours.append([(x, y) for x, y in contour])

    return sorted(contours, key=len)


def merge_contour_edges(contour: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged = contour.copy()

    for (x0, y0), (x1, y1), (x2, y2) in zip(contour, contour[1:], contour[2:]):
        area = x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1)
        if area == 0:
            merged.remove((x1, y1))

    return merged
