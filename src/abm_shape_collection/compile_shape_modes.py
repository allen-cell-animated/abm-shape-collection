import xml.etree.ElementTree as ET

from prefect import task


@task
def compile_shape_modes(
    shape_modes: dict,
    views: list[str],
    regions: list[str],
    variance: list[float],
    colors: dict,
    box: tuple[int, int],
    scale: float,
) -> str:
    # Extract layout parameters
    width, height = box
    components = sorted(list(shape_modes.keys()))
    points = sorted(list(shape_modes[components[0]].keys()))

    # Initialize output
    root = ET.fromstring("<svg></svg>")
    slice_group = add_group(root, "slices", f"translate({width},{height / 2})")
    row_group = add_group(root, "row_labels", f"translate(0,{height / 2})")
    col_group = add_group(root, "col_labels", f"translate({width},0)")

    # Add row labels
    for component_index, component in enumerate(components):
        transform = f"translate(0,{component_index * height})"
        add_row_labels(row_group, transform, component, variance[component_index], width, height)

    # Add col labels
    for point_index, point in enumerate(points):
        transform = f"translate({point_index * width * len(views)},0)"
        add_col_labels(col_group, transform, f"{point}σ", height, width, views)

    # Insert shape modes
    for component_index, component in enumerate(components):
        component_id = f"component[{component}]"
        component_group = add_group(
            slice_group, component_id, f"translate(0,{component_index * height})"
        )

        for point_index, point in enumerate(points):
            point_id = f"point[{point}]"
            point_group = add_group(
                component_group,
                f"{component_id}_{point_id}",
                f"translate({point_index * width * len(views)},0)",
            )

            for view_index, view in enumerate(views):
                view_transform = f"translate({view_index * width},0)"
                view_group = add_group(
                    point_group, f"{component_id}_{point_id}_{view}", view_transform
                )
                add_border(view_group, width, height)

                for region in regions:
                    rotate = 0 if view == "top" else 90
                    color = colors[region]
                    append_svg_element(
                        shape_modes[component][point][region][view],
                        view_group,
                        width,
                        height,
                        scale,
                        rotate,
                        color,
                    )

    # Clean namespaces
    clear_svg_namespaces(root)
    for element in root.findall(".//*"):
        clear_svg_namespaces(element)

    # Set SVG size and namespace
    root.set("xmlns", "http://www.w3.org/2000/svg")
    root.set("height", str(len(components) * height + height / 2))
    root.set("width", str(len(points) * width * len(views) + width))

    return ET.tostring(root, encoding="unicode")


def add_group(root: ET.Element, group_id: str, group_transform: str) -> ET.Element:
    return ET.SubElement(root, "g", {"id": group_id, "transform": group_transform})


def add_border(root: ET.Element, width: int, height: int) -> None:
    ET.SubElement(
        root,
        "rect",
        {
            "width": str(width),
            "height": str(height),
            "x": "0",
            "y": "0",
            "fill": "none",
            "stroke": "#eeeeee",
            "stroke-width": "0.5",
        },
    )


def add_col_labels(
    root: ET.Element, transform: str, text: str, height: int, width: int, views: list[str]
) -> None:
    group = ET.SubElement(root, "g", {"transform": transform})

    ET.SubElement(
        group,
        "text",
        {
            "font-family": "Helvetica",
            "font-size": "16pt",
            "text-anchor": "middle",
            "font-weight": "bold",
            "x": str(len(views) * width / 2),
            "y": str(20),
        },
    ).text = text

    for i, view in enumerate(views):
        ET.SubElement(
            group,
            "text",
            {
                "font-family": "Helvetica",
                "font-size": "14pt",
                "text-anchor": "middle",
                "font-style": "italic",
                "x": str((i + 0.5) * width),
                "y": str(height / 2 - 10),
            },
        ).text = f"{view.replace('_', ' ')} view"


def add_row_labels(
    root: ET.Element, transform: str, component: int, variance: float, width: int, height: int
) -> None:
    group = ET.SubElement(root, "g", {"transform": transform})

    ET.SubElement(
        group,
        "text",
        {
            "font-family": "Helvetica",
            "font-size": "16pt",
            "text-anchor": "middle",
            "font-weight": "bold",
            "x": str(width / 2),
            "y": str(height / 2 - 12),
        },
    ).text = "Shape"

    ET.SubElement(
        group,
        "text",
        {
            "font-family": "Helvetica",
            "font-size": "16pt",
            "text-anchor": "middle",
            "font-weight": "bold",
            "x": str(width / 2),
            "y": str(height / 2 + 8),
        },
    ).text = f"Mode {component}"

    ET.SubElement(
        group,
        "text",
        {
            "font-family": "Helvetica",
            "font-size": "14pt",
            "text-anchor": "middle",
            "x": str(width / 2),
            "y": str(height / 2 + 30),
        },
    ).text = f"{variance*100:.1f}%"


def append_svg_element(
    element: str,
    root: ET.Element,
    width: int,
    height: int,
    scale: float = 1.0,
    rotate: int = 0,
    color: str = "#555",
) -> None:
    """Append svg element to root."""
    path = ET.fromstring(element).findall(".//")[0]

    cx = width / 2
    cy = height / 2

    path.set("fill", "none")
    path.set("stroke", color)
    path.set("stroke-width", str(width / 80 / scale))
    path.set("transform", f"rotate({rotate},{cx},{cy}) translate({cx},{cy}) scale({scale})")
    root.insert(0, path)


def clear_svg_namespaces(svg: ET.Element) -> None:
    _, has_namespace, postfix = svg.tag.partition("}")
    if has_namespace:
        svg.tag = postfix
