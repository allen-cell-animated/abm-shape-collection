import xml.etree.ElementTree as ET

from prefect import task

from abm_shape_collection.compile_shape_modes import (
    add_border,
    add_col_labels,
    add_group,
    add_row_labels,
    append_svg_element,
    clear_svg_namespaces,
)


@task
def merge_shape_modes(
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
    points = sorted(list(shape_modes[components[0]].keys()), reverse=True)

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
    for region_index, region in enumerate(regions):
        transform = f"translate({region_index * width * len(views)},0)"
        add_col_labels(col_group, transform, region, height, width, views)

    # Insert shape modes
    for component_index, component in enumerate(components):
        component_id = f"component[{component}]"
        component_group = add_group(
            slice_group, component_id, f"translate(0,{component_index * height})"
        )

        for region_index, region in enumerate(regions):
            region_id = f"region[{region}]"
            region_group = add_group(
                component_group,
                f"{component_id}_{region_id}",
                f"translate({region_index * width * len(views)},0)",
            )

            for view_index, view in enumerate(views):
                view_transform = f"translate({view_index * width},0)"
                view_group = add_group(
                    region_group, f"{component_id}_{region_id}_{view}", view_transform
                )
                add_border(view_group, width, height)

                for point in points:
                    rotate = 0 if view == "top" else 90
                    color = "#ddd" if point == 0 else shade_color(colors[region], -point / 4)

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
    root.set("width", str(len(regions) * width * len(views) + width))

    return ET.tostring(root, encoding="unicode")


def shade_color(color: str, alpha: float) -> str:
    old_color = color.replace("#", "")
    old_red, old_green, old_blue = [int(old_color[i : i + 2], 16) for i in (0, 2, 4)]
    layer_color = 0 if alpha < 0 else 255

    new_red = round(old_red + (layer_color - old_red) * abs(alpha))
    new_green = round(old_green + (layer_color - old_green) * abs(alpha))
    new_blue = round(old_blue + (layer_color - old_blue) * abs(alpha))

    return f"#{new_red:02x}{new_green:02x}{new_blue:02x}"
