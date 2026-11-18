"""Write two small ABS fit-test coupons: a ladder of round recesses for the 608 bearing OD and a ladder of pegs for its bore.

Print them with the same material, nozzle, layer height, temperatures and orientation as the real parts, then try the bearing in each recess and each
peg in the bearing bore. The step that fits by hand is the modelled size to use in the fit sheet. Every hole and peg carries its modelled diameter as
raised numerals beneath it, and each plate is titled RECESS or PEG. Ascending order is left to right then top to bottom, from the corner that is cut off.
Writes the STLs plus a top view sliced from the finished meshes (not drawn from the constants that built them).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import trimesh
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from shapely.geometry import Polygon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logging.basicConfig(level=os.environ.get("AXEL_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_fit_coupon")

HERE = Path(__file__).resolve().parent
RECESS_STL = HERE / "Fit Test - Recess Ladder.stl"
PEG_STL = HERE / "Fit Test - Peg Ladder.stl"
LAYOUT_PNG = HERE / "fit_coupon_layout.png"

COLUMNS = 3
RECESS_DIAMETERS_MM = [22.00 + 0.05 * i for i in range(9)]  # the model's 22.00 printed 21.70 to 21.90; the bearing is 21.96
PEG_DIAMETERS_MM = [7.90 + 0.025 * i for i in range(9)]  # the model's 8.13 to 8.33 printed 8.08 or more; the bore is 7.95
RECESS_PLATE_MM = 5.0  # thick enough to grip a bearing, thin enough to print in minutes
PEG_BASE_MM = 3.0
PEG_HEIGHT_MM = 9.0  # the modelled straight stub is 9.65 mm; a bearing is 7 mm wide
PEG_SINK_MM = 0.5  # how far each peg is embedded in the base so the union is solid
PITCH_X_MM = 28.0  # centre to centre across; wider than the largest recess plus a wall
WALL_MM = 2.5  # plate left between a hole and its label, and between a label and the next row
MARGIN_MM = 4.0  # plate edge beyond the outermost features
CORNER_CUT_MM = 8.0  # the first (smallest) position's corner is cut off so the order cannot be mistaken
SECTIONS = 256  # chord error at 11 mm radius is about 0.001 mm
LABEL_HEIGHT_MM = 5.5  # numeral height; a bold face at this size has strokes near 1 mm, printable with a 0.4 or 0.6 mm nozzle
LABEL_RAISE_MM = 0.8  # how far the numerals stand above the plate
LABEL_SINK_MM = 0.2  # how far they are embedded in the plate so the union is solid
LABEL_FONT = FontProperties(family="DejaVu Sans Mono", weight="bold")


@dataclass
class Layout:
    """Plate size and where each hole or peg, its label and the title sit (x right, y up, origin at the plate centre)."""

    width: float
    depth: float
    centres: np.ndarray
    label_centres: np.ndarray
    title_centre: tuple[float, float]


def layout(count: int, widest_mm: float) -> Layout:
    """A COLUMNS-wide grid, ascending left to right then top to bottom, each cell holding the feature and its label beneath it."""
    rows = -(-count // COLUMNS)
    pitch_y = widest_mm + WALL_MM + LABEL_HEIGHT_MM + WALL_MM
    title_block = LABEL_HEIGHT_MM + WALL_MM
    width = (COLUMNS - 1) * PITCH_X_MM + widest_mm + 2 * MARGIN_MM
    depth = 2 * MARGIN_MM + title_block + (rows - 1) * pitch_y + widest_mm + WALL_MM + LABEL_HEIGHT_MM
    top = depth / 2 - MARGIN_MM
    xs = (np.arange(COLUMNS) - (COLUMNS - 1) / 2) * PITCH_X_MM
    centres, labels = [], []
    for r in range(rows):
        cy = top - title_block - widest_mm / 2 - r * pitch_y
        for x in xs:
            centres.append((x, cy))
            labels.append((x, cy - widest_mm / 2 - WALL_MM - LABEL_HEIGHT_MM / 2))
    return Layout(width, depth, np.array(centres[:count]), np.array(labels[:count]), (0.0, top - LABEL_HEIGHT_MM / 2))


def text_mesh(text: str, centre: tuple[float, float], z_bottom: float) -> trimesh.Trimesh:
    """Raised numerals: the text outline, scaled so digits are LABEL_HEIGHT_MM tall, centred on the point, extruded from z_bottom."""
    def outline(s: str) -> Polygon:
        shape = Polygon()
        for ring in TextPath((0, 0), s, size=1.0, prop=LABEL_FONT).to_polygons():
            if len(ring) >= 3:
                shape = shape.symmetric_difference(Polygon(ring).buffer(0))  # even-odd fill, so the holes in 0, 6, 8 and 9 stay open
        return shape
    digit = outline("0").bounds
    shape = outline(text)
    scale = LABEL_HEIGHT_MM / (digit[3] - digit[1])
    min_x, min_y, max_x, max_y = shape.bounds

    def place(points: object) -> list[tuple[float, float]]:
        """Scale glyph units to millimetres and centre the whole string on the requested point."""
        return [((x - (min_x + max_x) / 2) * scale + centre[0], (y - (min_y + max_y) / 2) * scale + centre[1]) for x, y in points]  # type: ignore[attr-defined]

    parts = []
    for poly in getattr(shape, "geoms", [shape]):
        moved = Polygon(place(poly.exterior.coords), [place(ring.coords) for ring in poly.interiors])
        mesh = trimesh.creation.extrude_polygon(moved, LABEL_RAISE_MM + LABEL_SINK_MM)
        mesh.apply_translation([0, 0, z_bottom - LABEL_SINK_MM])
        parts.append(mesh)
    return trimesh.util.concatenate(parts)


def corner_cut(lay: Layout, height: float) -> trimesh.Trimesh:
    """A triangular prism that removes the top-left corner of the plate."""
    prism = trimesh.creation.extrude_triangulation(np.array([[0, 0], [CORNER_CUT_MM, 0], [0, -CORNER_CUT_MM]], dtype=float), np.array([[0, 1, 2]]), height * 3)
    prism.apply_translation([-lay.width / 2, lay.depth / 2, -height])
    return prism


def labels_for(lay: Layout, diameters: list[float], decimals: int, title: str, z_top: float) -> list[trimesh.Trimesh]:
    """Numeral meshes for every position plus the plate title, standing on the plate's top face at z_top."""
    meshes = [text_mesh(f"{d:.{decimals}f}", (x, y), z_top) for d, (x, y) in zip(diameters, lay.label_centres)]
    meshes.append(text_mesh(title, lay.title_centre, z_top))
    return meshes


def build_recess_ladder(diameters: list[float] | None = None, tag: str = "") -> trimesh.Trimesh:
    """A plate with through-holes at the modelled recess diameters (default RECESS_DIAMETERS_MM) and raised labels; tag is added to the title."""
    diameters = diameters or RECESS_DIAMETERS_MM
    lay = layout(len(diameters), max(diameters))
    plate = trimesh.creation.box([lay.width, lay.depth, RECESS_PLATE_MM])
    plate.apply_translation([0, 0, RECESS_PLATE_MM / 2])
    cutters = [trimesh.creation.cylinder(radius=d / 2, height=RECESS_PLATE_MM * 3, sections=SECTIONS).apply_translation([x, y, RECESS_PLATE_MM / 2]) for d, (x, y) in zip(diameters, lay.centres)]
    cutters.append(corner_cut(lay, RECESS_PLATE_MM))
    cut = trimesh.boolean.difference([plate, *cutters], engine="manifold")
    return trimesh.boolean.union([cut, *labels_for(lay, diameters, 2, f"RECESS {tag}".strip(), RECESS_PLATE_MM)], engine="manifold")


def build_peg_ladder(diameters: list[float] | None = None, tag: str = "", footprint_mm: float | None = None, height_mm: float = PEG_HEIGHT_MM) -> trimesh.Trimesh:
    """A base plate with vertical pegs at the modelled stub diameters (default PEG_DIAMETERS_MM) and raised labels; tag is added to the title.

    footprint_mm is the widest feature the plate is laid out for, so a peg plate matches its recess plate (default: the widest default recess); height_mm is the peg height above the base.
    """
    diameters = diameters or PEG_DIAMETERS_MM
    lay = layout(len(diameters), footprint_mm or max(RECESS_DIAMETERS_MM))  # same footprint as the recess plate so the two print as a matched pair
    base = trimesh.creation.box([lay.width, lay.depth, PEG_BASE_MM])
    base.apply_translation([0, 0, PEG_BASE_MM / 2])
    pegs = [trimesh.creation.cylinder(radius=d / 2, height=height_mm + PEG_SINK_MM, sections=SECTIONS).apply_translation([x, y, PEG_BASE_MM - PEG_SINK_MM + (height_mm + PEG_SINK_MM) / 2]) for d, (x, y) in zip(diameters, lay.centres)]
    solid = trimesh.boolean.union([base, *pegs, *labels_for(lay, diameters, 3, f"PEG {tag}".strip(), PEG_BASE_MM)], engine="manifold")
    return trimesh.boolean.difference([solid, corner_cut(lay, PEG_BASE_MM)], engine="manifold")


def draw_layout(recess: trimesh.Trimesh, peg: trimesh.Trimesh) -> None:
    """Top views sliced from the finished meshes: the plate and its holes or pegs in grey, the raised labels in black."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    for ax, mesh, plate_z, title in ((axes[0], recess, 1.0, "Recess ladder, sliced from the STL"), (axes[1], peg, 1.0, "Peg ladder, sliced from the STL")):
        top_z = float(mesh.bounds[1, 2])
        for z, colour in ((plate_z, "#9aa4b5"), (RECESS_PLATE_MM if mesh is recess else PEG_BASE_MM + LABEL_RAISE_MM / 2, "#111")):
            section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
            if section is None:
                continue
            shape = Polygon()
            for loop in section.discrete:
                if len(loop) >= 3:
                    shape = shape.symmetric_difference(Polygon(np.asarray(loop)[:, :2]).buffer(0))
            for poly in getattr(shape, "geoms", [shape]):
                if poly.is_empty:
                    continue
                ax.fill(*poly.exterior.xy, color=colour, lw=0)
                for hole in poly.interiors:
                    ax.fill(*hole.xy, color="white", lw=0)
        ax.set_aspect("equal"); ax.set_title(f"{title} (top {top_z:.1f} mm)"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(LAYOUT_PNG, dpi=110)
    log.info("wrote %s", LAYOUT_PNG.name)


def check_diameters(mesh: trimesh.Trimesh, diameters: list[float], centres: np.ndarray, label: str) -> None:
    """Read each hole or peg diameter back from the finished mesh."""
    for d, (x, y) in zip(diameters, centres):
        radius_xy = np.hypot(mesh.vertices[:, 0] - x, mesh.vertices[:, 1] - y)
        ring = radius_xy[np.abs(radius_xy - d / 2) < 0.2]
        measured = 2 * float(ring.max())
        assert abs(measured - d) < 0.005, f"{label} {d}: read back {measured:.4f}"


def main() -> None:
    """Build both coupons, check them, write the STLs and the layout picture."""
    recess, peg = build_recess_ladder(), build_peg_ladder()
    for name, mesh in (("recess ladder", recess), ("peg ladder", peg)):
        if not mesh.is_watertight or mesh.volume <= 0:
            raise RuntimeError(f"{name} is not a closed solid")
    check_diameters(recess, RECESS_DIAMETERS_MM, layout(len(RECESS_DIAMETERS_MM), max(RECESS_DIAMETERS_MM)).centres, "recess")
    check_diameters(peg, PEG_DIAMETERS_MM, layout(len(PEG_DIAMETERS_MM), max(RECESS_DIAMETERS_MM)).centres, "peg")
    recess.export(RECESS_STL); peg.export(PEG_STL)
    log.info("wrote %s (%.1f x %.1f x %.1f mm) and %s (%.1f x %.1f x %.1f mm); diameters read back from the meshes within 0.005 mm", RECESS_STL.name, *recess.extents, PEG_STL.name, *peg.extents)
    draw_layout(recess, peg)


if __name__ == "__main__":
    main()
