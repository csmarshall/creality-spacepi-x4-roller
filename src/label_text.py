"""Recessed text for printed parts: the lettering is cut into a face that prints on the bed, mirrored in x so it reads correctly with the part turned over.

glyph(text, size, centre) builds the lettering as shapely polygons (even-odd fill, so the counters of 0, 6, 8 stay open) and cut(mesh, pieces, face_z, depth) removes them from
mesh from face_z up to depth. Check limit: cut() verifies the result is closed and that the removed volume equals lettering area x depth; it cannot tell whether a font's glyph shapes are right.
"""
from __future__ import annotations

import numpy as np
import trimesh
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.affinity import translate as shapely_translate

VOLUME_TOLERANCE = 0.05  # fraction of the lettering volume, plus 0.1 mm3


def glyph(text: str, size: float, centre: tuple[float, float] = (0.0, 0.0), rotation: float = 0.0) -> list[Polygon]:
    """The lettering centred on `centre`, as separate polygons, laid out as seen from below (the bed side) and turned `rotation` degrees counter-clockwise there, then mirrored in x to top-view coordinates."""
    path = TextPath((0, 0), text, size=size, prop=FontProperties(family="DejaVu Sans", weight="bold"))
    shape = Polygon()
    for ring in path.to_polygons():
        shape = shape.symmetric_difference(Polygon(ring).buffer(0))
    minx, miny, maxx, maxy = shape.bounds
    shape = scale(rotate(translate(shape, -(minx + maxx) / 2, -(miny + maxy) / 2), rotation, origin=(0, 0)), xfact=-1, yfact=1, origin=(0, 0))
    shape = translate(shape, centre[0], centre[1])
    return list(getattr(shape, "geoms", [shape]))


def max_radius(pieces: list[Polygon], about: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    """(smallest, largest) distance from `about` to the lettering's outline."""
    d = np.hypot(*(np.vstack([np.array(p.exterior.coords) for p in pieces]) - np.array(about)).T)
    return float(d.min()), float(d.max())


def cut(mesh: trimesh.Trimesh, pieces: list[Polygon], face_z: float, depth: float) -> trimesh.Trimesh:
    """Remove the lettering from `face_z` up to `depth` into the part (the face is the bottom, the part lies above it)."""
    cutters = [trimesh.creation.extrude_polygon(p, depth + 1.0).apply_translation([0, 0, face_z - 1.0]) for p in pieces]
    out = trimesh.boolean.difference([mesh, *cutters], engine="manifold")
    area = sum(p.area for p in pieces)
    if not out.is_watertight or abs((mesh.volume - out.volume) - area * depth) > VOLUME_TOLERANCE * area * depth + 0.1:
        raise RuntimeError(f"lettering cut is open or removed {mesh.volume - out.volume:.2f} mm3, expected {area * depth:.2f}")
    return out


LINE_GAP_MM = 1.2


def block(lines: list[tuple[str, float]]) -> list[Polygon]:
    """Stacked lines of (text, height in mm), the block centred on the origin."""
    total = sum(h for _, h in lines) + LINE_GAP_MM * (len(lines) - 1)
    pieces, top = [], total / 2
    for text, height in lines:
        pieces += glyph(text, height, (0.0, top - height / 2))
        top -= height + LINE_GAP_MM
    return pieces


def face_area(mesh: trimesh.Trimesh, z: float) -> Polygon:
    """The solid cross-section of a part at height z (even-odd over its loops, so holes are holes), in the part's own x and y."""
    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    area = Polygon()
    for loop in section.discrete:  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        area = area.symmetric_difference(Polygon(loop[:, :2]).buffer(0))
    return area


def find_spot(area, pieces: list[Polygon], prefer: tuple[float, float], clearance: float, avoid: tuple[float, float, float] | None = None, step: float = 1.0):
    """The grid point nearest `prefer` where the lettering (given centred on the origin), grown by `clearance`, lies wholly inside `area` and, if `avoid` = (x, y, radius) is given,
    wholly outside that circle; None if there is no such point."""
    ink = unary_union(pieces)
    grown = prep(area)
    best = None
    minx, miny, maxx, maxy = area.bounds
    for x in np.arange(minx, maxx, step):
        for y in np.arange(miny, maxy, step):
            if best is not None and np.hypot(x - prefer[0], y - prefer[1]) >= np.hypot(best[0] - prefer[0], best[1] - prefer[1]):
                continue
            spot = shapely_translate(ink, x, y).buffer(clearance)
            if grown.contains(spot) and (avoid is None or spot.distance(Point(avoid[0], avoid[1]).buffer(avoid[2])) > 0):
                best = (float(x), float(y))
    return best
