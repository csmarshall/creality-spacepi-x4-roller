"""Write "Stand Pi X4 - Flat Both Sides.stl": the Free Roll stand with both faces flat, for a single shared middle stand.

The source stand is a 4 mm plate: flat on the face towards the build plate, relief on the other (hex web recessed, bevelled ribs, a
conical lead-in on the nose groove). The new part keeps the source outline and every hole, fills both faces to the full plate
thickness, and keeps the groove's V notch, which is already symmetric about the mid-plane, so the double-sided wedge nose seats as before.
Everything is derived from the source 3MF, which is only read: the outline from the flat face, the groove circle by fitting its arc,
and the V notch by probing the groove edge at several depths.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import trimesh
from lxml import etree
from shapely.affinity import translate
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

logging.basicConfig(level=os.environ.get("AXEL_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_flat_stand")

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "SpacePi-X4+OG-PULLEY+NEW-WEDGE.3mf"
STAND_MODEL_PATTERN = re.compile(r"3D/Objects/Stand Pi X4\.stl_\d+\.model")  # the numeric suffix changes whenever the slicer re-saves
OUTPUT = HERE / "Stand Pi X4 - Flat Both Sides.stl"
CORE_NS = "{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}"
FACE_INSET_MM = 0.1  # how far inside a face the outline is sectioned, clear of the face itself
PROBE_HEIGHT_MM = 100.0  # ray origin above the part for the groove-edge probes
CYLINDER_SECTIONS = 360
ARC_FIT_TOLERANCE_MM = 0.02  # the source STL is tessellated; a true circle through its vertices stays within this
MIN_ARC_POINTS = 20
NOTCH_SAMPLES = 41
FOOT_HOLE_REGION_MM = (60.0, -40.0)  # (|x| above, z below): the two foot holes sit near the bottom corners of the outline
FOOT_HOLE_TOLERANCE_MM = 0.01  # the written hole is read back from the mesh and must match the requested size to this


def load_stand() -> trimesh.Trimesh:
    """The stand mesh from the 3MF: plate normal along y, groove opening towards +z."""
    with zipfile.ZipFile(SOURCE) as z:
        entry = max((i for i in z.infolist() if STAND_MODEL_PATTERN.fullmatch(i.filename)), key=lambda i: i.file_size)
        log.info("stand mesh: %s (%d bytes)", entry.filename, entry.file_size)
        root = etree.fromstring(z.read(entry))
    verts = np.array([[float(v.get(k)) for k in "xyz"] for v in root.iter(f"{CORE_NS}vertex")])
    tris = np.array([[int(t.get(k)) for k in ("v1", "v2", "v3")] for t in root.iter(f"{CORE_NS}triangle")])
    return trimesh.Trimesh(verts, tris, process=True)


def flat_face_outline(stand: trimesh.Trimesh) -> Polygon:
    """The outline in (x, z) with every through-hole, taken from just inside the flat face (-y), which has no relief."""
    section = stand.section(plane_origin=[0, stand.bounds[0, 1] + FACE_INSET_MM, 0], plane_normal=[0, 1, 0])
    loops = sorted((Polygon(loop[:, [0, 2]]) for loop in section.discrete), key=lambda p: p.area, reverse=True)
    outer, holes = loops[0], loops[1:]
    if not all(outer.contains(h) for h in holes):
        raise RuntimeError("outline has a loop that is not inside the outer boundary")
    return outer.difference(unary_union(holes))


def groove_circle(outline: Polygon) -> tuple[float, float, float]:
    """(x, z, radius) of the circular groove, fitted to the outline run that climbs from the groove floor to the horns."""
    ring = np.array(outline.exterior.coords)[:-1]
    centre_run = ring[np.abs(ring[:, 0]) < 1.0]
    floor = centre_run[np.argmax(centre_run[:, 1])]  # the highest outline point at the mid-line is the groove floor
    start = int(np.argmin(np.hypot(*(ring - floor).T)))
    arc = [ring[start]]
    for step in (1, -1):
        i = start
        while True:
            j = (i + step) % len(ring)
            if ring[j, 1] < ring[i, 1] - 1e-6:  # z stops climbing at the horn tip
                break
            arc.append(ring[j])
            i = j
    arc = np.array(arc)
    arc = arc[np.argsort(np.abs(arc[:, 0] - floor[0]))]  # nearest the floor first: grow the fit outwards until it stops being a circle

    def fit(points: np.ndarray) -> tuple[float, float, float, float]:
        design = np.c_[2 * points[:, 0], 2 * points[:, 1], np.ones(len(points))]
        cx, cz, k = np.linalg.lstsq(design, (points**2).sum(axis=1), rcond=None)[0]
        radius = float(np.sqrt(k + cx**2 + cz**2))
        return float(cx), float(cz), radius, float(np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cz) - radius).max())

    best = None
    for n in range(MIN_ARC_POINTS, len(arc) + 1):
        candidate = fit(arc[:n])
        if candidate[3] > ARC_FIT_TOLERANCE_MM:
            break
        best = (*candidate, n)
    if best is None:
        raise RuntimeError("no circular run found at the groove floor")
    cx, cz, radius, misfit, used = best
    log.info("groove circle: centre x %.3f z %.3f, radius %.3f mm, worst misfit %.4f mm over %d of %d arc points", cx, cz, radius, misfit, used, len(arc))
    return cx, cz, radius


def groove_notch_profile(stand: trimesh.Trimesh, cz: float) -> tuple[np.ndarray, np.ndarray]:
    """(y, radius) of the groove edge across the plate at x = 0, averaged about the mid-plane.

    The source groove is a cylinder at both faces with a V notch between them (the seat for the wedge nose), so the edge radius is
    measured at several depths with rays dropped onto the groove edge rather than assumed from the faces.
    """
    y_min, y_max = stand.bounds[0, 1], stand.bounds[1, 1]
    ys = np.linspace(y_min + FACE_INSET_MM / 2, y_max - FACE_INSET_MM / 2, NOTCH_SAMPLES)
    origins = np.c_[np.zeros(len(ys)), ys, np.full(len(ys), PROBE_HEIGHT_MM)]
    hits, rays, _ = stand.ray.intersects_location(origins, np.tile([0, 0, -1.0], (len(ys), 1)), multiple_hits=False)
    if len(rays) != len(ys):
        raise RuntimeError("a groove-edge probe missed the stand")
    radii = cz - hits[np.argsort(rays)][:, 2]
    asymmetry = float(np.abs(radii - radii[::-1]).max())
    log.info("groove edge radius %.3f mm at the faces, %.3f mm at the mid-plane, source asymmetry %.4f mm", radii[0], radii.max(), asymmetry)
    return ys, (radii + radii[::-1]) / 2


def resize_foot_holes(outline: Polygon, diameter: float) -> Polygon:
    """The outline with its two foot holes (the interior rings in FOOT_HOLE_REGION_MM) replaced by circles of the given diameter on the same centres."""
    kept, centres = [], []
    for ring in outline.interiors:
        c = Polygon(ring).centroid
        if abs(c.x) > FOOT_HOLE_REGION_MM[0] and c.y < FOOT_HOLE_REGION_MM[1]:
            centres.append((c.x, c.y))
        else:
            kept.append(ring)
    if len(centres) != 2:
        raise RuntimeError(f"expected two foot holes, found {len(centres)}")
    holes = [Point(x, z).buffer(diameter / 2, quad_segs=CYLINDER_SECTIONS // 4).exterior for x, z in centres]
    return Polygon(outline.exterior, [*kept, *holes])


def raise_cradle(outline: Polygon, raise_mm: float) -> Polygon:
    """The outline with everything above the neck's cut line moved up raise_mm and the neck lengthened between (the cross-section at the cut line repeated), so the cradle sits higher on unchanged feet and lattice.

    The cut line is the middle of the gap between the top of the lattice holes and the bottom of the groove, so it crosses neither."""
    cx, cz, radius = groove_circle(outline)
    z_cut = 0.5 * (max(Polygon(h).bounds[3] for h in outline.interiors) + cz - radius)
    line = outline.intersection(LineString([(outline.bounds[0] - 1, z_cut), (outline.bounds[2] + 1, z_cut)]))
    spans = [(g.bounds[0], g.bounds[2]) for g in getattr(line, "geoms", [line])]
    band = unary_union([box(x0, z_cut, x1, z_cut + raise_mm) for x0, x1 in spans])
    top = outline.intersection(box(outline.bounds[0] - 1, z_cut, outline.bounds[2] + 1, outline.bounds[3] + 1))
    bottom = outline.intersection(box(outline.bounds[0] - 1, outline.bounds[1] - 1, outline.bounds[2] + 1, z_cut))
    raised = unary_union([bottom, band, translate(top, 0.0, raise_mm)])
    if raised.geom_type != "Polygon" or not raised.is_valid:
        raise RuntimeError(f"raised outline is {raised.geom_type}, valid={raised.is_valid}")
    log.info("cradle raised %.2f mm: neck lengthened at z %.2f, outline height %.2f -> %.2f mm", raise_mm, z_cut, outline.bounds[3] - outline.bounds[1], raised.bounds[3] - raised.bounds[1])
    return raised


def build(stand: trimesh.Trimesh, foot_hole: float | None = None, raise_mm: float = 0.0) -> trimesh.Trimesh:
    """Extrude the outline to the source thickness and cut the symmetric V groove; foot_hole, if given, resizes the two foot holes."""
    y_min, y_max = stand.bounds[0, 1], stand.bounds[1, 1]
    thickness = y_max - y_min
    outline = flat_face_outline(stand)
    if foot_hole is not None:
        outline = resize_foot_holes(outline, foot_hole)
    cx, cz, radius = groove_circle(outline)
    notch_y, notch_r = groove_notch_profile(stand, cz)   # probed on the source stand, at the source groove centre
    if raise_mm:
        outline = raise_cradle(outline, raise_mm)
        cz += raise_mm
    plate = trimesh.creation.extrude_polygon(outline, thickness)  # (x, z, height); rotate +90 degrees about x to (x, -height, z)
    plate.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    plate.apply_translation([0, y_max, 0])
    pad = 1.0  # the cutter runs past both faces
    half = thickness / 2
    profile = np.r_[[(0, -half - pad)], [(notch_r[0], -half - pad)], np.c_[notch_r, notch_y - (y_max + y_min) / 2], [(notch_r[-1], half + pad)], [(0, half + pad)]]
    cutter = trimesh.creation.revolve(np.array(profile), sections=CYLINDER_SECTIONS)  # revolved about z with (r, z) = (r, height)
    cutter.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))  # height z -> y
    cutter.apply_translation([cx, 0, cz])
    part = trimesh.boolean.difference([plate, cutter], engine="manifold")
    if not part.is_watertight:
        raise RuntimeError("flat stand is not watertight")
    return part


def check_foot_hole(part: trimesh.Trimesh, diameter: float) -> None:
    """Fail unless the written mesh has round foot holes of the requested diameter, read from a section through the plate."""
    section = part.section(plane_origin=[0, 0, 0], plane_normal=[0, 1, 0])  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    sizes = [float(np.ptp(loop[:, 0])) for loop in section.discrete if abs(abs(loop[:, 0].mean()) - 64.67) < 0.5 and loop[:, 2].mean() < FOOT_HOLE_REGION_MM[1]]  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    if len(sizes) != 2 or max(abs(v - diameter) for v in sizes) > FOOT_HOLE_TOLERANCE_MM:
        raise RuntimeError(f"foot holes read {sizes}, expected {diameter}")
    log.info("foot holes read back %.3f and %.3f mm", *sizes)


def main() -> None:
    """Build and write the flat stand; with a diameter argument the foot holes are resized and the file name says so, leaving the standard stand alone."""
    foot_hole = float(sys.argv[1]) if len(sys.argv) > 1 else None
    stand = load_stand()
    part = build(stand, foot_hole)
    output = OUTPUT if foot_hole is None else OUTPUT.with_name(f"{OUTPUT.stem} - Foot Hole {foot_hole:.2f}.stl")
    if foot_hole is not None:
        check_foot_hole(part, foot_hole)
    part.export(output)
    log.info("extents %s mm, %d triangles, volume %.0f mm3 (source %.0f mm3), wrote %s", np.round(part.extents, 3), len(part.faces), part.volume, stand.volume, output.name)


if __name__ == "__main__":
    main()
