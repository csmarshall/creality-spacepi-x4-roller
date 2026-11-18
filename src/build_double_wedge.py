"""Write "Wedge - Double Sided.stl": the Free Roll wedge mirrored about its V-nose, an axle on each side, for a middle stand.

The wedge is a body of revolution, so its profile is taken from an axial section, cut at the nose, mirrored and revolved. Mirroring
about the nose keeps everything on the axle side of it: the nose still sits in the stand's single V-groove, and each pulley stays
the same distance from the stand as with the original wedge. The source 3MF is only read.
"""
from __future__ import annotations

import logging
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import trimesh
from lxml import etree
from shapely.geometry import LineString, Point, Polygon, box

import build_flat_stand as fs

logging.basicConfig(level=os.environ.get("AXEL_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_double_wedge")

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "SpacePi-X4+OG-PULLEY+NEW-WEDGE.3mf"
WEDGE_MODEL_PATTERN = re.compile(r"3D/Objects/Wedge - No AXLE\.stl_\d+\.model")  # the numeric suffix changes whenever the slicer re-saves the 3MF
OUTPUT = HERE / "Wedge - Double Sided.stl"
CORE_NS = "{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}"
SEGMENTS = 360  # chord error about 0.001 mm at the nose
PROFILE_TOLERANCE_MM = 0.002
MAX_DEVIATION_MM = 0.02  # allowed distance between the source wedge and the new part on the kept side
NEW_STUB_DOUBLE = HERE / "Wedge - Double Sided - New Stub.stl"
NEW_STUB_SINGLE = HERE / "Wedge - Single Sided - New Stub.stl"
STUB_DIAMETER_MM = 8.05  # All Tests print 2026-11-09: 8.05 stub was the one that fit a 608; earlier coupon: 7.900 to 8.075 slid on, 8.100 too tight
STUB_CHAMFER_MM = 0.5  # 45 degree lead-in at the tip, replacing the old 3 mm cone
STUB_FILLET_MM = 0.3  # root fillet; must stay within the 608's bore-edge chamfer (ISO minimum 0.3 mm) so the bearing still reaches the shoulder
SHOULDER_RADIUS_MM = 4.5  # where the profile leaves the stub: the first axial position whose radius exceeds this is the shoulder; a design choice, the old stub is under 4.2 and the shoulder climbs to 16 within 1 mm
SHOULDER_SCAN_MM = 0.01
BEARING_WIDTH_MM = 7.0  # ISO 608; build_assembly takes its value from here
STUB_PROTRUSION_MM = 2.5  # how far the stub's tip stands past the bearing's far face (was about 6 mm, a lever that snapped the stub at its root when the wedge was pulled out of the bearing); the tip chamfer is part of it
COLLAR_HEIGHT_MM = 0.56  # final wedge: from the flat axle-side face to the shoulder the bearing rests on (the old wedge's collar length)
COLLAR_RADIUS_MM = 5.4  # the collar's outer radius: out to the 608 inner race face (about 5.5 mm), so the whole race face sits on it
COLLAR_EDGE_MM = 0.2  # chamfer on the collar's outer top edge
RIM_INSET_MM = 0.02  # where the rim radius is read, just inside each face plane, so a ledge exactly on the plane does not decide it
RIM_SAMPLES = 41


def stand_half_thickness() -> float:
    """Half the stand's thickness, read from the stand mesh: the final wedge is cut flush with both faces of the stand it sits in."""
    return float(fs.load_stand().extents[1]) / 2


def load_wedge() -> trimesh.Trimesh:
    """The wedge mesh from the 3MF: axle tip towards -y, flat back towards +y, axis through the origin."""
    with zipfile.ZipFile(SOURCE) as z:
        # a re-saved 3MF holds several copies of the name; the stubs are a few hundred bytes, the real mesh is the largest
        entry = max((i for i in z.infolist() if WEDGE_MODEL_PATTERN.fullmatch(i.filename)), key=lambda i: i.file_size)
        log.info("wedge mesh: %s (%d bytes)", entry.filename, entry.file_size)
        root = etree.fromstring(z.read(entry))
    verts = np.array([[float(v.get(k)) for k in "xyz"] for v in root.iter(f"{CORE_NS}vertex")])
    tris = np.array([[int(t.get(k)) for k in ("v1", "v2", "v3")] for t in root.iter(f"{CORE_NS}triangle")])
    mesh = trimesh.Trimesh(verts, tris, process=True)
    return mesh.copy().apply_translation(-np.r_[mesh.bounds.mean(axis=0)[0], 0, mesh.bounds.mean(axis=0)[2]])


def half_profile(mesh: trimesh.Trimesh) -> tuple[Polygon, float]:
    """(section polygon on the +x side as (r, y), y of the nose)."""
    loops = mesh.section(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1]).discrete
    section = Polygon(max(loops, key=len)[:, :2])
    radial = section.intersection(box(0, -100, 100, 100))
    pts = np.array(radial.exterior.coords)
    r_max = pts[:, 0].max()
    nose_y = float(pts[pts[:, 0] > r_max - 0.001, 1].mean())
    return radial, nose_y


def stub_solid(low: float, shoulder_y: float) -> Polygon:
    """The uniform shaft from its chamfered tip at `low` to the shoulder at `shoulder_y`, with the root fillet below the shoulder (section, r and axial y)."""
    r, c, f = STUB_DIAMETER_MM / 2, STUB_CHAMFER_MM, STUB_FILLET_MM
    shaft = Polygon([(0, low), (r - c, low), (r, low + c), (r, shoulder_y + SHOULDER_SCAN_MM), (0, shoulder_y + SHOULDER_SCAN_MM)])
    corner = box(r, shoulder_y - f, r + f, shoulder_y).difference(Point(r + f, shoulder_y - f).buffer(f, quad_segs=64))
    return shaft.union(corner)


def refit_stub(radial: Polygon, nose_y: float) -> tuple[Polygon, float]:
    """The section with the axle stub replaced by a uniform STUB_DIAMETER_MM shaft: chamfered tip, root fillet, shoulder and wedge body untouched.

    Returns the new section and the axial (y) position where the shoulder starts.
    """
    low, high = radial.bounds[1], radial.bounds[3]
    scan = np.arange(low + SHOULDER_SCAN_MM, high, SHOULDER_SCAN_MM)
    reach = np.array([radial.intersection(LineString([(0, a), (radial.bounds[2] + 1, a)])).bounds[2] for a in scan])
    shoulder_y = float(scan[np.argmax(reach > SHOULDER_RADIUS_MM)])
    stub = stub_solid(low, shoulder_y)
    body = radial.intersection(box(0, shoulder_y, 100, high + 1))
    joined = stub.union(body)
    if joined.geom_type != "Polygon":
        raise RuntimeError(f"new stub section is {joined.geom_type}, not one polygon")
    log.info("new stub: %.2f mm shaft, %.2f mm chamfer, %.2f mm fillet, shoulder at %.2f mm from the tip", STUB_DIAMETER_MM, STUB_CHAMFER_MM, STUB_FILLET_MM, shoulder_y - low)
    return joined, shoulder_y


def final_profile(radial: Polygon, nose_y: float) -> tuple[Polygon, float]:
    """The production wedge section: the refitted stub on a disc cut flush with both faces of the stand and flat on the axle side, with a wider collar for the bearing.

    The source wedge is thicker than the stand (3.0 mm behind the nose, 2.72 mm in front, the axle side dished); here the disc runs from nose - h to nose + h (h = half the stand thickness),
    the rim between is the source's own, the axle-side face is flat out to the collar, and the shoulder sits COLLAR_HEIGHT_MM in front of that face. The stub keeps its length from the shoulder to the tip.
    Returns the section and the shoulder's axial y (the tip is towards lower y).
    """
    refit, old_shoulder = refit_stub(radial, nose_y)
    low = radial.bounds[1]
    stub_length = BEARING_WIDTH_MM + STUB_PROTRUSION_MM
    h = stand_half_thickness()
    front_y, back_y = nose_y - h, nose_y + h   # the axle side is towards lower y
    rim_ys = np.linspace(front_y, back_y, RIM_SAMPLES)

    def rim(y: float) -> float:
        inside = min(max(y, front_y + RIM_INSET_MM), back_y - RIM_INSET_MM)
        return float(radial.intersection(LineString([(0, inside), (radial.bounds[2] + 1, inside)])).bounds[2])

    shoulder_y = front_y - COLLAR_HEIGHT_MM
    pts = [(0.0, back_y), *[(rim(float(y)), float(y)) for y in rim_ys[::-1]], (COLLAR_RADIUS_MM, front_y), (COLLAR_RADIUS_MM, shoulder_y + COLLAR_EDGE_MM), (COLLAR_RADIUS_MM - COLLAR_EDGE_MM, shoulder_y), (0.0, shoulder_y)]
    body = Polygon(pts)
    joined = stub_solid(shoulder_y - stub_length, shoulder_y).union(body)
    if joined.geom_type != "Polygon" or not joined.is_valid:
        raise RuntimeError(f"final wedge section is {joined.geom_type}, valid={joined.is_valid}")
    log.info("final wedge section: disc %.3f mm thick (the stand's), flat axle-side face, collar r %.2f x %.2f mm, stub %.2f mm from the shoulder", 2 * h, COLLAR_RADIUS_MM, COLLAR_HEIGHT_MM, stub_length)
    return joined, shoulder_y


def doubled_profile(radial: Polygon, nose_y: float) -> np.ndarray:
    """Open polyline (r, axial) from axle tip to axle tip, axial 0 at the nose, starting and ending on the axis."""
    y_min = radial.bounds[1]
    kept = radial.intersection(box(0, y_min - 1, 100, nose_y))
    outline = np.array(kept.simplify(PROFILE_TOLERANCE_MM).exterior.coords)[:-1]
    on_axis = outline[:, 0] < 1e-6
    first = next(i for i in range(len(outline)) if on_axis[i - 1] and not on_axis[i])
    rolled = np.roll(outline, -first, axis=0)
    chain = rolled[~(rolled[:, 0] < 1e-6)]  # one run: the axis edge was the only on-axis part of the loop
    if chain[0, 1] > chain[-1, 1]:
        chain = chain[::-1]
    half = chain - [0, nose_y]
    if abs(half[-1, 1]) > PROFILE_TOLERANCE_MM or half[-1, 0] < half[:, 0].max() - MAX_DEVIATION_MM:
        raise RuntimeError(f"profile ends at r {half[-1, 0]:.3f}, {half[-1, 1]:.3f} mm from the nose")
    half[-1, 1] = 0.0
    half = np.r_[[[0.0, half[0, 1]]], half]
    return np.r_[half, half[-2::-1] * [1, -1]]


def main() -> None:
    """Build, check against the source wedge, and write the STL with its axis along z."""
    source = load_wedge()
    radial, nose_y = half_profile(source)
    line = doubled_profile(radial, nose_y)
    part = trimesh.creation.revolve(line, sections=SEGMENTS)
    if not part.is_watertight:
        raise RuntimeError("revolved part is not watertight")
    kept = source.vertices[source.vertices[:, 1] <= nose_y]
    samples = kept[np.random.default_rng(0).choice(len(kept), 20000, replace=False)]
    outline = LineString(line)  # part surface in (r, axial); a source vertex's distance to it is its distance to the revolved part
    deviation = max(outline.distance(Point(np.hypot(x, z), y - nose_y)) for x, y, z in samples)
    if deviation > MAX_DEVIATION_MM:
        raise RuntimeError(f"new part is {deviation:.3f} mm from the source wedge")
    part.export(OUTPUT)
    log.info("nose y %.3f, nose diameter %.2f mm, length %.2f mm, %d triangles, volume %.0f mm3, worst deviation from source %.4f mm, wrote %s", nose_y, 2 * line[:, 0].max(), float(np.ptp(part.vertices[:, 2])), len(part.faces), part.volume, deviation, OUTPUT.name)


if __name__ == "__main__":
    main()
