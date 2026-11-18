"""Write the refitted wedges (uniform 8.00 mm stub, chamfered tip, filleted root) and check them against the source wedge and the assembly.

Outputs "Wedge - Single Sided - New Stub.stl" and "Wedge - Double Sided - New Stub.stl"; the original wedges and the printed-part model are untouched.
Checks: the printed-body region still matches the source wedge, the shaft diameter is read back from the written meshes, the assembly built
with the new stub has no collisions above the interference tolerance, and its overall width is compared with the old assembly.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import trimesh
from shapely.geometry import LineString, Point, Polygon

import build_assembly as ba
import build_double_wedge as dw

logging.basicConfig(level=os.environ.get("AXEL_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_new_stub")

DIAMETER_PROBES_MM = (1.0, 3.0, 6.0, dw.BEARING_WIDTH_MM + dw.STUB_PROTRUSION_MM - 1.0)  # distances from the tip at which the shaft diameter is read back; the last is a millimetre short of the production stub's root
DIAMETER_TOLERANCE_MM = 0.01  # polygon chord error of a revolve with dw.SEGMENTS sides is far below this
BODY_MARGIN_MM = 0.5  # beyond the fillet, where the new part must still match the source wedge
SAMPLES = 20000


def section_diameter(mesh: trimesh.Trimesh, z: float) -> float:
    """Largest width in x of the mesh cut by the plane at z (the cut loop around the axis)."""
    loop = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    return float(np.ptp(loop.vertices[:, 0]))


def check_shaft(mesh: trimesh.Trimesh, tip_z: float, direction: float, name: str) -> None:
    """Fail unless the shaft reads STUB_DIAMETER_MM at every probe distance from the tip, and the chamfer is there."""
    for d in DIAMETER_PROBES_MM:
        read = section_diameter(mesh, tip_z + direction * d)
        if abs(read - dw.STUB_DIAMETER_MM) > DIAMETER_TOLERANCE_MM:
            raise RuntimeError(f"{name}: diameter {read:.3f} mm at {d} mm from the tip, expected {dw.STUB_DIAMETER_MM}")
    tip_face = section_diameter(mesh, tip_z + direction * 0.01)
    expected = dw.STUB_DIAMETER_MM - 2 * dw.STUB_CHAMFER_MM
    if abs(tip_face - expected) > 0.05:
        raise RuntimeError(f"{name}: tip face diameter {tip_face:.3f} mm, expected about {expected:.3f}")
    log.info("%s: shaft reads %.3f mm at %s mm from the tip, tip face %.3f mm", name, dw.STUB_DIAMETER_MM, DIAMETER_PROBES_MM, tip_face)


def width(parts: list[ba.Part]) -> float:
    """Distance between the outer faces of the outermost stands."""
    stands = [p for p in parts if p.name.startswith("stand")]
    return max(p.mesh.bounds[1, 1] for p in stands) - min(p.mesh.bounds[0, 1] for p in stands)


def main() -> None:
    """Write both wedges, verify them, and compare the assembly with the old one."""
    source = dw.load_wedge()
    radial, nose_y = dw.half_profile(source)
    new_radial, shoulder_y = dw.refit_stub(radial, nose_y)
    line = dw.doubled_profile(new_radial, nose_y)
    double = trimesh.creation.revolve(line, sections=dw.SEGMENTS)
    single = ba.single_wedge(new_radial, nose_y)
    for name, mesh in (("double", double), ("single", single)):
        if not mesh.is_watertight or mesh.volume <= 0:
            raise RuntimeError(f"new {name} wedge is not a closed solid")
    check_shaft(double, double.bounds[0, 2], +1.0, "double wedge, -z side")
    check_shaft(double, double.bounds[1, 2], -1.0, "double wedge, +z side")
    check_shaft(single, single.bounds[0, 2], +1.0, "single wedge")
    body = source.vertices[(source.vertices[:, 1] >= shoulder_y + BODY_MARGIN_MM) & (source.vertices[:, 1] <= nose_y)]
    samples = body[np.random.default_rng(0).choice(len(body), min(SAMPLES, len(body)), replace=False)]
    outline = LineString(line)
    deviation = max(outline.distance(Point(np.hypot(x, z), y - nose_y)) for x, y, z in samples)
    if deviation > dw.MAX_DEVIATION_MM:
        raise RuntimeError(f"wedge body is {deviation:.3f} mm from the source")
    log.info("wedge body beyond the shoulder is within %.4f mm of the source", deviation)
    double.export(dw.NEW_STUB_DOUBLE)
    single.export(dw.NEW_STUB_SINGLE)
    old_parts, _ = ba.build_parts()
    new_parts, _ = ba.build_parts(new_stub=True)
    found = ba.check_collisions(new_parts)
    for a, b, volume in found:
        log.error("COLLISION %s / %s: %.1f mm3", a, b, volume)
    log.info("new assembly: width %.2f mm (old %.2f mm), %d overlapping pair(s) thicker than %.3f mm", width(new_parts), width(old_parts), len(found), ba.INTERFERENCE_DEPTH_MM)
    log.info("wrote %s and %s", dw.NEW_STUB_SINGLE.name, dw.NEW_STUB_DOUBLE.name)


if __name__ == "__main__":
    main()
