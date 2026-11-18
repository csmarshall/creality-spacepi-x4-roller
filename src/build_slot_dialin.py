"""Write "Free Roll - Slot Fit Coupons (fast inner).3mf": the dovetail slot and the petal head as short, cheap coupons instead of a whole core, to find which slot / head pair slides in by hand and stays seated.

Slot coupons: a 24 x 10 mm block, full slot length (HALF_WIDTH_MM), the real tapered slot cross-section and the 1 mm end wall, each with its own clearance pair (per side, at the butt face and at the end wall; negative = press fit); 1 to 6 dimples on the top face say which.
Petal stubs: the real tapered dovetail head with a flat tab above it to hold, full petal length, three head sizes (offset per side from the nominal head), 2 of each; 1 to 3 dimples on the top end face. Both print standing, butt end on the bed, as the real parts do.
Check limit: geometry and file packing only; which pair fits is the print's answer, and the bed effect (elephant foot at the butt end) is in the print, not here.
"""
from __future__ import annotations

import logging
import os

import trimesh
from build123d import BuildLine, BuildPart, BuildSketch, Cylinder, Locations, Mode, Plane, Polyline, Rectangle, extrude, loft, make_face

import build_clipon_geometry as cg
import build_clipon_petal as cp
import build_clipon_plate as plate
import build_project_3mf as bp

logging.basicConfig(level=os.environ.get("AXEL_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_slot_dialin")

OUTPUT = bp.HERE / "Free Roll - Slot Fit Coupons (fast inner).3mf"
SLOT_CLEARANCES_MM = ((0.10, 0.00), (0.05, 0.00), (0.05, -0.05), (0.05, -0.10), (0.00, -0.05), (0.00, -0.10))   # (butt face, end wall) per side
HEAD_OFFSETS_MM = (-0.05, 0.0, 0.05)   # head size per side from the nominal head
STUB_COPIES = 2
BLOCK_W_MM, BLOCK_T_MM = 24.0, 10.0
TAB_W_MM, TAB_H_MM = 5.0, 4.0          # holding tab above the head
MARK_SPACING_MM = 2.4
SLOT = 0                               # slot 0 is centred at 90 degrees: radial direction +y


def exterior(poly) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in list(poly.exterior.coords)[:-1]]


def slot_coupon(k: int, length: float = cg.HALF_WIDTH_MM, clearance: tuple[float, float] | None = None):
    """Block whose top (y = core radius) carries slot k; the slot opens at z = 0 and stops the end wall short of the far face. `length` and `clearance` (butt, end) override the full-length coupon's."""
    mouth, end = clearance or SLOT_CLEARANCES_MM[k]
    with BuildPart() as c:
        with BuildSketch(Plane.XY):
            with Locations((0, cg.CLIP.r_core - BLOCK_T_MM / 2)):
                Rectangle(BLOCK_W_MM, BLOCK_T_MM)
        extrude(amount=length)
        for z in (-1.0, length - cg.SLOT_END_WALL_MM):
            with BuildSketch(Plane.XY.offset(z)):
                with BuildLine():
                    Polyline(*exterior(cg.slot_polygon(SLOT, z, mouth=mouth, end=end)), close=True)
                make_face()
        loft(mode=Mode.SUBTRACT)
        for j in range(k + 1):
            with Locations((cg.CLIP.r_core * 0 + (j - k / 2.0) * MARK_SPACING_MM, cg.CLIP.r_core - BLOCK_T_MM / 2, length)):
                Cylinder(cg.MARK_RADIUS_MM, 2 * cg.MARK_DEPTH_MM, mode=Mode.SUBTRACT)
    return c.part


def petal_stub(k: int, length: float = cp.PETAL_LENGTH_MM):
    """Tapered dovetail head of offset HEAD_OFFSETS_MM[k] with a holding tab, `length` long (the full petal length by default); k + 1 dimples on the top end."""
    r0 = cg.CLIP.r_core
    with BuildPart() as p:
        with BuildSketch(Plane.XY):
            with Locations((0, r0 + cg.HEAD_OVERLAP_MM + TAB_H_MM / 2 - 0.1)):
                Rectangle(TAB_W_MM, TAB_H_MM + 0.2)
        extrude(amount=length)
        for z in (0.0, length):
            with BuildSketch(Plane.XY.offset(z)):
                with BuildLine():
                    Polyline(*exterior(cg.head_polygon(SLOT, z, extra=HEAD_OFFSETS_MM[k])), close=True)
                make_face()
        loft()
        for j in range(k + 1):
            with Locations(((j - k / 2.0) * 1.8, r0 + cg.HEAD_OVERLAP_MM + TAB_H_MM / 2, length)):
                Cylinder(0.5, 2 * cg.MARK_DEPTH_MM, mode=Mode.SUBTRACT)
    return p.part


def main() -> None:
    parts: list[tuple[str, trimesh.Trimesh, int | None, int]] = []
    for k, (mouth, end) in enumerate(SLOT_CLEARANCES_MM):
        name = f"Slot Coupon {k + 1} (butt {mouth:+.2f}, end {end:+.2f})"
        parts.append((name, cp.write(slot_coupon(k), name), None, 1))
    for k, off in enumerate(HEAD_OFFSETS_MM):
        name = f"Head Stub {k + 1} (head {off:+.2f})"
        parts.append((name, cp.write(petal_stub(k), name), bp.WEDGE_WALLS, STUB_COPIES))
    plate.write_plate(parts, OUTPUT, "Slot fit")
    for ks, (mouth, end) in enumerate(SLOT_CLEARANCES_MM):
        log.info("slot %d: %s", ks + 1, ", ".join(f"stub {kp + 1} {mouth - off:+.2f} butt / {end - off:+.2f} end" for kp, off in enumerate(HEAD_OFFSETS_MM)))


if __name__ == "__main__":
    main()
