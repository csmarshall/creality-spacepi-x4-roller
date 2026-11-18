"""Write the design-size coupons for the layer height tolerance plates: a bearing recess ring (the core's recess exactly: 22.0 mm bore, 7.0 mm deep shelf, 0.5 mm mouth chamfer, 18.4 mm bore below) and a foot hole plate (the stand's thickness, the final 4.05 mm hole).

Together with the final wedge (8.05 mm stub), the 24-ridge groove coupon and the short dovetail blocks they hold every fit dimension of the current 0.12 mm design once, so printing them at another layer height shows whether the printed sizes and fits repeat.
Check limit: geometry only (cp.write checks each solid); the coupons' sizes come from the same constants as the real parts, so they prove the coupon matches the design, not that it prints to size. Run build_clipon_petal.py first.
"""
from __future__ import annotations

import logging
import os

from build123d import Axis, BuildLine, BuildPart, BuildSketch, Box, Cylinder, Locations, Mode, Plane, Polyline, make_face, revolve

import build_adaptive_half_step as hs
import build_assembly as ba
import build_clipon_geometry as cg
import build_clipon_petal as cp
import build_double_wedge as dw
import build_final_stand as bs

logging.basicConfig(level=os.environ.get("AXEL_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_design_coupons")

RECESS_NAME = "Bearing Recess Design"
FOOT_NAME = "Foot Hole Design"
RING_OD_MM = 30.0
RING_HEIGHT_MM = ba.BEARING_WIDTH_MM + cg.BEARING_FLOAT_MM + 3.0   # the recess depth (shelf) plus 3 mm of ring under the shelf
FOOT_PLATE_MM = (20.0, 16.0)


def recess_ring():
    """The core's bearing recess in a short ring, recess up as in the core: shelf at one bearing width down, chamfered mouth, the axle bore through."""
    recess_r = hs.RECESS_R_MM + cg.BEARING_RECESS_CLEARANCE_MM
    h = RING_HEIGHT_MM
    shelf = h - (ba.BEARING_WIDTH_MM + cg.BEARING_FLOAT_MM)
    pts = [(hs.BORE_R_MM, 0.0), (RING_OD_MM / 2, 0.0), (RING_OD_MM / 2, h), (recess_r + cg.BEARING_MOUTH_CHAMFER_MM, h), (recess_r, h - cg.BEARING_MOUTH_CHAMFER_MM), (recess_r, shelf), (hs.BORE_R_MM, shelf)]
    with BuildPart() as ring:
        with BuildSketch(Plane.XZ):
            with BuildLine():
                Polyline(*pts, close=True)
            make_face()
        revolve(axis=Axis.Z)
    return ring.part


def foot_plate():
    """A plate of the stand's thickness with the final foot hole."""
    thickness = 2 * dw.stand_half_thickness()
    with BuildPart() as plate:
        Box(FOOT_PLATE_MM[0], FOOT_PLATE_MM[1], thickness)
        with Locations((0, 0, 0)):
            Cylinder(bs.FOOT_HOLE_MM / 2, thickness + 2, mode=Mode.SUBTRACT)
    return plate.part


def main() -> None:
    cp.write(recess_ring(), RECESS_NAME)
    cp.write(foot_plate(), FOOT_NAME)


if __name__ == "__main__":
    main()
