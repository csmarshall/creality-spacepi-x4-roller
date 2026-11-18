"""Write the production stand: the flat both-sides stand with 4.05 mm foot holes (printed result) and the 24 ridge pockets cut in its groove (printed result: 24 ridges mate best).

Output "Stand Pi X4 - Final.stl". The pockets come from build_gear_test.pocket_cutter, the same cutter as the printed groove coupon, so the stand and the coupon cannot drift apart.
Check limit: the final wedge is seated in the written stand at three sag positions (no overlap) and turned half a pitch (overlap, the lock); that is the same clearance model the coupon used, so it proves
the geometry matches the coupon, not that ABS prints it (the coupon print is the evidence for that).
"""
from __future__ import annotations

import logging
import os

import trimesh

import build_flat_stand as fs
import build_gear_test as gt

logging.basicConfig(level=os.environ.get("AXEL_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_final_stand")

FOOT_HOLE_MM = 4.05
CRADLE_RAISE_MM = 2.5   # the cradle (groove and the neck and horns round it) sits this much higher on the same feet and lattice; the user asked for 5, cut to 2.5 because the lid height is unknown: with 2.5 the spool clears the 12 mm OEM spindle bodies by 2.9 mm and its top is 3.7 mm above a stock 200 mm spool's
TEETH_COUNT = 24
OUTPUT = fs.HERE / "Stand Pi X4 - Final.stl"
MAX_OVERLAP_MM3 = 1.0


def build() -> tuple[trimesh.Trimesh, float, float]:
    """The final stand mesh and the groove centre (x, z), the cradle raised CRADLE_RAISE_MM and the upper pockets opened into lift-out channels."""
    source = fs.load_stand()
    plate = fs.build(source, FOOT_HOLE_MM, CRADLE_RAISE_MM)
    fs.check_foot_hole(plate, FOOT_HOLE_MM)
    cx, cz, _ = fs.groove_circle(fs.flat_face_outline(source))   # the source groove centre: the pocket cutter probes the source stand there
    stand = trimesh.boolean.difference([plate, gt.pocket_cutter(TEETH_COUNT, source, cx, cz, lift=CRADLE_RAISE_MM, channels=True)], engine="manifold")
    if not stand.is_watertight:
        raise RuntimeError("the final stand is not watertight")
    return stand, float(cx), float(cz) + CRADLE_RAISE_MM


def main() -> None:
    stand, cx, cz = build()
    stand.export(OUTPUT)
    log.info("wrote %s: watertight, volume %.0f mm3, extents %s", OUTPUT.name, stand.volume, stand.extents.round(2))
    wedge = trimesh.load(fs.HERE / "Wedge - Single Sided - Final.stl")
    for sag in (0.0, 0.2, 0.3):   # seated at the nearest ridge, however far it sags in the clearance: no overlap
        volume = gt.overlap(stand, wedge, cx, cz, sag, 0.0)
        log.info("final wedge in final stand, sag %.1f: overlap %.3f mm3", sag, volume)
        if volume > MAX_OVERLAP_MM3:
            raise RuntimeError(f"the final wedge overlaps the final stand by {volume:.1f} mm3 at sag {sag}")
    half_pitch = 0.5 * 360.0 / TEETH_COUNT   # half a pitch round, ridges against pocket walls: it must overlap, which is the lock holding
    locked = gt.overlap(stand, wedge, cx, cz, 0.0, half_pitch)
    log.info("turned half a pitch (%.1f deg) the ridges hit the pocket walls: overlap %.1f mm3 (must be above zero, the lock)", half_pitch, locked)
    if locked < MAX_OVERLAP_MM3:
        raise RuntimeError("the wedge turns freely: the pockets do not lock it")

if __name__ == "__main__":
    main()
