"""Write the design-size dovetail coupons: the two slot clearances the real slot spans (0.10 mm at the butt, 0.00 mm at the end wall, as constant-clearance 12 mm blocks) and the nominal petal head, for the layer height tolerance plates.

A short section keeps the real slot cross-section, the 1.5 mm closed end wall (bridged roof) and the head profile but drops the taper along the full length. A nominal head in the 0.10 slot has 0.10 mm clearance per side, in the 0.00 slot none: the two ends of the real slot's range.
Dimples on top: 1 on the 0.10 slot, 2 on the 0.00 slot, 2 on the head. Check limit: geometry only (cp.write checks each solid); which pair slides and stays seated is the print's answer. Run build_clipon_petal.py first.
"""
from __future__ import annotations

import logging
import os

import build_clipon_geometry as cg
import build_clipon_petal as cp
import build_slot_dialin as sd

logging.basicConfig(level=os.environ.get("AXEL_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_short_dovetail")

SECTION_MM = 12.0                         # slot block length; a real slot is cg.HALF_WIDTH_MM
CLEARANCES_MM = (cg.SLOT_CLEARANCE_MOUTH_MM, cg.SLOT_CLEARANCE_END_MM)   # the design's two ends, each held constant
NOMINAL_HEAD = sd.HEAD_OFFSETS_MM.index(0.0)
SLOT_NAMES = [f"Short Slot {k + 1} (clearance {c:+.2f})" for k, c in enumerate(CLEARANCES_MM)]
HEAD_NAMES = ["Short Head (nominal)"]
HEAD_LENGTH_MM = SECTION_MM - cg.SLOT_END_WALL_MM - cg.PETAL_END_CLEARANCE_MM   # as the real petal is shorter than its slot by the end wall and the end clearance


def main() -> None:
    for k, (name, c) in enumerate(zip(SLOT_NAMES, CLEARANCES_MM)):
        cp.write(sd.slot_coupon(k, length=SECTION_MM, clearance=(c, c)), name)
    cp.write(sd.petal_stub(NOMINAL_HEAD, length=HEAD_LENGTH_MM), HEAD_NAMES[0])
    log.info("wrote %d short slots (%.0f mm) and the nominal head (%.1f mm)", len(SLOT_NAMES), SECTION_MM, HEAD_LENGTH_MM)


if __name__ == "__main__":
    main()
