"""Write the production single wedge: 24 ridges and the stub diameter chosen from the All Tests print (build_double_wedge.STUB_DIAMETER_MM).

Output "Wedge - Single Sided - Final.stl". Check limit: watertightness and the stub read-back only; fit is whatever the print showed for 24 teeth at 8.05.
"""
from __future__ import annotations

import logging
import os

import build_double_wedge as dw
import build_gear_test as gt

logging.basicConfig(level=os.environ.get("AXEL_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_final_wedge")

TEETH_COUNT = 24


def main() -> None:
    mesh = gt.toothed_wedge(TEETH_COUNT, dw.STUB_DIAMETER_MM, label=False)   # the production wedge carries no engraved text on its back
    out = dw.HERE / "Wedge - Single Sided - Final.stl"
    mesh.export(out)
    log.info("wrote %s: %d teeth, stub %.2f mm, watertight=%s, extents %s", out.name, TEETH_COUNT, dw.STUB_DIAMETER_MM, mesh.is_watertight, mesh.extents.round(2))


if __name__ == "__main__":
    main()
