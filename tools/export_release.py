"""Rebuild every model in this repo from the build123d/trimesh source in src/ and copy the results into stl/ and step/ under their release names.

Needs the upstream mesh `SpacePi-X4+OG-PULLEY+NEW-WEDGE.3mf` (see NOTICE.md) in src/; it is not redistributed here.
Usage: python3 tools/export_release.py
Check limit: the build scripts check watertightness, wedge/stand overlap and the 72.60 mm stack length; this tool only runs them and copies files. It cannot tell you the parts fit on your printer.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=os.environ.get("EXPORT_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("export_release")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
UPSTREAM = "SpacePi-X4+OG-PULLEY+NEW-WEDGE.3mf"
BUILD_ORDER = ("build_flat_stand", "build_double_wedge", "build_new_stub", "build_clipon_petal", "build_final_wedge", "build_final_stand", "build_gear_test",
               "build_short_dovetail", "build_design_coupons", "build_full_stack")
RELEASE_NAMES = {   # source file stem -> release file stem
    "Stand Pi X4 - Final": "stand_pi_x4",
    "Wedge - Single Sided - Final": "wedge_single_sided",
    "Adaptive Pulley Half (clip-on core)": "pulley_half_clip_on_core",
    "Adaptive Clip Petal soft ribbed": "clip_on_petal_soft_ribbed",
    "Groove Coupon - 24 Teeth": "coupon_groove_24_teeth",
    "Bearing Recess Design": "coupon_bearing_recess",
    "Foot Hole Design": "coupon_foot_hole",
    "Short Slot 1 (clearance +0.10)": "coupon_short_slot_clearance_0_10",
    "Short Slot 2 (clearance +0.00)": "coupon_short_slot_clearance_0_00",
    "Short Head (nominal)": "coupon_short_head_nominal",
}


def normalise_step(path: Path) -> None:
    """Fixed FILE_NAME timestamp so a rebuild that changes nothing produces identical bytes."""
    text = path.read_text(errors="ignore")
    path.write_text(re.sub(r"FILE_NAME\('([^']*)','[0-9T:\-]+'", r"FILE_NAME('\1','2000-01-01T00:00:00'", text, count=1))


def main() -> None:
    if not (SRC / UPSTREAM).exists():
        sys.exit(f"missing {SRC / UPSTREAM}: download the upstream model (see NOTICE.md) and put it in src/")
    for script in BUILD_ORDER:
        log.info("running %s", script)
        subprocess.run([sys.executable, f"{script}.py"], cwd=SRC, check=True, stdout=subprocess.DEVNULL)
    for out in ("stl", "step"):
        (ROOT / out).mkdir(exist_ok=True)
    for source, release in RELEASE_NAMES.items():
        for ext, out in ((".stl", "stl"), (".step", "step")):
            file = SRC / f"{source}{ext}"
            if file.exists():
                shutil.copyfile(file, ROOT / out / f"{release}{ext}")
                if ext == ".step":
                    normalise_step(ROOT / out / f"{release}{ext}")
            elif ext == ".stl":
                raise SystemExit(f"{file.name} was not produced")
    log.info("wrote %d STL and %d STEP files", len(list((ROOT / "stl").glob("*.stl"))), len(list((ROOT / "step").glob("*.step"))))


if __name__ == "__main__":
    main()
