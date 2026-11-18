"""Write the speed-sweep fit tests: five small Orca projects (A to E), each one recess ladder plus one peg ladder, differing only in print speed.

Print all five with the same filament, nozzle and orientation, then find, for each letter, the recess that takes the 608 bearing by hand and the peg that slides in
the bore. Where those sizes start to drift away from letter A is where speed starts costing accuracy. A is the recess-coupon profile that fit. B keeps A's outer wall
and speeds up everything inside it, to separate inner and infill speed from outer-wall speed. C to E raise the outer-wall speed and acceleration (E is the old project setting).
"""
from __future__ import annotations

import json
import logging
import os
import zipfile
from dataclasses import dataclass

import build_fit_coupon as bfc
import build_project_3mf as bp

logging.basicConfig(level=os.environ.get("AXEL_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_speed_tests")

RECESS_DIAMETERS_MM = [22.00 + 0.05 * i for i in range(6)]  # the 22.00 seated by hand at the slow profile; faster prints should shrink the holes, so the seat moves up this ladder
PEG_DIAMETERS_MM = [7.85 + 0.05 * i for i in range(6)]  # 7.90 to 8.075 slid at the slow profile; faster prints should grow the pegs, so the limit moves down this ladder
INNER_SPEED_KEYS = ("inner_wall_speed", "sparse_infill_speed", "internal_solid_infill_speed", "top_surface_speed", "gap_infill_speed", "default_acceleration")  # taken from the FlatStand project's own (fast) values
GAP_MM = 10.0
SHORT_RECESS_MM = [22.00, 22.05, 22.10]  # the short test: the known-good 22.00 and the two steps above it
SHORT_PEG_MM = [7.95, 8.00, 8.05]  # the new stub size and one step either side
SHORT_PEG_HEIGHT_MM = 8.0  # a bearing is 7 mm wide; the full ladder's pegs are 9 mm


@dataclass
class Level:
    """One speed profile: its letter, a short description for the file name, and the outer-wall speed (mm/s) and acceleration, or None to keep the coupon's."""

    letter: str
    label: str
    fast_inner: bool
    outer_speed: int | None
    outer_accel: int | None


LEVELS = [
    Level("A", "outer 60 slow inner", False, None, None),
    Level("B", "outer 60 fast inner", True, None, None),
    Level("C", "outer 100", True, 100, 3000),
    Level("D", "outer 150", True, 150, 4000),
    Level("E", "outer 200", True, 200, 5000),
]


def project_fast_values() -> dict[str, str]:
    """The FlatStand project's own values for INNER_SPEED_KEYS (first entry of each list)."""
    with zipfile.ZipFile(bp.BASE_3MF) as z:
        settings = json.loads(z.read("Metadata/project_settings.config"))
    return {k: str(settings[k][0] if isinstance(settings[k], list) else settings[k]) for k in INNER_SPEED_KEYS}


def main() -> None:
    """Write one project per speed level."""
    coupon_profile = bp.profile_values()
    fast = project_fast_values()
    for level in LEVELS:
        profile = dict(coupon_profile)
        if level.fast_inner:
            profile.update(fast)
        if level.outer_speed:
            profile["outer_wall_speed"], profile["outer_wall_acceleration"] = str(level.outer_speed), str(level.outer_accel)
        recess = bfc.build_recess_ladder(RECESS_DIAMETERS_MM, level.letter)
        peg = bfc.build_peg_ladder(PEG_DIAMETERS_MM, level.letter, max(RECESS_DIAMETERS_MM))
        bfc.check_diameters(recess, RECESS_DIAMETERS_MM, bfc.layout(len(RECESS_DIAMETERS_MM), max(RECESS_DIAMETERS_MM)).centres, "recess")
        bfc.check_diameters(peg, PEG_DIAMETERS_MM, bfc.layout(len(PEG_DIAMETERS_MM), max(RECESS_DIAMETERS_MM)).centres, "peg")
        kinds = {
            "recess": bp.Kind(f"Speed Test {level.letter} - Recess", bp.centred(recess), 1, float(recess.extents[2]), None),
            "peg": bp.Kind(f"Speed Test {level.letter} - Peg", bp.centred(peg), 2, float(peg.extents[2]), None),
        }
        width = kinds["recess"].mesh.extents[0]
        offset = (width + GAP_MM) / 2
        instances = [bp.Instance(3, kinds["recess"], 0, bp.BED_MM / 2 - offset, bp.BED_MM / 2), bp.Instance(4, kinds["peg"], 0, bp.BED_MM / 2 + offset, bp.BED_MM / 2)]
        output = bp.HERE / f"Speed Test {level.letter} - {level.label}.3mf"
        bp.write_project(output, kinds, instances, [f"Speed {level.letter} 0.12 mm"], profile)
        log.info("%s: outer wall %s mm/s accel %s, inner wall %s, sparse infill %s, default accel %s", level.letter, profile["outer_wall_speed"], profile["outer_wall_acceleration"], profile["inner_wall_speed"], profile["sparse_infill_speed"], profile["default_acceleration"])


def write_short(level: Level) -> None:
    """A three-hole, three-peg version of one level, for a quick check against the known-good results."""
    profile = {**bp.profile_values(), **(project_fast_values() if level.fast_inner else {})}
    recess = bfc.build_recess_ladder(SHORT_RECESS_MM, level.letter)
    peg = bfc.build_peg_ladder(SHORT_PEG_MM, level.letter, max(SHORT_RECESS_MM), SHORT_PEG_HEIGHT_MM)
    bfc.check_diameters(recess, SHORT_RECESS_MM, bfc.layout(len(SHORT_RECESS_MM), max(SHORT_RECESS_MM)).centres, "recess")
    bfc.check_diameters(peg, SHORT_PEG_MM, bfc.layout(len(SHORT_PEG_MM), max(SHORT_RECESS_MM)).centres, "peg")
    kinds = {
        "recess": bp.Kind(f"Speed Test {level.letter} short - Recess", bp.centred(recess), 1, float(recess.extents[2]), None),
        "peg": bp.Kind(f"Speed Test {level.letter} short - Peg", bp.centred(peg), 2, float(peg.extents[2]), None),
    }
    offset = (kinds["recess"].mesh.extents[0] + GAP_MM) / 2
    instances = [bp.Instance(3, kinds["recess"], 0, bp.BED_MM / 2 - offset, bp.BED_MM / 2), bp.Instance(4, kinds["peg"], 0, bp.BED_MM / 2 + offset, bp.BED_MM / 2)]
    bp.write_project(bp.HERE / f"Speed Test {level.letter} short - {level.label}.3mf", kinds, instances, [f"Speed {level.letter} short 0.12 mm"], profile)


def write_first_article(output_name: str, overrides: dict[str, str]) -> None:
    """One OG pulley and one single wedge (new stub) on one plate: the quickest complete fit check of bearing in recess and bearing on stub."""
    kinds = bp.build_kinds()
    instances = [bp.Instance(5, kinds["pulley"], 0, bp.BED_MM / 2 - 40, bp.BED_MM / 2), bp.Instance(6, kinds["single"], 0, bp.BED_MM / 2 + 40, bp.BED_MM / 2)]
    kept = {k: v for k, v in kinds.items() if k in ("pulley", "single")}
    bp.write_project(bp.HERE / output_name, kept, instances, ["First article 0.12 mm"], {**bp.profile_values(), **overrides})


def write_full_project_variant() -> None:
    """The three-plate pulley and wedge project with level B's profile, so the slicer's own time estimate can be compared with the all-slow project before printing anything."""
    bp.main(bp.HERE / "Free Roll - Fit Profile B (fast inner).3mf", project_fast_values())


def write_all_pulleys_project() -> None:
    """Fast-profile project with all 16 pulleys (8 OG, 8 concave, 54 mm across) in a 4 x 4 grid on one plate and the wedges (6 single, 2 double) on a second."""
    kinds = bp.build_kinds()
    instances = bp.placements(kinds, [(0, [("pulley", bp.PULLEYS), ("concave", bp.CONCAVES)], 4), (1, [("single", bp.SINGLES), ("double", bp.DOUBLES)], 3)])
    profile = {**bp.profile_values(), **project_fast_values()}
    names = [f"All Pulleys {profile['layer_height']} mm", f"Wedges {profile['layer_height']} mm"]
    bp.write_project(bp.HERE / "Free Roll - All Pulleys B (fast inner).3mf", kinds, instances, names, profile)


if __name__ == "__main__":
    main()
    write_all_pulleys_project()
    write_full_project_variant()
    write_short(next(level for level in LEVELS if level.letter == "B"))
    write_first_article("Free Roll - First Article B (fast inner).3mf", project_fast_values())
    write_first_article("Free Roll - First Article A (slow).3mf", {})
