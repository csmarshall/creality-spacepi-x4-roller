"""Write "Free Roll - Clip-on Petal Test (fast inner).3mf": two clip-on cores and six whole petals (PRINT_PETALS: three standard tapered 2.4 to 1.4 mm, one each of taper soft, taper firm and t2.0; the t2.6 and t3.0 STLs stay built for later), and bore gauges 50, 55 and 60 mm, one ABS plate at the project's fast inner settings.

The petals print standing (one end on the bed, the section is constant so nothing overhangs) with six walls so the 2 to 2.6 mm arms are solid. Run build_clipon_petal.py first: this reads its STLs.
Check limit: packing and file structure only; it does not slice the plate, so a first look in the slicer is still needed.
"""
from __future__ import annotations

import logging
import os
from typing import cast

import trimesh

import build_adaptive_coupon as ac
import build_clipon_petal as cp
import build_project_3mf as bp
import build_speed_tests as bst
import build_stack_plates as bss

logging.basicConfig(level=os.environ.get("AXEL_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_clipon_plate")

OUTPUT = bp.HERE / "Free Roll - Clip-on Petal Test (fast inner).3mf"
GAUGES_MM = (50.0, 55.0, 60.0)
CORES = 2                  # a full pulley is two identical cores
PRINT_PETALS = {"taper": 3, "taper soft": 1, "taper firm": 1, "t2.0": 1, "soft fillet": 1, "soft ribbed": 1}
ABS_SOFT = ("taper soft", "soft ribbed")   # also printed in ABS (the baseline and the corrugated winner), beside their PETG twins   # petal shape -> how many go on the plate: 3 standard tapered petals make the test pulley, one each of the others to compare by hand


def load(name: str) -> trimesh.Trimesh:
    return cast(trimesh.Trimesh, trimesh.load(bp.HERE / f"{name}.stl"))


PETG_SLOT = 2               # the printer profile's second filament slot is a Snapmaker PETG HF; check its nozzle temperature in Orca before slicing (the profile read from the printer lists unusual values)
PETG_PETALS = ("taper soft", "soft fillet", "soft ribbed")   # the soft family prints in PETG on plate 2; everything else prints in ABS on plate 1


def main() -> None:
    core_name = cp.hs.NAMES["clip"]
    abs_petals = [k for k in PRINT_PETALS if k not in PETG_PETALS or k in ABS_SOFT]
    parts: dict[str, tuple[trimesh.Trimesh, int | None, int, int]] = {core_name: (load(core_name), None, CORES, 1)}   # name -> (mesh, walls, count, extruder)
    for key in abs_petals:
        parts[cp.petal_name(key)] = (load(cp.petal_name(key)), bp.WEDGE_WALLS, PRINT_PETALS[key], 1)
    for key in PETG_PETALS:
        parts[cp.petal_name(key) + " (PETG)"] = (load(cp.petal_name(key)), bp.WEDGE_WALLS, 1, PETG_SLOT)
    for d in GAUGES_MM:
        mesh = ac.gauge_mesh(d)
        mesh.export(bp.HERE / f"Bore Gauge {d:.1f}.stl")
        parts[f"Bore Gauge {d:.1f}"] = (mesh, None, 1, 1)
    kinds = {name: bp.Kind(name, bp.centred(mesh), i + 1, float(mesh.extents[2]), walls, extruder) for i, (name, (mesh, walls, _, extruder)) in enumerate(parts.items())}
    plates = [("ABS parts", [[core_name], [cp.petal_name(k) for k in ABS_SOFT], [cp.petal_name(k) for k in ("taper", "taper firm", "t2.0")], [f"Bore Gauge {d:.1f}" for d in GAUGES_MM]]),
              ("PETG soft petals", [[cp.petal_name(k) + " (PETG)" for k in PETG_PETALS]])]
    instances = []
    oid = len(kinds) + 1
    for plate, (_, rows) in enumerate(plates):
        for k, x, y in bss.layout([[kinds[name] for name in row for _ in range(parts[name][2])] for row in rows]):
            instances.append(bp.Instance(oid, k, plate, x, y))
            oid += 1
    profile = {**bp.profile_values(), **bst.project_fast_values()}
    bp.write_project(OUTPUT, kinds, instances, [f"{name} {profile['layer_height']} mm" for name, _ in plates], profile)
    log.info("wrote %s: %d objects on %d plates (%s)", OUTPUT.name, len(instances), len(plates), "; ".join(f"{name}: {sum(parts[n][2] for row in rows for n in row)}" for name, rows in plates))


def write_plate(parts: list[tuple[str, trimesh.Trimesh, int | None, int]], output, label: str, rows: list[list[str]] | None = None) -> None:
    """Write the 3MF at the project's fast inner settings: parts is (name, mesh, walls override, count); rows lists the part names per row, top to bottom (each name once; its count of copies fills the row)."""
    kinds = {name: bp.Kind(name, bp.centred(mesh), i + 1, float(mesh.extents[2]), walls) for i, (name, mesh, walls, _) in enumerate(parts)}
    counts = {name: count for name, _, _, count in parts}
    if rows is None:   # no row plan: fill rows by name order, wrapping at the usable width
        usable = bp.BED_MM - 2 * ac.EDGE_MM
        grid: list[list[bp.Kind]] = [[]]
        for name, _, _, count in parts:
            for _ in range(count):
                k = kinds[name]
                if grid[-1] and sum(float(i.mesh.extents[0]) + ac.GAP_MM for i in grid[-1]) + float(k.mesh.extents[0]) > usable:
                    grid.append([])
                grid[-1].append(k)
    else:
        grid = [[kinds[name] for name in row for _ in range(counts[name])] for row in rows]
    placed = bss.layout(grid)
    oid = len(kinds) + 1
    instances = [bp.Instance(oid + i, k, 0, px, py) for i, (k, px, py) in enumerate(placed)]
    profile = {**bp.profile_values(), **bst.project_fast_values()}
    bp.write_project(output, kinds, instances, [f"{label} {profile['layer_height']} mm"], profile)
    log.info("wrote %s: %d objects in %d rows", output.name, len(instances), len(grid))


if __name__ == "__main__":
    main()
