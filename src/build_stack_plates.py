"""Write "Free Roll - Whole Stack (fast inner).3mf": every printed part of both rollers, packed onto as many plates as it takes, at the project's fast inner settings.

Parts (two rollers, the paired layout of build_full_stack): 4 final stands (lying flat), 4 final 24-ridge wedges, 4 clip-on cores and 12 t2.0 petals (3 per core). Bought parts are not in it:
the 608 bearings. Wedges and petals carry the 6-wall override. Run build_clipon_petal.py, build_final_wedge.py and build_final_stand.py first.
Check limit: bp.write_project verifies the placements (closed, on the bed, footprints at least 5 mm apart); packing is first-fit by footprint, so the plate split is a convenience, and nothing here
slices the plates or estimates print time.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import cast

import trimesh

import build_adaptive_coupon as ac
import build_clipon_petal as cp
import build_full_stack as fstack
import build_project_3mf as bp
import build_speed_tests as bst

logging.basicConfig(level=os.environ.get("AXEL_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_stack_plates")

LAYER_HEIGHT_MM = float(sys.argv[1]) if len(sys.argv) > 1 else None   # command line: `python3 build_stack_plates.py 0.20` writes the whole stack at that layer height; none keeps the project default (0.12)
OUTPUT = bp.HERE / ("Free Roll - Whole Stack (fast inner).3mf" if LAYER_HEIGHT_MM is None else f"Free Roll - Whole Stack {LAYER_HEIGHT_MM:.2f} (fast inner).3mf")
ROLLERS = 2
PETAL = fstack.PETAL


def parts(rollers: int = ROLLERS) -> list[tuple[str, trimesh.Trimesh, int | None, int]]:
    """(name, mesh in print orientation, wall override, count) for `rollers` rollers."""
    stand = cast(trimesh.Trimesh, trimesh.load(fstack.FINAL_STAND)).apply_transform(trimesh.transformations.rotation_matrix(3.141592653589793 / 2, [1, 0, 0]))   # thickness up: lies flat
    core_name = cp.hs.NAMES["clip"]
    return [
        ("Stand Pi X4 - Final", stand, None, 2 * rollers),
        ("Wedge - Single Sided - Final", fstack.load("Wedge - Single Sided - Final"), bp.WEDGE_WALLS, 2 * rollers),
        (core_name, fstack.load(core_name), None, 2 * rollers),
        (cp.petal_name(PETAL), fstack.load(cp.petal_name(PETAL)), bp.WEDGE_WALLS, 3 * rollers),   # three whole petals per roller
    ]


def layout(rows: list[list[bp.Kind]], gap: float = ac.GAP_MM) -> list[tuple[bp.Kind, float, float]]:
    """Place rows of kinds on one plate: rows run along x from the left edge, top to bottom in the order given, `gap` between footprints, the whole block centred on the bed."""
    usable = bp.BED_MM - 2 * ac.EDGE_MM
    depths = [max(float(k.mesh.extents[1]) for k in row) for row in rows]
    widths = [sum(float(k.mesh.extents[0]) for k in row) + gap * (len(row) - 1) for row in rows]
    block_w, block_d = max(widths), sum(depths) + gap * (len(rows) - 1)
    if block_w > usable or block_d > usable:
        raise RuntimeError(f"the rows do not fit one plate: {block_w:.0f} x {block_d:.0f} mm in {usable:.0f}")
    placed: list[tuple[bp.Kind, float, float]] = []
    y = (bp.BED_MM - block_d) / 2
    for row, depth, width in zip(rows, depths, widths):
        x = (bp.BED_MM - block_w) / 2
        for k in row:
            w = float(k.mesh.extents[0])
            placed.append((k, x + w / 2, y + depth / 2))
            x += w + gap
        y += depth + gap
    return placed


def main() -> None:
    plan = parts()
    kinds = {name: bp.Kind(name, bp.centred(mesh), i + 1, float(mesh.extents[2]), walls) for i, (name, mesh, walls, _) in enumerate(plan)}
    stand, wedge, core, petal = (kinds[name] for name, _, _, _ in plan)
    rows = [[stand, core, wedge], [stand, core, wedge], [petal, petal, petal]]   # one roller per plate: its two stands (each with its core and wedge beside it), then its three petals
    oid = len(kinds) + 1
    instances = []
    for p in range(ROLLERS):
        for k, x, y in layout(rows):
            instances.append(bp.Instance(oid, k, p, x, y))
            oid += 1
    log.info("each plate: 2 x %s, 2 x %s, 2 x %s, 3 x %s", stand.name, core.name, wedge.name, petal.name)
    profile = {**bp.profile_values(), **bst.project_fast_values()}
    if LAYER_HEIGHT_MM is not None:
        profile["layer_height"] = f"{LAYER_HEIGHT_MM:.2f}"
    bp.write_project(OUTPUT, kinds, instances, [f"Roller {i + 1}" for i in range(ROLLERS)], profile)
    log.info("wrote %s: %d objects on %d plates", OUTPUT.name, len(instances), ROLLERS)


if __name__ == "__main__":
    main()
