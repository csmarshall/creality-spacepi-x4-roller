"""Write "Free Roll - Parts By Group 0.12 (fast inner).3mf": every printed part of the rollers at 0.12 mm, one kind of part per plate so a batch can be printed, checked and reprinted on its own.

Packing: as many parts per plate as the bed allows, up to a per-kind cap that keeps a plate to roughly 4 to 7 h (MAX_PER_PLATE; the batches of a kind are split evenly). Stands lie flat (138 x 92 mm, 8 mm brim): two in rows plus a third turned 90 degrees beside them, three per plate.
Gaps between footprints are twice the brim plus 1 mm, and a part is never closer to the bed edge than its brim plus 2 mm. Walls, brims and shells come from the project writer (6-wall wedges and petals, 4 mm brim on cores and petals).
Check limit: bp.write_project verifies closed meshes, on the bed and spacing; per-plate times are slicer estimates made separately. Run build_clipon_petal.py, build_final_wedge.py and build_final_stand.py first. `python3 build_stack_by_part.py [rollers]`.
"""
from __future__ import annotations

import logging
import math
import os
import sys

import trimesh

import build_adaptive_coupon as ac
import build_project_3mf as bp
import build_speed_tests as bst
import build_stack_plates as bss

logging.basicConfig(level=os.environ.get("AXEL_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_stack_by_part")

LAYER_HEIGHT_MM = 0.12
ROLLERS = int(sys.argv[1]) if len(sys.argv) > 1 else bss.ROLLERS
OUTPUT = bp.HERE / (f"Free Roll - Parts By Group {LAYER_HEIGHT_MM:.2f} (fast inner).3mf" if ROLLERS == bss.ROLLERS else f"Free Roll - Parts By Group {LAYER_HEIGHT_MM:.2f} {ROLLERS} rollers (fast inner).3mf")
MAX_PER_PLATE = {"Stand": 3, "Adaptive Pulley": 4, "Wedge": 8, "Adaptive Clip": 6}   # stands: all the bed holds; the rest: a plate of about 2 to 6 h (a failed plate costs one batch)
GROUP_LABELS = {"Stand": "Stands", "Adaptive Pulley": "Cores", "Wedge": "Wedges", "Adaptive Clip": "Petals"}
BED_MARGIN_MM = 2.0   # beyond the brim


def brim_mm(name: str) -> float:
    if name.startswith(bp.STAND_NAME_PREFIX):
        return bp.STAND_BRIM_WIDTH_MM
    if name.startswith(bp.PETAL_NAME_PREFIX):
        return bp.PETAL_BRIM_WIDTH_MM
    return bp.BRIM_WIDTH_MM if name.startswith(bp.BRIM_NAME_PREFIXES) else 0.0


def rows_for(items: list[bp.Kind], gap: float, usable: float) -> list[list[bp.Kind]]:
    """Fill rows left to right inside the usable bed width."""
    rows: list[list[bp.Kind]] = [[]]
    width = 0.0
    for k in items:
        w = float(k.mesh.extents[0])
        if rows[-1] and width + gap + w > usable:
            rows.append([])
            width = 0.0
        width += (gap if rows[-1] else 0.0) + w
        rows[-1].append(k)
    return rows


def stand_plate(stand: bp.Kind, turned: bp.Kind, count: int, gap: float) -> list[tuple[bp.Kind, float, float]]:
    """Two stands in rows, a third turned 90 degrees beside them; the block is centred on the bed."""
    w, d = float(stand.mesh.extents[0]), float(stand.mesh.extents[1])
    flat = min(count, 2)
    cols_w = w + (gap + d if count > 2 else 0.0)
    block_d = max(flat * d + (flat - 1) * gap, w if count > 2 else 0.0)
    x0, y0 = (bp.BED_MM - cols_w) / 2, (bp.BED_MM - block_d) / 2
    placed = [(stand, x0 + w / 2, y0 + d / 2 + i * (d + gap)) for i in range(flat)]
    if count > 2:
        placed.append((turned, x0 + w + gap + d / 2, y0 + block_d / 2))
    return placed


def main() -> None:
    plan = bss.parts(ROLLERS)
    kinds = {name: bp.Kind(name, bp.centred(mesh), i + 1, float(mesh.extents[2]), walls) for i, (name, mesh, walls, _) in enumerate(plan)}
    stand_name = next(n for n, _, _, _ in plan if n.startswith(bp.STAND_NAME_PREFIX))
    stand_mesh = next(m for n, m, _, _ in plan if n == stand_name)
    turned_mesh = stand_mesh.copy().apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 0, 1]))
    turned = bp.Kind(f"{stand_name} (turned)", bp.centred(turned_mesh), len(kinds) + 1, kinds[stand_name].height, None)
    kinds[turned.name] = turned
    plates: list[tuple[str, list[bp.Kind]]] = []
    for name, _, _, count in plan:
        prefix = next(p for p in MAX_PER_PLATE if name.startswith(p))
        n_plates = math.ceil(count / MAX_PER_PLATE[prefix])
        sizes = [count // n_plates + (1 if i < count % n_plates else 0) for i in range(n_plates)]   # split evenly
        for n, size in enumerate(sizes, start=1):
            plates.append((f"{GROUP_LABELS[prefix]} {n} of {n_plates} ({size})", [kinds[name]] * size))
    oid = len(kinds) + 1
    instances = []
    for plate, (_, items) in enumerate(plates):
        brim = brim_mm(items[0].name)
        gap = max(2 * brim + 1.0, ac.GAP_MM)
        if items[0].name.startswith(bp.STAND_NAME_PREFIX):
            placed = stand_plate(items[0], turned, len(items), gap)
        else:
            placed = bss.layout(rows_for(items, gap, bp.BED_MM - 2 * (brim + BED_MARGIN_MM)), gap)
        for k, x, y in placed:
            if min(x - k.mesh.extents[0] / 2, y - k.mesh.extents[1] / 2, bp.BED_MM - x - k.mesh.extents[0] / 2, bp.BED_MM - y - k.mesh.extents[1] / 2) < brim + BED_MARGIN_MM - 1e-6:
                raise RuntimeError(f"plate {plate + 1}: {k.name} is closer to the bed edge than its brim")
            instances.append(bp.Instance(oid, k, plate, x, y))
            oid += 1
    profile = {**bp.profile_values(), **bst.project_fast_values(), "layer_height": f"{LAYER_HEIGHT_MM:.2f}"}
    bp.write_project(OUTPUT, kinds, instances, [label for label, _ in plates], profile)
    log.info("wrote %s: %d objects on %d plates (%s)", OUTPUT.name, len(instances), len(plates), "; ".join(label for label, _ in plates))


if __name__ == "__main__":
    main()
