"""Write the gear-lock test: four wedges whose rim carries radial ridges (8, 12, 18 and 24 of them) and four matching stand-groove coupons with pockets, as STLs and one project.

The wedge's rim is a rounded lens at radius 19.73 mm and the stand's groove is the same curve plus about 0.45 mm of clearance, so ridges on the wedge rim that drop into pockets cut in the
groove's lower arc stop the wedge rolling, and because the ridges are evenly spaced the wedge seats at whichever one is nearest. Everything is built in the wedge's frame (axis z, nose plane z = 0, which is the
stand's mid-plane) and moved into the stand's frame with the same transform the assembly uses (local z to stand y, local y to stand -z, centre at the groove circle).
Ridge: a trapezoid in plan (RIDGE_BASE to RIDGE_TIP radius and width), the count's ridge height (TEETH) proud of the rim along the rim's own profile. Pocket: the ridge outline grown by POCKET_SIDE_MM, cut to
the groove profile plus the ridge height plus POCKET_RADIAL_EXTRA_MM, over the lower arc only (ARC_HALF_DEG either side of straight down).
Check limit: the wedge-in-coupon overlap is computed on the written meshes at three sag positions and at two small turns, which shows the clearance is in the model; whether ABS prints it freely is what the coupons test.
"""
from __future__ import annotations

import logging
import os
from typing import cast

import numpy as np
import trimesh
from shapely.affinity import rotate as shapely_rotate
from shapely.affinity import translate as shapely_translate
from shapely.geometry import Polygon
from shapely.ops import unary_union

import build_double_wedge as dw
import build_flat_stand as fs
import build_project_3mf as bp
import build_speed_tests as bst
import build_stub_ladder as bsl
import label_text as lt

logging.basicConfig(level=os.environ.get("AXEL_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_gear_test")

HERE = bp.HERE
OUTPUT = HERE / "Free Roll - Gear Lock Test (fast inner).3mf"
TEETH = {8: (2.0, 1.0), 12: (1.0, 0.8), 18: (1.0, 0.8), 24: (1.0, 0.8)}  # count: (width scale on the ridge outline, ridge height mm); 8 is the coarse variant
TEETH_COUNTS = tuple(TEETH)
WEDGE_STUB_MM = 8.15  # mid-ladder; the stub does not matter for this test
RIDGE_BASE = (15.0, 3.5)  # (radius, width) of the ridge outline's inner end, inside the rim
RIDGE_TIP = (21.0, 1.0)  # and its outer end; the taper is the lead-in
POCKET_SIDE_MM = 0.3
CHANNEL_FROM_DEG = 30.0  # pockets this far or further round from straight down become lift-out channels: a ridge 20 degrees or more off the vertical binds when the wedge is pulled straight up (tan 20 x 0.8 mm ridge height is already the 0.3 mm side clearance)
CHANNEL_LENGTH_MM = 8.0  # how far up the channels run, past the top of the cradle
CHANNEL_STEP_MM = 0.25  # the channel is the union of the pocket moved up in steps this size
POCKET_RADIAL_EXTRA_MM = 0.1  # beyond the groove plus ridge height, so a wedge sagged by the full 0.45 mm of clearance still seats
ARC_HALF_DEG = 95.0  # reaches the ridges that sit level with the groove centre (90 degrees round), which the coarse 8-tooth wedge puts into the groove wall otherwise
Z_BACK_MM, Z_TOP_MM = -dw.stand_half_thickness(), dw.stand_half_thickness()  # ridge extent along the wedge axis: the final wedge's back flat (the stand's outer face plane) to the stand's inner face plane
COUPON_BELOW_MM, COUPON_ABOVE_MM = 33.0, 1.0  # coupon extent from the groove centre, down and up (stand z)
LABEL_SIZE_MM = 4.0
LABEL_DEPTH_MM = 0.36
SAG_STEPS_MM = (0.0, 0.2, 0.3)  # the groove clears the plain rim by about 0.45 mm; 0.4 already touches it with no teeth involved


def rim_radius(wedge: trimesh.Trimesh, z: float) -> float:
    """The wedge's outer radius at height z (print orientation, nose at z = 0)."""
    section = wedge.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    return float(max(np.hypot(loop[:, 0], loop[:, 1]).max() for loop in section.discrete))  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType, reportUnknownArgumentType]


def revolved(points: list[tuple[float, float]]) -> trimesh.Trimesh:
    """A body of revolution about z from (r, z) points on its outline, closed to the axis at both ends."""
    return trimesh.creation.revolve(np.array([(0.0, points[0][1]), *points, (0.0, points[-1][1])]), sections=fs.CYLINDER_SECTIONS)


def ridge_outline(angle_deg: float, grow: float, width_scale: float) -> Polygon:
    """The ridge's plan outline pointing along `angle_deg`, grown by `grow` all round."""
    k = width_scale
    shape = Polygon([(RIDGE_BASE[0], -k * RIDGE_BASE[1] / 2), (RIDGE_TIP[0], -k * RIDGE_TIP[1] / 2), (RIDGE_TIP[0], k * RIDGE_TIP[1] / 2), (RIDGE_BASE[0], k * RIDGE_BASE[1] / 2)])
    return shapely_rotate(shape.buffer(grow, join_style=2), angle_deg, origin=(0, 0))


def extruded(shape, z0: float, z1: float) -> list[trimesh.Trimesh]:
    """Prisms of the plan shape (a Polygon or MultiPolygon) from z0 to z1."""
    return [trimesh.creation.extrude_polygon(p, z1 - z0).apply_translation([0, 0, z0]) for p in getattr(shape, "geoms", [shape])]


def angles(count: int, lower_arc_only: bool) -> list[float]:
    """Ridge angles in the wedge frame, one at 90 degrees (straight down in the stand), evenly spaced; the lower arc only within ARC_HALF_DEG of it when asked."""
    pitch = 360.0 / count
    every = [90.0 + k * pitch for k in range(count)]
    return [a for a in every if abs(((a - 90.0 + 180.0) % 360.0) - 180.0) <= ARC_HALF_DEG] if lower_arc_only else every


def toothed_wedge(count: int, stub: float = WEDGE_STUB_MM, label: bool = True) -> trimesh.Trimesh:
    """The single wedge (print orientation) with a stub of the given diameter and `count` ridges on its rim, labelled "T<count> <stub>" and the ridge size on the back."""
    scale, height = TEETH[count]
    wedge = bsl.wedge_with_stub(stub, final=True)
    zs = np.linspace(Z_BACK_MM + 0.02, Z_TOP_MM, 41)
    envelope = revolved([(rim_radius(wedge, float(z)) + height, float(z)) for z in zs])
    prisms = [m for a in angles(count, False) for m in extruded(ridge_outline(a, 0.0, scale), Z_BACK_MM, Z_TOP_MM)]
    ridges = trimesh.boolean.intersection([trimesh.boolean.union(prisms, engine="manifold"), envelope], engine="manifold")
    out = trimesh.boolean.union([wedge, ridges], engine="manifold")
    if label:
        out = bsl.deboss_lines(out, [(f"T{count} {stub:.2f}", 5.0), (f"h{height:.1f} w{scale:.1f}", 3.4)])  # teeth, stub; ridge height and width scale
    out.vertices = out.vertices.astype(np.float32).astype(np.float64)   # an STL stores float32: snap first, or vertices that coincide in double precision split on export and the reloaded file is no longer a closed solid (the boolean in the label step did this implicitly)
    out.merge_vertices()
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    if not out.is_volume:
        raise RuntimeError(f"toothed wedge {count} is not a closed solid")
    return out


def stand_frame() -> trimesh.transformations:
    """Wedge frame to stand frame: local z to stand y, local y to stand -z, centred on the groove circle."""
    return trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])


def pocket_cutter(count: int, stand: trimesh.Trimesh, cx: float, cz: float, lift: float = 0.0, channels: bool = False) -> trimesh.Trimesh:
    """The ridge pockets for `count` ridges in the stand frame, ready to subtract from the stand plate (or a coupon of it).

    `stand` and `cz` are the source stand and its groove centre (the groove edge is probed there); `lift` moves the finished cutter up by that much, for a stand whose cradle was raised.
    With `channels`, every pocket more than CHANNEL_FROM_DEG round from straight down is swept straight up CHANNEL_LENGTH_MM, so the wedge can be pulled straight up out of the cradle without those ridges catching."""
    scale, height = TEETH[count]
    notch_y, notch_r = fs.groove_notch_profile(stand, cz)
    half = (stand.bounds[1, 1] - stand.bounds[0, 1]) / 2
    mid = float((stand.bounds[0, 1] + stand.bounds[1, 1]) / 2)
    pad = 1.0
    profile = [(float(notch_r[0] + height + POCKET_RADIAL_EXTRA_MM), -half - pad), *[(float(r + height + POCKET_RADIAL_EXTRA_MM), float(y - mid)) for r, y in zip(notch_r, notch_y)], (float(notch_r[-1] + height + POCKET_RADIAL_EXTRA_MM), half + pad)]
    envelope = revolved(profile)
    pieces = []
    for a in angles(count, True):
        prisms = extruded(ridge_outline(a, POCKET_SIDE_MM, scale), -half - pad, half + pad)
        pocket = trimesh.boolean.intersection([trimesh.boolean.union(prisms, engine="manifold"), envelope], engine="manifold")
        if channels and abs(((a - 90.0 + 180.0) % 360.0) - 180.0) >= CHANNEL_FROM_DEG:   # wedge frame: stand up is local -y
            pieces += [pocket.copy().apply_translation([0.0, -s, 0.0]) for s in np.arange(CHANNEL_STEP_MM, CHANNEL_LENGTH_MM + 1e-9, CHANNEL_STEP_MM)]
        pieces.append(pocket)
    pockets = trimesh.boolean.union(pieces, engine="manifold")
    return pockets.apply_transform(stand_frame()).apply_translation([cx, 0, cz + lift])


def coupon(count: int) -> tuple[trimesh.Trimesh, float, float]:
    """The stand's groove region with `count` pockets, in the stand frame; returns the mesh and the groove centre (x, z)."""
    stand = fs.load_stand()
    plate = fs.build(stand)
    outline = fs.flat_face_outline(stand)
    cx, cz, _ = fs.groove_circle(outline)
    half = (stand.bounds[1, 1] - stand.bounds[0, 1]) / 2
    mid = float((stand.bounds[0, 1] + stand.bounds[1, 1]) / 2)
    pockets = pocket_cutter(count, stand, cx, cz, channels=True)   # the final stand's pockets: the upper ones open straight up
    box = trimesh.creation.box([2 * (RIDGE_TIP[0] + 14.0), 2 * half + 2, COUPON_BELOW_MM + COUPON_ABOVE_MM]).apply_translation([cx, mid, cz + (COUPON_ABOVE_MM - COUPON_BELOW_MM) / 2])
    piece = trimesh.boolean.intersection([plate, box], engine="manifold")
    piece = trimesh.boolean.difference([piece, pockets], engine="manifold")
    if not piece.is_watertight:
        raise RuntimeError(f"coupon {count} is not watertight")
    return piece, float(cx), float(cz)


def overlap(piece: trimesh.Trimesh, wedge: trimesh.Trimesh, cx: float, cz: float, sag: float, turn_deg: float) -> float:
    """Volume (mm3) where the wedge, seated at the groove centre sagged `sag` mm and turned `turn_deg`, overlaps the coupon."""
    turned = wedge.copy().apply_transform(trimesh.transformations.rotation_matrix(np.radians(turn_deg), [0, 0, 1]))
    placed = turned.apply_transform(stand_frame()).apply_translation([cx, 0, cz - sag])
    both = trimesh.boolean.intersection([piece, placed], engine="manifold")
    return float(both.volume) if len(both.faces) else 0.0


def printable(piece: trimesh.Trimesh, count: int) -> trimesh.Trimesh:
    """The coupon turned to print flat (thickness up), with "T<count>" and the ridge height and width scale recessed into the bed-side face at the clear spot nearest its far edge, away from the groove."""
    flat = piece.copy().apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    flat.apply_translation(-flat.bounds.mean(axis=0))
    scale, height = TEETH[count]
    pieces = lt.block([(f"T{count}", LABEL_SIZE_MM), (f"h{height:.1f} w{scale:.1f}", 3.2)])
    area = lt.face_area(flat, float(flat.bounds[0, 2]) + 0.1)
    spot = lt.find_spot(area, pieces, (0.0, float(flat.bounds[1, 1])), 1.0, step=1.0)
    if spot is None:
        raise RuntimeError(f"no clear spot for the coupon {count} label")
    log.info("coupon %d label at (%.0f, %.0f) mm", count, *spot)
    return lt.cut(flat, [shapely_translate(p, *spot) for p in pieces], float(flat.bounds[0, 2]), LABEL_DEPTH_MM)


def main() -> None:
    """Build the wedges and coupons, check the overlap, write the STLs and the project."""
    kinds: dict[str, bp.Kind] = {}
    plan_names: list[str] = []
    for n, count in enumerate(TEETH_COUNTS, start=1):
        wedge = toothed_wedge(count)
        piece, cx, cz = coupon(count)
        for sag in SAG_STEPS_MM:
            v = overlap(piece, wedge, cx, cz, sag, 0.0)
            log.info("%d teeth: sag %.1f mm, overlap %.3f mm3", count, sag, v)
            if v > 1.0:
                raise RuntimeError(f"{count} teeth: the wedge overlaps the coupon by {v:.2f} mm3 at sag {sag} mm")
        turn = 0.5 * 360.0 / count
        log.info("%d teeth: wedge turned half a pitch (%.1f deg) overlaps by %.1f mm3 (it must, the ridges meet the pockets' sides)", count, turn, overlap(piece, wedge, cx, cz, 0.0, turn))
        wedge.export(HERE / f"Wedge - Single Sided - Teeth {count}.stl")
        flat = printable(piece, count)
        flat.export(HERE / f"Groove Coupon - {count} Teeth.stl")
        for name, mesh, walls in ((f"Wedge - Single - Teeth {count}", wedge, bp.WEDGE_WALLS), (f"Groove Coupon - {count} Teeth", flat, None)):
            kinds[name] = bp.Kind(name, bp.centred(mesh), 2 * n - (name.startswith("Wedge")), float(mesh.extents[2]), walls)
            plan_names.append(name)
    if not bp.PRINTER_HOST:
        log.info("PRINTER_HOST not set: STLs written, skipping the .3mf plate")
        return
    instances = bp.placements(kinds, [(0, [(name, 1) for name in plan_names[:4]], 2), (1, [(name, 1) for name in plan_names[4:]], 2)])
    profile = {**bp.profile_values(), **bst.project_fast_values()}
    bp.write_project(OUTPUT, kinds, instances, [f"Gear test 8 and 12 teeth {profile['layer_height']} mm", f"Gear test 18 and 24 teeth {profile['layer_height']} mm"], profile)


if __name__ == "__main__":
    main()
