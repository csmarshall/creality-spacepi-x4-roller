"""Write the Adaptive pulley half as STEP solids (and STLs of the same solids) in three designs: ABS springs, ABS springs with pad slots (plus the TPU pads that go in them), and an ABS carrier for TPU wing rings.

Axis is z. z = 0 is the butt face (the middle of the stack, where the two halves meet), z = WIDTH_MM is the outer face that looks at the wedge. Springs come from build_adaptive_coupon.py (same variants, same spiral
formulas), so the coupons and the halves cannot drift apart. The spring bands start at the butt face and end flush with the outer face, so the half prints butt face down with the springs standing on the bed.
The M3 holes and pockets are read off the OG pulley mesh: two 3.5 mm holes at x = +-15.5 through the 3 mm butt skin, then a round pocket (x = -15.5, 6.2 mm widening to 7.0 mm) and a hex pocket (x = +15.5, about
5.6 mm across flats widening to 6.2 mm) running out to the outer face.
Designs: "abs" is variant A (52 to 54 mm); "pads" is variant D (R 26 with a dovetail slot at every tip, and one TPU pad per wing, written as a separate placed mesh for display); "tpu" is a carrier with two retaining grooves
and no springs; the two TPU wing rings that sit on it are the ring from build_adaptive_coupon.py placed at both bands (display mesh only: the ring STL itself is the part to print).
Check limit: the spring flanks are 36-segment polylines (they match the STL exactly; they are not true splines); the pocket sizes are approximated from a noisy mesh section; the check below compares the
spring volume in the solid with shapely's spring area x band width, which tests the CAD build against the 2D polygons, not the physics of the springs.
"""
from __future__ import annotations

import logging
import os
from typing import cast

import numpy as np
import trimesh
from build123d import Align, Axis, BuildLine, BuildPart, BuildSketch, Circle, Cone, Cylinder, Locations, Mode, Plane, Polyline, Pos, RegularPolygon, export_step, export_stl, extrude, import_step, loft, make_face, revolve
from shapely.geometry import Point
from shapely.ops import unary_union

import build_adaptive_coupon as ac
import build_assembly as ba
import build_clipon_geometry as cg
import build_project_3mf as bp

logging.basicConfig(level=os.environ.get("AXEL_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_adaptive_half_step")

HERE = bp.HERE
WIDTH_MM = 33.0                   # pulley half width, as the OG and concave halves
FACE_R_MM = 22.1                  # outer face edge radius (OG/concave mesh)
BORE_R_MM = 9.2                   # axle clearance bore
RECESS_R_MM = 11.0                # 608 recess, 22.00 mm
RECESS_DEPTH_MM = 6.5
BUTT_SKIN_MM = 3.0                # thickness of the wall carrying the 3.5 mm screw holes
SCREW_X_MM = 15.5
SLOT_MARK_RING_R_MM = 13.0         # fit dial-in core: radius at which the slot-number dimples sit on the butt face
SCREW_HOLE_D_MM = 3.5
ROUND_POCKET_D_MM = (6.2, 7.0)    # at the skin, at the outer face
HEX_POCKET_AF_MM = (5.6, 6.2)     # across flats, same
BAND_START_MM = 0.0               # first band starts this far from the butt face (0: the springs stand on the bed when printed butt face down), the second ends this far from the outer face
CHECK_TOLERANCE = 0.02            # relative, spring volume against the polygon areas
DESIGNS = {"abs": ac.A, "pads": ac.D, "tpu": ac.F, "clip": cg.CLIP}
NAMES = {"abs": "Adaptive Pulley Half", "pads": "Adaptive Pulley Half (pad slots)", "tpu": "Adaptive Pulley Half (TPU carrier)", "clip": "Adaptive Pulley Half (clip-on core)"}


def half_width(mode: str) -> float:
    """Width of the half in this design: the clip-on core is wider than the older ones (cg.HALF_WIDTH_MM), which keep WIDTH_MM."""
    return cg.HALF_WIDTH_MM if mode == "clip" else WIDTH_MM


def band_starts() -> tuple[float, float]:
    return BAND_START_MM, WIDTH_MM - BAND_START_MM - ac.BAND_MM


def core_profile(mode: str) -> list[tuple[float, float]]:
    """(r, z) outline of the revolved body: bore, outer wall (with the carrier's two retaining grooves in "tpu" mode), face chamfer, recess."""
    v = DESIGNS[mode]
    core_r = ac.CARRIER_CORE_R_MM if mode == "tpu" else v.r_core
    face_r = min(FACE_R_MM, core_r)   # the pad-slot design's core (22.0) is already smaller than the face edge radius, so it has no chamfer
    chamfer = core_r - face_r
    outer: list[tuple[float, float]] = [(core_r, 0.0)]
    if mode == "tpu":
        for z0 in band_starts():
            g0 = z0 + (ac.BAND_MM - ac.GROOVE_WIDTH_MM) / 2
            outer += [(core_r, g0), (core_r - ac.GROOVE_DEPTH_MM, g0), (core_r - ac.GROOVE_DEPTH_MM, g0 + ac.GROOVE_WIDTH_MM), (core_r, g0 + ac.GROOVE_WIDTH_MM)]
    width = half_width(mode)
    outer += [(core_r, width - chamfer), (face_r, width)]
    recess = ba.BEARING_WIDTH_MM + cg.BEARING_FLOAT_MM if mode == "clip" else RECESS_DEPTH_MM   # the clip-on core's recess is one bearing width deep (plus the float, 0 now): the floor is a shelf the outer race lands on, face flush with the core mouth, held square by the recess wall
    recess_r = RECESS_R_MM + (cg.BEARING_RECESS_CLEARANCE_MM if mode == "clip" else 0.0)
    mouth = [(recess_r + cg.BEARING_MOUTH_CHAMFER_MM, width), (recess_r, width - cg.BEARING_MOUTH_CHAMFER_MM)] if mode == "clip" else [(recess_r, width)]
    points = [(BORE_R_MM, 0.0), *outer, *mouth, (recess_r, width - recess), (BORE_R_MM, width - recess), (BORE_R_MM, 0.0)]
    return [p for i, p in enumerate(points) if i == 0 or p != points[i - 1]]   # no zero-length edges


def build_half(mode: str, slot_extra: tuple[float, ...] = (0.0, 0.0, 0.0), slot_marks: bool = False, screws: bool = True):
    """The half pulley as a build123d solid."""
    v = DESIGNS[mode]
    with BuildPart() as half:
        with BuildSketch(Plane.XZ):
            with BuildLine():
                Polyline(*core_profile(mode))
            make_face()
        revolve(axis=Axis.Z)
        if mode == "clip":   # a bare core with three dovetail slots; the petals are separate parts (build_clipon_petal.py)
            slot_length = half_width(mode) - cg.SLOT_END_WALL_MM
            for i in range(len(cg.SLOT_CENTRES_DEG)):   # tapered slot: starts below the butt face so it opens there, ends at the end wall; the same linear taper continues below z = 0
                for z in (-1.0, slot_length):
                    slot = [(float(x), float(y)) for x, y in list(cg.slot_polygon(i, z, slot_extra[i]).exterior.coords)[:-1]]
                    with BuildSketch(Plane.XY.offset(z)):
                        with BuildLine():
                            Polyline(*slot, close=True)
                        make_face()
                loft(mode=Mode.SUBTRACT)
                for side in (1, -1):   # orientation dots on the outer surface beside the slot: one on the counter-clockwise side, two on the clockwise side
                    ang = np.radians(cg.SLOT_CENTRES_DEG[i] + side * cg.DOT_ANGLE_DEG)
                    for z in cg.dot_heights(1 if side > 0 else 2):
                        with BuildSketch(Plane(origin=(cg.CORE_RADIUS_MM * np.cos(ang), cg.CORE_RADIUS_MM * np.sin(ang), z), x_dir=(0, 0, 1), z_dir=(np.cos(ang), np.sin(ang), 0))):
                            Circle(cg.DOT_R_MM)
                        extrude(amount=-cg.DOT_DEPTH_MM, mode=Mode.SUBTRACT)
                if slot_marks:   # fit dial-in core: i + 1 dimples on the butt face in front of slot i, so the slots can be told apart
                    phi = np.radians(cg.SLOT_CENTRES_DEG[i])
                    for j in range(i + 1):
                        off = (j - i / 2.0) * 2.4
                        mx, my = SLOT_MARK_RING_R_MM * np.cos(phi) - off * np.sin(phi), SLOT_MARK_RING_R_MM * np.sin(phi) + off * np.cos(phi)
                        with Locations((mx, my, 0.0)):
                            Cylinder(cg.MARK_RADIUS_MM, 2 * cg.MARK_DEPTH_MM, mode=Mode.SUBTRACT)
        if mode not in ("tpu", "clip"):
            for z0 in band_starts():
                for i in range(v.n):
                    theta0 = i * 2 * np.pi / v.n
                    outline = [(float(x), float(y)) for x, y in list(ac.spring_polygon(v, theta0).exterior.coords)[:-1]]
                    with BuildSketch(Plane.XY.offset(z0)):
                        with BuildLine():
                            Polyline(*outline, close=True)
                        make_face()
                    extrude(amount=ac.BAND_MM)
                    if v.pad:   # the dovetail slot is cut after the wing is built: a slot drawn into the wing outline makes the sketch face fail
                        slot = ac.local_polygon(ac.slot_points(), ac.slot_frame(v, theta0))
                        with BuildSketch(Plane.XY.offset(z0)):
                            with BuildLine():
                                Polyline(*[(float(x), float(y)) for x, y in list(slot.exterior.coords)[:-1]], close=True)
                            make_face()
                        extrude(amount=ac.BAND_MM, mode=Mode.SUBTRACT)
        # M3 holes through the butt skin, then the tapered pockets out to the outer face
        if screws and (mode != "clip" or cg.CORE_M3_HOLES):
            for x in (-SCREW_X_MM, SCREW_X_MM):
                with Locations(Pos(x, 0, -1)):
                    Cylinder(SCREW_HOLE_D_MM / 2, BUTT_SKIN_MM + 2, align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT)
            depth = half_width(mode) - BUTT_SKIN_MM
            with Locations(Pos(-SCREW_X_MM, 0, BUTT_SKIN_MM)):
                Cone(ROUND_POCKET_D_MM[0] / 2, ROUND_POCKET_D_MM[1] / 2, depth + 1, align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT)
            taper_deg = float(np.degrees(np.arctan(((HEX_POCKET_AF_MM[1] - HEX_POCKET_AF_MM[0]) / 2) / depth)))
            with BuildSketch(Plane.XY.offset(BUTT_SKIN_MM).move(Pos(SCREW_X_MM, 0, 0))):
                RegularPolygon(radius=HEX_POCKET_AF_MM[0] / np.sqrt(3), side_count=6)
            extrude(amount=depth + 0.5, taper=-taper_deg, mode=Mode.SUBTRACT)
    return half.part


def placed_pads() -> trimesh.Trimesh:
    """The TPU pads of the "pads" design, sitting in their slots (display mesh)."""
    v = DESIGNS["pads"]
    length = ac.BAND_MM - ac.PAD_END_CLEARANCE_MM
    pieces = []
    for z0 in band_starts():
        for i in range(v.n):
            frame = ac.slot_frame(v, i * 2 * np.pi / v.n)
            pad = ac.local_polygon(list(ac.pad_section().exterior.coords)[:-1], frame)
            pieces.append(trimesh.creation.extrude_polygon(pad, length).apply_translation([0, 0, z0 + ac.PAD_END_CLEARANCE_MM / 2]))
    return trimesh.util.concatenate(pieces)


def placed_rings() -> trimesh.Trimesh:
    """The TPU wing ring at both bands of the carrier (display mesh)."""
    ring = ac.tpu_ring_mesh()
    return trimesh.util.concatenate([ring.copy().apply_translation([0, 0, z0]) for z0 in band_starts()])


def write_design(mode: str) -> trimesh.Trimesh:
    part = build_half(mode)
    log.info("%s: solid valid=%s volume=%.1f mm3", mode, part.is_valid(), part.volume)
    if not part.is_valid():
        raise RuntimeError(f"{mode}: the half is not a valid solid")
    step = HERE / f"{NAMES[mode]}.step"
    export_step(part, str(step))
    back = import_step(str(step))
    log.info("%s: STEP read back valid=%s volume=%.1f mm3 (written %.1f)", mode, back.is_valid(), back.volume, part.volume)
    stl = HERE / f"{NAMES[mode]}.stl"
    export_stl(part, str(stl))
    mesh = cast(trimesh.Trimesh, trimesh.load(stl))
    log.info("%s: STL watertight=%s volume=%.1f mm3, z %.2f to %.2f", mode, mesh.is_watertight, mesh.volume, mesh.bounds[0, 2], mesh.bounds[1, 2])
    return mesh


def check_spring_volume(mesh: trimesh.Trimesh, mode: str) -> None:
    """Spring volume outside the core, from shapely, against the same volume cut from the solid."""
    v = DESIGNS[mode]
    core = Point(0, 0).buffer(v.r_core, 360)
    spring_area = unary_union([ac.wing_with_slot(v, i * 2 * np.pi / v.n) for i in range(v.n)]).difference(core).area
    expected = 2 * spring_area * ac.BAND_MM
    outside = trimesh.creation.cylinder(radius=v.r_core + 0.001, height=WIDTH_MM + 2, sections=360).apply_translation([0, 0, WIDTH_MM / 2])
    measured = float(trimesh.boolean.difference([mesh, outside], engine="manifold").volume)
    log.info("%s: springs outside the core: %.1f mm3 in the solid, %.1f mm3 from the polygons (%.2f %% apart)", mode, measured, expected, 100 * (measured - expected) / expected)
    if abs(measured - expected) > CHECK_TOLERANCE * expected:
        raise RuntimeError(f"{mode}: spring volume in the solid disagrees with the polygon areas")


def main() -> None:
    for mode in ("abs", "pads"):
        check_spring_volume(write_design(mode), mode)
    carrier = write_design("tpu")
    pads, rings = placed_pads(), placed_rings()
    for name, m in (("pads", pads), ("rings", rings)):
        log.info("placed %s: watertight=%s volume=%.1f mm3", name, m.is_watertight, m.volume)
    overlap = trimesh.boolean.intersection([carrier, rings], engine="manifold").volume   # the rings sit on the carrier: only the stretch fit overlaps
    log.info("TPU rings overlap the carrier by %.1f mm3 (the %.1f mm stretch fit)", overlap, ac.RING_STRETCH_MM)


if __name__ == "__main__":
    main()
