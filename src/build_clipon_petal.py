"""Write the clip-on design: one pulley-half core with three dovetail slots, and the petal (print three per half), as STEP + STL, then check the fit and draw the end view.

The petal is one full-length extrusion of the arm-plus-dovetail section (PETAL_LENGTH_MM): the spool touches the pulley only with its two cardboard cheeks (at most CHEEK_MAX_MM thick), and a full-length band
meets them wherever the stack puts them. The slot opens at the butt face and the petal slides in from there; the end wall at the outer face and the petal behind it (the other half's) hold it in. All three
petals are the same part turned 120 degrees. Sag scales with 1/width in the beam model, so the figures here are build_adaptive_coupon's (14 mm band) times BAND_MM / PETAL_LENGTH_MM.
Check limit: the checks compare solids that come
from the same polygons (slot contains head; no volume overlap with the core), which proves the parts are consistent with each other and not that ABS prints to these clearances or that the springs behave.
"""
from __future__ import annotations

import logging
import os
from typing import cast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from build123d import BuildLine, Circle, BuildPart, BuildSketch, Plane, Polyline, Text, Vector, export_step, export_stl, extrude, loft, make_face, Cylinder, Locations, Mode
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

import build_adaptive_coupon as ac
import build_adaptive_half_step as hs
import build_clipon_geometry as cg

logging.basicConfig(level=os.environ.get("AXEL_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_clipon_petal")

HERE = hs.HERE
PETAL_NAMES = {"t2.0": ("Adaptive Clip Petal t2.0", cg.THIN), "t2.6": ("Adaptive Clip Petal t2.6", cg.CLIP), "t3.0": ("Adaptive Clip Petal t3.0", cg.THICK), "taper": ("Adaptive Clip Petal standard", cg.TAPERED),
               "taper soft": ("Adaptive Clip Petal soft", cg.tapered_variant(*cg.TAPER_SOFT)), "taper firm": ("Adaptive Clip Petal firm", cg.tapered_variant(*cg.TAPER_FIRM)),
               "soft fillet": ("Adaptive Clip Petal soft fillet", cg.tapered_variant(*cg.TAPER_SOFT, fillet=cg.FILLET_MM)),
               "soft ribbed": ("Adaptive Clip Petal soft ribbed", cg.tapered_variant(*cg.TAPER_SOFT, wave_amp=cg.WAVE_AMP_MM, fillet=cg.FILLET_MM))}
PETAL_LENGTH_MM = cg.PETAL_LENGTH_MM
PETAL_LABELS = {"t2.0": "T2", "t2.6": "T26", "t3.0": "T3", "taper": "M", "taper soft": "S", "taper firm": "F", "soft fillet": "SF", "soft ribbed": "SR"}   # engraved on both head end faces: S soft, SF soft + fillet, SR soft + fillet + corrugation, M standard (medium), F firm, T2 flat 2.0 mm
LABEL_SIZE_MM = 2.2            # character height; the head end face is about 4 to 6.4 mm wide and 2.4 mm deep
LABEL_DEPTH_MM = 0.4           # engraved, 3 layers at 0.12 mm; the end faces sit against the closed end of the slot, so the label cannot be felt in use
BORES_MM = (50.0, 56.0, 62.0)
MIN_PRESS_FIT_MM3 = 0.0        # lower bound: with slot clearance 0.10 / -0.05 the overlap is about 2.5 mm3 per half
MAX_PRESS_FIT_MM3 = 30.0       # the designed interference near an end wall (SLOT_CLEARANCE_END_MM < 0) shows up as this much petal/core overlap at most, per half
CONTACT_BANDS = 2              # per roller: the band of each half carries the spool (one cheek each)
WIDTH_FACTOR = ac.BAND_MM / PETAL_LENGTH_MM   # beam-model sag shrinks by this against the 14 mm band, if the load spreads over the whole width


def arm_dot_site(i: int, v: ac.Variant) -> tuple[float, float, float, float]:
    """(x, y, nx, ny): the point on the petal's top that sits DOT_GAP_MM + DOT_R_MM (measured along the surface) inside the arm root's end face, which is the radial line at cg.root_angle(i) on the clockwise side of the slot; and the outward direction there."""
    outer, _ = ac.spring_curves(v, cg.root_angle(i))
    pts = np.array(outer)
    ang = np.unwrap(np.arctan2(pts[:, 1], pts[:, 0]))
    radius = np.hypot(pts[:, 0], pts[:, 1])
    offset = cg.DOT_GAP_MM + cg.DOT_R_MM
    target = ang[0] + offset / cg.CORE_RADIUS_MM
    for _ in range(6):   # the surface sits a little above the core, so the same arc length is a slightly smaller angle
        r = float(np.interp(target, ang, radius))
        target = ang[0] + offset / r
    r = float(np.interp(target, ang, radius))
    return r * float(np.cos(target)), r * float(np.sin(target)), float(np.cos(target)), float(np.sin(target))


def exterior(poly) -> list[tuple[float, float]]:
    """Outline points, counter-clockwise (a mirrored polygon comes back clockwise, and the sketch face must not flip)."""
    return [(float(x), float(y)) for x, y in list(orient(poly, 1.0).exterior.coords)[:-1]]


def petal_name(key: str) -> str:
    """File and object name of petal shape key."""
    return PETAL_NAMES[key][0]


def head_sections(full: bool) -> list[tuple[float, float]]:
    """(z in the petal's own frame, distance from the butt plane) of the head's profile planes: a whole petal has its butt plane in the middle and tapers to both ends, a half petal (the old test parts) starts at the butt plane."""
    if full:
        return [(0.0, PETAL_LENGTH_MM), (PETAL_LENGTH_MM, 0.0), (cg.PETAL_SPAN_MM, PETAL_LENGTH_MM)]
    return [(0.0, 0.0), (PETAL_LENGTH_MM, PETAL_LENGTH_MM)]


def fillet_polygon(i: int, v: ac.Variant):
    """The material that rounds the concave corners where the arm meets the dovetail head (radius v.fillet), outside the core circle; None for a sharp corner."""
    if v.fillet <= 0.0:
        return None
    shape = unary_union([cg.wing_polygon(i, v), cg.head_polygon(i, 0.0)])
    extra = shape.buffer(v.fillet, 64).buffer(-v.fillet, 64).difference(shape).difference(Point(0, 0).buffer(cg.CLIP.r_core, 180))
    extra = extra.intersection(cg.head_polygon(i, 0.0).buffer(2 * v.fillet))   # only at the head corner: closing the whole arm would also fill every valley of a corrugation
    extra = extra.buffer(-1e-3).buffer(1e-3)   # drop numerical slivers
    return None if extra.is_empty or extra.area < 1e-3 else extra


def build_petal(i: int = 0, v: ac.Variant = cg.CLIP, head_extra: float = 0.0, marks: int = 0, taper: float = cg.HEAD_TAPER_MM, full: bool = True, label: str = ""):
    """Petal i of variant v: one piece spanning both halves (full) with its butt plane in the middle at z = PETAL_LENGTH_MM, or a half petal from the butt plane (z = 0) to PETAL_LENGTH_MM for the old single-core tests.

    The arm is a constant extrusion along all of the length and the dovetail head a ruled loft that is narrowest at the insertion ends and widest at the butt plane. head_extra offsets the head per side (fit dial-in); marks puts that many dimples on the top end."""
    top = cg.PETAL_SPAN_MM if full else PETAL_LENGTH_MM
    over = cg.ARM_OVERHANG_MM if full else 0.0   # the arm runs past the head at both ends of a whole petal
    fillet = fillet_polygon(i, v)
    arm = cg.wing_polygon(i, v)
    head_section = cg.head_polygon(i, 0.0, extra=head_extra, taper=taper)
    if fillet is not None:   # the fillet rides with the head prism, grown a little so it overlaps the arm and the head instead of only touching them
        head_section = unary_union([head_section, fillet.buffer(0.02)])
        if head_section.geom_type != "Polygon":
            head_section = max(head_section.geoms, key=lambda g: g.area)
    with BuildPart() as petal:
        if taper == 0.0:   # a straight head: the arm (with its fillet) is one extrusion and the head one prism over the middle part; they overlap in volume, so the union does not depend on coincident faces
            with BuildSketch(Plane.XY.offset(-over)):
                with BuildLine():
                    Polyline(*exterior(arm), close=True)
                make_face()
            extrude(amount=top + 2 * over)
            with BuildSketch(Plane.XY):
                with BuildLine():
                    Polyline(*exterior(head_section), close=True)
                make_face()
            extrude(amount=top)
        else:
            with BuildSketch(Plane.XY.offset(-over)):
                with BuildLine():
                    Polyline(*exterior(arm), close=True)
                make_face()
            extrude(amount=top + 2 * over)
            sections = head_sections(full)
            for (z0, d0), (z1, d1) in zip(sections, sections[1:]):
                for z, d in ((z0, d0), (z1, d1)):
                    with BuildSketch(Plane.XY.offset(z)):
                        with BuildLine():
                            Polyline(*exterior(cg.head_polygon(i, d, extra=head_extra, taper=taper)), close=True)
                        make_face()
                loft(ruled=True)
        if label:   # the label reads along the arm's tangent, centred on the head, on both end faces (the bottom one is read from below)
            head = cg.head_polygon(i, 0.0).centroid
            phi = np.radians(cg.SLOT_CENTRES_DEG[i])
            tangent = Vector(-np.sin(phi), np.cos(phi), 0)
            for z, normal in ((top, Vector(0, 0, 1)), (0.0, Vector(0, 0, -1))):
                with BuildSketch(Plane(origin=(head.x, head.y, z), x_dir=tangent, z_dir=normal)):
                    Text(label, font_size=LABEL_SIZE_MM)
                extrude(amount=-LABEL_DEPTH_MM, mode=Mode.SUBTRACT)
        if label and full:   # orientation dots on the petal's top just clockwise of the arm root's end face: a pair on the top half, one on the bottom half (z = top / 2 is the butt plane)
            x, y, nx, ny = arm_dot_site(i, v)
            for z in [top / 2 + h for h in cg.dot_heights(2, cg.DOT_AXIAL_MM)] + [top / 2 - cg.DOT_AXIAL_MM]:   # a pair in the middle of the top half (the half that goes in the first core), a single dot in the middle of the other
                with BuildSketch(Plane(origin=(x, y, z), x_dir=(0, 0, 1), z_dir=(nx, ny, 0))):
                    Circle(cg.DOT_R_MM)
                extrude(amount=-cg.DOT_DEPTH_MM, mode=Mode.SUBTRACT)
        for j in range(marks):
            mx, my = cg.mark_position(i, v, j)
            with Locations((mx, my, top)):
                Cylinder(cg.MARK_RADIUS_MM, 2 * cg.MARK_DEPTH_MM, mode=Mode.SUBTRACT)
    return petal.part


def write(part, name: str) -> trimesh.Trimesh:
    if not part.is_valid():
        raise RuntimeError(f"{name}: not a valid solid")
    export_step(part, str(HERE / f"{name}.step"))
    export_stl(part, str(HERE / f"{name}.stl"))
    mesh = cast(trimesh.Trimesh, trimesh.load(HERE / f"{name}.stl"))
    log.info("%s: valid, watertight=%s, volume %.1f mm3, extents %s mm", name, mesh.is_watertight, mesh.volume, np.round(mesh.extents, 2))
    return mesh


def draw_end_view(path) -> None:
    core = Point(0, 0).buffer(cg.CLIP.r_core, 180)
    for i in range(3):
        core = core.difference(cg.slot_polygon(i))
    petals = unary_union([cg.petal_arm_polygon(i) for i in range(3)])
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for geom, colour in ((core, "#7f8c8d"), (petals, "#c0392b")):
        for poly in getattr(geom, "geoms", [geom]):
            x, y = poly.exterior.xy
            ax.fill(x, y, color=colour)
            for hole in poly.interiors:
                hx, hy = hole.xy
                ax.fill(hx, hy, color="white")
    for bore, style in zip(BORES_MM, (":", "--", "-")):
        t = np.linspace(0, 2 * np.pi, 240)
        ax.plot(bore / 2 * np.cos(t), bore / 2 * np.sin(t), style, color="black", lw=1, label=f"{bore:.0f} mm bore")
    for deg in cg.M3_POCKET_ANGLES_DEG:
        ax.add_patch(plt.Circle((15.5 * np.cos(np.radians(deg)), 15.5 * np.sin(np.radians(deg))), 3.5, fill=False, color="#2980b9", lw=1))
    ax.set_aspect("equal")
    ax.set_xlim(-34, 34)
    ax.set_ylim(-34, 34)
    ax.axis("off")
    ax.legend(loc="lower center", fontsize=8, ncol=3)
    ax.set_title("Clip-on: grey core with 3 dovetail slots, red petals (free), blue M3 pockets", fontsize=9)
    fig.savefig(path, dpi=130, bbox_inches="tight")


def main() -> None:
    core_part = hs.build_half("clip")
    core = write(core_part, hs.NAMES["clip"])
    for stale in [*HERE.glob("Adaptive Clip Petal tapered*"), *HERE.glob("Adaptive Clip Petal * [AB].*")]:   # the old "tapered" names, renamed standard / soft / firm;   # the earlier mirror-image A and B petals (a petal is one piece now)
        stale.unlink()
    petals = {key: write(build_petal(0, v, label=PETAL_LABELS.get(key, "")), petal_name(key)) for key, (_, v) in PETAL_NAMES.items()}
    half_turn = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])   # the second half: the same core turned half a turn about the radial axis through slot 1, which keeps the slots in place and swaps 210 and 330
    core_b = core.copy().apply_transform(half_turn)
    for key, mesh in petals.items():
        placed = mesh.copy().apply_translation([0, 0, -PETAL_LENGTH_MM])   # the butt plane (the middle of the petal) at z = 0, half A above it and half B below
        overlaps = [trimesh.boolean.intersection([c, placed], engine="manifold").volume for c in (core, core_b)]
        log.info("%s petal: %.1f mm long, overlap with half A's core %.2f mm3 and half B's %.2f mm3 (the designed press fit near the two end walls)", key, mesh.extents[2], *overlaps)
        if max(overlaps) > MAX_PRESS_FIT_MM3 or min(overlaps) < MIN_PRESS_FIT_MM3:
            raise RuntimeError(f"{key}: the press fit with the cores is outside the designed range")
    log.info("designed clearance per side %.2f mm at the butt, %.2f mm at the end wall (negative = press fit)", cg.SLOT_CLEARANCE_MOUTH_MM, cg.SLOT_CLEARANCE_END_MM)
    if not cg.slot_polygon(0, 0.0).contains(cg.head_polygon(0, 0.0)):
        raise RuntimeError("the slot mouth does not contain the head at the butt plane")
    log.info("slot-to-M3-pocket clearance %.2f mm; petal %.1f mm long, arm along all of it", cg.slot_clear_of_pockets(), cg.PETAL_SPAN_MM)
    for key, (_, v) in PETAL_NAMES.items():
        q = ac.predict(v)
        log.info("%s predicted (beam model, guessed modulus): bores %.1f to %.1f, sag per half %.2f mm, two cheek contacts %.2f", key, q["bore_min"], q["bore_max"], q["sag"] * WIDTH_FACTOR, q["sag"] * WIDTH_FACTOR / CONTACT_BANDS)
    p = ac.predict(cg.CLIP)
    log.info("predicted (beam model, guessed modulus): bores %.1f to %.1f, sag per half %.2f mm, two cheek contacts %.2f", p["bore_min"], p["bore_max"], p["sag"] * WIDTH_FACTOR, p["sag"] * WIDTH_FACTOR / CONTACT_BANDS)
    draw_end_view(HERE / "adaptive_clipon_end_view.png")


if __name__ == "__main__":
    main()
