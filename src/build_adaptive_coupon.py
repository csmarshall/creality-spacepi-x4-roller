"""Write the Adaptive spring-lip test parts: ABS spring-ring coupons and bore gauges (one project) plus TPU pads and a TPU wing ring (STLs only), all for one test session.

The Adaptive pulley's lip is a ring of N curved cantilever springs ("wings") around a rigid core. Each wing is a spiral: its inner surface leaves the core at the root and rises to a free tip, its outer surface ends at
R_FREE, the largest bore it touches without being squeezed. A spool bore slid over the lip pushes the tips in and every wing presses back evenly, so the bore settles on the pulley's axis for any bore between
about 2 * (R_FREE - travel) and 2 * R_FREE. A coupon is one 14 mm slice of a lip on a hollow core (HUB_R hole, so it prints fast); the gauge rings stand in for a spool bore (ID 52, 54, 56).
Variants (ABS): A and B are the 5-wing rings for the 52 to 54 and 54 to 56 bands, C is the widest ring (4 wings, 2.2 mm, a 3.0 mm tip gap: about 52.0 to 56.1), D is a 5-wing ring with a dovetail slot at every tip for
printed TPU pads (a grip strip, the TPU is not bonded to ABS), and F is a bare ABS carrier for the TPU wing ring (8 wings, 3.5 mm, printed as its own TPU part and stretched onto the core, where a rib drops into a groove).
Check limit: closed solids, polygon validity, the pad-in-slot clearances and the predicted stiffness are computed here; the predictions are beam theory with guessed moduli (MATERIALS) and the real print, the root
fillet, creep, TPU-on-ABS retention and the TPU's own behaviour can differ, which is what the parts are for. Nothing here measures how any printed part behaves.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import cast

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

import build_project_3mf as bp
import build_speed_tests as bst
import label_text as lt

logging.basicConfig(level=os.environ.get("AXEL_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_adaptive_coupon")

HERE = bp.HERE
OUTPUT = HERE / "Free Roll - Adaptive Lip Test (fast inner).3mf"
BAND_MM = 14.0            # axial width of a spring band (coupon height); stiffness grows with it while travel does not
HUB_R_MM = 12.0           # hole through the coupon's core, to save plastic and give the label an annulus
MIN_GAP_MM = 0.5          # smallest slot the slicer keeps open
ROOT_EMBED_MM = 0.4       # the wing's inner surface starts this far inside the core, so root and core fuse
U_FIX = 0.06              # fraction of the wing (from the root) that stays inside the core
ROOT_TAPER = 0.6          # extra thickness at the root (fraction of t), fading out by U_TAPER
U_TAPER = 0.30
TIP_GAP_DEG = 4.0         # angular gap between one wing's tip and the next one's root
POINTS = 40
MATERIALS = {"ABS": (1600.0, 0.01), "TPU": (30.0, 0.10), "PETG": (1800.0, 0.025)}  # (modulus MPa, design strain at full travel): guesses; ABS yields near 2 % so stay at half, TPU 95A is rated far higher, PETG yields near 4 to 5 % (guess, not measured on this printer)
SPOOL_LOAD_N = 10.0       # a full 1 kg spool
GAUGE_IDS_MM = (52.0, 54.0, 56.0)
GAUGE_WALL_MM = 7.0
GAUGE_HEIGHT_MM = 8.0
LABEL_DEPTH_MM = 0.36
LABEL_CLEARANCE_MM = 0.8
GAP_MM = 11.0   # clear space between footprints: two 5 mm petal brims (bp.PETAL_BRIM_WIDTH_MM) plus 1 mm
EDGE_MM = 8.0
SLOT_U = 0.88             # where along the wing the pad's dovetail slot is centred
SLOT_OPENING_MM, SLOT_FLOOR_MM, SLOT_DEPTH_MM = 2.0, 3.2, 1.0   # dovetail: width at the surface, width at the bottom, depth
PAD_CLEARANCE_MM = 0.2    # TPU tongue to slot wall
PAD_HEAD_MM = (4.4, 1.1)  # grip head (width, height above the wing surface)
PAD_END_CLEARANCE_MM = 0.4  # pad is this much shorter than the band
RING_BASE_MM = 1.2        # TPU ring base thickness under the wings
RING_STRETCH_MM = 0.2     # TPU ring inner radius is this much smaller than the carrier core
GROOVE_DEPTH_MM, GROOVE_WIDTH_MM = 0.7, 8.0  # retaining groove in the ABS carrier; the TPU ring has a matching rib
RIB_WIDTH_CLEARANCE_MM = 0.4


@dataclass(frozen=True)
class Variant:
    """One spring ring: wing count, thickness, the radius its tips reach when free, the gap under a tip (the travel), the wing material, and whether the tips carry pad slots."""

    n: int
    t: float
    r_free: float
    gap: float = 2.0
    material: str = "ABS"
    pad: bool = False
    tip_gap_deg: float = 4.0   # angular gap between a wing's tip and the next wing's root (TIP_GAP_DEG unless set)
    tip_t: float = 0.0         # thickness at the tip when the wing tapers linearly from its root thickness to here (0 = the older shape: constant t after the root taper); the tip's outer edge stays at r_free

    wave_amp: float = 0.0      # corrugation: the arm's centreline swings this far (mm) in and out of the straight spiral, ridges running along the axle, to stiffen the arm against axle-direction bending (0 = plain arm)
    wave_len: float = 4.0      # mm along the arm per wave
    fillet: float = 0.0        # radius (mm) of the fillet where the arm meets the dovetail head (0 = sharp corner)

    @property
    def tapered(self) -> bool:
        return self.tip_t > 0.0

    @property
    def tip_t_eff(self) -> float:
        return self.tip_t if self.tapered else self.t

    @property
    def gap_eff(self) -> float:
        """Free gap under the tip: a thinner tip raises its inner surface so the outer edge still reaches r_free."""
        return self.gap + self.t - self.tip_t_eff

    @property
    def name(self) -> str:
        tail = " pad slots" if self.pad else ""
        mat = " TPU" if self.material == "TPU" else ""
        return f"Adaptive{mat} N{self.n} t{self.t:g} R{self.r_free:g} G{self.gap:g}{tail}"

    @property
    def r_core(self) -> float:
        return self.r_free - self.t - self.gap


A = Variant(5, 2.0, 27.0)
B = Variant(5, 2.0, 28.0)
C = Variant(4, 2.2, 28.05, gap=3.0)
D = Variant(5, 2.0, 26.0, pad=True)
CARRIER_CORE_R_MM = 21.0   # large enough that the groove bottom keeps a 1 mm wall around the half's M3 pockets (they reach 19.1 mm from the axle)
F = Variant(8, 3.2, 28.0, gap=28.0 - 3.2 - (CARRIER_CORE_R_MM + RING_BASE_MM), material="TPU")   # TPU wings sit on a base ring 1.2 mm thick around the carrier core
VARIANTS = (A, B, C, F)   # the ones the pages show with predictions


def predict(v: Variant) -> dict[str, float]:
    """Beam theory for one ring: free length, travel at the material's design strain, stiffness of the ring against a sideways shift, and the shift under SPOOL_LOAD_N (capped at the available travel)."""
    if v.tapered:
        return predict_profile(v)
    modulus, strain = MATERIALS[v.material]
    r_mid = v.r_core + v.t
    arc = float(r_mid * np.radians(360.0 / v.n - v.tip_gap_deg))
    free = float(arc * (1 - U_FIX) * (1 - (ROOT_EMBED_MM + MIN_GAP_MM) / (v.gap + ROOT_EMBED_MM)))
    inertia = BAND_MM * v.t**3 / 12.0
    k = float(3.0 * modulus * inertia / free**3)
    travel = float(strain * free**2 / (1.5 * v.t))
    geometric = v.gap - MIN_GAP_MM
    ring_k = float(v.n * k / 2.0)
    use = min(travel, geometric)
    return {"free_len": free, "k_beam": k, "travel": use, "geom": geometric, "ring_k": ring_k, "sag": SPOOL_LOAD_N / ring_k, "bore_min": 2 * (v.r_free - use), "bore_max": 2 * v.r_free}


def thickness_profile(v: Variant, u: np.ndarray) -> np.ndarray:
    """Wing thickness at u (0 root, 1 tip): the root taper then constant t, or for a tapered variant a straight line from the root thickness to tip_t."""
    root = v.t * (1 + ROOT_TAPER)
    if v.tapered:
        return root + (v.tip_t - root) * u
    return v.t * (1 + ROOT_TAPER * np.clip(1 - u / U_TAPER, 0.0, 1.0))


def predict_profile(v: Variant, points: int = 400) -> dict[str, float]:
    """Cantilever beam with the actual thickness profile, integrated numerically (tip load, strain limit at the worst section): same outputs as predict, for tapered wings and to compare shapes fairly. The wing is fixed at U_FIX and its free length runs to the tip."""
    modulus, strain = MATERIALS[v.material]
    r_mid = v.r_core + v.t
    arc = float(r_mid * np.radians(360.0 / v.n - v.tip_gap_deg))
    free = float(arc * (1 - U_FIX) * (1 - (ROOT_EMBED_MM + MIN_GAP_MM) / (v.gap_eff + ROOT_EMBED_MM)))
    s = np.linspace(0.0, free, points)
    h = thickness_profile(v, U_FIX + (1 - U_FIX) * s / free)
    inertia = BAND_MM * h**3 / 12.0
    arm = free - s
    flex = float(np.trapz(arm**2 / (modulus * inertia), s))   # tip deflection per newton
    k = 1.0 / flex
    strain_per_n = float(np.max(arm * (h / 2.0) / (modulus * inertia)))
    travel = strain / strain_per_n * flex
    geometric = v.gap_eff - MIN_GAP_MM
    ring_k = float(v.n * k / 2.0)
    use = min(travel, geometric)
    return {"free_len": free, "k_beam": k, "travel": use, "geom": geometric, "ring_k": ring_k, "sag": SPOOL_LOAD_N / ring_k, "bore_min": 2 * (v.r_free - use), "bore_max": 2 * v.r_free, "tip_t": v.tip_t_eff}


def spring_curves(v: Variant, theta0: float) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """(outer, inner) flank points of one wing from root to tip, root at angle theta0 (radians), running counter-clockwise."""
    span = np.radians(360.0 / v.n - v.tip_gap_deg)
    u = np.linspace(0.0, 1.0, POINTS)
    inner = np.where(u < U_FIX, v.r_core - ROOT_EMBED_MM, v.r_core - ROOT_EMBED_MM + (v.gap_eff + ROOT_EMBED_MM) * (u - U_FIX) / (1 - U_FIX))
    thick = thickness_profile(v, u)
    theta = theta0 + u * float(span)
    outer = inner + thick
    outer[-1] = inner[-1] + v.tip_t_eff  # the tip is the plain tip thickness, so its outer edge is R_FREE
    if v.wave_amp > 0.0:   # corrugation: shift both flanks together (the thickness is kept), fading to zero at the root and the tip so the embed and the free radius stay as designed
        mid = 0.5 * (inner + outer)
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(mid), mid[:-1] * np.diff(theta)))])
        clear = np.clip((inner - v.r_core) / (2 * v.wave_amp), 0.0, 1.0)   # no wave where the arm is within two swings of the core circle: a wave dipping into the circle pinches the section or splits it
        shift = v.wave_amp * np.sin(2 * np.pi * arc / v.wave_len) * np.sin(np.pi * u) * clear
        inner, outer = inner + shift, outer + shift
    o = [(float(r * np.cos(a)), float(r * np.sin(a))) for r, a in zip(outer, theta)]
    i = [(float(r * np.cos(a)), float(r * np.sin(a))) for r, a in zip(inner, theta)]
    return o, i


def spring_polygon(v: Variant, theta0: float) -> Polygon:
    outer, inner = spring_curves(v, theta0)
    return Polygon(outer + inner[::-1])


def slot_frame(v: Variant, theta0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centre point, tangent, outward normal) on the wing's outer surface where the pad slot sits."""
    outer, _ = spring_curves(v, theta0)
    k = int(round(SLOT_U * (POINTS - 1)))
    p = np.array(outer[k])
    tau = np.array(outer[k + 1]) - np.array(outer[k - 1])
    tau /= np.linalg.norm(tau)
    n = np.array([-tau[1], tau[0]])
    if float(n @ (p / np.linalg.norm(p))) < 0:
        n = -n
    return p, tau, n


def local_polygon(points: list[tuple[float, float]], frame: tuple[np.ndarray, np.ndarray, np.ndarray]) -> Polygon:
    p, tau, n = frame
    return Polygon([tuple(p + x * tau + y * n) for x, y in points])


def slot_points(extra_top: float = 0.3) -> list[tuple[float, float]]:
    return [(-SLOT_OPENING_MM / 2, extra_top), (SLOT_OPENING_MM / 2, extra_top), (SLOT_FLOOR_MM / 2, -SLOT_DEPTH_MM), (-SLOT_FLOOR_MM / 2, -SLOT_DEPTH_MM)]


def tongue_section() -> Polygon:
    """The pad's dovetail tongue, PAD_CLEARANCE_MM inside the slot walls, in the slot's frame (x along the wing, y outward)."""
    c = PAD_CLEARANCE_MM
    return Polygon([(-SLOT_OPENING_MM / 2 + c, 0.0), (SLOT_OPENING_MM / 2 - c, 0.0), (SLOT_FLOOR_MM / 2 - c, -SLOT_DEPTH_MM + c), (-SLOT_FLOOR_MM / 2 + c, -SLOT_DEPTH_MM + c)])


def pad_section() -> Polygon:
    """The TPU pad's cross-section: the tongue plus a rounded grip head on top."""
    head = box(-PAD_HEAD_MM[0] / 2, -0.001, PAD_HEAD_MM[0] / 2, PAD_HEAD_MM[1]).buffer(-0.35).buffer(0.35)
    return cast(Polygon, unary_union([tongue_section(), head]))


def wing_with_slot(v: Variant, theta0: float) -> Polygon:
    wing = spring_polygon(v, theta0)
    if not v.pad:
        return wing
    slot = local_polygon(slot_points(), slot_frame(v, theta0))
    cut = wing.difference(slot)
    if cut.geom_type != "Polygon":
        raise RuntimeError(f"{v.name}: the pad slot splits the wing")
    # the pad's tongue must lie wholly inside the slot, so it can slide in
    tongue = local_polygon(list(tongue_section().exterior.coords)[:-1], slot_frame(v, theta0))
    if tongue.difference(slot).area > 1e-6:
        raise RuntimeError("pad tongue outside the slot")
    return cast(Polygon, cut)


def coupon_section(v: Variant) -> Polygon:
    """The ring's cross-section: hollow core with N wings (each with a pad slot when v.pad)."""
    core = Point(0, 0).buffer(v.r_core, 180).difference(Point(0, 0).buffer(HUB_R_MM, 90))
    wings = [wing_with_slot(v, i * 2 * np.pi / v.n) for i in range(v.n)]
    shape = unary_union([core, *wings])
    if not shape.is_valid or shape.geom_type != "Polygon":
        raise RuntimeError(f"{v.name}: section is {shape.geom_type}, valid={shape.is_valid}")
    return cast(Polygon, shape)


def label_block(mesh: trimesh.Trimesh, lines: list[tuple[str, float]], prefer: tuple[float, float]) -> trimesh.Trimesh:
    """Cut the label into the bed face (z = 0) at the free spot nearest `prefer`."""
    pieces = lt.block(lines)
    spot = lt.find_spot(lt.face_area(mesh, 0.05), pieces, prefer, LABEL_CLEARANCE_MM, step=0.5)
    if spot is None:
        raise RuntimeError("no room for the label")
    placed = [Polygon(np.array(p.exterior.coords) + np.array(spot)) for p in pieces]
    return lt.cut(mesh, placed, 0.0, LABEL_DEPTH_MM)


def coupon_mesh(v: Variant) -> trimesh.Trimesh:
    mesh = trimesh.creation.extrude_polygon(coupon_section(v), BAND_MM)
    top = f"N{v.n} t{v.t:g}" + (" pad" if v.pad else "")
    return label_block(mesh, [(top, 3.2), (f"R{v.r_free:g} G{v.gap:g}", 3.2)], (0.0, -HUB_R_MM - 4.0))


def carrier_mesh() -> trimesh.Trimesh:
    """The bare ABS core the TPU wing ring is stretched onto: a plain ring with a retaining groove."""
    ring = Point(0, 0).buffer(CARRIER_CORE_R_MM, 180).difference(Point(0, 0).buffer(HUB_R_MM, 90))
    mesh = trimesh.creation.extrude_polygon(ring, BAND_MM)
    groove = Point(0, 0).buffer(CARRIER_CORE_R_MM + 1.0, 180).difference(Point(0, 0).buffer(CARRIER_CORE_R_MM - GROOVE_DEPTH_MM, 180))
    cutter = trimesh.creation.extrude_polygon(groove, GROOVE_WIDTH_MM).apply_translation([0, 0, (BAND_MM - GROOVE_WIDTH_MM) / 2])
    mesh = trimesh.boolean.difference([mesh, cutter], engine="manifold")
    return label_block(mesh, [("TPU C", 3.2)], (0.0, -HUB_R_MM - 3.5))


def tpu_ring_mesh() -> trimesh.Trimesh:
    """The TPU part: a base ring with a retaining rib and N wings."""
    inner_r = CARRIER_CORE_R_MM - RING_STRETCH_MM
    base = Point(0, 0).buffer(CARRIER_CORE_R_MM + RING_BASE_MM, 180).difference(Point(0, 0).buffer(inner_r, 180))
    wings = [spring_polygon(F, i * 2 * np.pi / F.n) for i in range(F.n)]
    shape = unary_union([base, *wings])
    if shape.geom_type != "Polygon" or not shape.is_valid:
        raise RuntimeError("TPU ring section is not one valid polygon")
    mesh = trimesh.creation.extrude_polygon(cast(Polygon, shape), BAND_MM)
    rib = Point(0, 0).buffer(inner_r + 0.2, 180).difference(Point(0, 0).buffer(CARRIER_CORE_R_MM - GROOVE_DEPTH_MM, 180))
    rib_w = GROOVE_WIDTH_MM - RIB_WIDTH_CLEARANCE_MM
    rib_mesh = trimesh.creation.extrude_polygon(rib, rib_w).apply_translation([0, 0, (BAND_MM - rib_w) / 2])
    return trimesh.boolean.union([mesh, rib_mesh], engine="manifold")


def pad_mesh(copies: int) -> trimesh.Trimesh:
    """TPU pads lying on their grip heads (the dovetail overhangs least that way), side by side."""
    section = pad_section()
    length = BAND_MM - PAD_END_CLEARANCE_MM
    one = trimesh.creation.extrude_polygon(section, length)            # profile in x, y; length along z
    one.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))  # length -> -y, profile y -> z
    zlo = float(one.bounds[0, 2])
    bottom = one.section(plane_origin=[0, 0, zlo + 0.05], plane_normal=[0, 0, 1])
    width_bottom = float(np.ptp(np.array(bottom.vertices)[:, 0])) if bottom is not None else 0.0
    if width_bottom < PAD_HEAD_MM[0] - 1.0:   # the narrow end is down: turn it over so the grip head is on the bed
        one.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
    one.apply_translation(-one.bounds[0])
    pitch = float(one.extents[0]) + 3.0
    pads = [one.copy().apply_translation([i * pitch, 0, 0]) for i in range(copies)]
    return trimesh.util.concatenate(pads)


def gauge_mesh(inner_d: float) -> trimesh.Trimesh:
    ring = Point(0, 0).buffer(inner_d / 2 + GAUGE_WALL_MM, 180).difference(Point(0, 0).buffer(inner_d / 2, 180))
    mesh = trimesh.creation.extrude_polygon(ring, GAUGE_HEIGHT_MM)
    return label_block(mesh, [(f"{inner_d:.1f}", 4.2)], (0.0, -(inner_d / 2 + GAUGE_WALL_MM / 2)))


def write_parts() -> list[tuple[str, trimesh.Trimesh, int | None]]:
    """Write every STL and return the ABS parts that go into the project."""
    for stale in [*HERE.glob("Adaptive N*.stl"), *HERE.glob("Adaptive TPU*.stl")]:   # earlier variants' files, never the Pulley Half
        stale.unlink()
    parts: list[tuple[str, trimesh.Trimesh, int | None]] = []
    for v in (A, B, C, D):
        mesh = coupon_mesh(v)
        mesh.export(HERE / f"{v.name}.stl")
        p = predict(v)
        log.info("%s: bores %.1f to %.1f, travel %.2f mm, ring k %.1f N/mm, sag %.2f mm at %.0f N (one ring), %.1f cm3", v.name, p["bore_min"], p["bore_max"], p["travel"], p["ring_k"], p["sag"], SPOOL_LOAD_N, mesh.volume / 1000)
        parts.append((v.name, mesh, bp.WEDGE_WALLS))
    carrier = carrier_mesh()
    carrier.export(HERE / "Adaptive TPU carrier (ABS).stl")
    parts.append(("Adaptive TPU carrier (ABS)", carrier, bp.WEDGE_WALLS))
    for d in GAUGE_IDS_MM:
        mesh = gauge_mesh(d)
        name = f"Bore Gauge {d:.1f}"
        mesh.export(HERE / f"{name}.stl")
        parts.append((name, mesh, None))
    ring = tpu_ring_mesh()
    ring.export(HERE / "Adaptive TPU wing ring (print in TPU).stl")
    pads = pad_mesh(5)
    pads.export(HERE / "Adaptive TPU pads x5 (print in TPU).stl")
    p = predict(F)
    log.info("TPU wing ring N%d t%g: bores %.1f to %.1f, travel %.2f mm, ring k %.1f N/mm, sag %.2f mm (one ring); closed=%s %.1f cm3", F.n, F.t, p["bore_min"], p["bore_max"], p["travel"], p["ring_k"], p["sag"], ring.is_watertight, ring.volume / 1000)
    log.info("TPU pads x5: closed=%s, %.0f x %.0f x %.1f mm", pads.is_watertight, *pads.extents)
    for name, m in (("TPU ring", ring), ("TPU pads", pads)):
        if not m.is_watertight or m.volume <= 0:
            raise RuntimeError(f"{name} is not a closed solid")
    return parts


def main() -> None:
    parts = write_parts()
    for name, mesh, _ in parts:
        if not mesh.is_watertight or mesh.volume <= 0:
            raise RuntimeError(f"{name} is not a closed solid")
    kinds = {name: bp.Kind(name, bp.centred(mesh), i + 1, float(mesh.extents[2]), walls) for i, (name, mesh, walls) in enumerate(parts)}
    ordered = sorted(kinds.values(), key=lambda k: -float(k.mesh.extents[1]))
    usable = bp.BED_MM - 2 * EDGE_MM
    rows: list[list[bp.Kind]] = [[]]
    width = 0.0
    for k in ordered:
        w = float(k.mesh.extents[0])
        if rows[-1] and width + GAP_MM + w > usable:
            rows.append([])
            width = 0.0
        width += (GAP_MM if rows[-1] else 0.0) + w
        rows[-1].append(k)
    placed: list[tuple[bp.Kind, float, float]] = []
    y = 0.0
    for row in rows:
        depth = max(float(k.mesh.extents[1]) for k in row)
        x = 0.0
        for k in row:
            w = float(k.mesh.extents[0])
            placed.append((k, x + w / 2, y + depth / 2))
            x += w + GAP_MM
        y += depth + GAP_MM
    left = min(px - k.mesh.extents[0] / 2 for k, px, _ in placed)
    right = max(px + k.mesh.extents[0] / 2 for k, px, _ in placed)
    low = min(py - k.mesh.extents[1] / 2 for k, _, py in placed)
    high = max(py + k.mesh.extents[1] / 2 for k, _, py in placed)
    if right - left > usable or high - low > usable:
        raise RuntimeError("the parts do not fit one plate")
    dx, dy = (bp.BED_MM - (right - left)) / 2 - left, (bp.BED_MM - (high - low)) / 2 - low
    oid = len(kinds) + 1
    instances = [bp.Instance(oid + i, k, 0, px + dx, py + dy) for i, (k, px, py) in enumerate(placed)]
    profile = {**bp.profile_values(), **bst.project_fast_values()}
    bp.write_project(OUTPUT, kinds, instances, [f"Adaptive lip {profile['layer_height']} mm"], profile)
    log.info("layout %.0f x %.0f mm centred, %d rows", right - left, high - low, len(rows))


if __name__ == "__main__":
    main()
