"""2D geometry of the clip-on petal design: three separate spiral petals, each one piece spanning both halves of the pulley, that slide axially into dovetail slots in the two cores (not printed in place).

The spiral arm is the same wing as build_adaptive_coupon (same formulas, one ring of three, free radius R_FREE_MM, so a bore up to twice that), clipped to outside the core circle;
its root carries a dovetail head that sits in a slot of the core. Slots are centred 90, 210 and 330 degrees, 30 degrees away from the M3 pockets (0 and 180 degrees), which reach 19.1 mm from the axle.
Check limit: this is geometry only; the spring numbers are the beam model with a guessed modulus and the fit clearance is a first guess for ABS on this printer.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

import build_adaptive_coupon as ac

CLIP_TIP_GAP_DEG = 16.0       # was 4: with 4 the tips ran past the next petal's root when squeezed toward a 50 mm bore, so the petals jammed before gripping the core
R_FREE_MM = 31.0              # free radius of the petals: a 62 mm reel is the largest that touches; a 60 mm reel is gripped with about 1 mm of preload per side (was 30, i.e. a 60 mm reel only touched)
CORE_RADIUS_MM = 21.9         # the clip core's plain cylinder; the same for every petal thickness so the slots never change
CLIP = ac.Variant(3, 2.6, R_FREE_MM, gap=R_FREE_MM - 2.6 - CORE_RADIUS_MM, tip_gap_deg=CLIP_TIP_GAP_DEG)
THIN = ac.Variant(3, 2.0, CLIP.r_free, gap=CLIP.r_free - 2.0 - CLIP.r_core, tip_gap_deg=CLIP_TIP_GAP_DEG)   # the thinner test petal: same free radius and same core, so the same slots
THICK = ac.Variant(3, 3.0, CLIP.r_free, gap=CLIP.r_free - 3.0 - CLIP.r_core, tip_gap_deg=CLIP_TIP_GAP_DEG)   # the stiffer test petal, same core
TAPER_ROOT_MM = 2.4           # the tapered petal's thickness at the root; it thins in a straight line to TAPER_TIP_MM so the bending strain spreads along the arm and the petal is more pliable (3.2 to 1.6 was tried and came out stiffer than t2.0)
TAPER_TIP_MM = 1.4            # thinner than about 1.2 mm would not print as solid walls


WAVE_AMP_MM = 0.6             # corrugation swing of the ribbed petal; 0.5 mm keeps the lost travel under the tip well below the 6 to 7 mm of travel
WAVE_LEN_MM = 5.0   # swing 0.6 mm / wave 5 mm won the corrugation sweep (petal_fea_sweep): 20 % softer, across-layer peak 17 % lower; 0.6 mm / 8 mm would split the cross-section
DOT_R_MM = 0.7                 # orientation dots (user's scheme): the seam is the radial line where the petal's arm root ends on the clockwise side of its slot; a core dot sits on the core's surface DOT_GAP_MM beyond it, a petal dot on the petal's top DOT_GAP_MM inside it, so the two sit the same distance either side of the seam. Each core has one dot on its counter-clockwise side and two on its clockwise side; a petal has two dots on its half that goes in the first core and one on the other
DOT_GAP_MM = 0.8               # edge of a dot to the seam (two 0.42 mm perimeters of wall beside the dot)
DOT_PITCH_MM = 1.6             # the two dots of a pair sit this far either side of the half's middle, along the axle
DOT_DEPTH_MM = 0.5
BEARING_RECESS_CLEARANCE_MM = 0.0     # radial slip clearance of the 608's recess: back to the plain 22.00 mm (user is fine with 11.0 mm radius); the mouth chamfer below does the starting work
BEARING_MOUTH_CHAMFER_MM = 0.5          # 45 degree lead-in at the recess mouth so the bearing starts square
FILLET_MM = 1.0              # arm-to-head fillet radius of the fillet and ribbed petals


def tapered_variant(root_mm: float, tip_mm: float, wave_amp: float = 0.0, fillet: float = 0.0) -> ac.Variant:
    """A petal that thins in a straight line from root_mm at the root to tip_mm at the tip; same core, same free radius, same slots as THIN."""
    t = root_mm / (1 + ac.ROOT_TAPER)   # the Variant's nominal t, whose root is t * (1 + ROOT_TAPER)
    return ac.Variant(3, t, CLIP.r_free, gap=CLIP.r_free - t - CLIP.r_core, tip_gap_deg=CLIP_TIP_GAP_DEG, tip_t=tip_mm, wave_amp=wave_amp, wave_len=WAVE_LEN_MM, fillet=fillet)


TAPERED = tapered_variant(TAPER_ROOT_MM, TAPER_TIP_MM)
TAPER_SOFT = (2.0, 1.2)    # (root, tip) mm: the softest, reaches a 50 mm bore in ABS, about 8 N per petal at a 52 mm bore
TAPER_FIRM = (2.8, 1.6)    # sags about 35 % less than TAPERED, reaches 52.8 mm in ABS, about 20 N per petal at a 52 mm bore
HEAD_MOUTH_HALF_MM = 2.0      # dovetail half width at the core surface
HEAD_FLOOR_HALF_MM = 3.2       # and at the slot floor (the undercut that holds the petal radially)
HEAD_DEPTH_MM = 2.4            # slot depth below the core surface
HEAD_ROOT_OFFSET_MM = 2.5      # head centre, measured along the arm from the wing's root start
HEAD_OVERLAP_MM = 0.3          # the head reaches this far above the core surface, into the wing root, so the two fuse
HEAD_TAPER_MM = 0.0            # the dovetail head is this much narrower per side at each end of the petal (the outer faces, which go into the slots first) than at the butt plane in the middle: a keystone taper, so the petal wedges as it seats
SLOT_CLEARANCE_MOUTH_MM = 0.10  # slot wall to head, per side, at the butt face (easy start); loosened from 0.05 after the first full-pulley print pushed together too stiffly
SLOT_CLEARANCE_END_MM = 0.00    # the same at the end wall (was -0.10: needed a lot of force from both ends and would not close completely): a small interference (negative), so the last stretch of the slide presses tight; clearance varies linearly between (0.05 / -0.10 is slot 4 with the nominal head: on the printed coupons stub 1 / slot 4 was good and stub 2 / slot 4 almost clicked into place)
SLOT_END_WALL_MM = 1.5         # the slot is open at the butt face and stops this far from the outer face (1.5 mm, up from 1.0 and the user's 0.5: the user pushes the petal into the channel against this wall to seat it, and it cracked at 1.0; before that back to 1.0 mm from the user's 0.5 mm: the closed end of the channel cracked on the test pulley, and this wall is a roof bridged over the channel when the core prints butt face down; the first test pulley also had 1.0 mm, so the wall alone is not the whole fix, see bp.CORE_* print overrides)
HALF_WIDTH_MM = 35.46          # clip-on half width (was 34.96 before the bearing went flush with the core mouth, which moved each mouth 0.5 mm towards the wedge): stand inner faces 72.60 apart = 2 x (pulley mouth 2.741 from the nose, set by the thinned wedge and the bearing seat + this) + END_SLACK_MM - stand thickness 4.004, checked by build_full_stack.py (it raises if the stack length drifts)
BEARING_FLOAT_MM = 0.0          # how much deeper the recess is than the bearing is wide: 0 makes the recess floor a shelf exactly one bearing width down, so a pressed-in 608 lands flush with the core mouth (was 0.2, which let it sink 0.2 mm below flush); the cost is that the outer race can meet the shelf while the wedge collar holds the inner race, i.e. a possible axial preload on a ceramic bearing
END_SLACK_MM = 0.2             # axial play of the two butted halves between the wedges, at the two ends together
CHEEK_MAX_MM = 5.0             # thickest spool cheek (the cardboard ring that touches the pulley), per the user: no more than 0.5 cm
PETAL_END_CLEARANCE_MM = 0.2   # petal is this much shorter than the slot
PETAL_LENGTH_MM = HALF_WIDTH_MM - SLOT_END_WALL_MM - PETAL_END_CLEARANCE_MM   # the part of a petal in one half's slot
ARM_OVERHANG_MM = SLOT_END_WALL_MM   # the arm lies outside the core, so it runs on past the head by the end wall's thickness at each end: the thick wall costs no spring length, and the petal's reach to the outer faces is the same as it was with a 0.0 mm wall
DOT_AXIAL_MM = PETAL_LENGTH_MM / 2   # the dots sit at the middle of each half of a petal, and at the same distance from the butt face on a core, so they line up when the petal is seated
DOT_ANGLE_DEG = float(np.degrees((HEAD_ROOT_OFFSET_MM + DOT_GAP_MM + DOT_R_MM) / CORE_RADIUS_MM))   # a core dot's angular offset from its slot axis, either side: the arm root's end (the seam) is HEAD_ROOT_OFFSET_MM from the axis
PETAL_SPAN_MM = 2 * PETAL_LENGTH_MM                                           # a whole petal: one piece across the butt plane, in the slots of both halves (the cores are identical, one turned half a turn, so their slots line up)
MARK_RADIUS_MM = 0.7           # the dimple on a test part's top end (fit dial-in parts carry 1 to 6 to say which size they are)
MARK_DEPTH_MM = 0.5
MARK_ARM_U = 0.15              # where along the arm (0 root, 1 tip) the dimple sits on a test petal: still on the thick root
SLOT_CENTRES_DEG = (90.0, 210.0, 330.0)
CORE_M3_HOLES = False         # the clip core carries no M3 screw holes or pockets (the dovetails and petals hold the halves together; user dropped the screws; saves about 13 minutes of print per core); True restores them
M3_POCKET_REACH_MM = 19.1      # farthest a screw pocket reaches from the axle
M3_POCKET_ANGLES_DEG = (0.0, 180.0)


def root_angle(i: int) -> float:
    """Angle (radians) at which petal i's wing starts, chosen so its head centre sits on SLOT_CENTRES_DEG[i]."""
    return float(np.radians(SLOT_CENTRES_DEG[i]) - HEAD_ROOT_OFFSET_MM / CLIP.r_core)


def head_grow_mm(z: float, clearance: bool, extra: float = 0.0, taper: float = HEAD_TAPER_MM, mouth: float = SLOT_CLEARANCE_MOUTH_MM, end: float = SLOT_CLEARANCE_END_MM) -> float:
    """Outward offset of the head (clearance False) or of its slot (True) at distance z from the butt face: the head narrows by HEAD_TAPER_MM over the petal length; the slot adds the clearance, which runs linearly from the mouth value to the end value."""
    t = z / PETAL_LENGTH_MM
    head = -taper * t
    return extra + head + (mouth + (end - mouth) * t if clearance else 0.0)


def head_polygon(i: int, z: float = 0.0, clearance: bool = False, extra: float = 0.0, taper: float = HEAD_TAPER_MM, mouth: float = SLOT_CLEARANCE_MOUTH_MM, end: float = SLOT_CLEARANCE_END_MM) -> Polygon:
    """The dovetail head of petal i in the plane at distance z from the butt face (clearance True gives the slot cross-section there). The top stays at the wing root so head and wing always fuse; the flanks and floor move out by the grow."""
    g = head_grow_mm(z, clearance, extra, taper, mouth, end)   # extra: a per-part offset for the fit dial-in coupon (0 for the real parts)
    k = 1.0 / np.cos(np.arctan((HEAD_FLOOR_HALF_MM - HEAD_MOUTH_HALF_MM) / (HEAD_DEPTH_MM + HEAD_OVERLAP_MM)))   # an angled flank moves sideways by g / cos(flank angle)
    phi = np.radians(SLOT_CENTRES_DEG[i])
    e_r = np.array([np.cos(phi), np.sin(phi)])
    e_t = np.array([-np.sin(phi), np.cos(phi)])
    r0 = CLIP.r_core
    local = [(r0 + HEAD_OVERLAP_MM, -(HEAD_MOUTH_HALF_MM + g * k)), (r0 + HEAD_OVERLAP_MM, HEAD_MOUTH_HALF_MM + g * k), (r0 - HEAD_DEPTH_MM - g, HEAD_FLOOR_HALF_MM + g * k), (r0 - HEAD_DEPTH_MM - g, -(HEAD_FLOOR_HALF_MM + g * k))]
    return Polygon([tuple(r * e_r + t * e_t) for r, t in local])


def slot_polygon(i: int, z: float = 0.0, extra: float = 0.0, mouth: float = SLOT_CLEARANCE_MOUTH_MM, end: float = SLOT_CLEARANCE_END_MM) -> Polygon:
    return head_polygon(i, z, clearance=True, extra=extra, mouth=mouth, end=end)


def dot_heights(count: int, middle: float = DOT_AXIAL_MM) -> tuple[float, ...]:
    """Axial positions of a group of dots about `middle`: one dot, or a pair DOT_PITCH_MM either side."""
    return (middle,) if count == 1 else (middle - DOT_PITCH_MM, middle + DOT_PITCH_MM)


def wing_polygon(i: int, v: ac.Variant = CLIP) -> Polygon:
    """The spiral arm outside the core circle."""
    wing = ac.spring_polygon(v, root_angle(i))
    return wing.difference(Point(0, 0).buffer(CLIP.r_core, 180))


def petal_arm_polygon(i: int, v: ac.Variant = CLIP, z: float = 0.0) -> Polygon:
    """Arm plus head as one polygon: the section of the petal at distance z from the butt face."""
    shape = unary_union([wing_polygon(i, v), head_polygon(i, z)])
    if shape.geom_type != "Polygon" or not shape.is_valid:
        raise RuntimeError(f"petal {i} arm section is not one valid polygon")
    return shape


def slot_clear_of_pockets() -> float:
    """Smallest distance between a slot cutter and an M3 pocket envelope (circle of 3.5 mm radius at 15.5 mm), mm; must be well above zero."""
    from shapely.geometry import Point
    worst = 1e9
    for deg in M3_POCKET_ANGLES_DEG:
        c = Point(15.5 * np.cos(np.radians(deg)), 15.5 * np.sin(np.radians(deg))).buffer(3.5, 90)
        worst = min(worst, min(slot_polygon(i).distance(c) for i in range(len(SLOT_CENTRES_DEG))))
    return float(worst)


MARK_ARM_STEP_U = 0.07         # spacing of extra dimples along the arm of a test petal


def mark_position(i: int, v: ac.Variant = CLIP, j: int = 0) -> tuple[float, float]:
    """Centre of dimple j (0 first) on the top end, midway through the arm's thickness at MARK_ARM_U + j * MARK_ARM_STEP_U."""
    outer, inner = ac.spring_curves(v, root_angle(i))
    k = int(round((MARK_ARM_U + j * MARK_ARM_STEP_U) * (ac.POINTS - 1)))
    return (float((outer[k][0] + inner[k][0]) / 2), float((outer[k][1] + inner[k][1]) / 2))
