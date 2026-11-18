"""Write "Free Roll - Stub Ladder (fast inner).3mf": five real single wedges that differ only in the modelled stub diameter, printed as the real parts are.

The wedge stub prints 7.7 to 7.85 mm from an 8.00 mm model, and the earlier peg coupons did not predict that, so the ladder uses the whole wedge (flat back
down, stub up, same walls, same fast profile). Whichever wedge's stub slides into the bearing without play is a usable part and sets STUB_DIAMETER_MM in
build_double_wedge. Objects are placed in rising diameter, left to right along the front row first, then the row behind it, and each carries its diameter in its name.
Check limit: the stub diameter is read back from each written mesh and the placement is verified, which says the files are right, not that they print to size.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import trimesh

import build_assembly as ba
import build_double_wedge as dw
import build_new_stub as bn
import build_project_3mf as bp
import label_text as lt
import build_speed_tests as bst

logging.basicConfig(level=os.environ.get("AXEL_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_stub_ladder")

OUTPUT = bp.HERE / "Free Roll - Stub Ladder (fast inner).3mf"
LADDER_START_MM = 8.05  # the coupon pegs print about 0.10 under the model (largest width) and the 8.00 wedge stub 0.15 to 0.3 under, so 8.05 to 8.25 brackets both estimates for the 7.95 mm bore
LADDER_STEP_MM = 0.05
LADDER_STEPS = 5
COLUMNS = 3
LABEL_SIZE_MM = 7.0  # lettering height on the back face
LABEL_LINE_GAP_MM = 1.5
LABEL_DEPTH_MM = 0.5  # four layers at 0.12 mm
LABEL_MAX_RADIUS_MM = 15.0  # the back face is flat to radius 16.2 mm


def deboss_lines(mesh: trimesh.Trimesh, lines: list[tuple[str, float]]) -> trimesh.Trimesh:
    """Cut stacked lines of (text, height in mm) into the flat back of a wedge printed back down, the block centred on that face and readable with the part turned over;
    it must stay inside LABEL_MAX_RADIUS_MM."""
    centre = mesh.bounds.mean(axis=0)
    total = sum(h for _, h in lines) + LABEL_LINE_GAP_MM * (len(lines) - 1)
    pieces, top = [], total / 2
    for text, height in lines:
        pieces += lt.glyph(text, height, (float(centre[0]), float(centre[1]) + top - height / 2))
        top -= height + LABEL_LINE_GAP_MM
    if lt.max_radius(pieces, (float(centre[0]), float(centre[1])))[1] > LABEL_MAX_RADIUS_MM:
        raise RuntimeError(f"label {[t for t, _ in lines]} does not fit on the back face")
    return lt.cut(mesh, pieces, float(mesh.bounds[0, 2]), LABEL_DEPTH_MM)


def deboss_text(mesh: trimesh.Trimesh, text: str) -> trimesh.Trimesh:
    """One line of the standard size on the flat back of a wedge."""
    return deboss_lines(mesh, [(text, LABEL_SIZE_MM)])


def wedge_with_stub(diameter: float, label: bool = False, final: bool = False) -> trimesh.Trimesh:
    """The single wedge with a uniform stub of the given diameter, flat back down and stub up, as build_kinds prints it; label=True debosses the stub size on the back; final=True gives the production section (flush with the stand, flat axle side, wider collar) instead of the source wedge's."""
    dw.STUB_DIAMETER_MM = diameter  # refit_stub reads this module constant
    source = dw.load_wedge()
    radial, nose_y = dw.half_profile(source)
    new_radial, _ = dw.final_profile(radial, nose_y) if final else dw.refit_stub(radial, nose_y)
    single = ba.single_wedge(new_radial, nose_y)
    bn.check_shaft(single, float(single.bounds[0, 2]), +1.0, f"single wedge {diameter:.2f}")
    printed = single.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    return deboss_text(printed, f"{diameter:.2f}") if label else printed


def main() -> None:
    """Build the ladder wedges, place them on one plate with the fast profile, and write the project."""
    original = dw.STUB_DIAMETER_MM
    try:
        diameters = [round(LADDER_START_MM + i * LADDER_STEP_MM, 3) for i in range(LADDER_STEPS)]
        kinds = {}
        for mesh_id, d in enumerate(diameters, start=1):
            kinds[f"stub {d:.2f}"] = bp.Kind(f"Wedge - Single - Stub {d:.2f}", bp.centred(wedge_with_stub(d, label=True)), mesh_id, 0.0, bp.WEDGE_WALLS)
    finally:
        dw.STUB_DIAMETER_MM = original
    for kind in kinds.values():
        kind.height = float(kind.mesh.extents[2])
    instances = bp.placements(kinds, [(0, [(name, 1) for name in kinds], COLUMNS)])
    profile = {**bp.profile_values(), **bst.project_fast_values()}
    bp.write_project(OUTPUT, kinds, instances, [f"Stub ladder {profile['layer_height']} mm"], profile)
    log.info("stub diameters %s mm, rising left to right along the front row (3), then the back row (2); each object is named with its diameter", diameters)


if __name__ == "__main__":
    main()
