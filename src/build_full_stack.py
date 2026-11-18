"""Assemble the whole Free Roll stack with the clip-on pulleys, final wedges and final stands, check it, and render it.

Two rollers (the paired layout of build_assembly): stand, wedge, half, half, wedge, stand, twice. Each roller's two halves are the clip-on core twice (the second turned half a turn) with three whole t2.0 petals spanning both; wedges are the 24-ridge
"Wedge - Single Sided - Final.stl", stands are "Stand Pi X4 - Final.stl". Run build_clipon_petal.py, build_final_wedge.py and build_final_stand.py first.
Checks: the distance between each roller's stand inner faces must be TARGET_STACK_MM within STACK_TOLERANCE_MM, and no pair of parts may overlap by more than build_assembly's tolerance.
Check limit: the stack length comes from build_assembly's bearing seat, which is read from the wedge model, so it checks the models against the 72.60 target, not a printed stack (shrink and
bearing seating are not covered); collisions are model volume only, not the flex of the petals.
"""
from __future__ import annotations

import logging
import os
from typing import cast

import numpy as np
import trimesh

import build_adaptive_half_step as hs
import build_assembly as ba
import build_clipon_geometry as cg
import build_clipon_petal as cp
import build_final_stand as bs

logging.basicConfig(level=os.environ.get("AXEL_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_full_stack")

TARGET_STACK_MM = 72.60   # between a roller's stand inner faces: the roller body length the stands rest on
STACK_TOLERANCE_MM = 0.03
PETAL = "soft ribbed"   # soft taper (reaches a 50 mm bore in ABS, up to the 62 mm free radius) with the 0.6 mm / 5 mm wave: 21 percent less push force than plain soft
FINAL_STAND = ba.HERE / "Stand Pi X4 - Final.stl"
FINAL_WEDGE = ba.HERE / "Wedge - Single Sided - Final.stl"


def load(name: str) -> trimesh.Trimesh:
    return cast(trimesh.Trimesh, trimesh.load(ba.HERE / f"{name}.stl"))


def build() -> tuple[list[ba.Part], float]:
    """The assembly parts and the groove-centre z."""
    core = load(hs.NAMES["clip"])
    shift = -float(core.extents[2]) / 2   # build_assembly wants the pulley centred on z = 0
    core.apply_translation([0, 0, shift])
    petal = load(cp.petal_name(PETAL)).apply_translation([0, 0, shift - cp.PETAL_LENGTH_MM])   # a whole petal: its middle (the butt plane) at the half's butt face, reaching into the other half
    turned = {f"petal {k}": petal.copy().apply_transform(trimesh.transformations.rotation_matrix(np.radians(120 * k), [0, 0, 1])) for k in range(3)}
    wedge = ba.rotate_x(cast(trimesh.Trimesh, trimesh.load(FINAL_WEDGE)), 180)   # final wedge prints stub up; the assembly frame has the stub towards -z
    stand = cast(trimesh.Trimesh, trimesh.load(FINAL_STAND)).apply_translation([0, 0, -bs.CRADLE_RAISE_MM])   # the assembly is built round the source groove centre; the final stand's cradle is CRADLE_RAISE_MM higher on its plate, so the plate sits that much lower
    return ba.build_parts(new_stub=True, paired=True, pulley_mesh=core, bridge_extras=turned, align_halves=True, bearing_float=cg.BEARING_FLOAT_MM, end_slack=cg.END_SLACK_MM, stand_mesh=stand, wedge_mesh=wedge)


def stack_lengths(parts: list[ba.Part]) -> list[float]:
    """Distance between the inner faces of each roller's two stands."""
    stands = sorted((p for p in parts if p.name.startswith("stand")), key=lambda p: p.mesh.bounds[0, 1])
    return [float(stands[i + 1].mesh.bounds[0, 1] - stands[i].mesh.bounds[1, 1]) for i in (0, 2)]


def main() -> None:
    parts, _ = build()
    lengths = stack_lengths(parts)
    log.info("stand inner faces: roller 1 %.3f mm, roller 2 %.3f mm (target %.2f +- %.2f)", *lengths, TARGET_STACK_MM, STACK_TOLERANCE_MM)
    if any(abs(v - TARGET_STACK_MM) > STACK_TOLERANCE_MM for v in lengths):
        raise RuntimeError(f"stack length {lengths} is not {TARGET_STACK_MM} +- {STACK_TOLERANCE_MM}: change cg.HALF_WIDTH_MM")
    collisions = ba.check_collisions(parts)
    for a, b, volume in collisions:
        log.error("COLLISION %s / %s: %.1f mm3", a, b, volume)
    log.info("collision check over %d parts: %d overlapping pair(s) thicker than %.3f mm", len(parts), len(collisions), ba.INTERFERENCE_DEPTH_MM)
    if collisions:
        raise RuntimeError("the stack has collisions")
    ba.render_iso(parts, ba.HERE / "full_stack_iso.png")
    ba.RENDER_SECTION = ba.HERE / "full_stack_section.png"
    stands = [p for p in parts if p.name.startswith("stand")]
    extent = (min(p.mesh.bounds[0, 1] for p in stands), max(p.mesh.bounds[1, 1] for p in stands))
    ba.render_section(parts, extent[1] - extent[0], extent)


if __name__ == "__main__":
    main()
