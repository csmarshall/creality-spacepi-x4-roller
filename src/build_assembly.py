"""Assemble the Free Roll set around the flat stand, check it for collisions, measure its width and render it.

Along the axle (y) the set is three identical flat stands with a pulley pair between each pair of stands:

    S1 [single wedge, pulley P1] [pulley P2, double wedge at S2, pulley P3] [pulley P4, single wedge] S3

Every wedge nose sits in its stand's V notch at the stand mid-plane with the wedge axis on the groove centre. Each pulley's bearing
pocket faces its wedge: the 608 bearing rides on the wedge axle stub, pushed up against the wedge until they touch, and the pulley is
seated so the bearing's far face is on the pocket floor. Positions come from the part meshes (pocket from the pulley STL, the stub
from the wedge profile); only the 608's own dimensions are constants, because they are a standard and the files cannot tell us them.
"""
from __future__ import annotations

import logging
import os
import zipfile
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np
import trimesh
from lxml import etree
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Polygon, box

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import build_double_wedge as dw  # noqa: E402
import build_flat_stand as fs  # noqa: E402

logging.basicConfig(level=os.environ.get("AXEL_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_assembly")

HERE = Path(__file__).resolve().parent
PULLEY_MODEL = "3D/Objects/Free Roll - X4 - OG Pulley.stl_11.model"
FLAT_STAND = HERE / "Stand Pi X4 - Flat Both Sides.stl"
DOUBLE_WEDGE = HERE / "Wedge - Double Sided.stl"
RENDER_ISO = HERE / "assembly_iso.png"
RENDER_SECTION = HERE / "assembly_section.png"
CORE_NS = "{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}"
BEARING_OD_MM, BEARING_WIDTH_MM = 22.0, dw.BEARING_WIDTH_MM  # ISO 608: a standard part, not something the files can tell us
MEASURED_BEARING_BORE_MM = 7.95  # measured by the user on both the ceramic and the steel bearing
POCKET_WALL_TOLERANCE_MM = 0.5  # how close to the bearing radius a pulley vertex must be to count as pocket wall (the pocket floor's outer corner is the deepest such vertex): the clip core's recess is the bearing plus a slip clearance well under this
CONTACT_AREA_MM2 = 1e-3  # the bearing and wedge count as touching below this overlap in the section
INTERFERENCE_DEPTH_MM = 0.05  # mean overlap thickness below which a touch fit is not a collision; a chosen design tolerance (finer than FDM accuracy), not derivable from the meshes
PULLEY_GAP_MM = 0.0  # between facing pulleys; a design choice, set to 0 to report the minimum width
ROLLER_SPACING_MM = 20.0  # paired layout only: clear space from one roller's second stand to the next roller's first stand; for display, the real container sets it
AXIAL_STEP_MM = 0.05
AXIS_TOLERANCE_MM = 1e-3  # profile points closer to the axis than this lie on it (the section leaves them a hair off)
MAX_STUB_SLOPE = 0.2  # radius change per mm of axial travel below which the profile counts as the stub (a chamfer or shoulder is steeper)
SECTIONS = 180
IMAGE_SIZE = (1800, 900)
VIEW_YAW_DEG, VIEW_PITCH_DEG = -35.0, 28.0
LIGHT = np.array([-0.4, -0.6, 0.7]) / np.linalg.norm([-0.4, -0.6, 0.7])


@dataclass
class Part:
    """A placed part: name, mesh in assembly coordinates, display colour."""

    name: str
    mesh: trimesh.Trimesh
    colour: tuple[int, int, int]


def load_pulley() -> trimesh.Trimesh:
    """The pulley from the 3MF: axis along z, the bearing pocket on the +z face."""
    with zipfile.ZipFile(dw.SOURCE) as z:
        root = etree.fromstring(z.read(PULLEY_MODEL))
    verts = np.array([[float(v.get(k)) for k in "xyz"] for v in root.iter(f"{CORE_NS}vertex")])
    tris = np.array([[int(t.get(k)) for k in ("v1", "v2", "v3")] for t in root.iter(f"{CORE_NS}triangle")])
    return trimesh.Trimesh(verts, tris, process=True)


def pocket_depth(pulley: trimesh.Trimesh) -> float:
    """Depth of the bearing pocket from the +z face, read from the vertices on the pocket wall (radius = bearing OD / 2)."""
    r = np.hypot(pulley.vertices[:, 0], pulley.vertices[:, 1])
    wall = pulley.vertices[np.abs(r - BEARING_OD_MM / 2) < POCKET_WALL_TOLERANCE_MM, 2]
    if len(wall) == 0:
        raise RuntimeError("no bearing pocket wall found in the pulley")
    return float(pulley.bounds[1, 2] - wall.min())


def stub_extent(radial: Polygon, nose_y: float) -> tuple[float, float, float, float]:
    """Widest radius, narrowest radius, axial length of the wedge axle stub and its distance from the stub tip to where the straight run starts: the longest axial run over which the profile is near-parallel to the axis (a slight draft is fine)."""
    low, high = radial.bounds[1], radial.bounds[3]
    axial = np.arange(low + AXIAL_STEP_MM, high - AXIAL_STEP_MM, AXIAL_STEP_MM)
    radii = np.array([radial.intersection(LineString([(0, a), (radial.bounds[2] + 1, a)])).bounds[2] for a in axial])
    best_run, best_radius, narrowest, run_start, best_start = 0, 0.0, 0.0, 0, 0
    for i in range(1, len(radii) + 1):
        if i == len(radii) or abs(radii[i] - radii[i - 1]) > MAX_STUB_SLOPE * AXIAL_STEP_MM:
            if i - run_start > best_run:
                best_run, best_radius, narrowest, best_start = i - run_start, float(radii[run_start:i].max()), float(radii[run_start:i].min()), run_start
            run_start = i
    length = best_run * AXIAL_STEP_MM
    tip_offset = float(axial[best_start] - low)
    log.info("axle stub: radius %.3f mm at its widest, %.3f mm at its narrowest, over %.2f mm", best_radius, narrowest, length)
    return best_radius, narrowest, length, tip_offset


def axle_radius(radial: Polygon, nose_y: float) -> float:
    """Largest radius of the wedge axle stub."""
    return stub_extent(radial, nose_y)[0]


def bearing_seat(radial: Polygon, nose_y: float, stub_radius: float) -> float:
    """Axial position (nose = 0, stub towards -axial) of the bearing's wedge-side face when it is pushed up against the wedge."""
    axial_wedge = Polygon([(r, y - nose_y) for r, y in radial.exterior.coords])
    low, high = axial_wedge.bounds[1], axial_wedge.bounds[3]
    for _ in range(60):  # bisect the face position: the overlap with the wedge grows monotonically towards the nose
        mid = (low + high) / 2
        ring = box(stub_radius, mid - BEARING_WIDTH_MM, BEARING_OD_MM / 2, mid)
        low, high = (mid, high) if axial_wedge.intersection(ring).area < CONTACT_AREA_MM2 else (low, mid)
    return low


def single_wedge(radial: Polygon, nose_y: float) -> trimesh.Trimesh:
    """The original one-sided wedge revolved from its section: axis along z, nose at z = 0, axle tip towards -z."""
    ring = np.array(radial.simplify(dw.PROFILE_TOLERANCE_MM).exterior.coords)[:-1]
    on_axis = ring[:, 0] < AXIS_TOLERANCE_MM
    start = next(i for i in range(len(ring)) if on_axis[i - 1] and not on_axis[i])
    chain = np.roll(ring, -start, axis=0)
    chain = chain[chain[:, 0] >= AXIS_TOLERANCE_MM]
    chain = chain - [0, nose_y]
    line = np.r_[[[0.0, chain[0, 1]]], chain, [[0.0, chain[-1, 1]]]]
    solid = trimesh.creation.revolve(line, sections=SECTIONS)
    if solid.volume < 0:  # the profile's winding decides which way the revolved normals face
        solid.invert()
    return solid


def rotate_x(mesh: trimesh.Trimesh, degrees: float) -> trimesh.Trimesh:
    """A copy rotated about x (a proper rotation, so nothing is mirrored)."""
    return mesh.copy().apply_transform(trimesh.transformations.rotation_matrix(np.radians(degrees), [1, 0, 0]))


def moved(mesh: trimesh.Trimesh, y: float, z: float) -> trimesh.Trimesh:
    """A copy translated to y, z."""
    return mesh.copy().apply_translation([0, y, z])


def rotate_z(mesh: trimesh.Trimesh, degrees: float) -> trimesh.Trimesh:
    """A copy rotated about z, the pulley's own axis (a proper rotation)."""
    return mesh.copy().apply_transform(trimesh.transformations.rotation_matrix(np.radians(degrees), [0, 0, 1]))


def build_parts(new_stub: bool = False, paired: bool = False, pulley_mesh: trimesh.Trimesh | None = None, bridge_extras: dict[str, trimesh.Trimesh] | None = None, align_halves: bool = False, bearing_float: float = 0.0, end_slack: float = 0.0,
                stand_mesh: trimesh.Trimesh | None = None, wedge_mesh: trimesh.Trimesh | None = None) -> tuple[list[Part], float]:
    """All parts in assembly coordinates, and the groove-centre z they are centred on.

    new_stub swaps in the uniform refitted stub (build_new_stub.py) and gives the bearing the measured bore. paired builds the layout of two separate
    rollers, stand / wedge / pulley / pulley / wedge / stand twice with single wedges only (four stands, no double wedge); the default is three stands sharing a middle one.
    pulley_mesh, stand_mesh and wedge_mesh replace the original pulley, stand and single wedge (pulley centred on z = 0 with the bearing pocket on +z, wedge in this module's frame: nose at z = 0, stub to -z);
    bridge_extras are further parts in the first pulley's frame (the clip-on petals) that span both halves of each roller, named by the dict key: each is placed once per roller, on the sign > 0 half, and reaches across the butt plane into the other half.
    align_halves turns the second half of each roller 180 degrees about its own axle first: together with the end-for-end spin that is a half turn about the radial axis through slot 1, which keeps the slot and M3 pocket angles and lines the two halves' slots up.
    """
    stand = stand_mesh if stand_mesh is not None else trimesh.load(FLAT_STAND)
    source_stand = fs.load_stand()
    _, cz, _ = fs.groove_circle(fs.flat_face_outline(source_stand))
    wedge_source = dw.load_wedge()
    radial, nose_y = dw.half_profile(wedge_source)
    if new_stub:
        radial, _ = dw.final_profile(radial, nose_y)
        stub_r = dw.STUB_DIAMETER_MM / 2  # the bearing face stops at the shaft's root, the bore is a hair smaller (interference)
    else:
        stub_r = axle_radius(radial, nose_y)
    seat = bearing_seat(radial, nose_y, stub_r)
    pulley = pulley_mesh if pulley_mesh is not None else load_pulley()
    depth = pocket_depth(pulley)
    face_axial = seat - BEARING_WIDTH_MM + (depth - bearing_float)  # pulley mouth: the bearing's far face would be on the pocket floor, but the recess is bearing_float deeper than that, so it floats
    reach = -face_axial  # pulley mouth distance from the nose plane
    thickness = float(np.ptp(pulley.vertices[:, 2]))
    log.info("axle radius %.3f mm, bearing seat at axial %.3f, pocket depth %.3f mm, pulley mouth %.3f mm from the nose, pulley %.2f mm", stub_r, seat, depth, reach, thickness)
    bearing = trimesh.creation.annulus(r_min=MEASURED_BEARING_BORE_MM / 2 if new_stub else stub_r, r_max=BEARING_OD_MM / 2, height=BEARING_WIDTH_MM, sections=SECTIONS)
    one_sided, two_sided = (wedge_mesh if wedge_mesh is not None else single_wedge(radial, nose_y)), trimesh.load(dw.NEW_STUB_DOUBLE if new_stub else DOUBLE_WEDGE)
    span = 2 * (reach + thickness) + PULLEY_GAP_MM + end_slack  # stand-to-stand spacing: two pulleys back to back, with end_slack of play at the two ends together
    stands_y = [0.0, span, span + ROLLER_SPACING_MM, 2 * span + ROLLER_SPACING_MM] if paired else [0.0, span, 2 * span]
    parts = [Part(f"stand {i + 1}", moved(stand, y, 0.0), (190, 196, 205)) for i, y in enumerate(stands_y)]

    def add_side(label: str, wedge_y: float, sign: int) -> None:
        """A pulley and its bearing on the stub pointing along sign * y from the wedge at wedge_y."""
        spin = 90 if sign > 0 else -90  # rotate the native +z (pocket face, wedge side) onto -sign * y
        pulley_y = wedge_y + sign * (reach + end_slack / 2 + thickness / 2)
        turn = (lambda m: rotate_z(m, 180)) if align_halves and sign < 0 else (lambda m: m)
        parts.append(Part(f"pulley {label}", moved(rotate_x(turn(pulley), spin), pulley_y, cz), (70, 130, 200)))
        for extra_name, extra in (bridge_extras or {}).items() if sign > 0 else ():
            parts.append(Part(f"{extra_name} {label}", moved(rotate_x(extra, spin), pulley_y, cz), (200, 80, 60)))
        bearing_y = wedge_y - sign * (seat - BEARING_WIDTH_MM / 2)
        parts.append(Part(f"bearing {label}", moved(rotate_x(bearing, spin), bearing_y, cz), (215, 170, 60)))

    parts.append(Part("single wedge 1", moved(rotate_x(one_sided, 90), stands_y[0], cz), (230, 120, 60)))
    add_side("P1", stands_y[0], +1)
    add_side("P2", stands_y[1], -1)
    if paired:
        parts.append(Part("single wedge 2", moved(rotate_x(one_sided, -90), stands_y[1], cz), (230, 120, 60)))
        parts.append(Part("single wedge 3", moved(rotate_x(one_sided, 90), stands_y[2], cz), (230, 120, 60)))
        add_side("P3", stands_y[2], +1)
        add_side("P4", stands_y[3], -1)
        parts.append(Part("single wedge 4", moved(rotate_x(one_sided, -90), stands_y[3], cz), (230, 120, 60)))
        return parts, cz
    parts.append(Part("double wedge", moved(rotate_x(two_sided, -90), stands_y[1], cz), (230, 120, 60)))
    add_side("P3", stands_y[1], +1)
    add_side("P4", stands_y[2], -1)
    parts.append(Part("single wedge 2", moved(rotate_x(one_sided, -90), stands_y[2], cz), (230, 120, 60)))
    return parts, cz


def check_collisions(parts: list[Part]) -> list[tuple[str, str, float]]:
    """Overlap volume for every pair of parts whose boxes overlap and whose overlap is thicker than the tangent-fit noise.

    A thin film over a tangent surface has volume = area x depth, and its closed mesh has about twice that area (both skins), so the mean
    depth is 2 x volume / area; the check scales with the size of the contact instead of using one absolute volume.
    """
    found = []
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            if not (np.all(a.mesh.bounds[0] < b.mesh.bounds[1]) and np.all(b.mesh.bounds[0] < a.mesh.bounds[1])):
                continue
            common = trimesh.boolean.intersection([a.mesh, b.mesh], engine="manifold")
            if not len(common.faces):
                continue
            volume = float(common.volume)
            mean_depth = 2 * volume / float(common.area)
            log.debug("overlap %s / %s: %.3f mm3, mean depth %.4f mm", a.name, b.name, volume, mean_depth)
            if mean_depth > INTERFERENCE_DEPTH_MM:
                found.append((a.name, b.name, volume))
    return found


def render_iso(parts: list[Part], output: Path = RENDER_ISO, background: tuple[int, int, int] = (250, 250, 250), upright: bool = False,
               yaw_deg: float = VIEW_YAW_DEG, pitch_deg: float = VIEW_PITCH_DEG) -> Callable[[np.ndarray], np.ndarray]:
    """Flat-shaded orthographic painter's-algorithm render of the parts; returns the projection from assembly coordinates to image pixels."""
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    if upright:  # axle (y) across the image, z up, x towards the viewer; yaw turns about the vertical, pitch tips the top towards the viewer
        about_vertical = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        about_across = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        view = about_across @ about_vertical @ np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    else:
        about_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        about_x = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        # the assembly's long axis is y and "up" is z, so look at it with y running across the image
        view = about_x @ about_z @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    tris, depths, shades, colours = [], [], [], []
    for part in parts:
        v = part.mesh.vertices @ view.T
        f = part.mesh.faces
        n = part.mesh.face_normals @ view.T
        tris.append(v[f][:, :, :2]); depths.append(v[f][:, :, 2].mean(axis=1)); shades.append(0.35 + 0.65 * np.clip(n @ LIGHT, 0, 1))
        colours.append(np.tile(part.colour, (len(f), 1)))
    tris, depths, shades, colours = map(np.concatenate, (tris, depths, shades, colours))
    lo, hi = tris.reshape(-1, 2).min(axis=0), tris.reshape(-1, 2).max(axis=0)
    scale = min((IMAGE_SIZE[0] - 80) / (hi[0] - lo[0]), (IMAGE_SIZE[1] - 80) / (hi[1] - lo[1]))
    pix = (tris - lo) * scale + 40
    pix[:, :, 1] = IMAGE_SIZE[1] - pix[:, :, 1]
    image = Image.new("RGB", IMAGE_SIZE, background)
    draw = ImageDraw.Draw(image)
    for k in np.argsort(depths):  # far to near
        draw.polygon([tuple(p) for p in pix[k]], fill=tuple(int(c) for c in colours[k] * shades[k]))
    image.save(output)
    log.info("wrote %s", output.name)

    def project(points: np.ndarray) -> np.ndarray:
        """Pixel coordinates (x right, y down) of assembly-coordinate points in this render."""
        flat = (np.atleast_2d(points) @ view.T)[:, :2]
        pixels = (flat - lo) * scale + 40
        pixels[:, 1] = IMAGE_SIZE[1] - pixels[:, 1]
        return pixels

    return project


def render_section(parts: list[Part], width_mm: float, stand_extent: tuple[float, float]) -> None:
    """Section through the axles (x = 0), parts coloured, with the overall width dimensioned."""
    fig, ax = plt.subplots(figsize=(18, 6))
    for part in parts:
        section = part.mesh.section(plane_origin=[0, 0, 0], plane_normal=[1, 0, 0])
        if section is None:
            continue
        shape = Polygon()
        for loop in section.discrete:
            shape = shape.symmetric_difference(Polygon(np.asarray(loop)[:, 1:3]))
        for poly in getattr(shape, "geoms", [shape]):
            if poly.is_empty:
                continue
            ax.fill(*poly.exterior.xy, color=np.array(part.colour) / 255, alpha=0.85, lw=0)
            for hole in poly.interiors:
                ax.fill(*hole.xy, color="white", lw=0)
    z_low = min(p.mesh.bounds[0, 2] for p in parts) - 8
    ax.annotate("", xy=(stand_extent[0], z_low), xytext=(stand_extent[1], z_low), arrowprops=dict(arrowstyle="<->"))
    ax.text(sum(stand_extent) / 2, z_low - 6, f"overall width {width_mm:.1f} mm", ha="center")
    ax.set_aspect("equal"); ax.set_xlabel("y along the axle (mm)"); ax.set_ylabel("z (mm)"); ax.grid(alpha=.25)
    ax.set_title("Section through the axles (x = 0): stands grey, pulleys blue, wedges orange, 608 bearings gold")
    fig.tight_layout(); fig.savefig(RENDER_SECTION, dpi=100)
    log.info("wrote %s", RENDER_SECTION.name)


def main() -> None:
    """Build the assembly, report collisions and width, and render."""
    parts, _ = build_parts()
    stands = [p for p in parts if p.name.startswith("stand")]
    extent = (min(p.mesh.bounds[0, 1] for p in stands), max(p.mesh.bounds[1, 1] for p in stands))
    width = extent[1] - extent[0]
    everything = (min(p.mesh.bounds[0, 1] for p in parts), max(p.mesh.bounds[1, 1] for p in parts))
    log.info("overall width %.2f mm between the outer stand faces; all parts span %.2f mm", width, everything[1] - everything[0])
    collisions = check_collisions(parts)
    for a, b, volume in collisions:
        log.error("COLLISION %s / %s: %.1f mm3", a, b, volume)
    log.info("collision check: %d overlapping pair(s) thicker than %.3f mm", len(collisions), INTERFERENCE_DEPTH_MM)
    render_section(parts, width, extent)
    render_iso(parts)


if __name__ == "__main__":
    main()
