"""Write "Free Roll - Fit Profile.3mf": an Orca project with the coupon print profile embedded, for two SpacePi X4 sets, one layer height per plate.

Plate 1 "Pulleys 0.12 mm": 7 OG pulleys (recess up). Plate 2 "Wedges 0.12 mm": 5 single wedges (flat back down) and 2 double wedges, the new 8.00 mm stub.
Plate 3 "Concave Pulleys 0.12 mm": 8 concave pulleys (recess up).
The process settings are the ones Orca wrote into the recess-ladder gcode that fit (read from the printer), applied over the settings of the user's FlatStand project,
so printer, filament and everything not listed stay as they were. Wedges get 6 walls as a per-object override; every object also carries its layer height explicitly.
Structure follows Orca's own 3MF as saved in the FlatStand project (mesh objects in 3D/Objects, instances as components, plates in model_settings.config).
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.parse
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import trimesh
from lxml import etree

import build_assembly as ba
import build_double_wedge as dw

logging.basicConfig(level=os.environ.get("AXEL_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_project_3mf")

HERE = Path(__file__).resolve().parent
BASE_3MF = HERE / "SpacePi-X4+FreeRoll-FlatStand.3mf"
OUTPUT = HERE / "Free Roll - Fit Profile.3mf"
PULLEY_ENTRY = "3D/Objects/Free Roll - X4 - OG Pulley.stl_33.model"
CONCAVE_ENTRY = "3D/Objects/Free Roll - X4 - Concave Pulley.stl_29.model"
RECESS_RADIUS_MM = ba.BEARING_OD_MM / 2  # the recess wall the fit coupon validated (22.00 mm)
RECESS_RADIUS_TOLERANCE_MM = 0.002
PRINTER_HOST = os.environ.get("PRINTER_HOST", "")   # host of a Moonraker printer that holds a sliced coupon gcode to read the print profile from; only the .3mf plate builders need it
PROFILE_GCODE = "Fit Test - Recess Ladder_ABS_1h16m.gcode"  # the coupon that seated the bearing; its settings block is the profile
TAIL_BYTES = 400_000
TIMEOUT_S = 30
PROFILE_KEYS = (
    "print_settings_id", "layer_height", "initial_layer_print_height", "bottom_shell_layers", "top_shell_layers", "top_shell_thickness", "wall_loops",
    "default_acceleration", "outer_wall_acceleration", "outer_wall_speed", "inner_wall_speed", "sparse_infill_speed", "internal_solid_infill_speed",
    "top_surface_speed", "gap_infill_speed", "overhang_1_4_speed", "overhang_2_4_speed", "overhang_3_4_speed", "overhang_4_4_speed",
    "sparse_infill_pattern", "sparse_infill_density", "internal_solid_infill_pattern", "min_width_top_surface", "seam_gap", "enable_prime_tower",
    "support_top_z_distance", "support_bottom_z_distance", "support_threshold_angle",
)
PETAL_BRIM_WIDTH_MM = 5.0   # outer brim on the petals ("wings"): at least 5 mm, several 4 mm ones came off the bed mid-print
BRIM_WIDTH_MM = 4.0   # outer brim on the pulley pieces (cores and petals): one of two printed cores warped badly off the bed without one
BRIM_NAME_PREFIXES = ("Adaptive Pulley", "Adaptive Clip Petal")  # the objects that get the brim; everything else prints as before
STAND_NAME_PREFIX = "Stand Pi X4"
STAND_BRIM_WIDTH_MM = 8.0   # outer brim on the stands: the flat 138 x 92 x 4 mm plates bowed and rubbed the spool, so they get a wider brim than the pulley pieces
CORE_NAME_PREFIX = "Adaptive Pulley"
CORE_WALLS = 4                        # the dovetail lips beside the channel are about 1.2 mm: 4 walls of 0.42 mm make them solid
CORE_TOP_SHELL_LAYERS = 14            # the slot's closed end is a roof bridged over the channel when the core prints butt face down; 14 layers of 0.12 mm (1.7 mm) make the 1.5 mm end wall solid with margin
PETAL_NAME_PREFIX = "Adaptive Clip Petal"
PETAL_INNER_WALL_SPEED = 60          # mm/s, as slow as the outer wall: the fast-inner profile prints inner walls at 300 mm/s with 10000 acceleration, and a petal is nearly all inner walls, so layer bonding suffers
PETAL_INNER_WALL_ACCEL = 2000        # mm/s2, matches the outer wall
WEDGE_WALLS = 6  # the stub is 8 mm across: 6 walls of 0.42 mm leave a 2 to 3 mm core, so the shaft is nearly solid
PULLEYS, CONCAVES, SINGLES, DOUBLES = 7, 8, 5, 2  # two SpacePi X4 sets minus the first article already printed and kept (1 OG pulley, 1 single wedge); 5 singles + the kept one = 6 so either stand layout can be built
BED_MM = 270.0
PLATE_GAP_FRACTION = 0.2  # Orca spaces plates by a fifth of the bed; read off the FlatStand project (plate 2 sits 324 mm right of plate 1)
OBJECT_GAP_MM = 10.0
MIN_GAP_MM = 5.0  # the verification fails below this footprint gap
KEEP_UPSTREAM_METADATA = ("Application", "CopyRight", "Snapmaker_Orca:3mfVersion")   # CopyRight is the upstream remix credit; every other upstream key (designer, profile and user ids, description) describes Alberto Maciel's MakerWorld listing, not this file
PROJECT_METADATA = {
    "Title": "Creality Space Pi X4 spool roller",
    "Designer": "Charles Marshall",
    "Description": "Free-rolling 608-bearing spool roller for the Creality Space Pi X4 dryer, adapted from SPACE PI - FREE ROLL MOD by Alberto Maciel. Source, STEP and build code: https://github.com/csmarshall/creality-spacepi-x4-roller",
    "License": "BY-NC-SA",
    "Origin": "remix",
}
NAMESPACE = uuid.UUID("6f1c0c2e-5a53-4b6e-9a77-3c1f4b0d9e01")
CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"


@dataclass
class Kind:
    """A mesh shared by several instances."""

    name: str
    mesh: trimesh.Trimesh  # centred on x, y and z
    mesh_id: int
    height: float
    walls: int | None
    extruder: int = 1   # the filament slot (1 to 4) the object prints with


@dataclass
class Instance:
    """One placed object: its top-level id, the kind it uses and the plate (0-based) and centre it sits at."""

    object_id: int
    kind: Kind
    plate: int
    x: float
    y: float


def centred(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """A copy with its bounding box centred on the origin."""
    return mesh.copy().apply_translation(-mesh.bounds.mean(axis=0))


def load_pulley(entry: str) -> trimesh.Trimesh:
    """A pulley mesh exactly as stored in the FlatStand project, turned if needed so its bearing recess faces up; the recess wall radius is checked against the validated 22.00 mm."""
    with zipfile.ZipFile(BASE_3MF) as z:
        root = etree.fromstring(z.read(entry))
    verts = np.array([[float(v.get(k)) for k in "xyz"] for v in root.iter(f"{{{CORE}}}vertex")])
    tris = np.array([[int(t.get(k)) for k in ("v1", "v2", "v3")] for t in root.iter(f"{{{CORE}}}triangle")])
    mesh = trimesh.Trimesh(verts, tris, process=False)
    radius = np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1])
    wall = mesh.vertices[np.abs(radius - RECESS_RADIUS_MM) < RECESS_RADIUS_TOLERANCE_MM, 2]
    if len(wall) == 0:
        raise RuntimeError(f"{entry}: no recess wall at radius {RECESS_RADIUS_MM} mm")
    up = wall.mean() > mesh.bounds.mean(axis=0)[2]
    log.info("%s: recess wall radius %.3f mm, %d wall vertices, recess faces %s", entry.split("/")[-1], RECESS_RADIUS_MM, len(wall), "up" if up else "down (turned over)")
    return mesh if up else mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))


@lru_cache(maxsize=1)
def profile_values() -> dict[str, str]:
    """PROFILE_KEYS read from the settings block at the end of the coupon gcode on the printer (a read-only range request)."""
    if not PRINTER_HOST:
        raise RuntimeError("set PRINTER_HOST to a Moonraker host holding the profile gcode; only the .3mf plate builders need it, the STL/STEP builders do not")
    request = urllib.request.Request(f"http://{PRINTER_HOST}/server/files/gcodes/{urllib.parse.quote(PROFILE_GCODE)}", headers={"Range": f"bytes=-{TAIL_BYTES}"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        text = response.read().decode("utf8", "ignore")
    found = {}
    for line in text.splitlines():
        if line.startswith("; ") and " = " in line:
            key, value = line[2:].split(" = ", 1)
            if key.strip() in PROFILE_KEYS:
                found[key.strip()] = value.strip()
    missing = [k for k in PROFILE_KEYS if k not in found]
    if missing:
        raise RuntimeError(f"settings missing from {PROFILE_GCODE}: {missing}")
    return found


def project_settings(profile: dict[str, str]) -> str:
    """The FlatStand project's settings with the profile applied; list-valued entries (standard / high-flow pairs) take the value in every slot."""
    with zipfile.ZipFile(BASE_3MF) as z:
        settings = json.loads(z.read("Metadata/project_settings.config"))
    for key, value in profile.items():
        current = settings.get(key)
        settings[key] = [value] * len(current) if isinstance(current, list) else value
        log.debug("%s: %r -> %r", key, current, settings[key])
    # Orca keeps only the options listed in different_settings_to_system (process slot first) as changes on top of the named preset and resets everything else to the preset's values on load,
    # so every option that differs from the preset the coupon was printed with must be listed. The coupon profile is that preset's own values.
    preset = profile_values()
    modified = ["layer_height"] + sorted(k for k, v in profile.items() if k != "layer_height" and v != preset[k])
    settings["different_settings_to_system"][0] = ";".join(modified)
    log.info("options marked as changed from the preset: %s", ";".join(modified))
    return json.dumps(settings, indent=4)


def mesh_xml(mesh: trimesh.Trimesh, object_id: int) -> str:
    """A sub-model file holding one mesh object."""
    verts = "\n".join(f'     <vertex x="{x:.6f}" y="{y:.6f}" z="{z:.6f}"/>' for x, y, z in mesh.vertices)
    tris = "\n".join(f'     <triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in mesh.faces)
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" xmlns="{CORE}" xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p">\n'
            f' <metadata name="Snapmaker_Orca:3mfVersion">1</metadata>\n <resources>\n  <object id="{object_id}" p:UUID="{uuid.uuid5(NAMESPACE, f"mesh{object_id}")}" type="model">\n   <mesh>\n    <vertices>\n{verts}\n    </vertices>\n'
            f'    <triangles>\n{tris}\n    </triangles>\n   </mesh>\n  </object>\n </resources>\n <build/>\n</model>\n')


def placements(kinds: dict[str, Kind], plan: list | None = None) -> list[Instance]:
    """Instances on a grid centred in each plate; plan is (plate, [(kind, count)], columns) per plate, by default pulleys on plate 0, wedges on 1, concave pulleys on 2."""
    plan = plan or [(0, [("pulley", PULLEYS)], 4), (1, [("single", SINGLES), ("double", DOUBLES)], 3), (2, [("concave", CONCAVES)], 4)]
    instances, next_id = [], len(kinds) + 1  # the mesh objects take ids 1 to len(kinds)
    for plate, groups, columns in plan:
        order = [name for name, count in groups for _ in range(count)]
        pitch = max(kinds[n].mesh.extents[0] for n in order) + OBJECT_GAP_MM
        rows = -(-len(order) // columns)
        for k, name in enumerate(order):
            col, row = k % columns, k // columns
            instances.append(Instance(next_id, kinds[name], plate, BED_MM / 2 + (col - (columns - 1) / 2) * pitch, BED_MM / 2 + (row - (rows - 1) / 2) * pitch))
            next_id += 1
    return instances


def plate_columns(count: int) -> int:
    """Orca's plate grid width for `count` plates: sqrt rounded to nearest, up when the root is above the rounded value (2 for 2 to 4 plates, 3 for 5 to 9)."""
    root = count ** 0.5
    return max(1, round(root) + (1 if root > round(root) else 0))


def plate_origin(plate: int, count: int) -> tuple[float, float]:
    """Where Orca puts a plate's (0, 0) when the project holds `count` plates: rows of plate_columns(count), spaced by a fifth of the bed."""
    stride = BED_MM * (1 + PLATE_GAP_FRACTION)
    columns = plate_columns(count)
    return (plate % columns) * stride, -(plate // columns) * stride


def model_xml(instances: list[Instance], paths: dict[int, str], header: str) -> str:
    """3D/3dmodel.model: one component object per instance, one build item each."""
    objects, items = [], []
    plate_count = 1 + max(i.plate for i in instances)
    for inst in instances:
        objects.append(f'  <object id="{inst.object_id}" p:UUID="{uuid.uuid5(NAMESPACE, f"obj{inst.object_id}")}" type="model">\n   <components>\n'
                       f'    <component p:path="{escape(paths[inst.kind.mesh_id])}" objectid="{inst.kind.mesh_id}" p:UUID="{uuid.uuid5(NAMESPACE, f"cmp{inst.object_id}")}" transform="1 0 0 0 1 0 0 0 1 0 0 0"/>\n   </components>\n  </object>')
        ox, oy = plate_origin(inst.plate, plate_count)
        items.append(f'  <item objectid="{inst.object_id}" p:UUID="{uuid.uuid5(NAMESPACE, f"item{inst.object_id}")}" transform="1 0 0 0 1 0 0 0 1 {ox + inst.x:.6f} {oy + inst.y:.6f} {inst.kind.height / 2:.6f}" printable="1"/>')
    return (header + " <resources>\n" + "\n".join(objects) + f'\n </resources>\n <build p:UUID="{uuid.uuid5(NAMESPACE, "build")}">\n' + "\n".join(items) + "\n </build>\n</model>\n")


def settings_xml(instances: list[Instance], plate_names: list[str], layer_height: str) -> str:
    """Metadata/model_settings.config: object names and per-object overrides, then the named plates."""
    out = ['<?xml version="1.0" encoding="UTF-8"?>', "<config>"]
    for inst in instances:
        extra = f'    <metadata key="wall_loops" value="{inst.kind.walls}"/>\n' if inst.kind.walls else ""
        if inst.kind.name.startswith(CORE_NAME_PREFIX):
            extra += f'    <metadata key="wall_loops" value="{CORE_WALLS}"/>\n    <metadata key="top_shell_layers" value="{CORE_TOP_SHELL_LAYERS}"/>\n'
        if inst.kind.name.startswith(PETAL_NAME_PREFIX):
            extra += f'    <metadata key="inner_wall_speed" value="{PETAL_INNER_WALL_SPEED}"/>\n    <metadata key="inner_wall_acceleration" value="{PETAL_INNER_WALL_ACCEL}"/>\n'
        if inst.kind.name.startswith(BRIM_NAME_PREFIXES):
            extra += f'    <metadata key="brim_type" value="outer_only"/>\n    <metadata key="brim_width" value="{PETAL_BRIM_WIDTH_MM if inst.kind.name.startswith(PETAL_NAME_PREFIX) else BRIM_WIDTH_MM}"/>\n'
        if inst.kind.name.startswith(STAND_NAME_PREFIX):
            extra += f'    <metadata key="brim_type" value="outer_only"/>\n    <metadata key="brim_width" value="{STAND_BRIM_WIDTH_MM}"/>\n'
        out.append(f'  <object id="{inst.object_id}">\n    <metadata key="name" value="{escape(inst.kind.name)}"/>\n    <metadata key="extruder" value="{inst.kind.extruder}"/>\n    <metadata key="layer_height" value="{layer_height}"/>\n{extra}'
                   f'    <part id="{inst.kind.mesh_id}" subtype="normal_part">\n      <metadata key="name" value="{escape(inst.kind.name)}"/>\n      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n'
                   f'      <mesh_stat edges_fixed="0" degenerate_facets="0" facets_removed="0" facets_reversed="0" backwards_edges="0"/>\n    </part>\n  </object>')
    for plate, name in enumerate(plate_names):
        out.append(f'  <plate>\n    <metadata key="plater_id" value="{plate + 1}"/>\n    <metadata key="plater_name" value="{escape(name)}"/>\n    <metadata key="locked" value="false"/>\n'
                   '    <metadata key="filament_map_mode" value="Auto For Flush"/>\n    <metadata key="filament_maps" value="1 1 1 1"/>')
        for inst in (i for i in instances if i.plate == plate):
            out.append(f'    <model_instance>\n      <metadata key="object_id" value="{inst.object_id}"/>\n      <metadata key="instance_id" value="0"/>\n      <metadata key="identify_id" value="{1000 + inst.object_id}"/>\n    </model_instance>')
        out.append("  </plate>")
    out.append("</config>\n")
    return "\n".join(out)


def verify(path: Path, instances: list[Instance], kinds: dict[str, Kind]) -> None:
    """Read the written file back: parts resolve, meshes are closed, every footprint lies inside its plate and clear of its neighbours, and everything sits on z = 0."""
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        root = etree.fromstring(z.read("3D/3dmodel.model"))
        comps = {o.get("id"): c.get("{http://schemas.microsoft.com/3dmanufacturing/production/2015/06}path").lstrip("/") for o in root.iter(f"{{{CORE}}}object") for c in o.iter(f"{{{CORE}}}component")}
        for oid, p in comps.items():
            if p not in names:
                raise RuntimeError(f"object {oid} points at missing {p}")
        meshes = {}
        for p in set(comps.values()):
            sub = etree.fromstring(z.read(p))
            v = np.array([[float(x.get(k)) for k in "xyz"] for x in sub.iter(f"{{{CORE}}}vertex")])
            f = np.array([[int(t.get(k)) for k in ("v1", "v2", "v3")] for t in sub.iter(f"{{{CORE}}}triangle")])
            m = trimesh.Trimesh(v, f, process=False)
            if not m.is_watertight or m.volume <= 0:
                raise RuntimeError(f"{p} is not a closed solid")
            meshes[p] = m
        boxes: dict[int, list[tuple[float, float, float, float, str]]] = {}
        for item in root.iter(f"{{{CORE}}}item"):
            t = [float(x) for x in item.get("transform").split()]
            m = meshes[comps[item.get("objectid")]]
            lo, hi = m.bounds[0] + t[9:12], m.bounds[1] + t[9:12]
            plate = next(i.plate for i in instances if str(i.object_id) == item.get("objectid"))
            ox, oy = plate_origin(plate, 1 + max(i.plate for i in instances))
            if abs(lo[2]) > 1e-4 or not (lo[0] >= ox and hi[0] <= ox + BED_MM and lo[1] >= oy and hi[1] <= oy + BED_MM):
                raise RuntimeError(f"object {item.get('objectid')} is off plate {plate + 1} or off the bed surface: {lo} {hi}")
            boxes.setdefault(plate, []).append((lo[0], hi[0], lo[1], hi[1], item.get("objectid")))
        for plate, bs in boxes.items():
            for i, a in enumerate(bs):
                for b in bs[i + 1:]:
                    gap = max(b[0] - a[1], a[0] - b[1], b[2] - a[3], a[2] - b[3])
                    if gap < MIN_GAP_MM:
                        raise RuntimeError(f"plate {plate + 1}: objects {a[4]} and {b[4]} are only {gap:.1f} mm apart")
        cfg = json.loads(z.read("Metadata/project_settings.config"))
    log.info("verified: %d objects on %d plates, all closed, on the bed, >= %.0f mm apart; profile %s, layer height %s, outer wall %s", len(instances), len(boxes), MIN_GAP_MM, cfg["print_settings_id"], cfg["layer_height"], cfg["outer_wall_speed"])


def project_header(original: str) -> str:
    """The base file's model header (XML declaration and <model> element) with only KEEP_UPSTREAM_METADATA kept and PROJECT_METADATA added."""
    header = re.sub(r"<build.*", "", original[: original.index("<resources>")], flags=re.S)
    kept = re.findall(r' *<metadata name="([^"]+)">.*?</metadata>\n?', header, flags=re.S)
    missing = [k for k in KEEP_UPSTREAM_METADATA if k not in kept]
    if missing:
        raise RuntimeError(f"base 3MF lacks metadata {missing}")
    header = re.sub(r' *<metadata name="([^"]+)">.*?</metadata>\n?', lambda m: m.group(0) if m.group(1) in KEEP_UPSTREAM_METADATA else "", header, flags=re.S)
    ours = "".join(f' <metadata name="{k}">{escape(v)}</metadata>\n' for k, v in PROJECT_METADATA.items())
    model_open = re.search(r"<model[^>]*>\n", header)
    if model_open is None:
        raise RuntimeError("base 3MF header has no <model> element")
    return header[: model_open.end()] + ours + header[model_open.end():]


def write_project(output: Path, kinds: dict[str, Kind], instances: list[Instance], plate_names: list[str], profile: dict[str, str]) -> None:
    """Write the project 3MF (meshes, instances, plates, profile applied over the FlatStand settings) and verify it by reading it back."""
    layer_height = profile["layer_height"]
    paths = {k.mesh_id: f"/3D/Objects/{k.name}.model" for k in kinds.values()}
    with zipfile.ZipFile(BASE_3MF) as base:
        original = base.read("3D/3dmodel.model").decode()
        header = project_header(original)
        filament_files = {n: base.read(n) for n in base.namelist() if n.startswith("Metadata/filament_settings_")}
        content_types = base.read("[Content_Types].xml")
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            ' <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n</Relationships>\n')
    model_rels = ('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
                  + "".join(f' <Relationship Target="{escape(paths[k.mesh_id])}" Id="rel-{k.mesh_id}" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n' for k in kinds.values()) + "</Relationships>\n")
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", model_xml(instances, paths, header))
        z.writestr("3D/_rels/3dmodel.model.rels", model_rels)
        for kind in kinds.values():
            z.writestr(paths[kind.mesh_id].lstrip("/"), mesh_xml(kind.mesh, kind.mesh_id))
        z.writestr("Metadata/project_settings.config", project_settings(profile))
        z.writestr("Metadata/model_settings.config", settings_xml(instances, plate_names, layer_height))
        for name, data in filament_files.items():
            z.writestr(name, data)
    verify(output, instances, kinds)
    log.info("wrote %s (%.1f MB)", output.name, output.stat().st_size / 1e6)


def build_kinds() -> dict[str, Kind]:
    """The four mesh kinds: OG pulley, single and double wedge with the new stub, concave pulley; each centred, with its height."""
    single = trimesh.load(dw.NEW_STUB_SINGLE).apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))  # flat back (+z in the file) down, stub up
    kinds = {
        "pulley": Kind("Free Roll - X4 - OG Pulley", centred(load_pulley(PULLEY_ENTRY)), 1, 0.0, None),
        "single": Kind("Wedge - Single Sided - New Stub", centred(single), 2, 0.0, WEDGE_WALLS),
        "double": Kind("Wedge - Double Sided - New Stub", centred(trimesh.load(dw.NEW_STUB_DOUBLE)), 3, 0.0, WEDGE_WALLS),
        "concave": Kind("Free Roll - X4 - Concave Pulley", centred(load_pulley(CONCAVE_ENTRY)), 4, 0.0, None),
    }
    for kind in kinds.values():
        kind.height = float(kind.mesh.extents[2])
    return kinds


def main(output: Path = OUTPUT, overrides: dict[str, str] | None = None) -> None:
    """Assemble the project and verify it; overrides replace profile values (for example faster inner speeds)."""
    profile = {**profile_values(), **(overrides or {})}
    layer_height = profile["layer_height"]
    kinds = build_kinds()
    instances = placements(kinds)
    plate_names = [f"Pulleys {layer_height} mm", f"Wedges {layer_height} mm", f"Concave Pulleys {layer_height} mm"]
    write_project(output, kinds, instances, plate_names, profile)
    log.info("profile keys applied: %s", {k: v[:24] for k, v in profile.items() if k in ("outer_wall_speed", "inner_wall_speed", "outer_wall_acceleration", "default_acceleration", "sparse_infill_pattern", "bottom_shell_layers")})


if __name__ == "__main__":
    main()
