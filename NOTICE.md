# Notice: upstream project and credits

This repository is an independent modification and extension of an existing model. It is not affiliated with or endorsed by its author, or by Creality. "Creality" and "Space Pi" are Creality's names, used here only to say which dryer the parts fit.

## Upstream

- **Title:** SPACE PI - FREE ROLL MOD
- **Author:** Alberto Maciel (MakerWorld: https://makerworld.com/@AlbertoMaciel, design ID 2741360)
- **Page:** https://makerworld.com/models/2741360-space-pi-free-roll-mod
- **Licence:** the upstream model file carries the licence tag **CC BY-NC** in its embedded metadata. The metadata does not state a version; check the page for the exact terms before you rely on them.

A second listing, **SPACE PI X4 - FREE ROLL MOD** (formerly https://makerworld.com/en/models/2771247-space-pi-x4-free-roll-mod, dated May 2026 on the 3DGo mirror https://3dgo.app/models/makerworld/2771247), was the X4 version and is probably the one the stand and wedge meshes were taken from. As of 2026-11-24 that MakerWorld page returns 404, so its author and licence could not be confirmed from the page; the CC BY-NC tag above is the one read from the model file. Both listings are credited here as upstream works under non-commercial terms.

## What comes from upstream

- The **stand** (`stl/stand_pi_x4.stl`) is the upstream stand mesh with changes (flat on both faces, 4.05 mm foot holes, 24 ridge pockets, raised cradle). It is an adaptation of the upstream work.
- The **wedge** (`stl/wedge_single_sided.stl`) reuses the upstream wedge's section, with a new stub and 24 ridges.
- The **pulley core and the soft petals**, and every test coupon, are new designs written for this repository, made to meet the upstream stand and wedge.

The upstream files are **not included** here. To rebuild, download the upstream model from the page above and put the 3MF that holds the stand and wedge meshes, named `SpacePi-X4+OG-PULLEY+NEW-WEDGE.3mf` as the scripts expect, into `src/`. Rename the file or change `SOURCE` in `src/build_flat_stand.py` and `src/build_double_wedge.py` if your download is named differently.

## Licences here

- Models, drawings and documents: **CC BY-NC-SA 4.0** (`LICENSES/CC-BY-NC-SA-4.0.txt`). Non-commercial use only, credit required, share alike. Not for resale. The adapted stand and wedge keep the upstream author's credit and the non-commercial condition.
- Code in `src/` and `tools/`: **MIT** (`LICENSES/MIT.txt`).
- `REUSE.toml` maps every path to its licence.
