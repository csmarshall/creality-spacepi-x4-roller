# Creality Space Pi X4 spool roller

Printable parts that let two filament spools roll freely on bearings inside a Creality Space Pi X4 dryer: two stands, two 608-bearing wedges, a clip-on pulley core and three soft petals per roller. It adapts the "SPACE PI X4 - FREE ROLL MOD" by Alberto Maciel (see [NOTICE.md](NOTICE.md)) and adds a clip-on pulley with soft petals that grip spool bores of about 50 to 60 mm.

Early and experimental: tested on one printer, some checks still open (see Status). NOT FOR RESALE.

## Parts

| Part | STL | STEP |
|---|---|---|
| Stand (138 x 92.5 x 4 mm, flat on both faces, 24 ridge pockets, 4.05 mm foot holes) | `stl/stand_pi_x4.stl` | none (built from the upstream mesh) |
| Wedge (single sided, 24 ridges, 8.05 mm stub for a 608 bearing) | `stl/wedge_single_sided.stl` | none (built from the upstream mesh) |
| Pulley half, clip-on core | `stl/pulley_half_clip_on_core.stl` | `step/pulley_half_clip_on_core.step` |
| Soft ribbed petal (print three per pulley) | `stl/clip_on_petal_soft_ribbed.stl` | `step/clip_on_petal_soft_ribbed.step` |

One roller is stand, wedge, two core halves with three petals, wedge, stand; the stack between the stand faces is 72.60 mm.

### Tolerance coupons

Small prints to tune the fits on your own printer before committing to the big parts. All in `stl/`, STEP for those built with build123d.

| Coupon | Tests | Design value |
|---|---|---|
| `coupon_bearing_recess` | 608 bearing pressed into the recess | 22.0 mm bore, 7.0 mm to the shelf, 0.5 mm chamfer |
| `coupon_foot_hole` | the dryer's 4 mm spindle peg | 4.05 mm hole |
| `coupon_groove_24_teeth` | wedge ridges seating and locking in the stand pockets | 24 ridges, 0.3 mm pocket side clearance |
| `coupon_short_slot_clearance_0_10`, `coupon_short_slot_clearance_0_00`, `coupon_short_head_nominal` | the dovetail that holds each petal in the core | slot clearance 0.10 mm at the open end, 0.00 mm at the closed end |

## Print settings used

ABS on a Snapmaker U1 with a 0.4 mm hardened steel nozzle, Snapmaker Orca 2.4.1, 0.12 mm layers, gyroid infill 15%. Walls: stands 2, cores 4, wedges and petals 6. Outer brims: petals 5 mm, cores 4 mm, stands 8 mm. Print the stands flat. Other materials are untested.

## Status

Seen on a print (one printer, 0.20 mm coupons): the 608 presses in flush by hand, the wedge stub is a good fit, the 4.05 mm foot hole is slightly tight but works, the dovetail slots slide and hold, and the wedge seats, locks and lifts out of the groove coupon.

Not yet verified: that a stand stays flat enough not to rub the spool, free rotation of the whole stack, clearance under the dryer lid above a 200 mm spool, long-term strength of the petal necks and wedge stub. The sizes above came from one printer and one filament; expect to adjust.

## Rebuild

Python 3 with build123d, trimesh, numpy, scipy, shapely, matplotlib and manifold3d. Put the upstream 3MF in `src/` (see [NOTICE.md](NOTICE.md)), then:

    python3 tools/export_release.py

This runs every build script in `src/`, checks watertightness, that the final wedge clears the stand while a half-pitch turn locks, and that the stack is 72.60 mm with no overlaps, then copies the models to `stl/` and `step/`. The `.3mf` print-plate builders in `src/` also need a Moonraker printer holding a sliced coupon gcode (set `PRINTER_HOST`); the model builders do not.

A passing build shows the parts agree with the design values, not that they fit your printer.

## Licences

Models and documents: CC BY-NC-SA 4.0. Code: MIT. See [NOTICE.md](NOTICE.md) and `REUSE.toml`.
