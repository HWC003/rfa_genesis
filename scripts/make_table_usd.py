"""
Author a simple box-and-legs table directly as a USD file, with exact dimensions.

No CAD / OBJ / conversion pipeline needed: this uses Pixar's USD Python API (`pxr`,
already installed alongside genesis) to write geometry straight to a .usd file.

Usage
-----
    python make_table_usd.py [--length 0.9] [--width 0.6] [--height 0.45] \
        [--top-thickness 0.03] [--leg-size 0.04] [--out ../assets/props/table.usd]

The table is authored as a single Xform root ("/Table") containing one box Mesh
per part (tabletop + 4 legs). Each box is authored as literal mesh geometry sized
to its exact extents (not a unit Cube with a non-uniform scale op -- Genesis's USD
importer collapses non-uniform scale to a single axis for collision geometry, which
would give the tabletop a cube-shaped collider and the legs stubby square nubs).
None of the parts carry their own PhysicsRigidBodyAPI, so Genesis loads the whole
thing as ONE fixed rigid entity via `gs.morphs.USD(..., fixed=True)` +
`scene.add_entity()` -- no need for `scene.add_stage()`.

The table sits on the ground plane (z=0) with its top surface at z=`height`,
centered at the stage origin in x/y.
"""

import argparse
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics, Gf

_BOX_FACE_VERTEX_COUNTS = [4] * 6
_BOX_FACE_VERTEX_INDICES = [
    0, 1, 3, 2,  # -X
    4, 6, 7, 5,  # +X
    0, 4, 5, 1,  # -Y
    2, 3, 7, 6,  # +Y
    0, 2, 6, 4,  # -Z
    1, 5, 7, 3,  # +Z
]


def add_box(stage, path, center, size, color):
    hx, hy, hz = (s / 2.0 for s in size)
    points = [
        Gf.Vec3f(x, y, z)
        for x in (-hx, hx)
        for y in (-hy, hy)
        for z in (-hz, hz)
    ]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(_BOX_FACE_VERTEX_COUNTS)
    mesh.CreateFaceVertexIndicesAttr(_BOX_FACE_VERTEX_INDICES)
    mesh.CreateExtentAttr([(-hx, -hy, -hz), (hx, hy, hz)])
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(*center))
    mesh.CreateDisplayColorAttr([color])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    return mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--length", type=float, default=0.9, help="Tabletop size along X (m)")
    parser.add_argument("--width", type=float, default=0.6, help="Tabletop size along Y (m)")
    parser.add_argument("--height", type=float, default=0.30, help="Height of the top surface off the ground (m)")
    parser.add_argument("--top-thickness", type=float, default=0.03, help="Tabletop slab thickness (m)")
    parser.add_argument("--leg-size", type=float, default=0.04, help="Leg cross-section side length (m)")
    parser.add_argument("--leg-inset", type=float, default=0.05, help="Leg inset from the tabletop edge (m)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "assets" / "props" / "table.usd",
        help="Output .usd path",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(args.out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Table")
    stage.SetDefaultPrim(root.GetPrim())

    leg_height = args.height - args.top_thickness
    top_center_z = leg_height + args.top_thickness / 2.0
    add_box(
        stage,
        "/Table/Top",
        center=(0.0, 0.0, top_center_z),
        size=(args.length, args.width, args.top_thickness),
        color=Gf.Vec3f(0.55, 0.38, 0.22),
    )

    half_l = args.length / 2.0 - args.leg_inset
    half_w = args.width / 2.0 - args.leg_inset
    leg_positions = [
        (half_l, half_w),
        (half_l, -half_w),
        (-half_l, half_w),
        (-half_l, -half_w),
    ]
    for i, (x, y) in enumerate(leg_positions):
        add_box(
            stage,
            f"/Table/Leg_{i}",
            center=(x, y, leg_height / 2.0),
            size=(args.leg_size, args.leg_size, leg_height),
            color=Gf.Vec3f(0.35, 0.24, 0.14),
        )

    stage.GetRootLayer().Save()
    print(f"Wrote table USD to {args.out}")
    print(f"  footprint: {args.length} x {args.width} m, top surface at z={args.height} m")


if __name__ == "__main__":
    main()
