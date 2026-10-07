"""Generate bowl-holder USD assets and marker sites from the two source STLs.

Run with Python providing numpy, scipy, trimesh, and pxr (no Isaac Sim needed).
Both outputs use the unmarked STL's coordinates, in meters.
"""

from pathlib import Path
import json

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt


WORKSPACE = Path(__file__).resolve().parents[2]
SOURCE = WORKSPACE / "openarm_mujoco/v2/assets/attachments"
OUTPUT = Path(__file__).resolve().parents[1] / "assets/props"


def fit_sphere(vertices, rng):
    """Reject marker stems using a sphere RANSAC fit, then refine inliers."""
    best = None
    for _ in range(2000):
        sample = vertices[rng.choice(len(vertices), 4, replace=False)]
        matrix = 2 * (sample[1:] - sample[0])
        if abs(np.linalg.det(matrix)) < 1e-12:
            continue
        center = np.linalg.solve(matrix, (sample[1:] ** 2).sum(1) - (sample[0] ** 2).sum())
        radius = np.linalg.norm(sample[0] - center)
        if not 0.003 < radius < 0.012:
            continue
        mask = abs(np.linalg.norm(vertices - center, axis=1) - radius) < 2e-6
        if best is None or mask.sum() > best[0]:
            best = (mask.sum(), center, radius, mask)
    assert best is not None and best[0] > len(vertices) / 2
    _, center, radius, mask = best
    fit = least_squares(
        lambda x: np.linalg.norm(vertices[mask] - x[:3], axis=1) - x[3],
        np.r_[center, radius], gtol=1e-14, xtol=1e-14, ftol=1e-14,
    ).x
    residual = np.linalg.norm(vertices[mask] - fit[:3], axis=1) - fit[3]
    return fit[:3], float(fit[3]), float(np.sqrt(np.mean(residual ** 2)))


def write_asset(name, parts, translation, sites):
    stage = Usd.Stage.CreateNew(str(OUTPUT / (name + ".usda")))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/" + name).GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    root.SetDocumentation("Frame: Bowl_holder_w_o_marker.STL. Sites are local reference frames, not physics links.")
    for folder in ("visuals", "collisions", "sites"):
        UsdGeom.Xform.Define(stage, f"/{name}/{folder}")
    for i, part in enumerate(parts):
        path = f"/{name}/visuals/part_{i:02d}"
        mesh = UsdGeom.Mesh.Define(stage, path)
        vertices = (part.vertices + translation).astype(np.float32)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(part.faces), 3, dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(part.faces.astype(np.int32).ravel()))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateExtentAttr([Gf.Vec3f(*vertices.min(0).tolist()), Gf.Vec3f(*vertices.max(0).tolist())])
        mesh.CreateDisplayColorAttr([Gf.Vec3f(0.65, 0.67, 0.72)])
        # Guide purpose suppresses rendering while keeping Genesis collision active.
        collision = stage.DefinePrim(f"/{name}/collisions/part_{i:02d}", "Mesh")
        collision.GetReferences().AddInternalReference(path)
        UsdGeom.Imageable(collision).CreatePurposeAttr("guide")
        UsdPhysics.CollisionAPI.Apply(collision)
        # Part 0 is the rectangular grasp block in both STLs. Its solid hull
        # provides stable jaw contacts; preserve the concave bowl and remaining parts.
        UsdPhysics.MeshCollisionAPI.Apply(collision).CreateApproximationAttr("convexHull" if i == 0 else "none")
    for label, data in sites.items():
        path = f"/{name}/sites/{label}"
        site = UsdGeom.Xform.Define(stage, path)
        site.AddTranslateOp().Set(Gf.Vec3d(*data["position_m"]))
        site.GetPrim().SetCustomDataByKey("reference_site", True)
        sphere = UsdGeom.Sphere.Define(stage, path + "/visual")
        sphere.CreateRadiusAttr(data["radius_m"])
        sphere.CreateDisplayColorAttr([Gf.Vec3f(0.2, 1.0, 0.2)])
        # Keep named reference frames, but never render or collide with site spheres.
        # Only the marked STL supplies visible, physical spheres and stems.
        sphere.CreateVisibilityAttr("invisible")
    stage.GetRootLayer().Save()
    stage.GetRootLayer().Export(str(OUTPUT / (name + ".usd")), args={"format": "usdc"})


def main():
    unmarked = trimesh.load(SOURCE / "Bowl_holder_w_o_marker.STL", force="mesh")
    marked = trimesh.load(SOURCE / "Bowl_holder_w_marker.STL", force="mesh")
    plain_parts = unmarked.split(only_watertight=False)
    marked_parts = marked.split(only_watertight=False)
    # The unique handle component establishes the common frame without marker bias.
    handle = min(plain_parts, key=lambda p: p.bounds[0, 0])
    handle_marked = min(marked_parts, key=lambda p: p.bounds[0, 0])
    translation = handle.vertices.mean(0) - handle_marked.vertices.mean(0)
    tree = cKDTree(unmarked.vertices)
    shared_error = cKDTree(marked.vertices + translation).query(unmarked.vertices)[0].max()
    assert shared_error < 1e-7, "Shared geometry changed; registration must be revisited."
    markers = [p for p in marked_parts if tree.query(p.vertices + translation)[0].max() > 1e-6]
    assert len(markers) == 5
    rng = np.random.default_rng(4)
    fitted = [fit_sphere(p.vertices + translation, rng) for p in markers]
    # Center sphere is recessed into the bowl, below all four rim spheres in Y.
    center = min(fitted, key=lambda item: item[0][1])
    rim = [item for item in fitted if item is not center]
    # Photo: handle points right (-X), top is +Z, viewing into the bowl from +Y.
    # Start with the rim marker nearest the +Z direction from the center.
    angle = lambda item: np.arctan2(item[0][2] - center[0][2], item[0][0] - center[0][0])
    top = min(rim, key=lambda item: abs(np.arctan2(np.sin(angle(item)-np.pi/2), np.cos(angle(item)-np.pi/2))))
    # Image horizontal is -X, so anticlockwise means decreasing this X/Z angle.
    rim.sort(key=lambda item: (angle(top) - angle(item)) % (2 * np.pi))
    ordered = [("Bowl_center", center)] + [(f"Bowl_rim_{i}", item) for i, item in enumerate(rim, 1)]
    sites = {label: {"position_m": c.tolist(), "radius_m": r, "sphere_fit_rms_m": error}
             for label, (c, r, error) in ordered}
    write_asset("Bowl_holder_w_o_marker", plain_parts, np.zeros(3), sites)
    write_asset("Bowl_holder_w_marker", marked_parts, translation, sites)
    report = {"marked_to_common_translation_m": translation.tolist(),
              "shared_geometry_max_error_m": float(shared_error), "sites": sites}
    (OUTPUT / "Bowl_holder_sites.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
