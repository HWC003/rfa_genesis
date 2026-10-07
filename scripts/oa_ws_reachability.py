#!/usr/bin/env python
"""
Sample the reachable Cartesian workspace of the OpenArm v2 pedestal's end-effectors
and plot it as a 3D point cloud.

Method: forward kinematics over randomized joint configurations (Monte Carlo
reachability), not inverse kinematics. For each random sample of the 7 arm
joints (drawn uniformly within their USD/URDF joint limits), we compute where
the end-effector link lands via FK. This is the standard way to visualize a
manipulator's reachable workspace, because:
  - IK-from-random-Cartesian-points mostly fails outside the workspace and
    tells you little about its shape.
  - FK-from-random-joints always succeeds and naturally densifies where the
    arm has more configurations mapping to the same region.

Each sampled point already carries a feasible orientation (the FK result), so
orientation is not ignored -- it's just not filtered by default. Pass
--target-rpy-deg to color/filter the cloud by how close each sample's
end-effector orientation is to a desired one (e.g. gripper pointing down for
scooping/feeding).

Caveats:
  - Self-collision and collisions with the pedestal/table are NOT checked.
    Some plotted points may be kinematically reachable but physically
    blocked by the robot's own body. Treat this as an upper bound on
    reachability, not a guarantee.
  - Uses Genesis's own USD loading + forward kinematics, so it needs to run
    in the `genesis` conda environment (same one oa_baseline.py uses).

Examples
--------
Both arms, default sample count, show interactively:
    python oa_workspace_reachability.py

Just the left arm, more samples, save a PNG, color by orientation error
against a straight-down gripper orientation:
    python oa_workspace_reachability.py --arms left --n-samples 60000 \\
        --target-rpy-deg 180 0 0 --output left_reach.png

Find the correct link names for a different pedestal variant:
    python oa_workspace_reachability.py --list-links
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)

import genesis as gs

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
DEFAULT_USD = ASSETS_DIR / "openarm_bimanual" / "openarm_pedestal.usd"

# DOF layout, matching oa_baseline.py's convention for this pedestal:
# 0-6 left arm, 7-8 left fingers, 9-15 right arm, 16-17 right fingers.
LEFT_ARM_DOFS = list(range(0, 7))
RIGHT_ARM_DOFS = list(range(9, 16))

DEFAULT_LEFT_LINK = "/openarm_pedestal_flat2/openarm_left_base_link/openarm_left_ee_base_link"
DEFAULT_RIGHT_LINK = "/openarm_pedestal_flat2/openarm_right_base_link/openarm_right_ee_base_link"


def build_scene(usd_path: Path, backend: str, n_envs: int):
    backend_obj = {"gpu": gs.gpu, "cpu": gs.cpu}[backend]
    try:
        gs.init(backend=backend_obj)
    except Exception as exc:
        if backend == "gpu":
            print(f"[warn] GPU backend init failed ({exc}); falling back to CPU.")
            gs.init(backend=gs.cpu)
        else:
            raise

    scene = gs.Scene(show_viewer=False)
    openarm = scene.add_entity(
        gs.morphs.USD(file=str(usd_path), pos=(0.0, 0.0, 0.0), euler=(0, 0, 0), fixed=True),
        name="OpenArm",
    )
    scene.build(n_envs=n_envs)
    return scene, openarm


def rpy_deg_to_quat_wxyz(roll_deg, pitch_deg, yaw_deg):
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y_ = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y_, z], dtype=np.float64)


def quat_angle_deviation_deg(quats: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
    """Smallest rotation angle (deg) between each row of `quats` (N,4 wxyz) and `target_quat`."""
    dots = np.abs(quats @ target_quat) / (np.linalg.norm(quats, axis=1) * np.linalg.norm(target_quat))
    dots = np.clip(dots, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def sample_arm_reachability(scene, openarm, arm_dofs, link_name, n_samples, batch_size):
    link = openarm.get_link(name=link_name)
    lower, upper = openarm.get_dofs_limit(dofs_idx_local=arm_dofs)
    if lower.dim() == 2:  # (n_envs, n_dofs) with identical rows -> take one
        lower, upper = lower[0], upper[0]
    lower, upper = lower.to(torch.float32), upper.to(torch.float32)

    positions, quats = [], []
    n_batches = int(np.ceil(n_samples / batch_size))
    for i in range(n_batches):
        n = min(batch_size, n_samples - i * batch_size) if i == n_batches - 1 else batch_size
        u = torch.rand((batch_size, len(arm_dofs)), dtype=torch.float32)
        q = lower + u * (upper - lower)
        openarm.set_qpos(q, qs_idx_local=arm_dofs)
        pos = openarm.get_links_pos(links_idx_local=[link.idx_local], relative=True)[:, 0, :]
        quat = openarm.get_links_quat(links_idx_local=[link.idx_local], relative=True)[:, 0, :]
        positions.append(pos[:n].cpu().numpy())
        quats.append(quat[:n].cpu().numpy())

    return np.concatenate(positions, axis=0), np.concatenate(quats, axis=0)


def print_position_stats(arm_name, pos):
    print(f"\n[{arm_name}] end-effector position stats over {len(pos)} reachable samples (m, robot base frame):")
    print(f"{'axis':>5} {'mean':>8} {'std':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'min':>8} {'max':>8}")
    for axis, label in enumerate(("x", "y", "z")):
        col = pos[:, axis]
        p25, p50, p75 = np.percentile(col, [25, 50, 75])
        print(f"{label:>5} {col.mean():8.3f} {col.std():8.3f} {p25:8.3f} {p50:8.3f} {p75:8.3f} {col.min():8.3f} {col.max():8.3f}")


def subsample(*arrays, max_points, rng):
    n = len(arrays[0])
    if n <= max_points:
        return arrays
    idx = rng.choice(n, size=max_points, replace=False)
    return tuple(a[idx] for a in arrays)


def main():
    parser = argparse.ArgumentParser(
        description="Plot the reachable Cartesian workspace of the OpenArm v2 pedestal's end-effectors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD, help="Path to the pedestal .usd/.usda file.")
    parser.add_argument("--arms", choices=["left", "right", "both"], default="both")
    parser.add_argument("--n-samples", type=int, default=20000, help="Random joint samples per arm.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Parallel Genesis envs per FK batch.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--left-link", default=DEFAULT_LEFT_LINK)
    parser.add_argument("--right-link", default=DEFAULT_RIGHT_LINK)
    parser.add_argument(
        "--list-links", action="store_true",
        help="Print all link names in the USD and exit (use this to find the right --left-link/--right-link).",
    )
    parser.add_argument(
        "--target-rpy-deg", type=float, nargs=3, default=None, metavar=("ROLL", "PITCH", "YAW"),
        help="If set, color points by orientation deviation (deg) from this target RPY.",
    )
    parser.add_argument("--orientation-tol-deg", type=float, default=30.0)
    parser.add_argument(
        "--filter-orientation", action="store_true",
        help="With --target-rpy-deg, drop points outside --orientation-tol-deg instead of just coloring them.",
    )
    parser.add_argument("--max-plot-points", type=int, default=15000, help="Subsample per arm before plotting.")
    parser.add_argument(
        "--no-stats", action="store_true",
        help="Skip printing per-axis mean/percentile stats (printed by default; useful for sizing table/bowl placement).",
    )
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None, help="Save the figure to this path (e.g. reach.png).")
    parser.add_argument("--no-show", action="store_true", help="Don't open an interactive window.")
    parser.add_argument("--save-npz", type=Path, default=None, help="Dump raw sampled positions/quats to this .npz.")
    args = parser.parse_args()

    scene, openarm = build_scene(args.usd, args.backend, args.batch_size)

    if args.list_links:
        for link in openarm.links:
            print(link.name)
        return

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    arm_specs = []
    if args.arms in ("left", "both"):
        arm_specs.append(("left", LEFT_ARM_DOFS, args.left_link, "navy"))
    if args.arms in ("right", "both"):
        arm_specs.append(("right", RIGHT_ARM_DOFS, args.right_link, "firebrick"))

    target_quat = None
    if args.target_rpy_deg is not None:
        target_quat = rpy_deg_to_quat_wxyz(*args.target_rpy_deg)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    all_pts_for_bounds = []
    scatter_for_colorbar = None
    for arm_name, dofs, link_name, color in arm_specs:
        print(f"[{arm_name}] sampling {args.n_samples} joint configurations...")
        pos, quat = sample_arm_reachability(scene, openarm, dofs, link_name, args.n_samples, args.batch_size)

        if target_quat is not None:
            dev_deg = quat_angle_deviation_deg(quat, target_quat)
            if args.filter_orientation:
                mask = dev_deg <= args.orientation_tol_deg
                pos, dev_deg = pos[mask], dev_deg[mask]
                print(f"[{arm_name}] {mask.sum()}/{len(mask)} samples within {args.orientation_tol_deg} deg of target orientation.")
            if not args.no_stats:
                print_position_stats(arm_name, pos)
            pos, dev_deg = subsample(pos, dev_deg, max_points=args.max_plot_points, rng=rng)
            scatter_for_colorbar = ax.scatter(
                pos[:, 0], pos[:, 1], pos[:, 2],
                c=dev_deg, cmap="viridis", vmin=0, vmax=max(dev_deg.max(), 1.0),
                s=args.point_size, alpha=min(args.alpha + 0.3, 1.0), label=f"{arm_name} EE",
            )
        else:
            if not args.no_stats:
                print_position_stats(arm_name, pos)
            (pos,) = subsample(pos, max_points=args.max_plot_points, rng=rng)
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=args.point_size, alpha=args.alpha, color=color, label=f"{arm_name} EE")

        all_pts_for_bounds.append(pos)

        if args.save_npz is not None:
            np.savez(
                args.save_npz.with_stem(f"{args.save_npz.stem}_{arm_name}"),
                positions=pos, orientation_deviation_deg=dev_deg if target_quat is not None else None,
            )

    ax.scatter([0], [0], [0], color="black", marker="^", s=60, label="pedestal base")

    all_pts = np.concatenate(all_pts_for_bounds, axis=0)
    center = all_pts.mean(axis=0)
    radius = np.max(np.linalg.norm(all_pts - center, axis=1)) * 1.05
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    title = f"OpenArm v2 pedestal — reachable EE workspace ({args.n_samples} samples/arm, FK, no collision check)"
    ax.set_title(title)
    ax.legend(loc="upper right")
    if scatter_for_colorbar is not None:
        fig.colorbar(scatter_for_colorbar, ax=ax, shrink=0.6, label="orientation deviation from target [deg]")

    fig.tight_layout()
    if args.output is not None:
        fig.savefig(args.output, dpi=200)
        print(f"Saved figure to {args.output}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
