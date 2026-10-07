"""Kinematic planning for the scooping replay: pure numpy, no physics stepping.

Candidate screening and the runtime replay plan share one continuous, bounded
pose IK (ArmKinematics), so a screened scooping point behaves the same in replay.

    python scooping_kinematics.py screen   # rank nearby scooping points
"""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

from trajectory_diagnostics import fit_pose

ROOT = Path(__file__).resolve().parents[1]
# Replay markers in world metres, written by feeding_mapping.py (crop/scale there). Only read here.
TRAJ_CSV = ROOT / 'data/trajectory/gb_0017_gen_frame_c.csv'
# Chosen by `screen` (data/diagnostics/2026-09-23/screening/screen_selection/screen_summary.json).
# The recording's orientation is unchanged; only the anchor moves.
SELECTED_POINT = np.array([0.34, -0.13, 0.43])
TABLE_TOP_Z = 0.30
# CSV rims run opposite to the USD site numbering; see oa_baseline_simple.py.
BOWL_NAMES = ['Bowl_center', 'Bowl_rim_4', 'Bowl_rim_3', 'Bowl_rim_2', 'Bowl_rim_1']
SPOON_NAMES = ['Spoon_front', 'Spoon_back', 'Spoon_left', 'Spoon_right']
LEFT = '/openarm_spoon/openarm_left_base_link/openarm_left_ee_base_link'
RIGHT = '/openarm_spoon/openarm_right_base_link/openarm_right_ee_base_link'
ARMS = np.r_[0:7, 9:16]
POS_TOL = 0.002             # m, per-frame wrist pose position error
ROT_TOL = np.deg2rad(1.0)   # rad, per-frame wrist pose orientation error
EXIT_DISTANCE = 0.15        # m, spoon-to-bowl distance that ends the scoop after it lifts out
SMOOTH_HZ = 8.0             # zero-phase joint low-pass: strips marker jitter, keeps the scoop (<3 Hz)


def load_trajectory():
    """TRAJ_CSV as written: times (s) and (N, 9, 3) world markers (m), BOWL_NAMES + SPOON_NAMES order."""
    rows = list(csv.DictReader(TRAJ_CSV.open()))
    times = np.array([float(r['Time']) for r in rows])
    markers = np.array([[[float(r[f'{n} {a}']) for a in 'XYZ'] for n in BOWL_NAMES + SPOON_NAMES] for r in rows])
    if len(times) < 2 or not np.isfinite(markers).all() or not np.all(np.diff(times) > 0):
        raise ValueError(f'{TRAJ_CSV.name} needs >= 2 rows, finite positions and increasing times')
    return times, markers


def anchor_point(trajectory=None):
    """First Bowl_center sample as written in TRAJ_CSV: the placement feeding_mapping chose."""
    return (load_trajectory() if trajectory is None else trajectory)[1][0, 0].copy()


def load_templates():
    """USD marker templates: bowl in the holder frame, spoon in the right wrist frame."""
    stage = Usd.Stage.Open(str(ROOT / 'assets/openarm_bimanual/openarm_spoon_w_o_markers.usd'))
    xf = UsdGeom.XformCache()
    inv = xf.GetLocalToWorldTransform(stage.GetPrimAtPath(RIGHT)).GetInverse()
    spoon = np.array([np.array((xf.GetLocalToWorldTransform(stage.GetPrimAtPath(
        f'{RIGHT}/sites/spoon_marker_{i}/spoon_marker_{i}')) * inv).ExtractTranslation()) for i in range(1, 5)])
    stage, xf = Usd.Stage.Open(str(ROOT / 'assets/props/Bowl_holder_w_o_marker.usd')), UsdGeom.XformCache()
    bowl = np.array([np.array(xf.GetLocalToWorldTransform(stage.GetPrimAtPath(
        '/Bowl_holder_w_o_marker/sites/' + n)).ExtractTranslation())
        for n in ['Bowl_center', 'Bowl_rim_1', 'Bowl_rim_2', 'Bowl_rim_3', 'Bowl_rim_4']])
    return bowl, spoon


def mapped_targets(point=None, trajectory=None, yaw_deg=0.0):
    """TRAJ_CSV moved so its first Bowl_center sample sits at point; (N, 9, 3) world markers.

    A pure translation (point=None keeps the CSV's placement), so orientations and
    all spoon-bowl relative motion are exactly as feeding_mapping wrote them.
    yaw_deg turns the whole recording about vertical through the scooping point.
    Gravity is yaw-invariant, so bowl tilt and all spoon-bowl relative motion keep.
    """
    times, markers = load_trajectory() if trajectory is None else trajectory
    point = markers[0, 0] if point is None else np.asarray(point, dtype=float)
    mapped = markers - markers[0, 0] + point
    if yaw_deg:
        mapped = Rotation.from_euler('z', yaw_deg, degrees=True).apply(
            (mapped - point).reshape(-1, 3)).reshape(mapped.shape) + point
    return times, mapped


def task_frames(targets):
    """Frames up to the spoon lifting clear of the bowl after the scoop.

    The recording then carries the spoon ~0.39 m sideways, beyond the right arm's
    reach at every screened placement (elbow fully extended), so replay ends here.
    """
    distance = np.linalg.norm(targets[:, 5:].mean(1) - targets[:, 0], axis=1)
    entered = np.argmax(distance < EXIT_DISTANCE / 2)
    exits = np.nonzero(distance[entered:] > EXIT_DISTANCE)[0]
    return len(targets) if len(exits) == 0 else entered + exits[0]


def pose_targets(targets, bowl_local, spoon_local):
    """Wrist poses whose attached marker templates best fit the world markers; quats wxyz."""
    positions, quats = [], []
    for pts in targets:
        ps, qs = [], []
        for local, world in [(bowl_local, pts[:5]), (spoon_local, pts[5:])]:
            centroid, rot, _ = fit_pose(local, world)
            ps.append(centroid - rot.apply(np.mean(local, axis=0)))
            qs.append(rot.as_quat()[[3, 0, 1, 2]])
        positions.append(ps)
        quats.append(qs)
    return np.array(positions), np.array(quats)


def measured_grasp(run='scoop_full'):
    """First held-pose row of a recorded run: joint state and bowl pose."""
    return next(csv.DictReader((ROOT / 'data/diagnostics/2026-09-23/runs' / run / 'diagnostics.csv').open()))


def screening_bowl_in_wrist(bowl_template, left, row):
    """Bowl markers in the left wrist frame for a grasp measured in a recorded run.
    Replay replaces this with the grasp actually achieved in that run."""
    T, _ = left.fk(np.array([float(row[f'actual_q{i}']) for i in range(7)]))
    bp = np.array([float(row[f'bowl_actual_p{a}_m']) for a in 'xyz'])
    br = Rotation.from_quat([float(row[f'bowl_actual_q{a}']) for a in 'xyzw'])
    world = bp + br.apply(bowl_template - bowl_template.mean(0))
    return (world - T[:3, 3]) @ T[:3, :3]


class ArmKinematics:
    """USD joint-frame FK with a bounded numerical pose solver.

    Planning bounds inset the unchanged physical limits; they do not modify USD
    or simulation joint limits. The weak seed term selects a continuous solution
    in the redundant seventh DOF. Pose errors are checked separately.
    """
    def __init__(self, side, margin=np.deg2rad(5)):
        from pxr import UsdPhysics
        stage = Usd.Stage.Open(str(ROOT / 'assets/openarm_bimanual/openarm_spoon_w_o_markers.usd'))
        joints = {p.GetName(): UsdPhysics.Joint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)}

        def frame(j, n):
            pos = np.array(getattr(j, f'GetLocalPos{n}Attr')().Get())
            q = getattr(j, f'GetLocalRot{n}Attr')().Get()
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()
            T[:3, 3] = pos
            return T
        root = joints[f'rootJoint_openarm_{side}_base_link']
        self.base = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(
            stage.GetPrimAtPath(root.GetBody1Rel().GetTargets()[0]))).T
        self.frames = []
        lo, hi = [], []
        for i in range(1, 8):
            joint = joints[f'openarm_{side}_joint{i}']
            p = joint.GetPrim()
            self.frames.append((frame(joint, 0), np.linalg.inv(frame(joint, 1))))
            lo.append(p.GetAttribute('physics:lowerLimit').Get())
            hi.append(p.GetAttribute('physics:upperLimit').Get())
        self.low, self.high, self.margin = np.deg2rad(lo), np.deg2rad(hi), margin

    def fk(self, q):
        T = self.base.copy()
        anchors, axes = [], []
        for a, (T0, T1) in zip(q, self.frames):
            T = T @ T0
            anchors.append(T[:3, 3].copy())
            axes.append(T[:3, 0].copy())
            c, s = np.cos(a), np.sin(a)
            T = T @ np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.]]) @ T1
        axes, anchors = np.array(axes).T, np.array(anchors)
        J = np.vstack((np.cross(axes.T, T[:3, 3] - anchors).T, axes))
        return T, J

    def solve(self, pos, quat, seed):
        from scipy.optimize import least_squares
        target = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]])
        seed = np.clip(seed, self.low + self.margin + 1e-8, self.high - self.margin - 1e-8)

        def residual(q):
            T, _ = self.fk(q)
            return np.r_[T[:3, 3] - pos, .2 * (Rotation.from_matrix(T[:3, :3]) * target.inv()).as_rotvec(),
                         .001 * (q - seed), .02 * (q - (self.low + self.high) / 2) / (self.high - self.low)]

        def jacobian(q):
            T, J = self.fk(q)
            v = (Rotation.from_matrix(T[:3, :3]) * target.inv()).as_rotvec()
            theta = np.linalg.norm(v)
            x, y, z = v
            K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            coeff = 1 / 12 if theta < 1e-5 else (1 - .5 * theta / np.tan(.5 * theta)) / theta**2
            inverse_left = np.eye(3) - .5 * K + coeff * K @ K
            return np.vstack((J[:3], .2 * inverse_left @ J[3:], .001 * np.eye(7), np.diag(.02 / (self.high - self.low))))
        sol = least_squares(residual, seed, jac=jacobian, bounds=(self.low + self.margin, self.high - self.margin),
                            max_nfev=80, ftol=1e-8, xtol=1e-8, gtol=1e-9)
        T, _ = self.fk(sol.x)
        err = np.r_[pos - T[:3, 3], (target * Rotation.from_matrix(T[:3, :3]).inv()).as_rotvec()]
        return sol.x, err


def marker_positions(kin, q, local):
    """World positions of wrist-frame markers at joint state q."""
    T, _ = kin.fk(q)
    return T[:3, 3] + np.asarray(local) @ T[:3, :3].T


def trace(kin, pos, quat, start):
    """Seed every sample with the previous solution: stays on one IK branch."""
    q, qs, errs = start, [], []
    for p, r in zip(pos, quat):
        q, e = kin.solve(p, r, q)
        qs.append(q)
        errs.append(e)
    return np.array(qs), np.array(errs)


def _rank(qs, errs, kin):
    pos, rot = np.linalg.norm(errs[:, :3], axis=1).max(), np.linalg.norm(errs[:, 3:], axis=1).max()
    feasible = pos <= POS_TOL and rot <= ROT_TOL
    margin = np.minimum(qs - kin.low, kin.high - qs).min()
    return (not feasible, pos + .2 * rot if not feasible else -margin)


def select_start(kin, pos, quat, seed, n_random=16, stride=10, rng=0):
    """Choose the start branch once, before playback. A coarse trace scores each
    start over the whole motion so the branch that avoids limits later wins."""
    rng = np.random.default_rng(rng)
    lo, hi = kin.low + kin.margin, kin.high - kin.margin
    best = None
    for s in [np.clip(seed, lo, hi)] + [rng.uniform(lo, hi) for _ in range(n_random)]:
        q = s
        for _ in range(3):   # re-seed so the seed term cannot bias the start
            q, _ = kin.solve(pos[0], quat[0], q)
        key = _rank(*trace(kin, pos[::stride], quat[::stride], q), kin)
        if best is None or key < best[0]:
            best = (key, q)
    return best[1]


def smooth(times, q):
    b, a = butter(2, SMOOTH_HZ / (0.5 / np.diff(times).mean()))
    return filtfilt(b, a, q, axis=0)


def plan_arm(kin, times, pos, quat, seed):
    """Full-rate continuous IK, then a smoothed spline for dense control."""
    qs, errs = trace(kin, pos, quat, select_start(kin, pos, quat, seed))
    spline = CubicSpline(times, np.clip(smooth(times, qs), kin.low, kin.high))
    q_s = spline(times)
    err_s = np.array([np.r_[p - T[:3, 3], (Rotation.from_quat(r[[1, 2, 3, 0]]) * Rotation.from_matrix(T[:3, :3]).inv()).as_rotvec()]
                      for p, r, T in ((p, r, kin.fk(q)[0]) for p, r, q in zip(pos, quat, q_s))])
    return dict(raw=qs, raw_err=errs, spline=spline, err=err_s)


def arm_metrics(kin, times, plan):
    """Feasibility uses the raw IK error; the smoothed error is what is commanded."""
    q, dq, ddq = plan['spline'](times), plan['spline'](times, 1), plan['spline'](times, 2)
    raw_pos, raw_rot = (np.linalg.norm(plan['raw_err'][:, s], axis=1) for s in (slice(0, 3), slice(3, 6)))
    return dict(max_pos_err_mm=1e3 * float(raw_pos.max()), max_rot_err_deg=float(np.rad2deg(raw_rot.max())),
                failing_frames=int(((raw_pos > POS_TOL) | (raw_rot > ROT_TOL)).sum()),
                smoothed_max_pos_err_mm=1e3 * float(np.linalg.norm(plan['err'][:, :3], axis=1).max()),
                smoothed_max_rot_err_deg=float(np.rad2deg(np.linalg.norm(plan['err'][:, 3:], axis=1).max())),
                min_limit_margin_deg=float(np.rad2deg(np.minimum(q - kin.low, kin.high - q).min())),
                max_raw_step_deg=float(np.rad2deg(np.abs(np.diff(plan['raw'], axis=0)).max())),
                max_vel_rad_s=float(np.abs(dq).max()), max_acc_rad_s2=float(np.abs(ddq).max()))


def plan_trajectory(times, targets, bowl_in_wrist, spoon_in_wrist, seed_q, kins=None):
    """Both arms' joint splines for the whole trajectory, before any playback."""
    kins = kins or (ArmKinematics('left'), ArmKinematics('right'))
    pos, quat = pose_targets(targets, bowl_in_wrist, spoon_in_wrist)
    plans = [plan_arm(k, times, pos[:, i], quat[:, i], seed_q[s]) for i, (k, s) in
             enumerate(zip(kins, (slice(0, 7), slice(9, 16))))]
    return plans, [arm_metrics(k, times, p) for k, p in zip(kins, plans)]


# World footprint of the table as placed in oa_baseline_simple (0.9 x 0.6 m, yawed 90 deg).
TABLE_X, TABLE_Y = (0.30, 0.90), (-0.45, 0.45)
CLEARANCE = 0.05
# Spoon collision mesh spans x 0.021-0.206 m along the right wrist frame.
SPOON_AXIS = np.c_[np.linspace(0.021, 0.206, 6), np.zeros((6, 2))]


def min_jerk(n):
    s = np.linspace(0, 1, n)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def transition_clearance(kins, path, bowl_local, spoon_local):
    """Worst table clearance and spoon-bowl separation of marker/handle proxies."""
    table, apart = np.inf, np.inf
    for q in path:
        left = np.r_[marker_positions(kins[0], q[:7], bowl_local), kins[0].fk(q[:7])[0][None, :3, 3]]
        right = np.r_[marker_positions(kins[1], q[9:16], spoon_local), marker_positions(kins[1], q[9:16], SPOON_AXIS)]
        pts = np.r_[left, right]
        over = ((pts[:, 0] > TABLE_X[0]) & (pts[:, 0] < TABLE_X[1]) & (pts[:, 1] > TABLE_Y[0]) & (pts[:, 1] < TABLE_Y[1]))
        if over.any():
            table = min(table, float(pts[over, 2].min() - TABLE_TOP_Z))
        apart = min(apart, float(np.linalg.norm(left[:, None] - right[None], axis=2).min()))
    return table, apart


def plan_transition(kins, q_from, q_to, bowl_local, spoon_local, steps=(250, 300, 200)):
    """Joint path from the held pose to the replay start.

    A straight joint ramp drags the spoon through the table edge. Instead the
    right arm lifts the spoon in place (it starts beside the table), traverses
    at height, then descends onto its start pose; the left arm moves directly
    during the traverse. The lift grows until the swept proxies clear the table.
    """
    right = kins[1]
    q_from, q_to = np.asarray(q_from, float), np.asarray(q_to, float)
    T_from, T_to = right.fk(q_from[9:16])[0], right.fk(q_to[9:16])[0]
    quat = Rotation.from_matrix(T_to[:3, :3]).as_quat()[[3, 0, 1, 2]]
    table = apart = float('nan')
    for lift in (0.15, 0.20, 0.25, 0.30):
        height = T_to[2, 3] + lift
        # Waypoints only have to be high; their exact poses do not matter.
        rise, over = q_from.copy(), q_to.copy()
        rise[9:16], _ = right.solve(np.r_[T_from[:2, 3], height], quat, q_to[9:16])
        over[9:16], _ = right.solve(np.r_[T_to[:2, 3], height], quat, q_to[9:16])
        path = [q_from]
        for (a, b), n in zip(((q_from, rise), (rise, over), (over, q_to)), steps):
            path.append(a + min_jerk(n)[1:, None] * (b - a))
        path = np.vstack(path)
        table, apart = transition_clearance(kins, path, bowl_local, spoon_local)
        if table >= CLEARANCE and apart >= CLEARANCE:
            return path, dict(lift_m=lift, table_clearance_m=table, spoon_bowl_separation_m=apart)
    raise RuntimeError(f'No lifted transition clears the table by {CLEARANCE} m '
                       f'(last: table {table:.3f} m, spoon-bowl {apart:.3f} m)')


def _screen_one(args):
    point, yaw, times, markers, bowl_in_wrist, spoon, seed = args
    _, targets = mapped_targets(point, (times, markers), yaw)
    n = task_frames(targets)
    times, targets = times[:n], targets[:n]
    _, (left, right) = plan_trajectory(times, targets, bowl_in_wrist, spoon, seed)
    return dict(point=np.round(point, 3).tolist(), yaw_deg=float(yaw), task_end_s=float(times[-1]), left=left, right=right,
                table_clearance_mm=1e3 * float(targets[..., 2].min() - TABLE_TOP_Z))


MAX_STEP_DEG = 5.0   # per 10 ms sample; larger means the trace switched IK branch


def feasible(r):
    """Strict: every frame within POS_TOL/ROT_TOL, no branch switch, table clear."""
    arms = (r['left'], r['right'])
    return (all(a['failing_frames'] == 0 and a['max_raw_step_deg'] < MAX_STEP_DEG for a in arms)
            and r['table_clearance_mm'] > 30)


def cost(r):
    """Worst-case pose error in tolerance units; branch switches are excluded.

    Used for ranking because the strict set is thin and the grasp differs run to
    run (33 mm / 5 deg between the recorded runs), so the chosen point should
    stay accurate across its neighbourhood rather than pass by a hair.
    """
    arms = (r['left'], r['right'])
    if any(a['max_raw_step_deg'] >= MAX_STEP_DEG for a in arms) or r['table_clearance_mm'] <= 30:
        return float('inf')
    return (max(a['max_pos_err_mm'] for a in arms) / (1e3 * POS_TOL)
            + max(a['max_rot_err_deg'] for a in arms) / np.rad2deg(ROT_TOL))


def screen(output, points, base_point, run='scoop_full'):
    from multiprocessing import Pool
    os.environ['OMP_NUM_THREADS'] = '1'
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    times, markers = load_trajectory()
    bowl, spoon = load_templates()
    left = ArmKinematics('left')
    row = measured_grasp(run)
    bowl_in_wrist = screening_bowl_in_wrist(bowl, left, row)
    seed = np.array([float(row[f'actual_q{i}']) for i in range(16)])
    with Pool(min(len(points), os.cpu_count())) as pool:
        records = pool.map(_screen_one, [(p, yaw, times, markers, bowl_in_wrist, spoon, seed) for p, yaw in points])
    for r in records:
        r['feasible'] = feasible(r)
        r['cost'] = cost(r)
        r['distance_from_baseline_mm'] = 1e3 * float(np.linalg.norm(np.array(r['point']) - base_point))
    (output / 'screen.json').write_text(json.dumps(records, indent=2))
    # Lowest worst-case error; ties go to less change (no yaw), then nearest.
    ranked = sorted((r for r in records if np.isfinite(r['cost'])),
                    key=lambda r: (round(r['cost'], 2), abs(r['yaw_deg']), r['distance_from_baseline_mm']))
    baseline = next(r for r in records if np.allclose(r['point'], np.round(base_point, 3)) and r['yaw_deg'] == 0)
    summary = dict(baseline=baseline, selected=ranked[0] if ranked else None,
                   feasible_count=sum(r['feasible'] for r in records), candidates=len(records),
                   ranked=[(r['point'], r['yaw_deg'], round(r['cost'], 2), r['feasible']) for r in ranked[:10]])
    (output / 'screen_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['screen'])
    parser.add_argument('--output', type=Path, default=ROOT / 'data/diagnostics' / datetime.now().strftime('%Y-%m-%d') / 'screening' / datetime.now().strftime('screen_%H%M%S'))
    parser.add_argument('--x', type=float, nargs='+', default=[.25, .30, .35, .40])
    parser.add_argument('--y', type=float, nargs='+', default=[-.10, -.05, 0., .05, .10])
    parser.add_argument('--z', type=float, nargs='+', default=[.35, .40, .45, .50])
    parser.add_argument('--yaw', type=float, nargs='+', default=[0.])
    parser.add_argument('--grasp-run', default='scoop_full', help='Run name in data/diagnostics/2026-09-23/runs, or absolute run directory')
    args = parser.parse_args()
    from itertools import product
    # Baseline: the placement feeding_mapping wrote into TRAJ_CSV.
    base = anchor_point()
    grid = [(base, 0.)] + [(np.array(p), yaw) for *p, yaw in product(args.x, args.y, args.z, args.yaw)
                           if not (np.allclose(p, base) and yaw == 0)]
    summary = screen(args.output, grid, base, args.grasp_run)
    for key in ('baseline', 'selected'):
        r = summary[key]
        print(key, None if r is None else {k: r[k] for k in ('point', 'yaw_deg', 'left', 'right', 'table_clearance_mm')})
    print(f"feasible {summary['feasible_count']}/{summary['candidates']}; top: {summary['ranked']}")
