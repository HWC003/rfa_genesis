"""Genesis scooping replay with the holder already rigid in the left gripper.

Step 6 of oa_full_sim.py on its own: no raise/approach/grasp/lift. The holder is geometry of the left wrist
link (openarm_spoon_bowl*.usd), in the pose oa_full_sim.py grasps it, so its markers are fixed wrist-frame
points read from USD. The whole joint trajectory is known before the scene is built; the robot starts at the
replay's first pose, the liquid settles in the bowl, then the trajectory streams (one target per physics step).

Defaults (since 2026-11-13) are the validated soybean-pudding setup: spoon v3, the soybean 0017 marker CSV
(sim_ready_trajs/sb_0017_cal_x1.6_yaw180.csv) at its screened placement (0.22, -0.04, 0.44), food 56.51 (= 41.7 mL
real, see README), rho 960 / mu 0.01 / gamma 0.01, 6 mm particles, hold 2 s, clearance abort -30 mm. The former
green-bean defaults (v1 spoon, rho 1000, hold 0, -2 mm, 60 mm food cube) are
  --spoon v1 --trajectory data/trajectory/real_trajs/gb_0017_gen_frame_cropped_corrected.csv
  --scooping-point 0.34 -0.13 0.43 --sph-rho 1000 --hold 0 --gate-clearance-mm -2 --food-volume 0

Trajectory sources:
  - marker CSV (--trajectory, default): planned with scooping_kinematics IK (continuity weight 0.012, validated
    in data/diagnostics/2026-09-28/cropped_contact_correction).
  - joint plan (--joint-plan file.npz): replayed as planned, no IK, e.g. the stationary-bowl wall scoop
      --joint-plan data/trajectory/gen_trajs/gen_scoop_traj_34deg.npz --weld-bowl
    as validated in data/diagnostics/2026-10-02 and 2026-10-06 (there: SPH 3 mm, --particle-size 0.003).
  - robot joint CSV (--joints file.csv, sb_robot_trajs format: time_from_start_s + 7 left + 7 right joints, rad):
    replayed as is, no IK; the same file the real robot gets (oa_real_scooping_replay.py). The joints must be for
    --spoon (sb_robot_trajs are v3). Pipeline plans (scooping_pipeline.py) already start at rest; the older v1 / v2
    CSVs start mid-motion and get the robot's smooth start only with --blend-window 0.3 (trajectory_smoothing).
      --joints data/trajectory/sb_robot_trajs/v2/sb_traj_0001.csv

Spoon (--spoon): v1 is the original spoon in line with joint 7; v2 is the CAD attachment perpendicular to joint 7
(spoon_mount_v2.usd). The v2 markers sit where v1's do relative to the dish, so any marker CSV runs unchanged with
--spoon v2 (IK puts the v2 dish where the v1 dish was). Joint plans and joint CSVs are joints of the spoon they
were planned for: a .npz with a 'spoon' entry must match --spoon; the gen_trajs .npz without one are v1 joints.

Scene construction: scooping_scene.py. Food/scoop metrics and the success test: scooping_metrics.py
(--evaluate, written to --diagnostics as metrics.csv, scoop_summary.json and particle snapshots).
"""

import argparse
import atexit
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

import scooping_kinematics as sk
import scooping_metrics as sm
import scooping_scene as ss
from trajectory_diagnostics import Recorder

TRAJ_DIR = Path(__file__).resolve().parent.parent / "data" / "trajectory"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--trajectory', type=Path, default=TRAJ_DIR / "sim_ready_trajs" / "sb_0017_cal_x1.6_yaw180.csv",
                    help='Marker trajectory CSV, planned with IK (default: soybean 0017, cal)')
parser.add_argument('--joint-plan', type=Path,
                    help='Precomputed joint plan (.npz with times, q); replaces --trajectory and IK. A CSV with the '
                    'same name supplies the marker targets for recording')
parser.add_argument('--joints', type=Path,
                    help='Robot joint CSV (time_from_start_s + 7 left + 7 right joints, rad), replayed as is; no IK. '
                    'Joints of --spoon')
parser.add_argument('--blend-window', type=float, default=0.0,
                    help='--joints: retime the first seconds from rest as the real robot does (0.3 there; 0 = off)')
parser.add_argument('--blend-duration', type=float, default=0.7, help='--joints: seconds the blend window is stretched to')
parser.add_argument('--continuity-weight', type=float, default=0.012,
                    help='IK pull toward the previous joints (scooping_kinematics default 0.001; 0.012 validated 2026-09-28)')
parser.add_argument('--joint-margin-deg', type=float, default=5.0,
                    help='Planner keeps every joint this far inside its limit (deg, scooping_kinematics default 5)')
parser.add_argument('--trim-after-exit', action='store_true',
                    help='End the replay once the spoon has lifted clear of the bowl (scooping_kinematics.task_frames); '
                    'needed for recordings that then carry the spoon out of reach')
parser.add_argument('--spoon', choices=sorted(ss.SPOONS), default=None,
                    help='Spoon attachment (default v3, or the spoon stored in a --joint-plan): v1 original (in line with joint 7), '
                         'v2 CAD attachment perpendicular to joint 7, '
                         'v3 = v2 3.14 mm further from the joint-7 axis, v4 / v5 = v3 turned +90 / -90 deg about the '
                         'spoon axis (dish face inward / outward; 2026-10-30 spoon_mount_v4)')
parser.add_argument('--weld-bowl', action='store_true',
                    help='Weld the left wrist (bowl) to the world at the first pose: a stationary bowl')
parser.add_argument('--sph-mu', type=float, default=0.01, help='SPH liquid viscosity mu (Genesis default 0.005; 0.01 matches the soybean pudding scoops, 2026-10-26 sph_param_validation)')
parser.add_argument('--sph-rho', type=float, default=960.0, help='SPH liquid rest density (kg/m^3; 960 = soybean pudding, Genesis default 1000)')
parser.add_argument('--sph-gamma', type=float, default=0.01, help='SPH liquid surface tension / cohesion (Genesis default 0.01)')
parser.add_argument('--particle-size', type=float, default=0.006, help='SPH particle size (m)')
parser.add_argument('--food-size', type=float, default=0.06, help='Edge of the food cube (m)')
parser.add_argument('--food-volume', type=float, default=56.51,
                    help='Food volume (mL; default 56.51 = soybean 0017, 41.7 mL real), overrides --food-size: particles = '
                         'volume / (0.8 particle_size^3) (Genesis particle volume, so the food mass is density x volume), in a '
                         'near-cubic box of whole particles. 0 = use --food-size')
parser.add_argument('--food-gap', type=float, default=0.03, help='Height of the food cube bottom above the rim centre (m)')
parser.add_argument('--food-vis', choices=['recon', 'particle'], default='recon', help='Food rendering')
parser.add_argument('--settle-time', type=float, default=2.0,
                    help='Seconds held at the first replay pose while the liquid settles (s)')
parser.add_argument('--hold', type=float, default=2.0, help='Seconds the final pose is held after the replay (s)')
parser.add_argument('--evaluate', action='store_true',
                    help='Food/scoop metrics every 20 ms and the success test (needs --diagnostics and liquid)')
parser.add_argument('--gate-clearance-mm', type=float, default=-30.0,
                    help='With --evaluate, abort when the spoon-bowl clearance (axial, inflated on the steep wall) drops '
                         'below this (mm); e.g. -30 for 0 mm clearance trajectories. The 30 N contact gate always applies')
parser.add_argument('--submerged-window', type=float, nargs=2, metavar=('T0', 'T1'),
                    help='Trajectory times when the dish is submerged, for submerged hollow contact (7.4 8.0 for gen_scoop_traj_34deg)')
parser.add_argument('--render-final', action='store_true',
                    help='Save final.png (spoon at the end) and pool_t0.png (bowl at t=0), rendered offscreen (needs --diagnostics)')
parser.add_argument('--headless', action='store_true', help='Disable viewer/debug drawing and exit after playback')
parser.add_argument('--diagnostics', type=Path, help='Directory for CSVs, plots and timing reports')
parser.add_argument('--max-frames', type=int, help='Limit recorded frames for a short smoke test')
parser.add_argument('--no-liquid', action='store_true', help='Rigid-only comparison; changes physics')
parser.add_argument('--backend', choices=['gpu', 'cpu'], default='gpu')
parser.add_argument('--scooping-point', type=float, nargs=3, default=[0.22, -0.04, 0.44],
                    metavar=('X', 'Y', 'Z'), help='Where the first Bowl_center sample of the trajectory CSV is moved to (pure translation); '
                    'default is the soybean placement (sim_ready_trajs/manifest.csv). Not used with --joint-plan / --joints')
parser.add_argument('--show-markers', action='store_true', help='Render the bowl and spoon marker spheres')
args = parser.parse_args()
if (args.evaluate or args.render_final) and not args.diagnostics:
    parser.error('--evaluate and --render-final need --diagnostics')
if args.evaluate and args.no_liquid:
    parser.error('--evaluate needs liquid')
if args.joint_plan and args.joints:
    parser.error('--joint-plan and --joints are alternatives')
if args.joint_plan:
    with np.load(args.joint_plan) as _plan:
        plan_spoon = str(_plan['spoon']) if 'spoon' in _plan.files else 'v1'     # gen_trajs plans: v1 joints
    if args.spoon is None:
        args.spoon = plan_spoon
    if args.spoon != plan_spoon:
        parser.error(f'{args.joint_plan.name} holds {plan_spoon} joints, --spoon is {args.spoon}; for another spoon '
                     f'pass the plan\'s marker CSV as --trajectory so IK plans that spoon')
args.spoon = args.spoon or 'v3'
if args.food_volume is not None and args.food_volume <= 0:
    args.food_volume = None                            # --food-volume 0: the --food-size cube
JOINT_INPUT = bool(args.joint_plan or args.joints)   # absolute joints, no IK
phase_timings = []
def report_phase(name, started):
    elapsed = time.perf_counter() - started
    phase_timings.append((name, elapsed))
    print(f'[Timing] {name}: {elapsed:.3f} s', flush=True)

DT = ss.DT
OPENARM_USD = ss.robot_usd(args.show_markers, args.spoon)
SCOOPING_POINT = np.array(args.scooping_point)
if args.diagnostics:
    args.diagnostics.mkdir(parents=True, exist_ok=True)
templates = ss.marker_templates(OPENARM_USD)
spoon_locals, bowl_locals = templates['spoon'], templates['bowl']
SPOON_TOOL = templates['spoon_tool']           # spoon frame {S} in the right wrist (identity for v1)
left_arm, left_finger, right_arm = ss.LEFT_ARM, ss.LEFT_FINGER, ss.RIGHT_ARM
kins = (sk.ArmKinematics('left', margin=np.deg2rad(args.joint_margin_deg), continuity=args.continuity_weight),
        sk.ArmKinematics('right', margin=np.deg2rad(args.joint_margin_deg), continuity=args.continuity_weight))

# ------------------------------------------------------------------ trajectory --
phase_start = time.perf_counter()
if args.joint_plan or args.joints:
    if args.joint_plan:
        TRAJ_SOURCE = args.joint_plan.resolve()
        trajectory_times, q_plan, trajectory = sk.load_joint_plan(TRAJ_SOURCE)
    else:
        import scooping_io as sio
        from trajectory_smoothing import smooth_start, start_speed, starts_at_rest
        TRAJ_SOURCE = args.joints.resolve()
        trajectory_times, q_left, q_right, _ = sio.read_joint_csv(TRAJ_SOURCE)
        if not starts_at_rest(trajectory_times, np.c_[q_left, q_right]) and args.blend_window <= 0:
            print(f"[Plan] {TRAJ_SOURCE.name} starts moving ({start_speed(trajectory_times, np.c_[q_left, q_right]):.2f} "
                  "rad/s); the real robot replays it with a 0.3 s smooth start (--blend-window 0.3 to simulate that)",
                  flush=True)
        trajectory_times, arms = smooth_start(trajectory_times, np.c_[q_left, q_right], args.blend_window,
                                              args.blend_duration)
        q_plan = np.tile(ss.SEED_Q, (len(trajectory_times), 1))       # fingers as in the 2026-11-06 sim plans
        q_plan[:, left_arm], q_plan[:, right_arm] = arms[:, :7], arms[:, 7:]
        trajectory = None
    task_end = len(trajectory_times) if args.max_frames is None else min(len(trajectory_times), args.max_frames)
    trajectory_times, q_plan = trajectory_times[:task_end], q_plan[:task_end]
    if trajectory is None:      # no marker CSV: record the plan's own forward kinematics
        trajectory = np.array([np.r_[sk.marker_positions(kins[0], q[left_arm], bowl_locals),
                                     sk.marker_positions(kins[1], q[right_arm], spoon_locals)] for q in q_plan])
    trajectory = trajectory[:task_end]
    joint_spline = CubicSpline(trajectory_times, q_plan)
    plan_metrics = None
    print(f"[Plan] joint plan {TRAJ_SOURCE.name}: replaying 0-{trajectory_times[-1]:.2f} s as planned (no IK)", flush=True)
else:
    TRAJ_SOURCE = args.trajectory.resolve()
    trajectory_times, trajectory = sk.mapped_targets(SCOOPING_POINT, trajectory=sk.load_trajectory(TRAJ_SOURCE))
    # The corrected CSV was validated in full; task_frames would trim it early.
    task_end = sk.task_frames(trajectory) if args.trim_after_exit else len(trajectory)
    if args.max_frames is not None:
        if args.max_frames < 2:
            raise ValueError("--max-frames must be at least 2")
        task_end = min(task_end, args.max_frames)
    trajectory_times, trajectory = trajectory_times[:task_end], trajectory[:task_end]
    print(f"[Plan] {TRAJ_SOURCE.name}; scooping point {SCOOPING_POINT.tolist()}; replaying "
          f"0-{trajectory_times[-1]:.2f} s{' (trimmed after exit)' if args.trim_after_exit else ''}; "
          f"continuity weight {args.continuity_weight}", flush=True)
    # Plan the whole joint trajectory before the scene exists: the liquid is placed
    # from where the plan puts the bowl.
    plans, plan_metrics = sk.plan_trajectory(trajectory_times, trajectory, np.array(bowl_locals),
                                             np.array(spoon_locals), ss.SEED_Q, kins)
    for side, metric in zip(("left/bowl", "right/spoon"), plan_metrics):
        print(f"  {side}: IK max {metric['max_pos_err_mm']:.1f} mm / {metric['max_rot_err_deg']:.2f} deg "
              f"({metric['failing_frames']} frames over {1e3 * sk.POS_TOL:.0f} mm / {np.rad2deg(sk.ROT_TOL):.0f} deg), "
              f"limit margin {metric['min_limit_margin_deg']:.1f} deg, peak {metric['max_vel_rad_s']:.2f} rad/s", flush=True)
    # Unreachable samples are not retried: the best residual is reported above and
    # executed as planned, so any shortfall stays visible rather than hidden.
    if any(m['failing_frames'] for m in plan_metrics):
        print("  WARNING: some samples exceed the IK tolerance; executing the best achievable plan.", flush=True)
frame_dt = trajectory_times[1] - trajectory_times[0]
if not np.isfinite(trajectory).all() or not np.all(np.diff(trajectory_times) > 0):
    raise ValueError("Trajectory must contain finite positions and increasing timestamps")
report_phase("trajectory_plan", phase_start)

dense_times = trajectory_times[0] + DT * np.arange(1, round((trajectory_times[-1] - trajectory_times[0]) / DT) + 1)
plan_times = np.r_[trajectory_times[0], dense_times]
if JOINT_INPUT:
    joint_plan = joint_spline(plan_times)
else:
    joint_plan = np.tile(ss.SEED_Q, (len(plan_times), 1))
    joint_plan[:, left_arm] = plans[0]['spline'](plan_times)
    joint_plan[:, right_arm] = plans[1]['spline'](plan_times)
    joint_plan[:, left_finger] = ss.HELD_FINGER_POS
joint_at = joint_spline if JOINT_INPUT else (lambda t: np.r_[plans[0]['spline'](t), [ss.HELD_FINGER_POS] * 2, plans[1]['spline'](t)])
planned_markers = np.array([np.r_[sk.marker_positions(kins[0], joint_at(t)[left_arm], bowl_locals),
                                  sk.marker_positions(kins[1], joint_at(t)[right_arm], spoon_locals)]
                            for t in trajectory_times])
planned_residual = np.concatenate([planned_markers - trajectory, np.zeros_like(trajectory)], axis=2)

# ----------------------------------------------------------------------- scene --
# Food cube starts food_gap above the rim of the bowl as held at the first replay pose, as in oa_full_sim.
start_wrist, _ = kins[0].fk(joint_plan[0, left_arm])
rim_center = start_wrist[:3, 3] + start_wrist[:3, :3] @ templates['rim_center_local']
food_box = (args.food_size,) * 3 if args.food_volume is None else ss.food_box(args.food_volume, args.particle_size)
food = None if args.no_liquid else dict(
    pos=rim_center + [0, 0, args.food_gap + food_box[2] / 2], size=food_box,
    particle_size=args.particle_size, mu=args.sph_mu, rho=args.sph_rho, gamma=args.sph_gamma, vis_mode=args.food_vis)
built = ss.build_scene(OPENARM_USD, joint_plan[0], food=food, backend=args.backend, headless=args.headless,
                       weld_bowl=args.weld_bowl, camera=args.render_final)
scene, openarm, liquid, cam = built['scene'], built['robot'], built['food'], built['cam']
left_gripper, right_gripper = built['left'], built['right']

# Static visual marker for the scooping point; not a physical entity, so it
# can't collide with or obstruct the gripper.
if not args.headless:
    scene.draw_debug_sphere(pos=SCOOPING_POINT, radius=0.01, color=(1, 0, 0, 1))
force_limits = ss.FORCE_LIMITS
low, high = (x.detach().cpu().numpy() for x in openarm.get_dofs_limit())
arm_dofs = np.r_[left_arm, right_arm]
if np.any(joint_plan[:, arm_dofs] < low[arm_dofs]) or np.any(joint_plan[:, arm_dofs] > high[arm_dofs]):
    raise RuntimeError("Planned joint trajectory leaves the joint limits; refusing to execute")
sim_step_count = 0
peak_contact = (0.0, 0.0)   # (force N, sim time s) of the largest spoon-bowl contact

# ------------------------------------------------------------------ evaluation --
evaluator, metrics_file, snaps = None, None, {}
if args.evaluate:
    geom = sm.ScoopGeometry(OPENARM_USD, spoon_tool=SPOON_TOOL)
    holder_to_link = templates['holder_to_link']
    particle_volume_mL = scene.sim.sph_solver.particle_volume * 1e6
    evaluator = sm.ScoopEvaluator(geom, liquid.n_particles, args.particle_size / 2,
                                  t_end=float(trajectory_times[-1] - trajectory_times[0]), hold=args.hold,
                                  submerged_window=args.submerged_window, bowl_fixed=args.weld_bowl)
    settled_volume_mL = sm.SETTLED_PARTICLE_VOLUME_ML * (args.particle_size / 0.006) ** 3
    aborted = None


def bowl_pose():
    return sm.pose(left_gripper) @ templates['holder_to_link']


def evaluate(trajectory_time):
    """One evaluation check (every 5 physics steps = 20 ms); returns False to abort."""
    global metrics_file, aborted
    P = liquid.get_particles_pos().detach().cpu().numpy().reshape(-1, 3)
    if not np.isfinite(P).all():
        aborted = 'non-finite particle positions'; return False
    vel = liquid.get_particles_vel().detach().cpu().numpy().reshape(-1, 3)
    R, B = sm.pose(right_gripper) @ SPOON_TOOL, bowl_pose()
    pool_was_set = evaluator.pool is not None
    row = evaluator.update(trajectory_time, P, vel, R, B, spoon_bowl_contact()[1])
    if metrics_file is None:
        handle = (args.diagnostics / 'metrics.csv').open('w', newline='')
        metrics_file = (handle, csv.DictWriter(handle, fieldnames=list(row)))
        metrics_file[1].writeheader()
    metrics_file[1].writerow(row); metrics_file[0].flush()
    if not pool_was_set and evaluator.pool is not None:
        np.savez_compressed(args.diagnostics / 'settled_particles.npz', world=P, bowl=B, spoon=R, in_bowl=evaluator.pool)
        if args.render_final:
            ss.render_bowl(cam, B, args.diagnostics / 'pool_t0.png')
    if sim_step_count % 50 == 0:
        snaps[f'P_{trajectory_time:.3f}'] = P.astype(np.float32); snaps[f'R_{trajectory_time:.3f}'] = R; snaps[f'B_{trajectory_time:.3f}'] = B
    if trajectory_time > 0 and (row['contact_peak_N'] > 30 or row['clearance_mm'] < args.gate_clearance_mm):
        aborted = 'rigid force/clearance safety gate'; return False
    return True


def spoon_bowl_contact():
    """Largest individual and resultant rigid contact between the two wrist links (N)."""
    return sm.spoon_bowl_contact(openarm, left_gripper, right_gripper)


def step():
    global sim_step_count, peak_contact
    scene.step()
    sim_step_count += 1
    # Every step: the README's peaks last only a few 4 ms steps.
    count, peak, resultant = spoon_bowl_contact()
    if peak > peak_contact[0]:
        peak_contact = (peak, sim_step_count * DT)
    if evaluator:
        evaluator.contact(peak)
    if recorder:
        recorder.write('dense_contacts.csv', dict(sim_time_s=sim_step_count * DT,
                       trajectory_time_s=sim_step_count * DT - args.settle_time, contact_count=count,
                       peak_contact_N=peak, resultant_N=resultant))
    if evaluator and sim_step_count % 5 == 0:
        return evaluate(sim_step_count * DT - args.settle_time)
    return True


recorder = None
if args.diagnostics:
    recorder = Recorder(args.diagnostics, {"bowl": templates['bowl_holder'], "spoon": spoon_locals},
                        dict(trajectory=str(TRAJ_SOURCE), joint_plan=JOINT_INPUT, blend_window_s=args.blend_window if args.joints else None,
                             scooping_point=None if JOINT_INPUT else SCOOPING_POINT.tolist(),
                             task_end_s=float(trajectory_times[-1]), headless=args.headless, liquid=not args.no_liquid,
                             dt=scene.sim.dt, substeps=scene.sim.substeps, pos_tol_m=sk.POS_TOL,
                             plan=plan_metrics, playback="precomputed joint spline, one target per physics step",
                             robot_usd=OPENARM_USD.name, spoon=args.spoon, joint_margin_deg=args.joint_margin_deg, continuity_weight=None if JOINT_INPUT else args.continuity_weight,
                             trim_after_exit=args.trim_after_exit,
                             food=None if args.no_liquid else dict(particle_size=args.particle_size, sph_mu=args.sph_mu,
                                                                   sph_rho=args.sph_rho, sph_gamma=args.sph_gamma, size=list(food_box),
                                                                   volume_mL=args.food_volume, gap_above_rim=args.food_gap),
                             bowl="left wrist welded to the world at the first pose" if args.weld_bowl
                             else "rigid geometry of the left wrist link (no weld)",
                             settle_time_s=args.settle_time, hold_s=args.hold,
                             pose_definition="Marker centroid + rotation from USD template; quaternion xyzw",
                             joint_order="left arm 0:7; fingers 7:9; right arm 9:16",
                             status_definition="Planned marker residual <= POS_TOL for every marker"))
    atexit.register(recorder.finish)
    for phase, seconds in phase_timings: recorder.event(phase, seconds)


def marker_world_positions():
    """Where the tracked markers actually ended up, in the solver's world frame.
    The holder is part of the left wrist link, so this is also the physical bowl."""
    positions = []
    for link, offsets in ((left_gripper, bowl_locals), (right_gripper, spoon_locals)):
        origin = link.get_pos(relative=False).detach().cpu().numpy()
        rotation = Rotation.from_quat(link.get_quat(relative=False).detach().cpu().numpy()[[1, 2, 3, 0]])
        positions.extend(origin + rotation.apply(offset) for offset in offsets)
    return np.array(positions)


def record_sample(index, command, physics_wall):
    if recorder:
        positions = marker_world_positions()
        recorder.sample(index, trajectory_times[index], sim_step_count * scene.sim.dt,
                        trajectory[index], positions, positions[:5],
                        command, openarm.get_qpos(), planned_residual[index], 0.0, physics_wall,
                        (low, high), openarm.get_dofs_control_force(), force_limits)
        # Rigid contacts on the holder's link (spoon, table, other arm). SPH
        # coupling forces are not included in this API.
        contacts = openarm.get_contacts()
        groups = {}
        for a, b, force in zip(contacts["link_a"].tolist(), contacts["link_b"].tolist(),
                               contacts["force_a"].detach().cpu().numpy()):
            if left_gripper.idx not in (a, b):
                continue
            pair = (scene.rigid_solver.links[a].name, scene.rigid_solver.links[b].name)
            count, total, maximum = groups.get(pair, (0, np.zeros(3), 0.0))
            groups[pair] = (count + 1, total + force, max(maximum, float(np.linalg.norm(force))))
        for (a, b), (count, total, maximum) in groups.items():
            recorder.write('bowl_contacts.csv', dict(index=index, trajectory_time_s=trajectory_times[index],
                           link_a=a, link_b=b, count=count, force_on_a_x_N=total[0],
                           force_on_a_y_N=total[1], force_on_a_z_N=total[2], max_contact_force_N=maximum))


# ------------------------------------------------------------------- timeline --
running = True
print(f"[Settle] Holding the first replay pose for {args.settle_time:.1f} s while the liquid settles", flush=True)
phase_start = time.perf_counter()
for _ in range(round(args.settle_time / DT)):
    running = step()
    if not running:
        break
report_phase("settle", phase_start)
if recorder: recorder.event('settle', args.settle_time)
record_sample(0, joint_plan[0], time.perf_counter() - phase_start)

print("[Replay] Streaming the planned trajectory", flush=True)
# Continuous playback: a new interpolated target every physics step, no IK.
# Samples are recorded at every second CSV frame, as in oa_full_sim, for comparison.
TRAJ_STRIDE = 2
sample_at_step = {round((trajectory_times[i] - trajectory_times[0]) / DT): i
                  for i in range(TRAJ_STRIDE, len(trajectory), TRAJ_STRIDE)}
worst_error = np.zeros(2)
traj_debug = None
started = time.perf_counter()
for step_index, command in enumerate(joint_plan[1:], start=1):
    if not running:
        break
    openarm.control_dofs_position(command)
    running = step()
    if recorder:
        recorder.joints(int((dense_times[step_index - 1] - trajectory_times[0]) / frame_dt), dense_times[step_index - 1],
                        sim_step_count * scene.sim.dt, command, openarm.get_qpos())
    index = sample_at_step.get(step_index)
    if index is None:
        continue
    record_sample(index, command, time.perf_counter() - started)
    started = time.perf_counter()
    error = np.linalg.norm(marker_world_positions() - trajectory[index], axis=1)
    worst_error = np.maximum(worst_error, [error[:5].max(), error[5:].max()])
    if not args.headless:
        if traj_debug is not None:
            scene.clear_debug_object(traj_debug)
        traj_debug = scene.draw_debug_spheres(poss=trajectory[index], radius=0.005, color=(1, 0.9, 0.2, 0.6))
    if index % 100 == 0:
        print(f'[Replay] frame {index}, t={trajectory_times[index]:.2f}s, '
              f'max marker error={1000*error.max():.1f}mm, spoon-bowl contact={spoon_bowl_contact()[1]:.1f}N',
              flush=True)
if args.hold > 0 and running:
    print(f"[Hold] Holding the final pose for {args.hold:.1f} s", flush=True)
    openarm.control_dofs_position(joint_plan[-1])
    for _ in range(round(args.hold / DT)):
        running = step()
        if not running:
            break

print(f"Worst actual marker error during replay: bowl {1000 * worst_error[0]:.1f} mm, "
      f"spoon {1000 * worst_error[1]:.1f} mm")
print(f"Peak spoon-bowl contact: {peak_contact[0]:.1f} N at replay t={peak_contact[1] - args.settle_time:.2f} s "
      "(README, rigid CPU: 2.72 N corrected vs 155.7 N original)")
if evaluator:
    if aborted:
        print(f'ABORTED: {aborted}', flush=True)
    summary = dict(trajectory=str(TRAJ_SOURCE), joint_plan=JOINT_INPUT, spoon=args.spoon, weld_bowl=args.weld_bowl,
                   particle_size_m=args.particle_size, sph_mu=args.sph_mu, sph_rho=args.sph_rho, sph_gamma=args.sph_gamma,
                   food_size_m=args.food_size,
                   food_box_m=list(food_box), food_volume_mL=args.food_volume,
                   food_gap_m=args.food_gap, settle_s=args.settle_time, hold_s=args.hold, aborted=aborted,
                   **evaluator.summary(particle_volume_mL, settled_volume_mL), peak_contact_every_step_N=peak_contact[0])
    if metrics_file:
        metrics_file[0].close()
    P, R, B = evaluator.last
    np.savez_compressed(args.diagnostics / 'snapshots.npz', **snaps, final_particles=P, final_spoon=R, final_bowl=B,
                        hold_ids=evaluator.hold_ids if evaluator.hold_ids is not None else np.zeros(len(P), bool),
                        spoon_through=evaluator.trk.spoon_through, bowl_through=evaluator.trk.bowl_through,
                        embedded_ever=evaluator.trk.spoon_embedded_ever)
    (args.diagnostics / 'scoop_summary.json').write_text(json.dumps(summary, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
    print('[Evaluate]', json.dumps({k: summary[k] for k in ('success', 'failed_checks', 'retained_same_ids', 'retained_volume_mL',
                                                             'initial_in_bowl', 'initial_volume_mL',
                                                             'final_gap_med_mm', 'final_food_contact_fraction', 'spoon_through',
                                                             'min_clearance_mm')}), flush=True)
    evaluator = None            # the viewer loop below keeps stepping; metrics.csv is closed
if args.render_final:
    ss.render_spoon(cam, sm.pose(right_gripper) @ SPOON_TOOL, args.diagnostics / 'final.png')
if recorder:
    recorder.finish()
    atexit.unregister(recorder.finish)
    print(f'Diagnostics saved to {args.diagnostics.resolve()}', flush=True)
    recorder = None             # files are closed; the viewer loop below keeps stepping without recording

# Hold the final (lifted-clear) pose.
while not args.headless and scene.viewer.is_alive():
    step()
