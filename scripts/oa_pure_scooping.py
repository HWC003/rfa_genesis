"""Genesis scooping replay with the holder already rigid in the left gripper.

Step 6 of oa_full_sim.py on its own: no raise/approach/grasp/lift. The holder is geometry of the left wrist
link (openarm_spoon_bowl*.usd), in the pose oa_full_sim.py grasps it, so its markers are fixed wrist-frame
points read from USD. The whole joint trajectory is known before the scene is built; the robot starts at the
replay's first pose, the liquid settles in the bowl, then the trajectory streams (one target per physics step).

Trajectory sources:
  - marker CSV (--trajectory, default): planned with scooping_kinematics IK. Defaults are the validated
    "corrected CSV + continuity solver" setup of data/diagnostics/2026-09-28/cropped_contact_correction:
    real_trajs/gb_0017_gen_frame_cropped_corrected.csv (x1.6 time, 9.584 s), in full, IK continuity weight
    0.012. The former oa_pure_scooping.py behaviour is
      --trajectory data/trajectory/real_trajs/gb_0017_gen_frame_cropped.csv --continuity-weight 0.001 --trim-after-exit
  - joint plan (--joint-plan file.npz): replayed as planned, no IK, e.g. the stationary-bowl wall scoop
      --joint-plan data/trajectory/gen_trajs/gen_scoop_traj_34deg.npz --weld-bowl
    as validated in data/diagnostics/2026-10-02 and 2026-10-06 (there: SPH 3 mm, --particle-size 0.003).

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
parser.add_argument('--trajectory', type=Path, default=TRAJ_DIR / "real_trajs" / "gb_0017_gen_frame_cropped_corrected.csv",
                    help='Marker trajectory CSV, planned with IK (default: the corrected cropped scoop)')
parser.add_argument('--joint-plan', type=Path,
                    help='Precomputed joint plan (.npz with times, q); replaces --trajectory and IK. A CSV with the '
                    'same name supplies the marker targets for recording')
parser.add_argument('--continuity-weight', type=float, default=0.012,
                    help='IK pull toward the previous joints (scooping_kinematics default 0.001; 0.012 validated 2026-09-28)')
parser.add_argument('--trim-after-exit', action='store_true',
                    help='End the replay once the spoon has lifted clear of the bowl (scooping_kinematics.task_frames); '
                    'needed for recordings that then carry the spoon out of reach')
parser.add_argument('--weld-bowl', action='store_true',
                    help='Weld the left wrist (bowl) to the world at the first pose: a stationary bowl')
parser.add_argument('--sph-mu', type=float, default=0.005, help='SPH liquid viscosity mu (Genesis default 0.005)')
parser.add_argument('--particle-size', type=float, default=0.006, help='SPH particle size (m)')
parser.add_argument('--food-size', type=float, default=0.06, help='Edge of the food cube (m)')
parser.add_argument('--food-gap', type=float, default=0.03, help='Height of the food cube bottom above the rim centre (m)')
parser.add_argument('--food-vis', choices=['recon', 'particle'], default='recon', help='Food rendering')
parser.add_argument('--settle-time', type=float, default=2.0,
                    help='Seconds held at the first replay pose while the liquid settles (s)')
parser.add_argument('--hold', type=float, default=0.0, help='Seconds the final pose is held after the replay (s)')
parser.add_argument('--evaluate', action='store_true',
                    help='Food/scoop metrics every 20 ms and the success test (needs --diagnostics and liquid)')
parser.add_argument('--submerged-window', type=float, nargs=2, metavar=('T0', 'T1'),
                    help='Trajectory times when the dish is submerged, for submerged hollow contact (7.4 8.0 for gen_scoop_traj_34deg)')
parser.add_argument('--render-final', action='store_true',
                    help='Save final.png (spoon at the end) and pool_t0.png (bowl at t=0), rendered offscreen (needs --diagnostics)')
parser.add_argument('--headless', action='store_true', help='Disable viewer/debug drawing and exit after playback')
parser.add_argument('--diagnostics', type=Path, help='Directory for CSVs, plots and timing reports')
parser.add_argument('--max-frames', type=int, help='Limit recorded frames for a short smoke test')
parser.add_argument('--no-liquid', action='store_true', help='Rigid-only comparison; changes physics')
parser.add_argument('--backend', choices=['gpu', 'cpu'], default='gpu')
parser.add_argument('--scooping-point', type=float, nargs=3, default=sk.SELECTED_POINT.tolist(),
                    metavar=('X', 'Y', 'Z'), help='Where the first Bowl_center sample of the trajectory CSV is moved to (pure translation); '
                    'default is the screened placement. Not used with --joint-plan (absolute joints)')
parser.add_argument('--show-markers', action='store_true', help='Render the bowl and spoon marker spheres')
args = parser.parse_args()
if (args.evaluate or args.render_final) and not args.diagnostics:
    parser.error('--evaluate and --render-final need --diagnostics')
if args.evaluate and args.no_liquid:
    parser.error('--evaluate needs liquid')
phase_timings = []
def report_phase(name, started):
    elapsed = time.perf_counter() - started
    phase_timings.append((name, elapsed))
    print(f'[Timing] {name}: {elapsed:.3f} s', flush=True)

DT = ss.DT
OPENARM_USD = ss.robot_usd(args.show_markers)
SCOOPING_POINT = np.array(args.scooping_point)
if args.diagnostics:
    args.diagnostics.mkdir(parents=True, exist_ok=True)
templates = ss.marker_templates(OPENARM_USD)
spoon_locals, bowl_locals = templates['spoon'], templates['bowl']
left_arm, left_finger, right_arm = ss.LEFT_ARM, ss.LEFT_FINGER, ss.RIGHT_ARM
kins = (sk.ArmKinematics('left', continuity=args.continuity_weight),
        sk.ArmKinematics('right', continuity=args.continuity_weight))

# ------------------------------------------------------------------ trajectory --
phase_start = time.perf_counter()
if args.joint_plan:
    TRAJ_SOURCE = args.joint_plan.resolve()
    trajectory_times, q_plan, trajectory = sk.load_joint_plan(TRAJ_SOURCE)
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
    # sk.load_trajectory reads sk.TRAJ_CSV; point it at the chosen CSV for this process only.
    TRAJ_SOURCE = sk.TRAJ_CSV = args.trajectory.resolve()
    trajectory_times, trajectory = sk.mapped_targets(SCOOPING_POINT)
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
if args.joint_plan:
    joint_plan = joint_spline(plan_times)
else:
    joint_plan = np.tile(ss.SEED_Q, (len(plan_times), 1))
    joint_plan[:, left_arm] = plans[0]['spline'](plan_times)
    joint_plan[:, right_arm] = plans[1]['spline'](plan_times)
    joint_plan[:, left_finger] = ss.HELD_FINGER_POS
joint_at = joint_spline if args.joint_plan else (lambda t: np.r_[plans[0]['spline'](t), [ss.HELD_FINGER_POS] * 2, plans[1]['spline'](t)])
planned_markers = np.array([np.r_[sk.marker_positions(kins[0], joint_at(t)[left_arm], bowl_locals),
                                  sk.marker_positions(kins[1], joint_at(t)[right_arm], spoon_locals)]
                            for t in trajectory_times])
planned_residual = np.concatenate([planned_markers - trajectory, np.zeros_like(trajectory)], axis=2)

# ----------------------------------------------------------------------- scene --
# Food cube starts food_gap above the rim of the bowl as held at the first replay pose, as in oa_full_sim.
start_wrist, _ = kins[0].fk(joint_plan[0, left_arm])
rim_center = start_wrist[:3, 3] + start_wrist[:3, :3] @ templates['rim_center_local']
food = None if args.no_liquid else dict(
    pos=rim_center + [0, 0, args.food_gap + args.food_size / 2], size=(args.food_size,) * 3,
    particle_size=args.particle_size, mu=args.sph_mu, vis_mode=args.food_vis)
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
    geom = sm.ScoopGeometry()
    holder_to_link = templates['holder_to_link']
    particle_volume_mL = scene.sim.sph_solver.particle_volume * 1e6
    evaluator = sm.ScoopEvaluator(geom, liquid.n_particles, args.particle_size / 2,
                                  t_end=float(trajectory_times[-1] - trajectory_times[0]), hold=args.hold,
                                  submerged_window=args.submerged_window)
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
    R, B = sm.pose(right_gripper), bowl_pose()
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
    if trajectory_time > 0 and (row['contact_peak_N'] > 30 or row['clearance_mm'] < -2):
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
                        dict(trajectory=str(TRAJ_SOURCE), joint_plan=bool(args.joint_plan),
                             scooping_point=None if args.joint_plan else SCOOPING_POINT.tolist(),
                             task_end_s=float(trajectory_times[-1]), headless=args.headless, liquid=not args.no_liquid,
                             dt=scene.sim.dt, substeps=scene.sim.substeps, pos_tol_m=sk.POS_TOL,
                             plan=plan_metrics, playback="precomputed joint spline, one target per physics step",
                             robot_usd=OPENARM_USD.name, continuity_weight=None if args.joint_plan else args.continuity_weight,
                             trim_after_exit=args.trim_after_exit,
                             food=None if args.no_liquid else dict(particle_size=args.particle_size, sph_mu=args.sph_mu,
                                                                   size=args.food_size, gap_above_rim=args.food_gap),
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
    summary = dict(trajectory=str(TRAJ_SOURCE), joint_plan=bool(args.joint_plan), weld_bowl=args.weld_bowl,
                   particle_size_m=args.particle_size, sph_mu=args.sph_mu, food_size_m=args.food_size,
                   food_gap_m=args.food_gap, settle_s=args.settle_time, hold_s=args.hold, aborted=aborted,
                   **evaluator.summary(particle_volume_mL), peak_contact_every_step_N=peak_contact[0])
    if metrics_file:
        metrics_file[0].close()
    P, R, B = evaluator.last
    np.savez_compressed(args.diagnostics / 'snapshots.npz', **snaps, final_particles=P, final_spoon=R, final_bowl=B,
                        hold_ids=evaluator.hold_ids if evaluator.hold_ids is not None else np.zeros(len(P), bool),
                        spoon_through=evaluator.trk.spoon_through, bowl_through=evaluator.trk.bowl_through,
                        embedded_ever=evaluator.trk.spoon_embedded_ever)
    (args.diagnostics / 'scoop_summary.json').write_text(json.dumps(summary, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
    print('[Evaluate]', json.dumps({k: summary[k] for k in ('success', 'retained_same_ids', 'retained_volume_mL', 'initial_in_bowl',
                                                             'final_gap_med_mm', 'final_food_contact_fraction', 'spoon_through',
                                                             'min_clearance_mm')}), flush=True)
    evaluator = None            # the viewer loop below keeps stepping; metrics.csv is closed
if args.render_final:
    ss.render_spoon(cam, sm.pose(right_gripper), args.diagnostics / 'final.png')
if recorder:
    recorder.finish()
    atexit.unregister(recorder.finish)
    print(f'Diagnostics saved to {args.diagnostics.resolve()}', flush=True)

# Hold the final (lifted-clear) pose.
while not args.headless and scene.viewer.is_alive():
    step()
