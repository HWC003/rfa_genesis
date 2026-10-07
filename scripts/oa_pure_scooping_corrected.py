"""oa_pure_scooping.py with the contact-corrected cropped trajectory and continuity IK.

Implements "corrected CSV + continuity solver" from
data/diagnostics/2026-09-28/cropped_contact_correction/README.md:
  1. Replays gb_0017_gen_frame_cropped_corrected.csv (timestamps stretched x1.6, 9.584 s).
  2. Plans with the IK previous-joint weight raised 0.001 -> 0.012 (ContinuousArmKinematics);
     the corrected CSV with the unchanged solver still peaks at 124.7 N.
  3. Replays the complete CSV (no task_frames trim), as validated.
Everything else -- robot USD, gains, settling, physics -- is oa_pure_scooping.py unchanged.
Validated there only rigid on CPU: --no-liquid --backend cpu.
"""

import argparse
import atexit
import time
from pathlib import Path

import genesis as gs
import numpy as np
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation

import scooping_kinematics as sk
from trajectory_diagnostics import Recorder

CORRECTION_DIR = (Path(__file__).resolve().parent.parent / "data" / "diagnostics" / "2026-09-28"
                  / "cropped_contact_correction")
CONTINUITY_WEIGHT = 0.012   # scooping_kinematics uses 0.001; value from the README (kinematics_continuous_v2)


class ContinuousArmKinematics(sk.ArmKinematics):
    """sk.ArmKinematics.solve with a stronger pull toward the previous joints.

    Only the previous-joint residual/Jacobian weight differs. It discourages the
    37 deg left-arm branch jump near 1.96 s; joint limits and physics are unchanged.
    """
    def solve(self, pos, quat, seed):
        from scipy.optimize import least_squares
        target = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]])
        seed = np.clip(seed, self.low + self.margin + 1e-8, self.high - self.margin - 1e-8)

        def residual(q):
            T, _ = self.fk(q)
            return np.r_[T[:3, 3] - pos, .2 * (Rotation.from_matrix(T[:3, :3]) * target.inv()).as_rotvec(),
                         CONTINUITY_WEIGHT * (q - seed), .02 * (q - (self.low + self.high) / 2) / (self.high - self.low)]

        def jacobian(q):
            T, J = self.fk(q)
            v = (Rotation.from_matrix(T[:3, :3]) * target.inv()).as_rotvec()
            theta = np.linalg.norm(v)
            x, y, z = v
            K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            coeff = 1 / 12 if theta < 1e-5 else (1 - .5 * theta / np.tan(.5 * theta)) / theta**2
            inverse_left = np.eye(3) - .5 * K + coeff * K @ K
            return np.vstack((J[:3], .2 * inverse_left @ J[3:], CONTINUITY_WEIGHT * np.eye(7),
                              np.diag(.02 / (self.high - self.low))))
        sol = least_squares(residual, seed, jac=jacobian, bounds=(self.low + self.margin, self.high - self.margin),
                            max_nfev=80, ftol=1e-8, xtol=1e-8, gtol=1e-9)
        T, _ = self.fk(sol.x)
        err = np.r_[pos - T[:3, 3], (target * Rotation.from_matrix(T[:3, :3]).inv()).as_rotvec()]
        return sol.x, err


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--trajectory', type=Path,
                    default=CORRECTION_DIR / "trajectories" / "gb_0017_gen_frame_cropped_corrected.csv",
                    help='Trajectory CSV (default: the corrected cropped scoop)')
parser.add_argument('--headless', action='store_true', help='Disable viewer/debug drawing and exit after playback')
parser.add_argument('--diagnostics', type=Path, help='Directory for CSVs, plots and timing reports')
parser.add_argument('--max-frames', type=int, help='Limit recorded frames for a short smoke test')
parser.add_argument('--no-liquid', action='store_true', help='Rigid-only comparison; changes physics')
parser.add_argument('--backend', choices=['gpu', 'cpu'], default='gpu')
parser.add_argument('--scooping-point', type=float, nargs=3, default=sk.SELECTED_POINT.tolist(),
                    metavar=('X', 'Y', 'Z'), help='Where the first Bowl_center sample of the trajectory CSV is moved to (pure translation); '
                    'default is the screened placement')
parser.add_argument('--show-markers', action='store_true', help='Render the bowl and spoon marker spheres')
parser.add_argument('--settle-time', type=float, default=2.0,
                    help='Seconds held at the first replay pose while the liquid settles (s)')
args = parser.parse_args()
phase_timings = []
def report_phase(name, started):
    elapsed = time.perf_counter() - started
    phase_timings.append((name, elapsed))
    print(f'[Timing] {name}: {elapsed:.3f} s', flush=True)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
OPENARM_USD = ASSETS_DIR / "openarm_bimanual" / (
    "openarm_spoon_bowl_w_markers.usd" if args.show_markers else "openarm_spoon_bowl_w_o_markers.usd")
# sk.load_trajectory reads sk.TRAJ_CSV; point it at the corrected CSV for this process only.
TRAJ_CSV = sk.TRAJ_CSV = args.trajectory.resolve()
if args.diagnostics:
    args.diagnostics.mkdir(parents=True, exist_ok=True)

DT = 0.004
LIQUID_SIZE = (0.06, 0.06, 0.06)
# Bowl opening at rim height on its central axis, in the holder STL frame (m); see oa_full_sim.py.
BOWL_RIM_CENTER_LOCAL = (0.1549, 0.0191, 0.0)
# Finger angle measured on the handle block in oa_full_sim (scoop_full run). Finger-to-wrist
# contact is filtered as adjacent, so a closed (0.0) target would pass through the block.
HELD_FINGER_POS = 0.20
# IK seed: the replay-start joints oa_full_sim reached (scoop_full run, first held row).
# plan_trajectory still screens random starts; this keeps it on oa_full_sim's branch.
SEED_Q = np.array([0.3784, -0.3537, 0.0445, 1.6384, -0.3253, 0.3495, -0.8398, HELD_FINGER_POS, HELD_FINGER_POS,
                   0.2241, 0.5404, -0.8492, 0.9817, 1.2600, 0.7853, -0.4997])
SCOOPING_POINT = np.array(args.scooping_point)

# Fixed-base robot order: left arm 0:7, left fingers 7:9, right arm 9:16.
left_arm = np.arange(7)
left_finger = np.arange(7, 9)
right_arm = np.arange(9, 16)

# Marker offsets come straight from USD: the spoon is rigid with the right wrist and
# the holder with the left. Spoon site Xforms are identity -- the marker sphere nested
# under each carries the offset -- so resolve the child prim, not the site.
LEFT_GRIPPER_PATH = sk.LEFT
RIGHT_GRIPPER_PATH = sk.RIGHT
HOLDER_PATH = LEFT_GRIPPER_PATH + "/bowl_holder"
SPOON_MARKER_COLUMNS = {"spoon_marker_1": "Spoon_front", "spoon_marker_2": "Spoon_back",
                        "spoon_marker_3": "Spoon_left", "spoon_marker_4": "Spoon_right"}
# The rims share the CSV's names but not its numbering: the capture walks the rim
# in the opposite direction, so pairing them 1:1 warps the constellation by 37 mm
# and the IK cannot resolve an orientation. Reversing the rims fits it to 2.2 mm.
BOWL_MARKER_COLUMNS = {"Bowl_center": "Bowl_center", "Bowl_rim_1": "Bowl_rim_4",
                       "Bowl_rim_2": "Bowl_rim_3", "Bowl_rim_3": "Bowl_rim_2",
                       "Bowl_rim_4": "Bowl_rim_1"}
stage = Usd.Stage.Open(str(OPENARM_USD))
xforms = UsdGeom.XformCache()

def local_points(link_path, prim_paths):
    link_inv = xforms.GetLocalToWorldTransform(stage.GetPrimAtPath(link_path)).GetInverse()
    return [np.array((xforms.GetLocalToWorldTransform(stage.GetPrimAtPath(p)) * link_inv).ExtractTranslation())
            for p in prim_paths]

spoon_locals = local_points(RIGHT_GRIPPER_PATH, [f"{RIGHT_GRIPPER_PATH}/sites/{m}/{m}" for m in SPOON_MARKER_COLUMNS])
bowl_locals = local_points(LEFT_GRIPPER_PATH, [f"{HOLDER_PATH}/sites/{s}" for s in BOWL_MARKER_COLUMNS])
# Holder-frame template (as in oa_full_sim) so diagnostics poses are comparable.
bowl_marker_local = local_points(HOLDER_PATH, [f"{HOLDER_PATH}/sites/{s}" for s in BOWL_MARKER_COLUMNS])
holder_to_link = np.array(xforms.GetLocalToWorldTransform(stage.GetPrimAtPath(HOLDER_PATH))
                          * xforms.GetLocalToWorldTransform(stage.GetPrimAtPath(LEFT_GRIPPER_PATH)).GetInverse()).T
rim_center_local = holder_to_link[:3, :3] @ BOWL_RIM_CENTER_LOCAL + holder_to_link[:3, 3]

MARKER_COLUMNS = list(BOWL_MARKER_COLUMNS.values()) + list(SPOON_MARKER_COLUMNS.values())
if MARKER_COLUMNS != sk.BOWL_NAMES + sk.SPOON_NAMES:
    raise ValueError("Marker order differs from the planner's CSV order")
trajectory_times, trajectory = sk.mapped_targets(SCOOPING_POINT)
# The corrected CSV was validated in full; task_frames would trim it early.
task_end = len(trajectory)
if args.max_frames is not None:
    if args.max_frames < 2:
        raise ValueError("--max-frames must be at least 2")
    task_end = min(task_end, args.max_frames)
trajectory_times, trajectory = trajectory_times[:task_end], trajectory[:task_end]
frame_dt = trajectory_times[1] - trajectory_times[0]
if not np.isfinite(trajectory).all() or not np.all(np.diff(trajectory_times) > 0):
    raise ValueError("Trajectory must contain finite positions and increasing timestamps")
print(f"[Plan] {TRAJ_CSV.name}; scooping point {SCOOPING_POINT.tolist()}; replaying "
      f"0-{trajectory_times[-1]:.2f} s; continuity weight {CONTINUITY_WEIGHT}", flush=True)

# Plan the whole joint trajectory before the scene exists: the liquid is placed
# from where the plan puts the bowl.
phase_start = time.perf_counter()
kins = (ContinuousArmKinematics('left'), ContinuousArmKinematics('right'))
plans, plan_metrics = sk.plan_trajectory(trajectory_times, trajectory, np.array(bowl_locals),
                                         np.array(spoon_locals), SEED_Q, kins)
report_phase("trajectory_plan", phase_start)
for side, metric in zip(("left/bowl", "right/spoon"), plan_metrics):
    print(f"  {side}: IK max {metric['max_pos_err_mm']:.1f} mm / {metric['max_rot_err_deg']:.2f} deg "
          f"({metric['failing_frames']} frames over {1e3 * sk.POS_TOL:.0f} mm / {np.rad2deg(sk.ROT_TOL):.0f} deg), "
          f"limit margin {metric['min_limit_margin_deg']:.1f} deg, peak {metric['max_vel_rad_s']:.2f} rad/s", flush=True)
dense_times = trajectory_times[0] + DT * np.arange(1, round((trajectory_times[-1] - trajectory_times[0]) / DT) + 1)
joint_plan = np.tile(SEED_Q, (len(dense_times) + 1, 1))
joint_plan[:, left_arm] = plans[0]['spline'](np.r_[trajectory_times[0], dense_times])
joint_plan[:, right_arm] = plans[1]['spline'](np.r_[trajectory_times[0], dense_times])
joint_plan[:, left_finger] = HELD_FINGER_POS
# Unreachable samples are not retried: the best residual is reported above and
# executed as planned, so any shortfall stays visible rather than hidden.
if any(m['failing_frames'] for m in plan_metrics):
    print("  WARNING: some samples exceed the IK tolerance; executing the best achievable plan.", flush=True)

planned_markers = np.array([np.r_[sk.marker_positions(kins[0], plans[0]['spline'](t), bowl_locals),
                                  sk.marker_positions(kins[1], plans[1]['spline'](t), spoon_locals)]
                            for t in trajectory_times])
planned_residual = np.concatenate([planned_markers - trajectory, np.zeros_like(trajectory)], axis=2)

# Liquid starts 3 cm above the rim of the bowl as held at the first replay pose, as in oa_full_sim.
start_wrist, _ = kins[0].fk(joint_plan[0, left_arm])
rim_center = start_wrist[:3, 3] + start_wrist[:3, :3] @ rim_center_local
LIQUID_POS = tuple(rim_center + [0, 0, 0.03 + LIQUID_SIZE[2] / 2])

gs.init(backend=gs.gpu if args.backend == "gpu" else gs.cpu, seed=0)
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(1.49, 0.68, 0.96), camera_lookat=(0.35, 0.10, 0.33),
        enable_gui=not args.headless,
    ),
    sim_options=gs.options.SimOptions(dt=DT, substeps=60),
    sph_options=gs.options.SPHOptions(
        particle_size=0.006, lower_bound=(0.1, -0.5, 0.0), upper_bound=(1.0, 0.5, 0.9),
    ),
    vis_options=gs.options.VisOptions(visualize_sph_boundary=True),
    show_viewer=not args.headless,
    profiling_options=gs.options.ProfilingOptions(show_FPS=not args.headless),
)
plane = scene.add_entity(gs.morphs.Plane(), name="Plane")
openarm = scene.add_entity(
    gs.morphs.USD(file=str(OPENARM_USD), fixed=True),
    # Cancel robot self-weight so the position controller tracks the plan more closely.
    # The 1 mm SDF cell is the standalone holder's in oa_full_sim; the concave bowl now lives on the robot.
    material=gs.materials.Rigid(sdf_cell_size=0.001, sdf_min_res=64, sdf_max_res=256, gravity_compensation=1.0),
    name="OpenArm",
)
table = scene.add_entity(
    gs.morphs.USD(file=str(ASSETS_DIR / "props" / "table.usd"),
                  pos=(0.6, 0, 0), euler=(0, 0, 90), fixed=True),
    name="Table",
)
if not args.no_liquid:
    liquid = scene.add_entity(
        material=gs.materials.SPH.Liquid(mu=0.01),
        morph=gs.morphs.Box(pos=LIQUID_POS, size=LIQUID_SIZE),
        surface=gs.surfaces.Default(color=(0.68, 0.62, 0.49), vis_mode="recon"),
    )
scene.build()

# Static visual marker for the scooping point; not a physical entity, so it
# can't collide with or obstruct the gripper.
if not args.headless:
    scene.draw_debug_sphere(pos=SCOOPING_POINT, radius=0.01, color=(1, 0, 0, 1))

openarm.set_dofs_kp(np.array([230, 230, 190, 190, 30, 30, 30, 30, 30,
                             230, 230, 190, 190, 30, 30, 30]))
openarm.set_dofs_kv(np.array([2.7, 2.7, 2.2, 2.2, 1.5, 1.5, 1.5, 0.2, 0.2,
                             2.7, 2.7, 2.2, 2.2, 1.5, 1.5, 1.5]))
force_limits = np.array([40, 40, 27, 27, 7, 7, 7, 10, 10, 40, 40, 27, 27, 7, 7, 7])
openarm.set_dofs_force_range(-force_limits, force_limits)
low, high = (x.detach().cpu().numpy() for x in openarm.get_dofs_limit())
arm_dofs = np.r_[left_arm, right_arm]
if np.any(joint_plan[:, arm_dofs] < low[arm_dofs]) or np.any(joint_plan[:, arm_dofs] > high[arm_dofs]):
    raise RuntimeError("Planned joint trajectory leaves the joint limits; refusing to execute")

# Start at the replay's first pose before any physics step: at zero joints the
# attached holder would sit below the floor.
openarm.set_qpos(joint_plan[0])
openarm.control_dofs_position(joint_plan[0])
left_gripper = openarm.get_link(LEFT_GRIPPER_PATH)
right_gripper = openarm.get_link(RIGHT_GRIPPER_PATH)
sim_step_count = 0
peak_contact = (0.0, 0.0)   # (force N, sim time s) of the largest spoon-bowl contact


def spoon_bowl_contact():
    """Largest individual and resultant rigid contact between the two wrist links (N)."""
    contacts = openarm.get_contacts()
    forces = [f if a == left_gripper.idx else -f
              for a, b, f in zip(contacts["link_a"].tolist(), contacts["link_b"].tolist(),
                                 contacts["force_a"].detach().cpu().numpy())
              if {a, b} == {left_gripper.idx, right_gripper.idx}]
    if not forces:
        return 0, 0.0, 0.0
    return len(forces), max(float(np.linalg.norm(f)) for f in forces), float(np.linalg.norm(np.sum(forces, axis=0)))


def step():
    global sim_step_count, peak_contact
    scene.step()
    sim_step_count += 1
    # Every step: the README's peaks last only a few 4 ms steps.
    count, peak, resultant = spoon_bowl_contact()
    if peak > peak_contact[0]:
        peak_contact = (peak, sim_step_count * DT)
    if recorder:
        recorder.write('dense_contacts.csv', dict(sim_time_s=sim_step_count * DT,
                       trajectory_time_s=sim_step_count * DT - args.settle_time, contact_count=count,
                       peak_contact_N=peak, resultant_N=resultant))


recorder = None
if args.diagnostics:
    recorder = Recorder(args.diagnostics, {"bowl": bowl_marker_local, "spoon": spoon_locals},
                        dict(trajectory=str(TRAJ_CSV), scooping_point=SCOOPING_POINT.tolist(),
                             task_end_s=float(trajectory_times[-1]), headless=args.headless, liquid=not args.no_liquid,
                             dt=scene.sim.dt, substeps=scene.sim.substeps, pos_tol_m=sk.POS_TOL,
                             plan=plan_metrics, playback="precomputed joint spline, one target per physics step",
                             robot_usd=OPENARM_USD.name, continuity_weight=CONTINUITY_WEIGHT, bowl="rigid geometry of the left wrist link (no weld)",
                             settle_time_s=args.settle_time,
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


print(f"[Settle] Holding the first replay pose for {args.settle_time:.1f} s while the liquid settles", flush=True)
phase_start = time.perf_counter()
for _ in range(round(args.settle_time / DT)):
    step()
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
    openarm.control_dofs_position(command)
    step()
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

print(f"Worst actual marker error during replay: bowl {1000 * worst_error[0]:.1f} mm, "
      f"spoon {1000 * worst_error[1]:.1f} mm")
print(f"Peak spoon-bowl contact: {peak_contact[0]:.1f} N at replay t={peak_contact[1] - args.settle_time:.2f} s "
      "(README, rigid CPU: 2.72 N corrected vs 155.7 N original)")
if recorder:
    recorder.finish()
    atexit.unregister(recorder.finish)
    print(f'Diagnostics saved to {args.diagnostics.resolve()}', flush=True)

# Hold the final (lifted-clear) pose.
while not args.headless and scene.viewer.is_alive():
    step()
