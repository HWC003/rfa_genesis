"""Replay a recorded joint-angle CSV (data/trajectory/sb_robot_trajs/sb_traj_XXXX.csv) in Genesis.

No IK, no markers: the 14 arm joints are interpolated onto the 4 ms physics step and streamed as position
targets. Same scene as oa_pure_scooping.py (scooping_scene.build_scene, spoon v3 by default, SPH food placed above
the bowl at the first pose). The food volume comes from data/trajectory/sim_ready_trajs/manifest.csv unless given.
The start is smoothed as on the real robot (oa_real_scooping_replay.py, trajectory_smoothing.py); --blend-window 0
replays the CSV unchanged.

  GS_SPH_RIGID_PRESSURE_PUSH=0 python scripts/oa_joint_replay.py data/trajectory/sb_robot_trajs/sb_traj_0017.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import scooping_kinematics as sk
import scooping_scene as ss
from trajectory_smoothing import smooth_start, peak_velocity

MANIFEST = Path(__file__).resolve().parent.parent / 'data' / 'trajectory' / 'sim_ready_trajs' / 'manifest.csv'

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('csv', type=Path, help='Joint CSV: time_from_start_s, 7 left joints, 7 right joints (rad)')
p.add_argument('--spoon', choices=sorted(ss.SPOONS), default='v3')
p.add_argument('--food-volume', type=float, help='Food volume (mL); default: food_volume_arg from manifest.csv')
p.add_argument('--no-liquid', action='store_true')
p.add_argument('--particle-size', type=float, default=0.006)
p.add_argument('--sph-rho', type=float, default=960.0)
p.add_argument('--sph-mu', type=float, default=0.01)
p.add_argument('--sph-gamma', type=float, default=0.01)
p.add_argument('--food-gap', type=float, default=0.03, help='Food bottom above the bowl rim (m)')
p.add_argument('--settle-time', type=float, default=2.0, help='Hold the first pose while the food settles (s)')
p.add_argument('--hold', type=float, default=2.0, help='Hold the last pose (s)')
p.add_argument('--blend-window', type=float, default=0.3, help='CSV start seconds retimed to start from rest (0 = off)')
p.add_argument('--blend-duration', type=float, default=0.7, help='Seconds the blend window is stretched to (<= 2.5 x window)')
p.add_argument('--backend', choices=['gpu', 'cpu'], default='gpu')
p.add_argument('--headless', action='store_true')
args = p.parse_args()

# ---- joints: CSV -> 16-DOF targets at every physics step
data = np.loadtxt(args.csv, delimiter=',', skiprows=1)
t, arms = smooth_start(data[:, 0], data[:, 1:15], args.blend_window, args.blend_duration)
if args.blend_window > 0:
    print(f'[Replay] smooth start: peak joint speed {peak_velocity(data[:, 0], data[:, 1:15]):.2f} -> '
          f'{peak_velocity(t, arms):.2f} rad/s', flush=True)
steps = np.arange(t[0], t[-1] + 1e-9, ss.DT)
q = np.tile(ss.SEED_Q, (len(steps), 1))
for j, dof in enumerate(np.r_[ss.LEFT_ARM, ss.RIGHT_ARM]):
    q[:, dof] = np.interp(steps, t, arms[:, j])
print(f'[Replay] {args.csv.name}: {len(t)} rows, {t[-1]:.2f} s', flush=True)

# ---- food volume from the manifest
food = None
if not args.no_liquid:
    volume = args.food_volume
    if volume is None:
        rows = [r for r in csv.DictReader(MANIFEST.open()) if r['robot_joint_file'] == args.csv.name]
        if not rows:
            p.error(f'{args.csv.name} not in {MANIFEST.name}; pass --food-volume')
        volume = float(rows[0]['food_volume_arg'])
    box = ss.food_box(volume, args.particle_size)
    usd = ss.robot_usd(False, args.spoon)
    wrist, _ = sk.ArmKinematics('left').fk(q[0, ss.LEFT_ARM])
    rim = wrist[:3, 3] + wrist[:3, :3] @ ss.marker_templates(usd)['rim_center_local']
    food = dict(pos=rim + [0, 0, args.food_gap + box[2] / 2], size=box, particle_size=args.particle_size,
                mu=args.sph_mu, rho=args.sph_rho, gamma=args.sph_gamma)
    print(f'[Food] {volume} mL', flush=True)

# ---- scene + playback
built = ss.build_scene(ss.robot_usd(False, args.spoon), q[0], food=food, backend=args.backend, headless=args.headless)
scene, robot = built['scene'], built['robot']

for _ in range(round(args.settle_time / ss.DT)):
    scene.step()
for i, target in enumerate(q):
    robot.control_dofs_position(target)
    scene.step()
    if i % 250 == 0:
        print(f'[Replay] t={steps[i]:.2f} s', flush=True)
for _ in range(round(args.hold / ss.DT)):
    scene.step()
print('[Replay] done', flush=True)

while not args.headless and scene.viewer.is_alive():
    scene.step()
