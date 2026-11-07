# Selected soybean joint trajectories

| Folder | Contents |
|---|---|
| `v1/` | The batch exports described below (21 recordings, 2026-10-21 planner). What the real robot replayed up to 2026-11-06 (`real_runs/`) |
| [`v2/`](#v2-bowl-tilt-then-hold-and-spin-2026-11-06) | Re-plan of 2026-11-06: bowl tilt-then-hold and spin, settled start; 19 recordings whose food simulation executes |

Same CSV format in both. Pick one explicitly, e.g. `data/trajectory/sb_robot_trajs/v2/sb_traj_0008.csv`.

## v1

One joint trajectory per successful recording, selected by `Replay Used` and
`Result` in [`181026_soybean_state_remarks.csv`](../real_trajs/181026_soybean_state_remarks.csv).
Selection is also checked against the October 21 `results/batch_best.csv`.

21 recordings are included: 12 calibrated (`cal`) and 9 calibrated with wall
margin (`calwall`). Recordings **0015, 0020, 0022** have no trajectory because both
configurations jammed. When both variants work, the remarks determine the choice;
for example, 0017 uses `cal`, and 0019 uses `calwall`.

## Files and format

- `v1/sb_traj_0001.csv`, etc.: processed commanded joint positions for both arms.
- [`v1/manifest.csv`](v1/manifest.csv): selection and result for all 24 recordings,
  filenames, sample counts, duration, placement, yaw, and source paths.
- [`v1/provenance.json`](v1/provenance.json): source/output SHA-256 hashes, initial-pose
  sources, and source run metadata including IK and motion metrics.
- [`tools/export.py`](tools/export.py): reproducible exporter (Python standard library), writes `v1/`.

Each trajectory has 15 columns: `time_from_start_s`, then
`openarm_left_joint1` through `openarm_left_joint7`, then
`openarm_right_joint1` through `openarm_right_joint7`.
Angles are radians; time is seconds. Both arms share the same timestamp per row.

The exporter copies `command_q0`–`command_q6` and `command_q9`–`command_q15`
from the selected `sim/rigid_<take>/joint_trajectory.csv` or
`sim/rigid_calwall_<take>/joint_trajectory.csv`. It adds the exact `t=0` command
from that run's first `diagnostics.csv` row. Subsequent rows retain the source
4 ms spacing, values, and timestamps. No IK, smoothing, retiming, or coordinate
transform is applied. Simulation actual positions, forces, wall-clock timings,
and simulated finger joints are omitted. Original recordings remain in diagnostics.

## Replay interpretation

These are the v3 spoon trajectories with the spoon mounted at the right wrist
and the bowl holder fixed relative to the left wrist by its simulated grasp.
Positions already encode the selected placement and yaw; times already encode
the original ×1.6 slowdown. Do not apply these transformations again.

For ROS replay, split each row into left/right seven-joint position arrays and
use the shared times as `JointTrajectoryPoint.time_from_start`, with a common
start timestamp for both arm controllers. The robot must first be moved to the
initial pose. Bowl gripping is controlled separately; the simulator's finger
positions are not exported as physical gripper commands. Settling and final-hold
intervals are not part of the exported replay timeline.

This folder supplies joint position data, not a ROS execution node. Success means
the selected simulation replay succeeded; hardware replay has not been tested.
The original speeds are preserved. Check the source metrics in `provenance.json`
when choosing hardware replay speed, and apply any retiming equally to both arms.

Regenerate or verify from the repository root:

```bash
python data/trajectory/sb_robot_trajs/tools/export.py
```

The exporter validates all inputs before writing, accepts identical existing
outputs, and refuses to replace different existing outputs.

## v2: bowl tilt-then-hold and spin (2026-11-06)

Made by `data/diagnostics/2026-11-06/bowl_smoothing_and_spin/tools/replan_bowl.py`, variant `smooth_spin`. The
details are in that session's README. It re-plans from the same `sim_ready_trajs` markers with the same IK.

| Change vs v1 | What |
|---|---|
| Bowl | Flat demos (0002, 0006, 0012, 0016, 0021): one level pose. Tilted demos: one smooth tilt, then hold still (mean scoop pose); 0024 tips back after the spoon has left |
| Bowl spin | Bowl turned about its own axis so the gripper side (X_B) points along world +Y; every joint at least 5 deg from its limit (fixes the awkward 0008 / 0009 holds) |
| Spoon | Same pose relative to the bowl, de-spiked and low-passed at 4 Hz |
| Start | Settled first frame: no start swing of the redundant DOF (elbow swivel) |

- **Selection:** the food simulation executes. That means no jam or abort, food kept, and the spoon above the rim
  plane at the end. 19 of 21 pass.
- **Excluded:** 0003 and 0017 jam on the bowl floor (zero planned floor margin).
- **Files:**
  - `v2/sb_traj_XXXX.csv`: byte copies of the session's `trajectories/smooth_spin/`.
  - `v2/manifest.csv`: placement, tilt, spin, margins, planned clearance and the food-sim result per recording.
  - `v2/provenance.json`: sources and SHA-256.
- **Exporter:** `python data/trajectory/sb_robot_trajs/tools/export_v2.py` (standard library; needs the git-ignored
  session).
- **Status:** simulation only. Not yet run on the real robot.
