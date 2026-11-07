# Simulation-ready soybean trajectories

The marker trajectories that `scripts/oa_pure_scooping.py` replays for the 21 successful soybean recordings. These
are the same recordings, configuration (`cal` / `calwall`), placement and turn as the robot joint trajectories in
[`../sb_robot_trajs`](../sb_robot_trajs/README.md). Recordings 0015, 0020 and 0022 are absent because they jam.

They are byte-identical copies of `data/diagnostics/2026-10-21/soybean_batch/mapped/`. That folder is not in git, so
these copies let any checkout run the simulation.

## Files
- `sb_<take>_<cal|calwall>_x1.6_yaw<turn>.csv`: bowl and spoon marker positions over time (9 markers, world frame, m).
  Already turned (180°, or 200° for 0009) and slowed ×1.6. `cal` = calibrated spoon; `calwall` = calibrated plus the
  0.5 mm floor / 3 mm wall margin.
- [`manifest.csv`](manifest.csv): the runner values for each recording:

  | Column | Meaning |
  |---|---|
  | `scooping_point` | `--scooping-point` (bowl centre at the first frame, m) |
  | `food_volume_arg` | `--food-volume` that gives the real starting food amount (`initial_food_ml`) in the simulation |
  | `real_scooped_ml` | the human scoop, weighed |
  | `sim_scooped_ml` | the simulated scoop with μ 0.01, γ 0.01, 6 mm (2026-10-26 validation) |
  | `robot_joint_file` | the matching robot joint trajectory in `../sb_robot_trajs/v1` |
- [`provenance.json`](provenance.json): source path and SHA-256 of every file.
- [`tools/export.py`](tools/export.py): regenerates the folder (needs the diagnostics; standard library only).

## Run one
From the repository root, with the values from `manifest.csv` (here 0017):
```bash
GS_SPH_RIGID_PRESSURE_PUSH=0 python scripts/oa_pure_scooping.py --spoon v3 \
  --trajectory data/trajectory/sim_ready_trajs/sb_0017_cal_x1.6_yaw180.csv \
  --scooping-point 0.22 -0.04 0.44 --hold 2 --food-volume 56.51 \
  --sph-rho 960 --sph-mu 0.01 --sph-gamma 0.01 --food-gap 0.030 \
  --gate-clearance-mm -30 --evaluate --diagnostics sim_runs/0017
```
Drop the food arguments and add `--no-liquid` to see the motion only. See the main [README](../../../README.md) for
every runner argument.

The simulated portion is trustworthy for 41.7–105 mL of food. At 20.8 mL the simulation keeps far less than the real
scoop on the tilted takes (0023, 0024).
