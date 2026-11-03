# rfa_genesis
Exploration into Genesis Simulation for Robot Assisted Feeding (RAF) Applications

Diagnostic runs, placement searches and replay comparisons: [dated sessions](data/diagnostics/README.md).

## Scooping replay: `scripts/oa_pure_scooping.py`

Replays one scoop on the OpenArm in Genesis, with the bowl holder already held in the left gripper and the spoon on
the right wrist. The whole joint trajectory is planned before the scene is built:
1. the robot starts at the first pose;
2. the food (SPH liquid) settles in the bowl;
3. the trajectory streams, one target per physics step (dt 4 ms).

It can open the viewer, or run headless and write diagnostics and scoop metrics.

Scene construction is in `scripts/scooping_scene.py`, IK planning in `scripts/scooping_kinematics.py`, and the
food/scoop metrics in `scripts/scooping_metrics.py`.

### Typical commands
Run from the repository root. The soybean trajectories are in
[`data/trajectory/sim_ready_trajs/`](data/trajectory/sim_ready_trajs/README.md); its `manifest.csv` gives each
recording's file, `--scooping-point` and `--food-volume`.

Viewer, rigid only (no food):
```bash
python scripts/oa_pure_scooping.py --spoon v3 \
  --trajectory data/trajectory/sim_ready_trajs/sb_0017_cal_x1.6_yaw180.csv \
  --scooping-point 0.22 -0.04 0.44 --no-liquid --hold 1
```

Viewer with food and scoop evaluation (the validated soybean pudding settings):
```bash
GS_SPH_RIGID_PRESSURE_PUSH=0 python scripts/oa_pure_scooping.py --spoon v3 \
  --trajectory data/trajectory/sim_ready_trajs/sb_0017_cal_x1.6_yaw180.csv \
  --scooping-point 0.22 -0.04 0.44 --hold 2 --food-volume 56.51 \
  --sph-rho 960 --sph-mu 0.01 --sph-gamma 0.01 --food-gap 0.030 \
  --gate-clearance-mm -30 --evaluate --diagnostics sim_runs/0017
```

Headless jam check (rigid, CPU), as the batch scripts run it:
```bash
python scripts/oa_pure_scooping.py --spoon v3 --trajectory <csv> --scooping-point 0.22 -0.04 0.44 \
  --headless --no-liquid --backend cpu --hold 1 --diagnostics <dir>
```
A run passes the jam check when the peak contact in `dense_contacts.csv` is under 10 N and stays above 5 N for less
than 50 ms.

### Arguments

**Trajectory and planning**

| Argument | Default | What it does |
|---|---|---|
| `--trajectory CSV` | green-bean 0017¹ | Marker trajectory (bowl and spoon markers over time), turned into joints by IK |
| `--joint-plan NPZ` | none | Precomputed joint plan (`times`, `q`). Replaces `--trajectory` and IK; a CSV with the same name supplies the markers for recording |
| `--scooping-point X Y Z` | `0.34 -0.13 0.43` | Where the bowl centre at the first frame is placed (m, world). Pure translation of the whole trajectory. The soybean batch uses `0.22 -0.04 0.44`. Ignored with `--joint-plan` |
| `--spoon {v1..v5}` | `v1` | Spoon attachment. v1: original, in line with joint 7. v2: CAD attachment, perpendicular to joint 7. **v3: v2 3.14 mm further from the joint-7 axis (the one used for the soybean work).** v4 / v5: v3 turned +90° / −90° about the spoon axis (dish facing inward / outward). The markers sit in the same place relative to the dish for every version, so any marker CSV runs with any spoon |
| `--continuity-weight W` | `0.012` | How strongly the IK prefers joints close to the previous frame's (prevents branch jumps) |
| `--joint-margin-deg D` | `5` | The planner keeps every joint at least this far inside its limit (deg) |
| `--trim-after-exit` | off | End the replay once the spoon has lifted clear of the bowl (for recordings that then move out of reach) |
| `--weld-bowl` | off | Weld the left wrist to the world at the first pose (stationary bowl) |

¹ `data/trajectory/real_trajs/gb_0017_gen_frame_cropped_corrected.csv`, the corrected cropped green-bean scoop.

**Food (SPH liquid)**

| Argument | Default | What it does |
|---|---|---|
| `--no-liquid` | off | No food: rigid robot and bowl only. Used for jam checks; changes the physics |
| `--food-volume ML` | none | Food amount. The runner places `volume / (0.8 × particle_size³)` particles (Genesis's particle volume). The settled pool is smaller: 0.1275 mL per particle at 6 mm, so **pass V_real / 0.1275 × 0.1728** (e.g. 41.7 mL → `56.51`, 83.3 → `112.84`, 105.2 → `142.56`) |
| `--food-size M` | `0.06` | Edge of the food cube (m) when `--food-volume` is not given |
| `--food-gap M` | `0.03` | Height of the food cube's bottom above the bowl rim centre when it is dropped (m) |
| `--particle-size M` | `0.006` | SPH particle size (m). Calibration and validation were done at 6 mm; 3 mm behaves like a stickier food with the same μ / γ (2026-11-02 particle_size_3mm) |
| `--sph-rho` | `1000` | Food density (kg/m³). Soybean pudding: `960` |
| `--sph-mu` | `0.01` | Viscosity. `0.01` matches the soybean scoops at 41.7–105 mL (2026-10-26 sph_param_validation) |
| `--sph-gamma` | `0.01` | Surface tension / cohesion |
| `--settle-time S` | `2.0` | Seconds the first pose is held while the food settles |
| `--food-vis {recon,particle}` | `recon` | Food rendering: reconstructed surface or particles |

**Evaluation and output**

| Argument | Default | What it does |
|---|---|---|
| `--diagnostics DIR` | none | Write CSVs, plots and timings to this folder |
| `--evaluate` | off | Food/scoop metrics every 20 ms and the success test. Needs food and `--diagnostics` |
| `--gate-clearance-mm MM` | `-2` | With `--evaluate`, abort if the spoon-bowl clearance drops below this. Use `-30` for trajectories that scrape the bowl (the soybean ones). A 30 N contact abort always applies |
| `--hold S` | `0` | Seconds the final pose is held after the replay (the soybean runs use 2 with food, 1 without) |
| `--submerged-window T0 T1` | none | Times when the dish is submerged, for the submerged-contact metric |
| `--render-final` | off | Save `final.png` (spoon at the end) and `pool_t0.png` (bowl at t = 0). Needs `--diagnostics` |
| `--max-frames N` | none | Stop after N frames (smoke test) |

**Viewer and backend**

| Argument | Default | What it does |
|---|---|---|
| `--headless` | off | No viewer; exit after playback |
| `--show-markers` | off | Draw the bowl and spoon marker spheres |
| `--backend {gpu,cpu}` | `gpu` | Physics backend. Rigid-only jam checks run on CPU; food runs use the GPU (run at most 3 at once) |

### Environment variable
`GS_SPH_RIGID_PRESSURE_PUSH=0` turns off the SPH pressure push on moving rigid bodies. Every food run here uses it.

Stock Genesis ignores this variable: it only works after you apply the local patch in
[`patches/genesis_sph_rigid_pressure_push.patch`](patches/genesis_sph_rigid_pressure_push.patch). Without the patch
the run completes without error, but the push stays on (airgap between food and spoon). Apply it again on every
new machine and after every `pip install`/upgrade of `genesis-world`.

**What the patch does** (`genesis/engine/couplers/legacy_coupler.py`, made against genesis-world 1.3.3):
- `LegacyCoupler.build()` creates a 0-d runtime field `sph_rigid_pressure_push` from the variable:
  `0` turns the push off; unset or any other value keeps the upstream behaviour.
- In `_func_collide_with_rigid_geom_robust`, the static-pressure push on non-fixed links runs only when the field
  is `1`.
- It is a runtime field rather than a Python constant, so cached compiled kernels still pick up the setting.

**Apply** (inside the `genesis` conda env, from this package's directory):
```bash
GENESIS_DIR=$(python -c "import genesis, os; print(os.path.dirname(genesis.__file__))")
cp "$GENESIS_DIR/engine/couplers/legacy_coupler.py" "$GENESIS_DIR/engine/couplers/legacy_coupler.py.orig"
patch -p1 -d "$GENESIS_DIR" < patches/genesis_sph_rigid_pressure_push.patch
```
If `patch` reports a rejected hunk, the Genesis version has changed the file. Make the same two edits by hand,
then regenerate the patch.

**Check** that the patch is there and reads the variable:
```bash
grep -c sph_rigid_pressure_push "$GENESIS_DIR/engine/couplers/legacy_coupler.py"   # expect 3
```
After `scene.build()`, `scene.sim._coupler.sph_rigid_pressure_push[None]` should be `0` when the run sets
`GS_SPH_RIGID_PRESSURE_PUSH=0`.

**Viewer setup:** don't export `MUJOCO_GL=egl` globally (e.g. in `~/.bashrc`). It switches pyglet to headless EGL
and the Genesis viewer fails with `Unable to initialize an OpenGL 3+ context`. Set it only for the commands that
need headless MuJoCo rendering.

### Output in `--diagnostics`

| File | Contents |
|---|---|
| `metadata.json` | Arguments, planner metrics (IK error, joint-limit margin, velocities), robot asset |
| `joint_trajectory.csv` | Commanded joints over time (`command_q0–6` left arm, `command_q9–15` right arm). Source of the real-robot joint CSVs (`data/trajectory/sb_robot_trajs/tools/export.py`) |
| `diagnostics.csv`, `tracking_errors.png`, `joint_positions.png` | Actual vs commanded joints and marker tracking |
| `dense_contacts.csv`, `bowl_contacts.csv` | Rigid contact forces every step (jam check) |
| `timings.json`, `timing.png` | Wall time per phase |
| With `--evaluate`: `scoop_summary.json` | `success`, `failed_checks`, `retained_volume_mL` (kept food at the settled 0.1275 mL per particle), `initial_volume_mL`, particle counts, peak contact |
| With `--evaluate`: `metrics.csv`, `snapshots.npz`, `settled_particles.npz` | Metrics over time, particle positions at key moments, the settled pool |
