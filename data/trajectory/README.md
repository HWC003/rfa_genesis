# Trajectories

| Folder | Contents |
|---|---|
| `real_trajs/` | Recorded (Qualisys) feeding demonstrations and their direct mappings to the robot world |
| `gen_trajs/` | Generated (designed / planned) trajectories |
| `old_tb_traj/` | Older corrective trajectories, kept for reference |

## real_trajs/

| File | What it is | Made by |
|---|---|---|
| `green_bean0017.csv` / `.tsv` | Raw Qualisys recording (mm) | capture |
| `green_bean0017_cropped.csv` | The recording cropped to the scoop (crop tool not recorded) | manual |
| `gb_0017_gen_frame.csv` | Uncropped recording mapped to the robot world: mm→m, (x, y, z) → (−y, x, z), first `Bowl_center` at (0.30, 0, 0.40) | `scripts/feeding_mapping.py` |
| `gb_0017_gen_frame_cropped.csv` | Same mapping of the cropped recording (600 rows, 5.99 s). `scooping_kinematics.TRAJ_CSV` | `scripts/feeding_mapping.py` |
| `gb_0017_gen_frame_cropped_corrected.csv` | The cropped mapping with the spoon path corrected for spoon–bowl contact and timestamps stretched ×1.6 (9.584 s). Default of `oa_pure_scooping.py` | copied unchanged from `data/diagnostics/2026-09-28/cropped_contact_correction/trajectories/` (see its README) |

## gen_trajs/

### `gen_scoop_traj_34deg.{npz,csv,json}` — stationary-bowl wall scoop, bowl tilted 34.5°

- **Source:** byte-identical copies of `data/diagnostics/2026-09-29/stationary_bowl_acquisition/trajectories/wallscoop_v9.*`
  (design `v9` in that session; SHA-256 prefixes: npz `59d12f76e964117c`, csv `ffb3f5b073410ef5`, json `8f273901808deb90`).
  The json still says `"design": "v9"`; that is its provenance, not a naming convention.
- **Generator** (diagnostic, not migrated): `2026-09-29/.../tools/wallscoop_generate.py v9`, waypoints in `tools/wallscoop_designs.py`.
- **Derivation:** only frame 0 of `real_trajs/gb_0017_gen_frame_cropped.csv` is used (moved to the screened placement
  (0.34, −0.13, 0.43)): it fixes the bowl pose, frozen for the whole trajectory, tilt 34.47°, and the spoon's start pose.
  The scoop itself is designed in bowl coordinates (downhill *d*, lateral *l*, bowl-normal *e*; heading/pitch/roll)
  with the spoon kept 3 mm above the bowl's inner surface, then planned with `scooping_kinematics` IK (continuity weight 0.012).
- **Contents:**
  - `npz`: `times` (691 samples, 0–13.8 s, 20 ms), `q` (691 × 16 joint plan, both arms; the left arm is constant),
    `target_spoon`, `target_bowl`, `holder_to_link`, `bowl_markers`, `spoon_markers`, `planned_gap`, `hug_shift`;
  - `csv`: the same plan as world marker positions (the usual 9-marker format);
  - `json`: design waypoints, IK metrics, planned clearance (≥ 3.0 mm), bowl tilt.
- **Intended use:** replay the joint plan with the bowl welded at its frame-0 pose:
  `oa_pure_scooping.py --joint-plan data/trajectory/gen_trajs/gen_scoop_traj_34deg.npz --weld-bowl`.
- **Phases (trajectory time):**

  | t (s) | Phase |
  |---|---|
  | 0–1 | hold the start pose |
  | 1–3 | approach |
  | 3–3.4 | entry at the low wall: tip down 42°, dish on its side (roll −95°), heading 62° |
  | 3.4–7.4 | sweep 30 mm across the trough, rolling face-up, hugging the bowl at 3 mm |
  | 7.4–8 | hold, dish level and submerged |
  | 8–11.8 | tip down 17°, point uphill (heading 78°), slide 30 mm uphill out of the pool, roll +6° while leaving, pause |
  | 11.8–13.8 | lift 95 mm along the bowl axis |
- **Validated in:** 2026-09-29 (SPH 6 mm, 6/6), 2026-10-02 (SPH 3 mm 2/2, MPM viscous 2/2), 2026-10-06 (SPH 3 mm push off, 1 run).
  It needs a brim-full pool: with 55 mL of food v9 does not acquire (2026-10-06).
