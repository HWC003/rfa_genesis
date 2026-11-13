"""Diagram of the implemented post-processing pipeline (docs/scooping_pipeline/README.md).

Colours follow the user's 'Post Processing Pipeline Proposal' sketch: grey = raw inputs, blue = RGB / FT (not used
yet), red = metadata, purple = post-processing, green = simulation validation, teal = robot, orange = policy data.
Dashed border = not implemented yet.

    python docs/scooping_pipeline/make_figure.py      # -> docs/scooping_pipeline/pipeline.png (matplotlib only)
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / 'pipeline.png'
INK, SUB, BG = '#111111', '#3d3c38', '#fbfaf3'
K = dict(raw=('#e9e8e3', '#222222'), wip=('#e3eefb', '#2a62b8'), meta=('#fde3e1', '#d23b2f'),
         post=('#f1e4f4', '#8b3fa0'), sim=('#dff1e7', '#1f8a5b'), rl=('#fff0dc', '#d97a00'),
         ros=('#e2f4f1', '#138a7a'), fix=('#fff8d6', '#a07800'))


def box(ax, x, y, w, h, kind, title, body, dashed=False):
    fc, ec = K[kind]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.12', fc=fc,
                                ec=ec, lw=1.6, ls=(0, (5, 3)) if dashed else '-'))
    ax.text(x - w / 2 + .15, y + h / 2 - .14, title, ha='left', va='top', fontsize=10.5, fontweight='bold', color=ec)
    ax.text(x - w / 2 + .15, y + h / 2 - .52, body, ha='left', va='top', fontsize=8, color=SUB, linespacing=1.35)
    return dict(x=x, y=y, w=w, h=h)


def P(n, s, f=0.5):
    x0, y0 = n['x'] - n['w'] / 2, n['y'] - n['h'] / 2
    return dict(b=(x0 + f * n['w'], y0), t=(x0 + f * n['w'], y0 + n['h']), l=(x0, y0 + f * n['h']),
                r=(x0 + n['w'], y0 + f * n['h']))[s]


def arrow(ax, pts, color=INK, ls='-', label=None, lpos=None):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.3, ls=ls)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle='-|>', mutation_scale=13, lw=1.3, color=color, ls=ls,
                                 shrinkA=0, shrinkB=1))
    if label:
        ax.text(*lpos, label, fontsize=7.8, color=color, ha='center', va='center', style='italic',
                bbox=dict(fc=BG, ec='none', pad=1.2))


fig, ax = plt.subplots(figsize=(17, 21), facecolor=BG)
ax.set_xlim(0, 17); ax.set_ylim(0, 21); ax.axis('off')
ax.text(8.5, 20.6, 'Scooping post-processing pipeline (implemented, 2026-11-13)', ha='center', fontsize=19,
        fontweight='bold', color=INK, family='serif')
ax.text(8.5, 20.2, 'scripts/scooping_pipeline.py {calibrate | ingest | plan | validate | export} SESSION.yaml'
        '   ·   dashed = not implemented yet', ha='center', fontsize=10, color=SUB)

# inputs
raw = box(ax, 2.6, 18.6, 4.4, 1.9, 'raw', 'Raw demos (QTM TSV)',
          '14 markers @ 100 Hz per take\n5 bowl-holder + 4 spoon markers used\nsession.yaml: raw_dir, tsv pattern, takes')
calt = box(ax, 7.4, 18.6, 4.0, 1.9, 'raw', 'Calibration take',
           'spoon tip rubbed over the empty bowl\n(soybean0029); layout takes\nsession.yaml: calibration:')
meta = box(ax, 11.8, 18.6, 3.8, 1.9, 'meta', 'Metadata',
           'metadata.csv: take, initial_food_ml,\nreal_scooped_ml (weighed)\nsession.yaml: food type, spoon')
wip = box(ax, 15.3, 18.6, 3.0, 1.9, 'wip', 'RGB / FT', 'spoon camera 30 Hz,\nFT 200 Hz (scoop_08)\nnot ingested yet',
          dashed=True)

# post-processing
cal = box(ax, 7.4, 15.9, 4.4, 2.0, 'post', '1  calibrate  (per session)',
          'scooping_calibration: bowl marker layout,\nspoon-vs-bowl ICP (bowl shift t, spoon shift s)\n'
          'gate: cross-validated RMS < 1 mm\n-> calibration.json')
ing = box(ax, 2.6, 15.9, 4.4, 2.0, 'post', '2  ingest  (per take)',
          'scooping_ingest: robust Kabsch poses H(t), S(t),\ncalibration applied, phases (standard), crop\n'
          '[entry - 0.5 s, lift + 0.5 s], depth\n-> episodes/<take>.npz  (canonical record)')

# plan
plan = box(ax, 4.6, 12.3, 8.6, 2.6, 'post', '3  plan  (retarget to the robot; per take, rung k)',
           'scooping_motion: spoon margin (cal | calwall) -> yaw / placement -> x time_scale\n'
           '-> bowl hold (min-jerk tilt, held; flat = level) + spoon-in-bowl 4 Hz -> spin to 180 + delta\n'
           'scooping_kinematics: IK both arms (settled frame 0, bowl-arm J2 rule), check_plan\n'
           'ladder rungs: floor lift (spoon raised along e where clearance < 1.5 / 5 mm)\n'
           'start ramp (quintic, 0.3 s over 0.7 s) written into the file + demo_time_s column\n'
           '-> plans/<take>/r<k>.csv + .json (metrics)')

# validate
val = box(ax, 4.6, 8.6, 8.6, 2.6, 'sim', '4  validate  (simulation + gate)',
          'plan checks (scooping_gate): plan_ok both arms, starts at rest, rim-entry clearance (warn < 4 mm)\n'
          'food simulation: oa_pure_scooping.py --joints r<k>.csv --evaluate (SPH soybean, <= 3 GPU runs)\n'
          'sim checks: no abort, jam (< 10 N, > 5 N for < 50 ms), food kept, spoon above the rim at the end\n'
          '-> sim/<take>/r<k>/, gate/<take>.json, gate.json')
lad = box(ax, 12.4, 10.4, 3.5, 2.0, 'fix', 'Correction ladder',
          'fail -> next rung (session ladder:)\nr1 floor lift 1.5 mm, r2 floor lift 5 mm\n'
          '(any plan setting can be a rung)\nall rungs fail -> reject with reasons')

# export
rob = box(ax, 3.2, 4.9, 5.6, 2.2, 'ros', '5a  export robot',
          'accepted plans, byte copies (= the simulated file)\nexport/<name>/robot/sb_traj_<take>.csv\n'
          'manifest.csv (plan + sim + weights) and\nprovenance.json (SHA-256, code revision)')
pol = box(ax, 12.3, 4.9, 5.6, 2.2, 'rl', '5b  export policy (bowl frame)',
          'every episode, whatever the gate says\nexport/<name>/policy/<take>.npz: T_BS, tip_B, dish_B,\n'
          'gravity_B, phases, t_qtm + <take>.json labels\n(food, weights, gate, sim kept mL); schema.json')
real = box(ax, 3.2, 1.8, 5.6, 2.0, 'ros', 'Real robot',
           'oa_real_scooping_replay.py --preview-approach / --execute\nCSV starts at rest -> no extra smoothing\n'
           '-> real_runs/<traj>_<time>/ commanded + measured')
rl = box(ax, 12.3, 1.8, 5.6, 2.0, 'rl', 'RL / VLA dataset writer',
         'model-specific format, sync of RGB / FT\n(migration steps 6-7)', dashed=True)

arrow(ax, [P(calt, 'b'), P(cal, 't')])
arrow(ax, [P(raw, 'b'), P(ing, 't')])
arrow(ax, [P(cal, 'l'), P(ing, 'r')], label='calibration.json', lpos=(4.98, 17.15))
arrow(ax, [P(ing, 'b', 0.3), P(plan, 't', 0.1)], label='episode', lpos=(1.55, 14.3))
arrow(ax, [P(plan, 'b'), P(val, 't')], label='plans/<take>/r<k>.csv', lpos=(4.6, 10.4))
arrow(ax, [P(val, 'r', 0.75), (12.4, 8.6 + 0.65), P(lad, 'b')], color=K['fix'][1], label='reject', lpos=(11.5, 9.25))
arrow(ax, [P(lad, 't'), (12.4, 12.3), P(plan, 'r')], color=K['fix'][1], label='re-plan with rung k + 1',
      lpos=(10.3, 12.62))
arrow(ax, [P(val, 'b', 0.2), P(rob, 't', 0.5)], color=K['sim'][1], label='accept', lpos=(3.0, 6.6))
arrow(ax, [P(rob, 'b'), P(real, 't')], color=K['ros'][1], label='sb_traj_<take>.csv', lpos=(3.2, 3.25))
arrow(ax, [P(ing, 'b', 0.85), (2.6 + 1.54, 14.55), (14.35, 14.55), (14.35, 6.0)], color=K['rl'][1],
      label='episodes (all takes)', lpos=(12.2, 14.55))
arrow(ax, [P(val, 'r', 0.25), (11.0, 8.6 - 0.65), (11.0, 6.0)], color=K['rl'][1], ls='--', label='labels',
      lpos=(11.0, 7.2))
arrow(ax, [P(meta, 'b', 0.7), (12.46, 16.9), (14.85, 16.9), (14.85, 6.0)], color=K['meta'][1], label='weights',
      lpos=(14.85, 11.5))
arrow(ax, [P(meta, 'b', 0.1), (10.28, 9.0), (8.9, 9.0)], color=K['meta'][1], label='food volume',
      lpos=(10.28, 13.4))
arrow(ax, [P(pol, 'b'), P(rl, 't')], color=K['rl'][1], ls='--')
arrow(ax, [P(real, 'r'), P(rl, 'l')], color=K['ros'][1], ls='--', label='robot replays (possibly)', lpos=(7.6, 2.05))
arrow(ax, [P(wip, 'b', 0.8), (16.2, 2.5), P(rl, 'r', 0.35)], color=K['wip'][1], ls='--')

fig.savefig(OUT, dpi=110, bbox_inches='tight', facecolor=BG)
print('written', OUT)
