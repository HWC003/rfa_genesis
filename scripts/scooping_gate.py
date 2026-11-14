"""Success gate: is a plan accepted for the robot? Plan checks, then food-simulation checks; reasons when not.

Plan checks (scooping_planner metrics, no physics):
  plan_ok             both arms: IK within tolerance, joints >= 5 deg inside the limits, raw IK step <= 5 deg,
                      bowl-arm J2 >= -90 deg (scooping_kinematics.check_plan)                          reject
  start_at_rest       first-step joint speed < trajectory_smoothing.REST_SPEED, when the plan has a start ramp
                                                                                                        reject
  entry_rim_clearance planned spoon-bowl clearance while the dish enters the bowl < entry_warn_mm (4 mm)  warn
                      (2026-11-13 rim_entry_check: 0023 / 0024 at 2.6 / 3.1 mm touched the rim on the robot in
                      v1 and v2; v1 failed, v2 succeeded, so a warning, not a reject)
Food-simulation checks (oa_pure_scooping.py --joints --evaluate: scoop_summary.json, dense_contacts.csv); the
2026-11-06 v2 selection (bowl_smoothing_and_spin/tools/plot_results.py::scoop_ok, jam):
  not_aborted         the simulation's safety gate did not abort                                          reject
  jam                 peak contact < 10 N and > 5 N for < 50 ms (CLAUDE.md)                               reject
  food_kept           food on the spoon at the end (retained_same_ids > 0)                                reject
  end_above_rim       at the end the spoon is above the rim plane in the bowl frame (plan)                 reject
  kept_vs_real        sim kept volume >= kept_warn_frac (50 %) of the weighed real scoop (metadata)       warn
                      (user decision 2026-11-14: small scoops are a known food-acquisition limit, e.g. 0020 at
                      r2 keeps 2.0 of 9.4 mL; decide pass / fail only after a real-robot replay)
Kept volume is otherwise not gated: one GPU run has about +-1 mL of noise (2026-11-12 migration_checks step 3).

decide() -> ('accept' | 'reject', reasons, warnings). Thresholds: GATE_DEFAULTS, overridable in session.yaml gate:.
"""
import csv
import json
from pathlib import Path

import numpy as np

from trajectory_smoothing import REST_SPEED

GATE_DEFAULTS = dict(jam_peak_N=10.0, jam_above_N=5.0, jam_max_ms=50.0, entry_warn_mm=4.0, rest_speed=REST_SPEED,
                     min_kept_particles=1, end_above_rim_mm=0.0, kept_warn_frac=0.5)
SIM_DT = 0.004                                     # scooping_scene.DT: one dense_contacts row per physics step


def check(value, ok, limit, level='reject'):
    return dict(value=value, ok=bool(ok), limit=limit, level=level)


def plan_checks(rep, start_speed, g=GATE_DEFAULTS):
    """rep: the plan's JSON report (scooping_pipeline plan); start_speed: first-step joint speed of its CSV (rad/s)."""
    m = rep['metrics']
    c = dict(plan_ok=check(rep['info']['plan_ok'], rep['info']['plan_ok'], 'both arms pass check_plan'))
    if rep['settings'].get('start_ramp'):
        c['start_at_rest'] = check(round(start_speed, 4), start_speed < g['rest_speed'], f"< {g['rest_speed']} rad/s")
    e = m.get('entry_min_clearance_mm')
    if e is not None and np.isfinite(e):
        c['entry_rim_clearance'] = check(e, e >= g['entry_warn_mm'], f">= {g['entry_warn_mm']} mm", 'warn')
    return c


def jam(dense_contacts_csv, g=GATE_DEFAULTS):
    """(peak N, longest stretch above jam_above_N in ms, pass) over trajectory time >= 0."""
    rows = list(csv.DictReader(Path(dense_contacts_csv).open()))
    t = np.array([float(r['trajectory_time_s']) for r in rows])
    f = np.array([float(r['peak_contact_N']) for r in rows])[t >= 0]
    run = best = 0
    for a in f > g['jam_above_N']:
        run = run + 1 if a else 0; best = max(best, run)
    ms = best * SIM_DT * 1000
    return float(f.max()), ms, bool(f.max() < g['jam_peak_N'] and ms < g['jam_max_ms'])


def sim_checks(sim_dir, end_above_rim_mm, g=GATE_DEFAULTS, real_ml=None):
    """sim_dir: oa_pure_scooping --diagnostics folder of an --evaluate run; real_ml: weighed real scoop (mL) or None."""
    sim_dir = Path(sim_dir)
    s = json.loads((sim_dir / 'scoop_summary.json').read_text())
    peak, ms, ok = jam(sim_dir / 'dense_contacts.csv', g)
    c = dict(
        not_aborted=check(s.get('aborted') or 'no', not s.get('aborted'), 'no abort'),
        jam=check(dict(peak_N=round(peak, 2), ms_above=round(ms, 1)), ok,
                  f"peak < {g['jam_peak_N']} N, > {g['jam_above_N']} N for < {g['jam_max_ms']} ms"),
        food_kept=check(s['retained_same_ids'], s['retained_same_ids'] >= g['min_kept_particles'],
                        f">= {g['min_kept_particles']} particles"),
        end_above_rim=check(end_above_rim_mm, end_above_rim_mm > g['end_above_rim_mm'], f"> {g['end_above_rim_mm']} mm"),
    )
    if real_ml:
        lim = g['kept_warn_frac'] * real_ml
        c['kept_vs_real'] = check(round(s['retained_volume_mL'], 2), s['retained_volume_mL'] >= lim,
                                  f">= {lim:.1f} mL ({g['kept_warn_frac']:.0%} of the real {real_ml} mL)", 'warn')
    return c, dict(kept_ml=round(s['retained_volume_mL'], 2), kept_particles=s['retained_same_ids'],
            peak_contact_N=round(s['peak_rigid_contact_N'], 2), peak_contact_every_step_N=round(peak, 2),
            ms_above_5N=round(ms, 1), sim_failed_checks=s.get('failed_checks'))


def decide(checks):
    """('accept' | 'reject', reasons, warnings) from {name: check}."""
    reasons = [f"{k}: {v['value']} (needs {v['limit']})" for k, v in checks.items() if not v['ok'] and v['level'] == 'reject']
    warnings = [f"{k}: {v['value']} (wants {v['limit']})" for k, v in checks.items() if not v['ok'] and v['level'] == 'warn']
    return ('reject' if reasons else 'accept'), reasons, warnings
