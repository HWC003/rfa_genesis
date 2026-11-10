"""Export v2: the 2026-11-06 bowl re-plan (smooth_spin) for every recording whose food simulation executes.

Source (git-ignored): data/diagnostics/2026-11-06/bowl_smoothing_and_spin/
  trajectories/<variant>/sb_traj_XXXX.csv   joint CSVs, same format as v1 (written by tools/replan_bowl.py)
  results/summary.csv                         selection: sim_proposed_scoop_ok == True (food sim executes)
  results/metrics.csv                         plan metrics copied into the manifest
Files are copied byte for byte. Writes v2/manifest.csv and v2/provenance.json (SHA-256 of every source).
Standard library only; existing outputs are replaced only if byte-identical.

Revisions (REVISIONS): a recording taken from another session variant. Its existing v2 file is replaced only if it
is byte-identical to the superseded source (recorded in provenance as 'replaces').
  2026-11-10  0010 -> smooth_spin_j2: bowl-arm J2 rule (usual arm pose instead of shoulder-up; same bowl path).
              Real-robot runs in real_runs/v2/ dated before 2026-11-10 13:00 used the superseded file.
v1 placement/yaw come from v1/manifest.csv when present, else from sim_ready_trajs/manifest.csv.

  python data/trajectory/sb_robot_trajs/tools/export_v2.py
"""
import csv
import hashlib
import io
import json
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / 'v2'
ROOT = DEST.parents[3]
SESSION = ROOT / 'data/diagnostics/2026-11-06/bowl_smoothing_and_spin'
VARIANT = 'smooth_spin'
V1 = DEST.parent / 'v1'
SIM_READY = ROOT / 'data/trajectory/sim_ready_trajs/manifest.csv'
REVISIONS = {'0010': dict(variant='smooth_spin_j2', metrics='metrics_0010_0001_j2.csv', date='2026-11-10',
                          reason='bowl-arm J2 rule: J2 < -90 deg (shoulder-up branch) re-solved from the usual pose; '
                                 'same bowl path within 0.4 mm / 0.14 deg; food sim executes (11.9 mL, 1.4 N)')}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def write(path, data, superseded=None):
    if path.exists() and path.read_bytes() != data:
        assert superseded is not None and path.read_bytes() == superseded, f'{path} exists and differs; not replaced'
    path.write_bytes(data)


def main():
    DEST.mkdir(exist_ok=True)
    summary = {r['recording']: r for r in csv.DictReader((SESSION / 'results/summary.csv').open())}
    metrics = {(r['recording'], r['variant']): r for r in csv.DictReader((SESSION / 'results/metrics.csv').open())}
    for rec, rv in REVISIONS.items():
        for r in csv.DictReader((SESSION / 'results' / rv['metrics']).open()):
            if r['recording'] == rec:
                metrics[(rec, r['variant'] + rv['variant'][len(VARIANT):])] = r     # e.g. smooth_spin -> smooth_spin_j2
    if (V1 / 'manifest.csv').exists():
        v1 = {r['recording']: r for r in csv.DictReader((V1 / 'manifest.csv').open())}
    else:
        v1 = {r['recording']: dict(configuration=r['configuration'], placement_m=r['scooping_point'],
                                   yaw_deg=r['yaw_deg'], file=r['robot_joint_file'])
              for r in csv.DictReader(SIM_READY.open())}
    rows, prov = [], []
    for rec, s in sorted(summary.items()):
        if s.get('sim_proposed_scoop_ok') != 'True':
            continue
        rv = REVISIONS.get(rec)
        var = rv['variant'] if rv else VARIANT
        src = SESSION / 'trajectories' / var / f'sb_traj_{rec}.csv'
        old = SESSION / 'trajectories' / VARIANT / f'sb_traj_{rec}.csv'
        data = src.read_bytes()
        lines = data.decode().strip().splitlines()
        write(DEST / src.name, data, superseded=old.read_bytes() if rv else None)
        m = metrics[(rec, var)]
        food, peak = s['sim_proposed_retained_ml'], s['sim_proposed_peak_contact_N']
        if rv:
            sim = json.loads((SESSION / 'sim_runs' / var / rec / 'scoop_summary.json').read_text())
            food, peak = f"{sim['retained_volume_mL']:.2f}", f"{sim['peak_contact_every_step_N']:.2f}"
        rows.append(dict(
            recording=rec, file=src.name, variant=var, revision=rv['date'] if rv else '', configuration=v1[rec]['configuration'],
            samples=len(lines) - 1, duration_s=lines[-1].split(',')[0],
            placement_m=v1[rec]['placement_m'], yaw_deg=v1[rec]['yaw_deg'], v1_file=f'../v1/{v1[rec]["file"]}',
            flat_bowl=s['flat'], tilt_start_hold_end_deg=m['tilt_start_hold_end_deg'],
            tipback_fallback=m['tipback_fallback'], spin_delta_deg=m['spin_delta_deg'],
            xb_from_world_y_deg=m['xb_from_world_y_deg'],
            min_margin_left_deg=m['min_margin_left_deg'], min_margin_right_deg=m['min_margin_right_deg'],
            plan_min_clearance_mm=m['min_clearance_mm'],
            sim_food_kept_ml=food, sim_peak_contact_N=peak,
            real_robot_tested='see data/diagnostics/2026-11-10/real_runs_v2_analysis'))
        prov.append(dict(recording=rec, source=str(src.relative_to(ROOT)), source_sha256=sha(data),
                         output=f'v2/{src.name}', output_sha256=sha(data),
                         **(dict(revision=rv['date'], reason=rv['reason'],
                                 replaces=dict(source=str(old.relative_to(ROOT)), sha256=sha(old.read_bytes())))
                            if rv else {})))
    buf = io.StringIO(newline='')
    w = csv.DictWriter(buf, list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
    mf = DEST / 'manifest.csv'
    write(mf, buf.getvalue().encode(), superseded=mf.read_bytes() if mf.exists() and REVISIONS else None)
    srcs = {k: SESSION / 'results' / k for k in ('summary.csv', 'metrics.csv')}
    pv = DEST / 'provenance.json'
    write(pv, (json.dumps(dict(
        generator='data/diagnostics/2026-11-06/bowl_smoothing_and_spin/tools/replan_bowl.py (variant smooth_spin)',
        session_readme='data/diagnostics/2026-11-06/bowl_smoothing_and_spin/README.md',
        selection='results/summary.csv sim_proposed_scoop_ok == True (food simulation executes: no jam/abort, '
                  'food kept, spoon above the rim plane at the end)',
        selection_sources={k: dict(path=str(p.relative_to(ROOT)), sha256=sha(p.read_bytes())) for k, p in srcs.items()},
        excluded=sorted(set(summary) - {r['recording'] for r in rows}),
        revisions={k: dict(v) for k, v in REVISIONS.items()},
        files=prov), indent=2) + '\n').encode(), superseded=pv.read_bytes() if pv.exists() and REVISIONS else None)
    print(f'{len(rows)} trajectories -> {DEST}; excluded {sorted(set(summary) - {r["recording"] for r in rows})}')


if __name__ == '__main__':
    main()
