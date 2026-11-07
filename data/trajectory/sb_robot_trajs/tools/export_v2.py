"""Export v2: the 2026-11-06 bowl re-plan (smooth_spin) for every recording whose food simulation executes.

Source (git-ignored): data/diagnostics/2026-11-06/bowl_smoothing_and_spin/
  trajectories/<variant>/sb_traj_XXXX.csv   joint CSVs, same format as v1 (written by tools/replan_bowl.py)
  results/summary.csv                         selection: sim_proposed_scoop_ok == True (food sim executes)
  results/metrics.csv                         plan metrics copied into the manifest
Files are copied byte for byte. Writes v2/manifest.csv and v2/provenance.json (SHA-256 of every source).
Standard library only; existing outputs are replaced only if byte-identical.

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


def sha(b):
    return hashlib.sha256(b).hexdigest()


def write(path, data):
    if path.exists():
        assert path.read_bytes() == data, f'{path} exists and differs; not replaced'
        return
    path.write_bytes(data)


def main():
    DEST.mkdir(exist_ok=True)
    summary = {r['recording']: r for r in csv.DictReader((SESSION / 'results/summary.csv').open())}
    metrics = {(r['recording'], r['variant']): r for r in csv.DictReader((SESSION / 'results/metrics.csv').open())}
    v1 = {r['recording']: r for r in csv.DictReader((V1 / 'manifest.csv').open())}
    rows, prov = [], []
    for rec, s in sorted(summary.items()):
        if s.get('sim_proposed_scoop_ok') != 'True':
            continue
        src = SESSION / 'trajectories' / VARIANT / f'sb_traj_{rec}.csv'
        data = src.read_bytes()
        lines = data.decode().strip().splitlines()
        write(DEST / src.name, data)
        m = metrics[(rec, VARIANT)]
        rows.append(dict(
            recording=rec, file=src.name, variant=VARIANT, configuration=v1[rec]['configuration'],
            samples=len(lines) - 1, duration_s=lines[-1].split(',')[0],
            placement_m=v1[rec]['placement_m'], yaw_deg=v1[rec]['yaw_deg'], v1_file=f'../v1/{v1[rec]["file"]}',
            flat_bowl=s['flat'], tilt_start_hold_end_deg=m['tilt_start_hold_end_deg'],
            tipback_fallback=m['tipback_fallback'], spin_delta_deg=m['spin_delta_deg'],
            xb_from_world_y_deg=m['xb_from_world_y_deg'],
            min_margin_left_deg=m['min_margin_left_deg'], min_margin_right_deg=m['min_margin_right_deg'],
            plan_min_clearance_mm=m['min_clearance_mm'],
            sim_food_kept_ml=s['sim_proposed_retained_ml'], sim_peak_contact_N=s['sim_proposed_peak_contact_N'],
            real_robot_tested='no'))
        prov.append(dict(recording=rec, source=str(src.relative_to(ROOT)), source_sha256=sha(data),
                         output=f'v2/{src.name}', output_sha256=sha(data)))
    buf = io.StringIO(newline='')
    w = csv.DictWriter(buf, list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
    write(DEST / 'manifest.csv', buf.getvalue().encode())
    srcs = {k: SESSION / 'results' / k for k in ('summary.csv', 'metrics.csv')}
    write(DEST / 'provenance.json', (json.dumps(dict(
        generator='data/diagnostics/2026-11-06/bowl_smoothing_and_spin/tools/replan_bowl.py (variant smooth_spin)',
        session_readme='data/diagnostics/2026-11-06/bowl_smoothing_and_spin/README.md',
        selection='results/summary.csv sim_proposed_scoop_ok == True (food simulation executes: no jam/abort, '
                  'food kept, spoon above the rim plane at the end)',
        selection_sources={k: dict(path=str(p.relative_to(ROOT)), sha256=sha(p.read_bytes())) for k, p in srcs.items()},
        excluded=sorted(set(summary) - {r['recording'] for r in rows}),
        files=prov), indent=2) + '\n').encode())
    print(f'{len(rows)} trajectories -> {DEST}; excluded {sorted(set(summary) - {r["recording"] for r in rows})}')


if __name__ == '__main__':
    main()
