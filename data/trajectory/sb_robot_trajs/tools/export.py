"""Export the successful soybean replays selected by the state remarks CSV.

Run from any directory with Python 3; only the standard library is needed.
Sources are never changed. Existing outputs are replaced only if byte-identical.
"""
import csv
import hashlib
import io
import json
import math
from pathlib import Path


DEST = Path(__file__).resolve().parents[1] / 'v1'   # v1 = these batch exports; v2 = 2026-11-06 re-plan
ROOT = DEST.parents[3]
REMARKS = ROOT / 'data/trajectory/real_trajs/181026_soybean_state_remarks.csv'
SESSION = ROOT / 'data/diagnostics/2026-10-21/soybean_batch'
JOINTS = [f'openarm_{side}_joint{i}' for side in ('left', 'right') for i in range(1, 8)]
INDICES = list(range(7)) + list(range(9, 16))
FIELDS = ['time_from_start_s'] + JOINTS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_bytes(fields, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def converted(row):
    return dict(zip(FIELDS, [row['trajectory_time_s']] + [row[f'command_q{i}'] for i in INDICES]))


def main():
    text = REMARKS.read_text()
    remarks = list(csv.DictReader(io.StringIO(text[text.index('Recording,'):])))
    assert sorted(int(r['Recording']) for r in remarks) == list(range(1, 25))
    best = {r['take']: r for r in read_csv(SESSION / 'results/batch_best.csv')}
    variants = {'calibrated': 'cal', 'calibrated + wall margin': 'calwall', 'none': ''}
    outputs, manifest = {}, []
    provenance = {'selection_source': str(REMARKS.relative_to(ROOT)),
                  'selection_sha256': digest(REMARKS), 'joint_names': JOINTS,
                  'position_units': 'rad', 'time_units': 's', 'recordings': []}
    for row in remarks:
        take = f"{int(row['Recording']):04d}"
        variant = variants[row['Replay Used'].strip()]
        success = row['Result'].strip() == 'success'
        assert success == bool(variant), (take, 'inconsistent remarks')
        assert best[take]['source'] == variant and (best[take]['success'] == 'True') == success
        entry = dict(recording=take, result=row['Result'], configuration=variant,
                     calibrated_replay=row['Calibrated Replay'], wall_margin_replay=row['Wall Margin Replay'],
                     file='', samples='', duration_s='', placement_m=row['Placement (x y z m)'],
                     yaw_deg=row['Turn (deg)'], initial_food_volume_ml=row['Initial Food Volume (ml)'],
                     source='')
        detail = dict(entry)
        if success:
            run = SESSION / 'sim' / (f'rigid_{take}' if variant == 'cal' else f'rigid_calwall_{take}')
            source, initial = run / 'joint_trajectory.csv', run / 'diagnostics.csv'
            metadata = json.loads((run / 'metadata.json').read_text())
            assert metadata['spoon'] == 'v3' and metadata['liquid'] is False
            assert all(p['failing_frames'] == 0 for p in metadata['plan'])
            start = read_csv(initial)[0]
            assert float(start['trajectory_time_s']) == 0
            rows = [converted(start)] + [converted(r) for r in read_csv(source)]
            times = [float(r[FIELDS[0]]) for r in rows]
            assert all(math.isfinite(float(r[f])) for r in rows for f in FIELDS)
            assert all(b > a and math.isclose(b-a, metadata['dt'], abs_tol=1e-8)
                       for a, b in zip(times, times[1:]))
            assert math.isclose(times[-1], metadata['task_end_s'], abs_tol=1e-8)
            assert metadata['scooping_point'] == [float(v) for v in entry['placement_m'].split()]
            assert f'_{variant}_x1.6_yaw{entry["yaw_deg"]}.csv' in metadata['trajectory']
            name = f'sb_traj_{take}.csv'
            outputs[DEST / name] = csv_bytes(FIELDS, rows)
            entry.update(file=name, samples=len(rows), duration_s=times[-1], source=str(source.relative_to(ROOT)))
            detail.update(entry)
            detail.update(source_sha256=digest(source), initial_pose_source=str(initial.relative_to(ROOT)),
                          initial_pose_source_sha256=digest(initial), metadata=metadata,
                          output_sha256=hashlib.sha256(outputs[DEST / name]).hexdigest())
        manifest.append(entry)
        provenance['recordings'].append(detail)
    outputs[DEST / 'manifest.csv'] = csv_bytes(list(manifest[0]), manifest)
    outputs[DEST / 'provenance.json'] = (json.dumps(provenance, indent=2) + '\n').encode()
    expected = {p.name for p in outputs if p.name.startswith('sb_traj_')}
    assert not ({p.name for p in DEST.glob('sb_traj_*.csv')} - expected), 'Unexpected trajectory files'
    # Validate everything before writing; refuse to overwrite a different export.
    for path, data in outputs.items():
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError(f'Existing output differs: {path}')
    for path, data in outputs.items():
        if not path.exists():
            path.write_bytes(data)
    print(f'Validated and exported {len(expected)} trajectories; {len(remarks)-len(expected)} excluded.')


if __name__ == '__main__':
    main()
