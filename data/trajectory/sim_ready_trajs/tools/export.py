"""Copy the simulation inputs of the selected soybean replays out of data/diagnostics (which git ignores).

One marker CSV per recording in ../sb_robot_trajs/manifest.csv, in the same configuration (cal / calwall), placement
and turn: data/diagnostics/2026-10-21/soybean_batch/mapped/sb_<take>_<configuration>_x1.6_yaw<yaw>.csv, copied
byte for byte. Writes manifest.csv (runner values per recording) and provenance.json (SHA-256 of every source).

Run from any directory with Python 3 on the machine that has the diagnostics; only the standard library is needed.
Sources are never changed. Existing outputs are replaced only if byte-identical.
"""
import csv
import hashlib
import io
import json
from pathlib import Path

DEST = Path(__file__).resolve().parents[1]
ROOT = DEST.parents[2]
ROBOT = ROOT / 'data/trajectory/sb_robot_trajs/manifest.csv'
STATE = ROOT / 'data/trajectory/real_trajs/181026_soybean_state.csv'
MAPPED = ROOT / 'data/diagnostics/2026-10-21/soybean_batch/mapped'
VALIDATION = ROOT / 'data/diagnostics/2026-10-26/sph_param_validation/results/validation.csv'
V_SET, P_NOM = 0.1275, 0.8 * 0.006 ** 3 * 1e6     # settled mL per particle (6 mm); mL per particle as placed
FIELDS = ['recording', 'configuration', 'file', 'scooping_point', 'yaw_deg', 'initial_food_ml', 'food_volume_arg',
          'real_scooped_ml', 'sim_scooped_ml', 'robot_joint_file']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    if path.exists() and path.read_bytes() != data:
        raise SystemExit(f'refusing to replace {path}: it differs from the new output')
    path.write_bytes(data)


def real_scooped():
    rows = list(csv.reader(STATE.open(newline='')))
    head = next(i for i, r in enumerate(rows) if r and r[0] == 'Recording')
    return {f'{int(r[0]):04d}': r[4] for r in rows[head + 1:] if r and r[0].strip().isdigit()}


def main():
    robot = [r for r in csv.DictReader(ROBOT.open(newline='')) if r['result'] == 'success']
    real = real_scooped()
    sim = {r['take']: r for r in csv.DictReader(VALIDATION.open(newline=''))} if VALIDATION.exists() else {}
    rows, prov = [], []
    for r in robot:
        take, cfg, yaw = r['recording'], r['configuration'], r['yaw_deg']
        src = MAPPED / f'sb_{take}_{cfg}_x1.6_yaw{yaw}.csv'
        data = src.read_bytes()
        write(DEST / src.name, data)
        V = float(r['initial_food_volume_ml'])
        v = sim.get(take, {})
        rows.append(dict(recording=take, configuration=cfg, file=src.name, scooping_point=r['placement_m'], yaw_deg=yaw,
                         initial_food_ml=f'{V:g}', food_volume_arg=f'{round(V / V_SET) * P_NOM:.2f}',
                         real_scooped_ml=real.get(take, ''),
                         sim_scooped_ml=f'{float(v["after_mL"]):.1f}' if v.get('replay') == cfg and v.get('after_mL') else '',
                         robot_joint_file=r['file']))
        prov.append(dict(recording=take, file=src.name, source=str(src.relative_to(ROOT)), sha256=digest(data)))
    stream = io.StringIO(newline='')
    w = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n'); w.writeheader(); w.writerows(rows)
    write(DEST / 'manifest.csv', stream.getvalue().encode())
    write(DEST / 'provenance.json', (json.dumps(dict(
        note='marker CSVs copied unchanged; already turned (yaw_deg) and slowed x1.6. food_volume_arg = round(V / 0.1275) '
             'x 0.1728 mL; sim_scooped_ml = 2026-10-26 sph_param_validation set A (mu 0.01, gamma 0.01, 6 mm) when it '
             'used the same configuration', files=prov), indent=1) + '\n').encode())
    print(f'{len(rows)} trajectories -> {DEST}')


if __name__ == '__main__':
    main()
