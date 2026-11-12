"""Raw recordings in, episode records out.

Raw Qualisys QTM TSV (v2.0.0, mm, lab frame) -> world markers (m) in the planner order, with a per-frame flag for
gap-filled markers. Episode = the canonical record of one human demo (see scooping_ingest.py), saved as .npz without
pickled objects, plus a .json summary next to it.

Migrated unchanged from data/diagnostics/2026-10-13/greenbean_batch/tools/gb_common.py (read_tsv, world_markers,
write_csv) and data/diagnostics/2026-10-19/soybean_calibration/tools/sb_common.py (read_types, load).
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scooping_kinematics import BOWL_NAMES, SPOON_NAMES

NAMES = BOWL_NAMES + SPOON_NAMES                 # planner / CSV order: 5 bowl, then 4 spoon markers
ANCHOR = np.array([0.30, 0.0, 0.40])             # the first Bowl_center sample is placed here (world, m)


def read_tsv(path):
    """QTM TSV: times (s) and {marker name: (N, 3) mm, lab frame}."""
    rows = list(csv.reader(open(path), delimiter='\t'))
    h = next(i for i, r in enumerate(rows) if r and r[0] == 'Frame')
    head, data = rows[h], rows[h + 1:]
    times = np.array([float(r[1]) for r in data])
    mk = {head[i][:-2]: np.array([[float(r[i]), float(r[i + 1]), float(r[i + 2])] for r in data])
          for i, c in enumerate(head) if c.endswith(' X')}
    return times, mk


def read_header(path):
    """QTM TSV header lines before the 'Frame' row, as {key: [values]} (e.g. TIME_STAMP, FREQUENCY)."""
    out = {}
    for r in csv.reader(open(path), delimiter='\t'):
        if r and r[0] == 'Frame':
            break
        if r:
            out[r[0]] = r[1:]
    return out


def read_types(path):
    """(N, 9) bool in NAMES order: True where the marker was measured or rigid-body virtual, False where QTM
    gap-filled it (interpolated while hidden)."""
    rows = list(csv.reader(open(path), delimiter='\t'))
    h = next(i for i, r in enumerate(rows) if r and r[0] == 'Frame'); head = rows[h]
    cols = [head.index(f'{n} Type') for n in NAMES]
    return np.array([[r[c] != 'Gap-filled' for c in cols] for r in rows[h + 1:]])


def world_markers(mk, anchor=ANCHOR):
    """(N, 9, 3) world metres in NAMES order: (x, y, z)_world = (-y, x, z)_lab / 1000, then the first Bowl_center
    sample moved to `anchor`."""
    P = np.stack([mk[n] for n in NAMES], axis=1)
    W = P[..., [1, 0, 2]] / 1000.0
    W[..., 0] *= -1
    return W - W[0, 0] + anchor


def load_take(path):
    """times (s), (N, 9, 3) world markers in NAMES order, and (N, 9) bool: not gap-filled."""
    times, mk = read_tsv(path)
    return times, world_markers(mk), read_types(path)


def write_marker_csv(path, times, M):
    """Marker CSV (Frame, Time, <marker> X/Y/Z) as read by oa_pure_scooping.py --trajectory."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Frame', 'Time'] + [f'{n} {a}' for n in NAMES for a in 'XYZ'])
        for k, (t, P) in enumerate(zip(times, M)):
            w.writerow([k + 1, f'{t:.5f}'] + [f'{v:.7f}' for v in P.ravel()])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_episode(path, arrays, summary):
    """arrays: name -> numpy array (no object arrays); summary: JSON-able dict written to <path>.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        if np.asarray(v).dtype == object:
            raise TypeError(f'episode array {k} is an object array; store strings as a fixed-width str array')
    np.savez_compressed(path, **arrays)
    path.with_suffix('.json').write_text(json.dumps(summary, indent=1))


def load_episode(path):
    """(arrays dict, summary dict) of an episode written by save_episode."""
    path = Path(path)
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    return arrays, json.loads(path.with_suffix('.json').read_text())


JOINT_COLUMNS = ['time_from_start_s'] + [f'openarm_{s}_joint{i}' for s in ('left', 'right') for i in range(1, 8)]


def write_joint_csv(path, tt, q_left, q_right, extra=None):
    """Robot joint CSV (sb_robot_trajs format): time_from_start_s, left 1-7, right 1-7 (rad), '%.6f'.
    extra: optional {column: (n,) values} appended after the 14 joints (e.g. demo_time_s)."""
    cols, data = list(JOINT_COLUMNS), [np.asarray(tt)[:, None], q_left, q_right]
    for k, v in (extra or {}).items():
        cols.append(k); data.append(np.asarray(v)[:, None])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.hstack(data), delimiter=',', header=','.join(cols), comments='', fmt='%.6f')


def read_joint_csv(path):
    """Robot joint CSV -> (times (n,), q_left (n, 7), q_right (n, 7), {other column: (n,)}), columns by name."""
    with open(path, newline='') as f:
        head = next(csv.reader(f))
    data = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    col = {n: data[:, i] for i, n in enumerate(head)}
    missing = [n for n in JOINT_COLUMNS if n not in col]
    if missing:
        raise ValueError(f'{path}: missing columns {missing}')
    q = np.stack([col[n] for n in JOINT_COLUMNS[1:]], axis=1)
    return col['time_from_start_s'], q[:, :7], q[:, 7:], {n: v for n, v in col.items() if n not in JOINT_COLUMNS}
