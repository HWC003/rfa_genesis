"""Measurements for oa_baseline_simple; no robot control or trajectory replay."""
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation


def numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def fit_pose(local, world):
    """Rigid least-squares fit. Position is marker centroid; quaternion is xyzw.

    Orientation maps the USD template axes into world axes. No scale is fitted.
    The residual measures non-rigidity/calibration error, independent of robot IK.
    """
    local, world = np.asarray(local), np.asarray(world)
    a, b = local - local.mean(0), world - world.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rot = Rotation.from_matrix(vt.T @ correction @ u.T)
    residual = np.linalg.norm(rot.apply(a) + world.mean(0) - world, axis=1)
    return world.mean(0), rot, float(np.max(residual))


class Recorder:
    def __init__(self, directory, templates, metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.templates = templates
        self.rows = []
        self.started = time.perf_counter()
        self.files = []
        self.writers = {}
        self.events = []
        (self.directory / 'metadata.json').write_text(json.dumps(metadata, indent=2))

    def write(self, name, row):
        if name not in self.writers:
            f = (self.directory / name).open('w', newline='')
            self.files.append(f)
            self.writers[name] = csv.DictWriter(f, fieldnames=list(row))
            self.writers[name].writeheader()
        self.writers[name].writerow(row)
        for f in self.files:
            f.flush()

    def event(self, phase, seconds):
        self.events.append(dict(phase=phase, wall_s=seconds))
        (self.directory / 'timings.json').write_text(json.dumps(self.events, indent=2))

    def joints(self, index, trajectory_time, sim_time, command, actual):
        row = dict(index=index, trajectory_time_s=trajectory_time, sim_time_s=sim_time,
                   elapsed_wall_s=time.perf_counter()-self.started)
        for i, (c, a) in enumerate(zip(numpy(command), numpy(actual))):
            row[f'command_q{i}'] = float(c)
            row[f'actual_q{i}'] = float(a)
        self.write('joint_trajectory.csv', row)

    def sample(self, index, timestamp, sim_time, targets, actual, wrist_bowl, command, qactual,
               residual, ik_wall, physics_wall, limits, force, force_limits):
        residual = numpy(residual).reshape(-1, 6)
        positional = np.linalg.norm(residual[:, :3], axis=1)
        row = dict(index=index, trajectory_time_s=timestamp, sim_time_s=sim_time,
                   elapsed_wall_s=time.perf_counter()-self.started, ik_wall_s=ik_wall,
                   stepping_wall_s=physics_wall,
                   ik_status='converged' if np.all(positional <= 0.001) else 'residual_above_tolerance',
                   ik_max_position_residual_m=float(positional.max()))
        for name, sl in [('bowl', slice(0, 5)), ('spoon', slice(5, 9))]:
            tp, tr, fit = fit_pose(self.templates[name], targets[sl])
            ap, ar, _ = fit_pose(self.templates[name], actual[sl])
            row[f'{name}_position_error_m'] = float(np.linalg.norm(ap-tp))
            row[f'{name}_orientation_error_deg'] = float((tr.inv()*ar).magnitude()*180/np.pi)
            row[f'{name}_marker_max_error_m'] = float(np.linalg.norm(actual[sl]-targets[sl], axis=1).max())
            row[f'{name}_rigid_fit_max_residual_m'] = fit
            row[f'{name}_ik_max_residual_m'] = float(positional[sl].max())
            for label, pos, rot in [('target', tp, tr), ('actual', ap, ar)]:
                for axis, v in zip('xyz', pos): row[f'{name}_{label}_p{axis}_m'] = float(v)
                for axis, v in zip('xyzw', rot.as_quat()): row[f'{name}_{label}_q{axis}'] = float(v)
        row['weld_marker_max_error_m'] = float(np.linalg.norm(actual[:5]-wrist_bowl, axis=1).max())
        command, qactual = numpy(command), numpy(qactual)
        low, high = map(numpy, limits)
        for i, (c, a, f, limit) in enumerate(zip(command, qactual, numpy(force), force_limits)):
            row[f'command_q{i}'] = float(c)
            row[f'actual_q{i}'] = float(a)
            row[f'command_limit_margin_q{i}_rad'] = float(min(c-low[i], high[i]-c))
            row[f'control_force_q{i}_Nm'] = float(f)
            row[f'force_limit_ratio_q{i}'] = float(abs(f)/limit)
        self.rows.append(row)
        self.write('diagnostics.csv', row)

    def finish(self):
        for f in self.files: f.close()
        if not self.rows: return
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        rows = self.rows
        t = [r['trajectory_time_s'] for r in rows]
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        for ax, key, label, factor in zip(axes,
                ['spoon_position_error_m', 'spoon_orientation_error_deg', 'bowl_position_error_m'],
                ['Spoon centroid error (mm)', 'Spoon orientation error (deg)', 'Bowl centroid error (mm)'], [1000, 1, 1000]):
            ax.plot(t, [r[key]*factor for r in rows]); ax.set_ylabel(label); ax.grid()
        axes[-1].set_xlabel('Recorded trajectory time (s)')
        fig.tight_layout(); fig.savefig(self.directory/'tracking_errors.png'); plt.close(fig)
        fig, axes = plt.subplots(4, 4, figsize=(16, 10), sharex=True)
        for i, ax in enumerate(axes.flat):
            ax.plot(t, [r[f'command_q{i}'] for r in rows], label='command')
            ax.plot(t, [r[f'actual_q{i}'] for r in rows], label='actual')
            ax.set_title(f'q{i} (rad)'); ax.grid()
        axes[0, 0].legend(); fig.supxlabel('Recorded trajectory time (s)')
        fig.tight_layout(); fig.savefig(self.directory/'joint_positions.png'); plt.close(fig)
        fig, ax = plt.subplots(figsize=(11, 4))
        for key in ['ik_wall_s', 'stepping_wall_s']: ax.plot(t, [r[key] for r in rows], label=key)
        ax.set(xlabel='Recorded trajectory time (s)', ylabel='Wall time per target (s)'); ax.legend(); ax.grid()
        fig.tight_layout(); fig.savefig(self.directory/'timing.png'); plt.close(fig)
        summary = {'samples': len(rows), 'ik_failed_samples': sum(r['ik_status']!='converged' for r in rows)}
        for key in ['spoon_position_error_m', 'spoon_orientation_error_deg', 'bowl_position_error_m',
                    'weld_marker_max_error_m', 'spoon_ik_max_residual_m', 'bowl_ik_max_residual_m',
                    'spoon_rigid_fit_max_residual_m', 'bowl_rigid_fit_max_residual_m', 'ik_wall_s', 'stepping_wall_s']:
            summary['max_'+key] = max(r[key] for r in rows)
        (self.directory/'summary.json').write_text(json.dumps(summary, indent=2))
