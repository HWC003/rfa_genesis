"""Compare the original waypoint-IK replay with the precomputed joint replay.

    python compare_replays.py            # writes data/diagnostics/2026-09-23/comparison/
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scooping_kinematics import ArmKinematics, ROOT

DIAG = ROOT / 'data/diagnostics/2026-09-23/runs'
OUT = ROOT / 'data/diagnostics/2026-09-23/comparison'
RUNS = {  # label: (directory, colour, line style)
    'original, liquid': ('scoop_full', 'tab:red', '-'),
    'improved, liquid': ('replay_selected_liquid', 'tab:blue', '-'),
    'original, rigid': ('scoop_rigid_contacts', 'tab:red', ':'),
    'improved, rigid': ('replay_selected_rigid', 'tab:blue', ':'),
    'new planner at original point, rigid': ('replay_original_point_rigid', 'tab:gray', '--'),
}
ARM_JOINTS = {'left': list(range(7)), 'right': list(range(9, 16))}


def read_csv(path):
    """Columns as arrays: floats where numeric, strings otherwise."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for key in rows[0]:
        values = [r[key] for r in rows]
        try:
            out[key] = np.array(values, float)
        except ValueError:
            out[key] = np.array(values)
    return out


def cols(table, names):
    return np.column_stack([table[n] for n in names])


def spoon_contacts(contacts):
    """Rows where the bowl touches the right arm/spoon."""
    mask = np.char.find(np.char.add(contacts['link_a'], contacts['link_b']), 'right') >= 0
    return {k: v[mask] for k, v in contacts.items()}


def load(directory):
    d = DIAG / directory
    if not (d / 'diagnostics.csv').exists():
        return None
    run = dict(samples=read_csv(d / 'diagnostics.csv'), steps=read_csv(d / 'joint_trajectory.csv'))
    run['contacts'] = read_csv(d / 'bowl_contacts.csv') if (d / 'bowl_contacts.csv').exists() else None
    return run


def limits():
    kins = ArmKinematics('left'), ArmKinematics('right')
    low, high = np.zeros(16), np.zeros(16)
    for kin, idx in zip(kins, ARM_JOINTS.values()):
        low[idx], high[idx] = kin.low, kin.high
    return low, high


def summarize(run, low, high):
    s, st = run['samples'], run['steps']
    cmd = cols(st, [f'command_q{i}' for i in range(16)])
    act = cols(st, [f'actual_q{i}' for i in range(16)])
    out = {}
    for arm, idx in ARM_JOINTS.items():
        margin_cmd = np.minimum(cmd[:, idx] - low[idx], high[idx] - cmd[:, idx])
        margin_act = np.minimum(act[:, idx] - low[idx], high[idx] - act[:, idx])
        out[arm] = dict(
            max_command_step_deg=float(np.rad2deg(np.abs(np.diff(cmd[:, idx], axis=0)).max())),
            max_command_change_per_20ms_deg=float(np.rad2deg(np.abs(np.diff(cmd[::5, idx], axis=0)).max())),
            min_command_margin_deg=float(np.rad2deg(margin_cmd.min())),
            min_actual_margin_deg=float(np.rad2deg(margin_act.min())),
            steps_within_1deg_of_limit=int((margin_act < np.deg2rad(1)).any(axis=1).sum()),
            max_force_ratio=float(cols(s, [f'force_limit_ratio_q{i}' for i in idx]).max()))
    for key in ('spoon_position_error_m', 'bowl_position_error_m'):
        out[f'max_{key[:-2]}_mm'] = 1e3 * float(s[key].max())
        out[f'median_{key[:-2]}_mm'] = 1e3 * float(np.median(s[key]))
    out['max_spoon_orientation_error_deg'] = float(s['spoon_orientation_error_deg'].max())
    if run['contacts'] is not None:
        spoon = spoon_contacts(run['contacts'])
        n = len(spoon['index'])
        out['peak_spoon_bowl_contact_N'] = float(spoon['max_contact_force_N'].max()) if n else 0.0
        out['spoon_bowl_contact_samples'] = int(len(np.unique(spoon['index'])))
        out['spoon_bowl_contact_time_s'] = [float(spoon['trajectory_time_s'].min()), float(spoon['trajectory_time_s'].max())] if n else None
    out['replayed_s'] = float(s['trajectory_time_s'].max())
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    low, high = limits()
    runs = {k: (load(d), c, ls) for k, (d, c, ls) in RUNS.items()}
    runs = {k: v for k, v in runs.items() if v[0] is not None}
    summary = {k: summarize(r, low, high) for k, (r, _, _) in runs.items()}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))

    # 1. Commanded joint positions against the joint limits (liquid runs).
    fig, axes = plt.subplots(2, 7, figsize=(22, 7), sharex=True)
    for row, (arm, idx) in zip(axes, ARM_JOINTS.items()):
        for ax, j in zip(row, idx):
            ax.axhspan(np.rad2deg(high[j]), np.rad2deg(high[j]) + 30, color='k', alpha=.12)
            ax.axhspan(np.rad2deg(low[j]) - 30, np.rad2deg(low[j]), color='k', alpha=.12)
            for label in ('original, liquid', 'improved, liquid'):
                if label in runs:
                    r, c, ls = runs[label]
                    ax.plot(r['steps']['trajectory_time_s'], np.rad2deg(r['steps'][f'command_q{j}']), color=c, ls=ls, lw=1, label=label)
            ax.set_ylim(np.rad2deg(low[j]) - 10, np.rad2deg(high[j]) + 10)
            ax.set_title(f'{arm} joint {j - idx[0] + 1} (q{j})')
            ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=8)
    fig.supxlabel('Recorded trajectory time (s)')
    fig.supylabel('Commanded joint position (deg); grey = beyond limit')
    fig.tight_layout()
    fig.savefig(OUT / '1_joint_positions_limits.png', dpi=110)
    plt.close(fig)

    # 2. Largest commanded joint change per 20 ms window (one original waypoint), per arm.
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, (arm, idx) in zip(axes, ARM_JOINTS.items()):
        for label, (r, c, ls) in runs.items():
            if 'liquid' not in label:
                continue
            cmd = cols(r['steps'], [f'command_q{j}' for j in idx])[::5]
            ax.semilogy(r['steps']['trajectory_time_s'][::5][1:], np.rad2deg(np.abs(np.diff(cmd, axis=0)).max(1)),
                        color=c, ls=ls, lw=.8, label=label)
        ax.set_ylabel(f'{arm} arm: max |dq| per 20 ms (deg)')
        ax.grid(alpha=.3, which='both')
    axes[0].legend()
    axes[1].set_xlabel('Recorded trajectory time (s)')
    fig.tight_layout()
    fig.savefig(OUT / '2_adjacent_joint_change.png', dpi=110)
    plt.close(fig)

    # 3. Spoon target-vs-actual pose error.
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for label, (r, c, ls) in runs.items():
        s = r['samples']
        axes[0].plot(s['trajectory_time_s'], 1e3 * s['spoon_position_error_m'], color=c, ls=ls, label=label)
        axes[1].plot(s['trajectory_time_s'], s['spoon_orientation_error_deg'], color=c, ls=ls, label=label)
    axes[0].set_ylabel('Spoon position error (mm)')
    axes[1].set_ylabel('Spoon orientation error (deg)')
    axes[1].set_xlabel('Recorded trajectory time (s)')
    for ax in axes:
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / '3_spoon_tracking_error.png', dpi=110)
    plt.close(fig)

    # 4. Spoon-bowl contact force (rigid runs) and right-arm effort (liquid runs).
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for label, (r, c, ls) in runs.items():
        if r['contacts'] is not None and 'rigid' in label:
            k = spoon_contacts(r['contacts'])
            times = np.unique(k['trajectory_time_s'])
            peak = [k['max_contact_force_N'][k['trajectory_time_s'] == t].max() for t in times]
            axes[0].plot(times, peak, 'o', ms=3, color=c, label=label)
        if 'liquid' in label:
            s = r['samples']
            axes[1].plot(s['trajectory_time_s'], cols(s, [f'force_limit_ratio_q{j}' for j in ARM_JOINTS['right']]).max(1),
                         color=c, ls=ls, label=label)
    axes[0].set_ylabel('Peak spoon-bowl contact (N)')
    axes[1].set_ylabel('Right arm max |effort| / limit')
    axes[1].set_xlabel('Recorded trajectory time (s)')
    for ax in axes:
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / '4_contact_and_effort.png', dpi=110)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
