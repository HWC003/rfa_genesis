"""Post-processing pipeline entry point: raw human demos -> episodes -> (later) plans, validation, exports.

A session is described by a YAML file; outputs go to --out (default: the YAML file's folder).

    session.yaml
      name: 2026-10-17_soybean
      raw_dir: data/trajectory/real_trajs        # relative to the repository root, or absolute
      tsv: soybean{take}.tsv                     # file name pattern of one take
      takes: ['0001', '0002', ...]
      spoon: v3                                  # robot spoon version whose mesh measures the demo
      calibration:
        take: '0029'                             # spoon tip rubbed over the empty bowl
        layout_takes: ['0001', '0012', '0017', '0029']
        parts: [t, s]
        frame_select_spoon: v2                   # optional; default = spoon (v2 reproduces the 2026-10-21 x_v3)

      plan:                                      # robot plan settings (PLAN_DEFAULTS), per-take overrides under takes:
        spoon_margin: cal                        # cal | calwall (scooping_motion.wall_margin)
        bowl: hold                               # hold | demo
        spin: true
        yaw_deg: 0
        placement: [0.22, -0.04, 0.44]
        time_scale: 1.6
        skip: ['0015']                           # takes not planned
        takes: {'0009': {yaw_deg: 200, placement: [0.22, 0.0, 0.44]}}

Stages (run from the repository root, genesis env):
    python scripts/scooping_pipeline.py calibrate SESSION.yaml     # -> calibration.json
    python scripts/scooping_pipeline.py ingest SESSION.yaml        # -> episodes/<take>.npz + .json, ingest.json
    python scripts/scooping_pipeline.py plan SESSION.yaml          # -> plans/<take>.csv + .json, plan.json

Existing outputs are never overwritten (stage refuses) unless --overwrite.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_session(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    raw = Path(cfg['raw_dir'])
    cfg['raw_dir'] = raw if raw.is_absolute() else ROOT / raw
    cfg['_file'] = path
    return cfg


def tsv(cfg, take):
    return cfg['raw_dir'] / cfg['tsv'].format(take=take)


def guard(paths, overwrite):
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing and not overwrite:
        sys.exit(f'refusing to overwrite (pass --overwrite): {existing}')


def cmd_calibrate(cfg, out, args):
    import scooping_calibration as sc
    import scooping_metrics as sm
    c = cfg['calibration']
    dest = out / 'calibration.json'
    guard([dest], args.overwrite)
    geo = sm.DemoGeometry(cfg['spoon'])
    layout_H, layout_rep = sc.learn_layout([tsv(cfg, t) for t in c['layout_takes']], geo.bowl_h)
    print('layout', json.dumps(layout_rep))
    cal, extra = sc.fit(tsv(cfg, c['take']), layout_H, cfg['spoon'], c.get('parts', ['t', 's']),
                        c.get('frame_select_spoon'), cv=not args.no_cv)
    cal['layout'] = layout_rep
    cal['layout_takes'] = list(c['layout_takes'])
    out.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(cal, indent=1))
    np.savez_compressed(out / 'calibration_frames.npz', **extra)
    print('x_mm', cal['x_mm'], 'fit', cal['fit'], 'gate', cal.get('gate'))
    print('written', dest)


def cmd_ingest(cfg, out, args):
    import scooping_ingest as si
    import scooping_io as sio
    import scooping_metrics as sm
    cal_path = Path(args.calibration) if args.calibration else out / 'calibration.json'
    cal = json.loads(cal_path.read_text())
    takes = args.takes or cfg['takes']
    dests = [out / 'episodes' / f'{t}.npz' for t in takes]
    guard(dests, args.overwrite)
    geo = sm.DemoGeometry(cfg['spoon'])
    pts = sm.subsample(geo.spoon_V)
    report = {}
    for take, dest in zip(takes, dests):
        try:
            arrays, summ = si.ingest_take(geo, tsv(cfg, take), cal, pts)
        except ValueError as e:
            report[take] = dict(error=str(e)); print(take, report[take]); continue
        summ['calibration']['file'] = str(cal_path)
        sio.save_episode(dest, arrays, summ)
        report[take] = {k: summ[k] for k in ('duration_s', 'phases_s', 'raw', 'cal', 'tilt_scoop_deg', 'curl_deg')}
        print(take, json.dumps({k: summ[k] for k in ('duration_s', 'raw', 'cal')}), flush=True)
    rep_path = out / ('ingest.json' if not args.takes else f'ingest_{"_".join(takes)}.json')
    rep_path.write_text(json.dumps(report, indent=1))
    print('written', rep_path)


PLAN_DEFAULTS = dict(spoon_margin='cal', bowl='hold', spin=True, yaw_deg=0.0, placement=[0.22, -0.04, 0.44],
                     time_scale=1.6, flat=None)


def plan_settings(cfg, take):
    """Plan settings of one take: PLAN_DEFAULTS < session plan: < plan: takes: <take>:"""
    p = dict(cfg.get('plan', {}))
    per = (p.pop('takes', None) or {}).get(take, {})
    p.pop('skip', None)
    return {**PLAN_DEFAULTS, **p, **per}


def _plan_one(job):
    """Worker: one take -> joint CSV + plan JSON (runs in a fresh process per pool worker)."""
    import scooping_io as sio
    import scooping_metrics as sm
    import scooping_motion as mo
    import scooping_planner as sp
    take, ep_path, dest, s, spoon = job
    ep, summ = sio.load_episode(ep_path)
    rel = ep['cal']
    shift_max = 0.0
    if s['spoon_margin'] == 'calwall':
        rel, shift = mo.wall_margin(sm.DemoGeometry(spoon), rel)
        shift_max = float(np.linalg.norm(shift, axis=1).max() * 1e3)
    elif s['spoon_margin'] != 'cal':
        raise ValueError(f"spoon_margin must be cal or calwall, not {s['spoon_margin']}")
    pl = sp.Planner(spoon)
    t, T, WR = pl.targets_from_episode(ep, rel, s['yaw_deg'], s['placement'], s['time_scale'])
    plans, _, info = pl.plan(t, T, WR, flat=s['flat'], bowl=s['bowl'], spin=s['spin'])
    inside = info.pop('inside')
    tt, ql, qr = pl.sample(t, plans)
    sio.write_joint_csv(dest, tt, ql, qr)
    rep = dict(take=take, settings=s, spoon=spoon, wall_shift_max_mm=shift_max,
               info={k: v for k, v in info.items() if k not in ('left', 'right')},
               arm_metrics=dict(left=info['left'], right=info['right']), metrics=pl.metrics(t, plans, inside),
               episode=dict(file=str(ep_path), sha256=sio.sha256(ep_path)), output_sha256=sio.sha256(dest))
    dest.with_suffix('.json').write_text(json.dumps(rep, indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
    return take, info['plan_ok'], rep['metrics']


def cmd_plan(cfg, out, args):
    from multiprocessing import get_context
    skip = set((cfg.get('plan') or {}).get('skip', []))
    takes = [t for t in (args.takes or cfg['takes']) if t not in skip]
    jobs = []
    for take in takes:
        ep = out / 'episodes' / f'{take}.npz'
        if not ep.exists():
            print(take, 'no episode, skipped'); continue
        jobs.append((take, ep, out / 'plans' / f'{take}.csv', plan_settings(cfg, take), cfg['spoon']))
    guard([j[2] for j in jobs], args.overwrite)
    report = {}
    with get_context('spawn').Pool(min(args.jobs, len(jobs))) as pool:   # spawn: workers re-import cleanly
        for take, ok, m in pool.imap_unordered(_plan_one, jobs):
            report[take] = dict(plan_ok=ok, **m)
            print(take, 'plan_ok' if ok else 'PLAN FAILED', json.dumps(m), flush=True)
    rep_path = out / ('plan.json' if not args.takes else f'plan_{"_".join(takes)}.json')
    rep_path.write_text(json.dumps(dict(sorted(report.items())), indent=1, default=str))
    print('written', rep_path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=['calibrate', 'ingest', 'plan'])
    p.add_argument('session', type=Path)
    p.add_argument('--out', type=Path, help='output folder (default: the session file folder)')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--no-cv', action='store_true', help='calibrate: skip the cross-validation (and the gate)')
    p.add_argument('--calibration', help='ingest: calibration.json to use (default: <out>/calibration.json)')
    p.add_argument('--takes', nargs='+', help='ingest / plan: only these takes')
    p.add_argument('--jobs', type=int, default=12, help='plan: parallel takes')
    args = p.parse_args()
    cfg = load_session(args.session)
    out = (args.out or cfg['_file'].parent).resolve()
    dict(calibrate=cmd_calibrate, ingest=cmd_ingest, plan=cmd_plan)[args.stage](cfg, out, args)


if __name__ == '__main__':
    main()
