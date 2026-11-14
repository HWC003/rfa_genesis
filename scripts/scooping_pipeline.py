"""Post-processing pipeline: raw human demos -> episodes -> robot plans -> simulation gate -> exports.

A session is described by a YAML file; outputs go to --out (default: the YAML file's folder).
Overview and data layout: docs/scooping_pipeline/README.md.

    session.yaml
      name: 2026-10-17_soybean
      raw_dir: data/trajectory/real_trajs        # relative to the repository root, or absolute
      tsv: soybean{take}.tsv                     # file name pattern of one take
      takes: ['0001', '0002', ...]
      spoon: v3                                  # robot spoon version whose mesh measures the demo
      metadata: metadata.csv                     # per take: take, initial_food_ml, real_scooped_ml (session folder)
      food: {type: soybean_pudding}              # simulation food model (FOOD_DEFAULTS), per food type
      calibration:
        take: '0029'                             # spoon tip rubbed over the empty bowl
        layout_takes: ['0001', '0012', '0017', '0029']
        parts: [t, s]
        frame_select_spoon: v2                   # optional; default = spoon (v2 reproduces the 2026-10-21 x_v3)
      plan:                                      # robot plan settings (PLAN_DEFAULTS); per-take overrides under takes:
        spoon_margin: cal                        # cal | calwall (scooping_motion.wall_margin)
        bowl: hold                               # hold | demo
        spin: true
        yaw_deg: 0
        placement: [0.22, -0.04, 0.44]
        time_scale: 1.6
        start_ramp: [0.3, 0.7]                   # trajectory_smoothing.smooth_start written into the plan; null = none
        floor_lift_mm: 0
        skip: []                                 # takes not planned
        takes: {'0009': {yaw_deg: 200, placement: [0.22, 0.0, 0.44]}}
      ladder:                                    # correction ladder: tried in order when a plan fails the gate; each
        - {floor_lift_mm: 1.5}                   # rung = overrides on the take's plan settings
        - {floor_lift_mm: 5.0}
      gate: {entry_warn_mm: 4.0}                 # scooping_gate.GATE_DEFAULTS overrides

Stages (repository root, genesis env):
    python scripts/scooping_pipeline.py calibrate SESSION.yaml    # -> calibration.json (+ gate: CV RMS < 1 mm)
    python scripts/scooping_pipeline.py ingest SESSION.yaml       # -> episodes/<take>.npz + .json, ingest.json
    python scripts/scooping_pipeline.py plan SESSION.yaml         # -> plans/<take>/r0.csv + .json, plan.json
    python scripts/scooping_pipeline.py validate SESSION.yaml     # -> sim/<take>/r<k>/, gate/<take>.json, gate.json
    python scripts/scooping_pipeline.py export SESSION.yaml --name NAME
        # -> export/NAME/robot/sb_traj_<take>.csv + manifest.csv + provenance.json (accepted takes)
        #    export/NAME/policy/<take>.npz + index.csv + schema.json (every episode, bowl frame)

validate runs the ladder in rounds: plan rung k for the takes not yet accepted, check the plan, simulate it
(oa_pure_scooping.py --joints --evaluate, at most 3 GPU runs at once), check the result; a take that fails goes to
rung k + 1. Existing plans and simulations are reused, never overwritten (plan / ingest / calibrate / export refuse
without --overwrite); gate files are recomputed from them on every validate.
"""
import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
PARTICLE_ML_SETTLED, PARTICLE_ML_SIM = 0.1275, 0.1728      # 6 mm particle: settled real volume / Genesis volume (README)
SIM_JOBS = 3                                              # food runs on the GPU: at most 3 at once (CLAUDE.md)


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


def jdump(obj):
    return json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))


def metadata(cfg):
    """{take: row} from the session metadata CSV (take, initial_food_ml, real_scooped_ml, ...), or {}."""
    if not cfg.get('metadata'):
        return {}
    p = Path(cfg['metadata'])
    p = p if p.is_absolute() else cfg['_file'].parent / p
    return {r['take']: r for r in csv.DictReader(p.open())}


def git_revision():
    try:
        rev = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--', 'scripts'], cwd=ROOT, capture_output=True,
                               text=True).stdout.strip()
        return rev + (' (scripts/ modified)' if dirty else '')
    except OSError:
        return 'unknown'


# ------------------------------------------------------------------ calibrate / ingest ---
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


# ------------------------------------------------------------------------------- plan ---
PLAN_DEFAULTS = dict(spoon_margin='cal', bowl='hold', spin=True, yaw_deg=0.0, placement=[0.22, -0.04, 0.44],
                     time_scale=1.6, flat=None, start_ramp=[0.3, 0.7], floor_lift_mm=0.0)


def plan_settings(cfg, take, rung=0):
    """Plan settings of one take at ladder rung k: PLAN_DEFAULTS < session plan: < plan: takes: <take>: < ladder[k-1]"""
    p = dict(cfg.get('plan') or {})
    per = (p.pop('takes', None) or {}).get(take, {})
    p.pop('skip', None)
    ladder = cfg.get('ladder') or []
    if rung > len(ladder):
        raise ValueError(f'rung {rung}: the ladder has {len(ladder)} rungs')
    return {**PLAN_DEFAULTS, **p, **per, **(ladder[rung - 1] if rung else {})}


def plan_path(out, take, rung):
    return out / 'plans' / take / f'r{rung}.csv'


def write_plan(pl, dest, take, t, T, WR, s, source, t_demo0=None, extra_info=None):
    """Plan both arms for (t, T, WR) with settings s, write the joint CSV and its JSON report.

    The CSV starts at rest when s['start_ramp'] is set (trajectory_smoothing.smooth_start written into the file).
    t_demo0: QTM time of the episode's first frame; then a demo_time_s column (QTM time of the take shown by every
    row, ramp included) is appended after the 14 joints. Returns the report."""
    import scooping_io as sio
    from trajectory_smoothing import smooth_start, start_speed
    plans, _, info = pl.plan(t, T, WR, flat=s['flat'], bowl=s['bowl'], spin=s['spin'],
                             floor_lift_mm=s.get('floor_lift_mm') or 0.0)
    inside = info.pop('inside')
    tt, ql, qr = pl.sample(t, plans)
    q = np.c_[ql, qr]
    src = tt
    if s.get('start_ramp'):
        tt, q, src = smooth_start(tt, q, *s['start_ramp'], return_source=True)
    extra = None if t_demo0 is None else dict(demo_time_s=src / s['time_scale'] + t_demo0)
    sio.write_joint_csv(dest, tt, q[:, :7], q[:, 7:], extra)
    metrics = pl.metrics(t, plans, inside)
    metrics.update(start_speed_rad_s=round(start_speed(tt, q), 4), duration_with_ramp_s=round(float(tt[-1]), 3),
                   samples_with_ramp=int(len(tt)))
    rep = dict(take=take, settings=s, spoon=pl.spoon, **(extra_info or {}),
               info={k: v for k, v in info.items() if k not in ('left', 'right')},
               arm_metrics=dict(left=info['left'], right=info['right']), metrics=metrics, source=source,
               output_sha256=sio.sha256(dest))
    dest.with_suffix('.json').write_text(jdump(rep))
    return rep


def _plan_one(job):
    """Worker: one take at one rung -> joint CSV + plan JSON (runs in a fresh process per pool worker)."""
    import scooping_io as sio
    import scooping_metrics as sm
    import scooping_motion as mo
    import scooping_planner as sp
    take, rung, ep_path, dest, s, spoon = job
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
    rep = write_plan(pl, dest, take, t, T, WR, s, dict(episode=str(ep_path), episode_sha256=sio.sha256(ep_path)),
                     t_demo0=float(ep['t_qtm0']), extra_info=dict(rung=rung, wall_shift_max_mm=shift_max))
    return take, rung, rep['info']['plan_ok'], rep['metrics']


def plan_jobs(cfg, out, takes, rung):
    jobs = []
    for take in takes:
        ep = out / 'episodes' / f'{take}.npz'
        if not ep.exists():
            print(take, 'no episode, skipped'); continue
        jobs.append((take, rung, ep, plan_path(out, take, rung), plan_settings(cfg, take, rung), cfg['spoon']))
    return jobs


def run_plans(jobs, n):
    from multiprocessing import get_context
    if not jobs:
        return {}
    res = {}
    with get_context('spawn').Pool(min(n, len(jobs))) as pool:   # spawn: workers re-import cleanly
        for take, rung, ok, m in pool.imap_unordered(_plan_one, jobs):
            res[take] = dict(plan_ok=ok, **m)
            print(f'{take} r{rung}', 'plan_ok' if ok else 'PLAN CHECK FAILED', json.dumps(m), flush=True)
    return res


def cmd_plan(cfg, out, args):
    skip = set((cfg.get('plan') or {}).get('skip', []))
    takes = [t for t in (args.takes or cfg['takes']) if t not in skip]
    jobs = plan_jobs(cfg, out, takes, args.rung)
    guard([j[3] for j in jobs], args.overwrite)
    report = run_plans(jobs, args.jobs)
    rep_path = out / (f'plan_r{args.rung}.json' if not args.takes else f'plan_r{args.rung}_{"_".join(takes)}.json')
    rep_path.write_text(json.dumps(dict(sorted(report.items())), indent=1, default=str))
    print('written', rep_path)


# --------------------------------------------------------------------------- validate ---
FOOD_DEFAULTS = dict(type='soybean_pudding', sph_rho=960.0, sph_mu=0.01, sph_gamma=0.01, particle_size=0.006,
                     food_gap=0.030, hold=2.0, gate_clearance_mm=-30.0)    # validated 2026-10-26 (CLAUDE.md)


def food_volume_arg(initial_ml):
    """--food-volume for a real food volume (6 mm particles): the settled particle count x the Genesis particle
    volume (as sim_ready_trajs/manifest.csv food_volume_arg)."""
    return round(float(initial_ml) / PARTICLE_ML_SETTLED) * PARTICLE_ML_SIM


def sim_dir(out, take, rung):
    return out / 'sim' / take / f'r{rung}'


def run_sim(job):
    """One oa_pure_scooping.py --joints --evaluate run (skipped if its scoop_summary.json exists)."""
    take, rung, csv_path, dest, food, volume, spoon = job
    if (dest / 'scoop_summary.json').exists():
        return f'{take} r{rung} sim exists'
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SCRIPTS / 'oa_pure_scooping.py'), '--joints', str(csv_path), '--spoon', spoon,
           '--food-volume', f'{volume:.4f}', '--sph-rho', str(food['sph_rho']), '--sph-mu', str(food['sph_mu']),
           '--sph-gamma', str(food['sph_gamma']), '--particle-size', str(food['particle_size']),
           '--food-gap', str(food['food_gap']), '--hold', str(food['hold']),
           '--gate-clearance-mm', str(food['gate_clearance_mm']), '--evaluate', '--headless', '--diagnostics', str(dest)]
    (dest / 'command.txt').write_text(' '.join(cmd) + '\n')
    with (dest / 'run.log').open('w') as log:
        r = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, GS_SPH_RIGID_PRESSURE_PUSH='0'))
    return f'{take} r{rung} sim exit {r.returncode}'


def gate_plan(out, take, rung, g):
    import scooping_gate as sg
    import scooping_io as sio
    from trajectory_smoothing import start_speed
    p = plan_path(out, take, rung)
    rep = json.loads(p.with_suffix('.json').read_text())
    tt, ql, qr, _ = sio.read_joint_csv(p)
    return rep, sg.plan_checks(rep, start_speed(tt, np.c_[ql, qr]), g)


def cmd_validate(cfg, out, args):
    import scooping_gate as sg
    g = {**sg.GATE_DEFAULTS, **(cfg.get('gate') or {})}
    food = {**FOOD_DEFAULTS, **(cfg.get('food') or {})}
    meta = metadata(cfg)
    skip = set((cfg.get('plan') or {}).get('skip', []))
    takes = [t for t in (args.takes or cfg['takes']) if t not in skip]
    n_rungs = 1 + len(cfg.get('ladder') or [])
    state = {t: dict(take=t, status='pending', rungs=[]) for t in takes}
    for rung in range(n_rungs):
        todo = [t for t in takes if state[t]['status'] == 'pending']
        if not todo:
            break
        run_plans([j for j in plan_jobs(cfg, out, todo, rung) if not j[3].exists()], args.jobs)
        sims = []
        for t in todo:
            if not plan_path(out, t, rung).exists():
                state[t]['status'] = 'reject'; state[t]['rungs'].append(dict(rung=rung, decision='no plan')); continue
            rep, checks = gate_plan(out, t, rung, g)
            r = dict(rung=rung, settings=rep['settings'], plan=str(plan_path(out, t, rung).relative_to(out)),
                     plan_checks=checks)
            state[t]['rungs'].append(r)
            decision, reasons, warnings = sg.decide(checks)
            if decision == 'reject':
                r.update(decision='reject (plan)', reasons=reasons, warnings=warnings)
                continue
            if t not in meta:
                sys.exit(f'take {t}: no initial_food_ml in the session metadata')
            vol = food_volume_arg(meta[t]['initial_food_ml'])
            r['food_volume_arg'] = round(vol, 4)
            sims.append((t, rung, plan_path(out, t, rung), sim_dir(out, t, rung), food, vol, cfg['spoon']))
        with ThreadPoolExecutor(SIM_JOBS) as ex:
            for msg in ex.map(run_sim, sims):
                print(msg, flush=True)
        for t, rung_, *_ in sims:
            r = state[t]['rungs'][-1]
            d = sim_dir(out, t, rung_)
            if not (d / 'scoop_summary.json').exists():
                r.update(decision='reject (sim did not finish)', reasons=[f'see {d / "run.log"}'], warnings=[])
                continue
            end_above = json.loads(plan_path(out, t, rung_).with_suffix('.json').read_text())['metrics']['end_spoon_above_rim_mm']
            real = meta.get(t, {}).get('real_scooped_ml')
            checks, summ = sg.sim_checks(d, end_above, g, float(real) if real else None)
            r.update(sim=str(d.relative_to(out)), sim_checks=checks, sim_summary=summ)
            decision, reasons, warnings = sg.decide({**r['plan_checks'], **checks})
            r.update(decision=decision, reasons=reasons, warnings=warnings)
            if decision == 'accept':
                state[t].update(status='accept', accepted_rung=rung_)
        for t in todo:
            if state[t]['status'] == 'pending' and rung == n_rungs - 1:
                state[t]['status'] = 'reject'
            print(f'{t} r{rung}:', state[t]['rungs'][-1].get('decision'), state[t]['rungs'][-1].get('reasons', ''),
                  flush=True)
    for t in takes:
        if state[t]['status'] == 'pending':
            state[t]['status'] = 'reject'
    (out / 'gate').mkdir(exist_ok=True)
    summary = {}
    for t, s in state.items():
        (out / 'gate' / f'{t}.json').write_text(jdump(dict(gate=g, food=food, **s)))
        last = s['rungs'][-1] if s['rungs'] else {}
        acc = next((r for r in s['rungs'] if r.get('decision') == 'accept'), None)
        summary[t] = dict(status=s['status'], rung=s.get('accepted_rung'), tried=len(s['rungs']),
                          reasons=None if acc else last.get('reasons'), warnings=(acc or last).get('warnings'),
                          kept_ml=(acc or last).get('sim_summary', {}).get('kept_ml'),
                          peak_contact_N=(acc or last).get('sim_summary', {}).get('peak_contact_every_step_N'))
    (out / 'gate.json').write_text(jdump(dict(gate=g, food=food, takes=summary)))
    acc = [t for t, s in summary.items() if s['status'] == 'accept']
    print(f'accepted {len(acc)}/{len(summary)}:', acc)
    print('rejected:', {t: s['reasons'] for t, s in summary.items() if s['status'] != 'accept'})


# ----------------------------------------------------------------------------- export ---
def manifest_row(out, take, gate_t, meta_t):
    acc = next(r for r in gate_t['rungs'] if r.get('decision') == 'accept')
    rep = json.loads((out / acc['plan']).with_suffix('.json').read_text())
    s, m, i = rep['settings'], rep['metrics'], rep['info']
    sim = acc['sim_summary']
    return dict(
        recording=take, file=f'sb_traj_{take}.csv', rung=acc['rung'],
        configuration=s['spoon_margin'], bowl=s['bowl'], spin=s['spin'], placement_m=' '.join(map(str, s['placement'])),
        yaw_deg=s['yaw_deg'], time_scale=s['time_scale'],
        start_ramp_s='' if not s.get('start_ramp') else '/'.join(map(str, s['start_ramp'])),
        floor_lift_mm=s.get('floor_lift_mm') or 0, samples=m['samples_with_ramp'], duration_s=m['duration_with_ramp_s'],
        flat_bowl=i['flat'], tilt_start_hold_end_deg=m.get('tilt_start_hold_end_deg', ''), tipback_fallback=i['tipback'],
        spin_delta_deg=i.get('spin_delta_deg', ''), xb_from_world_y_deg=m['xb_from_world_y_deg'],
        min_margin_left_deg=m['min_margin_left_deg'], min_margin_right_deg=m['min_margin_right_deg'],
        plan_min_clearance_mm=m['min_clearance_mm'], entry_min_clearance_mm=m['entry_min_clearance_mm'],
        peak_speed_left_rad_s=m['peak_speed_left_rad_s'], peak_speed_right_rad_s=m['peak_speed_right_rad_s'],
        sim_food_kept_ml=f"{sim['kept_ml']:.2f}", sim_peak_contact_N=f"{sim['peak_contact_every_step_N']:.2f}",
        initial_food_ml=meta_t.get('initial_food_ml', ''), real_scooped_ml=meta_t.get('real_scooped_ml', ''),
        gate_warnings='; '.join(acc.get('warnings') or []), real_robot_tested='')


def export_robot(cfg, out, dest, gate):
    import scooping_io as sio
    meta = metadata(cfg)
    rows, files = [], []
    (dest / 'robot').mkdir(parents=True, exist_ok=True)
    for take, s in sorted(gate['takes'].items()):
        if s['status'] != 'accept':
            continue
        gt = json.loads((out / 'gate' / f'{take}.json').read_text())
        acc = next(r for r in gt['rungs'] if r.get('decision') == 'accept')
        src = out / acc['plan']
        data = src.read_bytes()
        (dest / 'robot' / f'sb_traj_{take}.csv').write_bytes(data)        # byte copy: the simulated file itself
        rows.append(manifest_row(out, take, gt, meta.get(take, {})))
        rep = json.loads(src.with_suffix('.json').read_text())
        files.append(dict(recording=take, rung=acc['rung'], plan=str(src.relative_to(out)), plan_sha256=sio.sha256(src),
                          output=f'robot/sb_traj_{take}.csv', output_sha256=sio.sha256(dest / 'robot' / f'sb_traj_{take}.csv'),
                          plan_source=rep.get('source'), sim=acc['sim'],
                          sim_summary_sha256=sio.sha256(out / acc['sim'] / 'scoop_summary.json')))
    if rows:
        buf = io.StringIO(newline='')
        w = csv.DictWriter(buf, list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
        (dest / 'robot' / 'manifest.csv').write_text(buf.getvalue())
    rejected = {t: s['reasons'] for t, s in gate['takes'].items() if s['status'] != 'accept'}
    (dest / 'robot' / 'provenance.json').write_text(jdump(dict(
        generator='scripts/scooping_pipeline.py export', code_revision=git_revision(),
        session=str(cfg['_file']), session_sha256=sio.sha256(cfg['_file']),
        calibration=str(out / 'calibration.json'),
        calibration_sha256=sio.sha256(out / 'calibration.json') if (out / 'calibration.json').exists() else None,
        selection='scooping_gate: plan checks + food simulation (gate.json)', gate=gate['gate'], food=gate['food'],
        gate_sha256=sio.sha256(out / 'gate.json'), rejected=rejected, files=files)))
    print(f'robot: {len(rows)} trajectories -> {dest / "robot"}; rejected {sorted(rejected)}')


POLICY_SCHEMA = dict(
    description='One human demo per file, in the bowl frame B; robot-independent (no yaw, placement or time scale). '
                'Written by scripts/scooping_pipeline.py export from the episodes (every ingested take, whatever the '
                'gate says); labels carry the robot / simulation outcome when there is one.',
    frames=dict(
        B='bowl frame: origin = bowl floor centre, z_B = bowl axis (out of the bowl), x_B = -x_holder '
          '(scooping_metrics.T_HB); rim at z_B = 0.041 m, rim radius 0.0727 m',
        S='spoon frame {S} of the robot spoon (scooping_scene.SPOONS); dish centre = scooping_metrics.DISH_CENTRE, '
          'dish tip = scooping_metrics.TIP',
        world='demo world (QTM lab frame mapped to Genesis axes, first Bowl_center at (0.30, 0, 0.40)); z up'),
    arrays=dict(
        t='(N,) s, demo time from the first kept frame (100 Hz)', t_qtm='(N,) s, QTM Time of the raw take (sync key)',
        T_BS='(N, 4, 4) spoon {S} in B, calibrated (the pose the robot is planned from)',
        tip_B='(N, 3) m, dish tip in B', dish_B='(N, 3) m, dish centre in B',
        gravity_B='(N, 3) unit world -z in B (bowl tilt; independent of the planning yaw)',
        T_WB='(N, 4, 4) B in the demo world', depth_mm='(N,) deepest spoon point inside the bowl surface (mm, + = '
             'into the wall; NaN = spoon away from the bowl), calibrated',
        phase='(N,) str: approach, descent, sweep, curl, slide, lift, out (scooping_ingest phase standard)',
        phase_idx='JSON str: entry, low0, deep, c0, c1, low1, lift frame indices'),
    labels='<take>.json: food, initial_food_ml, real_scooped_ml, demo summary (phases_s, tilt, curl), gate status, '
           'accepted rung, simulated kept mL / peak contact, robot file (if exported)')


def export_policy(cfg, out, dest, gate):
    import scooping_io as sio
    import scooping_metrics as sm
    meta = metadata(cfg)
    food = {**FOOD_DEFAULTS, **(cfg.get('food') or {})}
    pdir = dest / 'policy'
    pdir.mkdir(parents=True, exist_ok=True)
    rows = []
    T_HB_inv = np.linalg.inv(sm.T_HB)
    for ep_path in sorted((out / 'episodes').glob('*.npz')):
        take = ep_path.stem
        ep, summ = sio.load_episode(ep_path)
        T_BS = T_HB_inv @ ep['cal']
        T_WB = ep['H'] @ sm.T_HB
        g_B = np.einsum('nji,j->ni', T_WB[:, :3, :3], np.array([0, 0, -1.]))
        arrays = dict(t=ep['t'], t_qtm=ep['t'] + float(ep['t_qtm0']), T_BS=T_BS,
                      tip_B=(T_BS @ np.r_[sm.TIP, 1])[:, :3], dish_B=(T_BS @ np.r_[sm.DISH_CENTRE, 1])[:, :3],
                      gravity_B=g_B, T_WB=T_WB, depth_mm=ep['depth_cal'] * 1e3, phase=ep['phase'], phase_idx=ep['idx'])
        gs = gate['takes'].get(take, dict(status='not planned'))
        m = meta.get(take, {})
        labels = dict(take=take, food=food['type'], initial_food_ml=m.get('initial_food_ml'),
                      real_scooped_ml=m.get('real_scooped_ml'), demo=dict(
                          duration_s=summ['duration_s'], phases_s=summ['phases_s'], tilt_scoop_deg=summ['tilt_scoop_deg'],
                          curl_deg=summ['curl_deg'], cal=summ['cal']),
                      robot=dict(gate=gs['status'], rung=gs.get('rung'), reasons=gs.get('reasons'),
                                 sim_kept_ml=gs.get('kept_ml'), sim_peak_contact_N=gs.get('peak_contact_N'),
                                 file=f'robot/sb_traj_{take}.csv' if gs['status'] == 'accept' else None),
                      source=dict(episode=str(ep_path), episode_sha256=sio.sha256(ep_path), raw=summ['source']))
        sio.save_episode(pdir / f'{take}.npz', arrays, labels)
        rows.append(dict(take=take, frames=len(ep['t']), duration_s=round(float(ep['t'][-1]), 2), food=food['type'],
                         initial_food_ml=labels['initial_food_ml'], real_scooped_ml=labels['real_scooped_ml'],
                         gate=gs['status'], sim_kept_ml=gs.get('kept_ml')))
    with (pdir / 'index.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
    (pdir / 'schema.json').write_text(jdump(POLICY_SCHEMA))
    print(f'policy: {len(rows)} episodes -> {pdir}')


def cmd_export(cfg, out, args):
    if not args.name:
        sys.exit('export needs --name')
    dest = out / 'export' / args.name
    if dest.exists():
        if not args.overwrite:
            sys.exit(f'refusing to overwrite (pass --overwrite): {dest}')
        shutil.rmtree(dest)
    gate = json.loads((out / 'gate.json').read_text())
    if 'robot' in args.what:
        export_robot(cfg, out, dest, gate)
    if 'policy' in args.what:
        export_policy(cfg, out, dest, gate)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=['calibrate', 'ingest', 'plan', 'validate', 'export'])
    p.add_argument('session', type=Path)
    p.add_argument('--out', type=Path, help='output folder (default: the session file folder)')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--no-cv', action='store_true', help='calibrate: skip the cross-validation (and the gate)')
    p.add_argument('--calibration', help='ingest: calibration.json to use (default: <out>/calibration.json)')
    p.add_argument('--takes', nargs='+', help='ingest / plan / validate: only these takes')
    p.add_argument('--rung', type=int, default=0, help='plan: ladder rung (0 = the base settings)')
    p.add_argument('--jobs', type=int, default=12, help='plan / validate: parallel planning processes')
    p.add_argument('--name', help='export: export folder name (export/<name>/)')
    p.add_argument('--what', nargs='+', choices=['robot', 'policy'], default=['robot', 'policy'], help='export: formats')
    args = p.parse_args()
    cfg = load_session(args.session)
    out = (args.out or cfg['_file'].parent).resolve()
    sys.path.insert(0, str(SCRIPTS))
    dict(calibrate=cmd_calibrate, ingest=cmd_ingest, plan=cmd_plan, validate=cmd_validate,
         export=cmd_export)[args.stage](cfg, out, args)


if __name__ == '__main__':
    main()
