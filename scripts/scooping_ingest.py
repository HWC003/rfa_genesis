"""Ingest: one raw demo (QTM TSV) -> one episode record (robot-independent; no yaw, no time scale, no placement).

Per take, in order:
  1. scooping_io.load_take          lab mm -> world m (NAMES order), gap-filled flags
  2. robust_poses                    bowl holder H(t) and spoon {S} S(t) by robust Kabsch (measured bowl layout from
                                     the session calibration; a mislabelled bowl marker dropped; bad frames Slerped)
  3. apply                           calibration (bowl shift t, spoon shift s): rel_cal = C @ inv(H) @ S @ Ts
  4. analyse + check_phases          phases from the calibrated dish tip vs the bowl (standard below)
  5. crop                            [entry - 0.5 s, lift + 0.5 s]
  6. depth (scooping_metrics.deepest) before / after calibration
Episode arrays (scooping_io.save_episode): t, H, raw, cal, depth_raw, depth_cal, phase, idx (as prep_XXXX.npz of
2026-10-21/soybean_batch) plus t_qtm0 (QTM time of the first kept frame), crop, bowl_fit_res, spoon_fit_res,
not_gap_filled. The .json summary holds the prepare.json fields and provenance.

Phase standard (holder frame H, bowl frame B = H @ T_HB, dish tip = scooping_metrics.TIP in {S}; all on the
calibrated spoon pose rel_cal, i.e. the pose the robot is planned from):
  entry      first frame with the tip inside the rim cylinder (below the rim plane in B, radius < RIM_R)
  low0/low1  first / last frame of the longest stretch after entry with the tip within LOW (10 mm) of the inner
             bowl surface; stretches separated by <= LOW_GAP_S (0.2 s) are merged
  deep       lowest tip (relative to the surface) between low0 and low1
  c0/c1      curl: the spoon's face-up angle (world vertical vs {S} +z) has dropped by 10 % / 90 % of its total drop
             after its peak (peak searched between entry and lift)
  lift       first frame after low1 with the tip above the rim plane
  Required order (check_phases): entry <= low0 <= deep <= low1 < lift and entry <= c0 <= c1 <= lift. The curl may
  start before low0 (0010). Labels per frame: approach < entry <= descent < low0 <= sweep < c0 <= curl <= c1 < slide
  <= low1 < lift-label <= lift < out; when c0 < low0 the curl label wins.
Changed 2026-11-13 (user decision): phases used to run on poses rebuilt as markers and refitted with the USD bowl
template, on the uncalibrated spoon (2026-10-21 prepare.py). See data/diagnostics/2026-11-13/migration_steps_4_5.

Migrated from the diagnostics:
  pose, analyse, crop      2026-10-13/greenbean_batch/tools/gb_common.py
  robust_poses              2026-10-19/soybean_calibration/tools/sb_common.py::poses
  correction, apply         2026-10-19/soybean_calibration/tools/calibrate.py
  ingest_take               2026-10-21/soybean_batch/tools/prepare.py::main (per take, without the yaw / x1.6 CSV)
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

import scooping_io as sio
import scooping_metrics as sm
from trajectory_diagnostics import fit_pose

BAD_FIT = 0.004            # bowl fit residual (m) above which a marker is suspect
LOW = 0.010                # dish tip within this of the surface = low (scooping) frames
LOW_GAP_S = 0.2            # low stretches closer than this are one stretch
CROP_BEFORE, CROP_AFTER = 0.5, 0.5


def pose(local, world):
    """4x4 pose placing the template `local` on the points `world` (Kabsch), and the fit residual (m)."""
    c, r, res = fit_pose(local, world)
    T = np.eye(4); T[:3, :3] = r.as_matrix(); T[:3, 3] = c - r.apply(local.mean(0))
    return T, res


def robust_poses(M, bowl_tpl, spoon_tpl, ok=None):
    """Bowl holder H and spoon {S} world poses per frame, with the rigid-fit residuals (m).

    ok: (N, 9) markers to use (gap-filled excluded); the bowl needs >= 3 good markers, else all are used. A bowl fit
    above BAD_FIT drops the marker that disagrees; frames still above it are interpolated (position linear, Slerp).
    """
    H, S, rb, rs = [], [], [], []
    for k, P in enumerate(M):
        use = np.ones(5, bool) if ok is None else ok[k, :5].copy()
        if use.sum() < 3:
            use = np.ones(5, bool)
        h, a = pose(bowl_tpl[use], P[:5][use])
        if a > BAD_FIT and use.sum() > 3:
            idx = np.nonzero(use)[0]
            for m in range(len(idx) - 1, 2, -1):
                best = min((pose(bowl_tpl[list(c)], P[:5][list(c)]) for c in combinations(idx, m)), key=lambda x: x[1])
                if best[1] < a:
                    h, a = best
                if a <= BAD_FIT:
                    break
        s, b = pose(spoon_tpl, P[5:])
        H.append(h); S.append(s); rb.append(a); rs.append(b)
    H, rb = np.array(H), np.array(rb)
    bad = rb > BAD_FIT
    if bad.any() and (~bad).sum() > 2:
        good = np.nonzero(~bad)[0]; kk = np.arange(len(H)).clip(good[0], good[-1])
        H[:, :3, 3] = np.stack([np.interp(kk, good, H[good, a, 3]) for a in range(3)], axis=1)
        H[:, :3, :3] = Slerp(good, Rot.from_matrix(H[good, :3, :3]))(kk).as_matrix()
    return H, np.array(S), rb, np.array(rs)


def correction(x, parts):
    """Bowl correction C and spoon correction Ts (4x4) from x; parts names x's 3-vectors in order:
    't' bowl shift (holder frame), 'w' bowl rotation vector about the bowl floor centre, 's' spoon shift ({S})."""
    t = w = s = np.zeros(3)
    for i, p in enumerate(parts):
        v = x[3 * i:3 * i + 3]
        if p == 't':
            t = v
        elif p == 'w':
            w = v
        else:
            s = v
    O = sm.T_HB[:3, 3]
    R = Rot.from_rotvec(w).as_matrix()
    C = np.eye(4); C[:3, :3] = R; C[:3, 3] = O - R @ O + t
    Ts = np.eye(4); Ts[:3, 3] = s
    return C, Ts


def apply(rel, x, parts):
    """Calibrated spoon pose(s) in the holder frame: C @ rel @ Ts (rel: one 4x4 or a stack)."""
    C, Ts = correction(x, parts)
    return C @ rel @ Ts


def analyse(geo, times, H, S):
    """Phase indices, per-frame signals and a summary dict from the bowl holder and spoon poses (module docstring:
    phase standard). Raises ValueError when a phase cannot be found or the order is violated."""
    g = geo.g
    B = H @ sm.T_HB
    rel = np.linalg.inv(H) @ S
    tipH = (rel @ np.r_[sm.TIP, 1])[:, :3]
    tipB = (np.linalg.inv(sm.T_HB) @ np.c_[tipH, np.ones(len(tipH))].T).T[:, :3]
    h_tip = -geo.inner.depth(tipH)
    n = len(times)
    pen = np.full(n, np.nan); pen_part = np.full(n, np.nan); pen_rad = np.full(n, np.nan)
    for k in range(n):
        v = geo.spoon_V @ rel[k, :3, :3].T + rel[k, :3, 3]
        near = np.nonzero(v[:, 1] - g.height(v[:, [0, 2]]) < 0.015)[0]
        if len(near):
            d = geo.inner.depth(v[near])
            if np.isfinite(d).any():
                j = int(np.nanargmax(d))
                pen[k] = d[j]; pen_part[k] = geo.spoon_V[near[j], 0]
                pen_rad[k] = np.hypot(v[near[j], 0] - sm.BOWL_AXIS_X, v[near[j], 2])
    face_up = np.degrees(np.arccos(np.clip(S[:, 2, 2], -1, 1)))
    tilt = np.degrees(np.arccos(np.clip(B[:, 2, 2], -1, 1)))
    a = np.arctan2(-B[:, 2, 1], -B[:, 2, 0])                 # low-rim direction in B
    low_w = np.einsum('nij,nj->ni', B[:, :3, :3], np.c_[np.cos(a), np.sin(a), np.zeros_like(a)])
    rad = np.hypot(tipB[:, 0], tipB[:, 1])
    inside = (tipB[:, 2] < sm.RIM_Z) & (rad < sm.RIM_R)
    if not inside.any():
        raise ValueError('the dish tip never enters the rim cylinder')
    i_entry = int(np.argmax(inside))
    low = np.nonzero((h_tip < LOW) & (np.arange(n) >= i_entry))[0]
    if not len(low):
        raise ValueError(f'the dish tip never comes within {LOW * 1e3:.0f} mm of the bowl surface')
    runs = np.split(low, np.nonzero(np.diff(times[low]) > LOW_GAP_S + 1e-9)[0] + 1)
    main = max(runs, key=len); i_low0, i_low1 = int(main[0]), int(main[-1])
    above = np.nonzero((tipB[:, 2] > sm.RIM_Z) & (np.arange(n) > i_low1))[0]
    i_lift = int(above[0]) if len(above) else n - 1
    seg = np.arange(i_entry, i_lift)
    i_pk = int(seg[np.argmax(face_up[seg])]); after = np.arange(i_pk, i_lift)
    i_min = int(after[np.argmin(face_up[after])]); drop = face_up[i_pk] - face_up[i_min]
    i_c0 = int(after[np.argmax(face_up[after] <= face_up[i_pk] - 0.1 * drop)])
    i_c1 = int(after[np.argmax(face_up[after] <= face_up[i_pk] - 0.9 * drop)])
    phase = np.full(n, 'out', dtype='<U8')
    phase[:i_entry] = 'approach'; phase[i_entry:i_low0] = 'descent'; phase[i_low0:max(i_c0, i_low0)] = 'sweep'
    phase[i_c0:i_c1 + 1] = 'curl'; phase[i_c1 + 1:i_low1 + 1] = 'slide'; phase[max(i_low1, i_c1) + 1:i_lift + 1] = 'lift'
    sw = np.arange(i_low0, min(max(i_c0, i_low0), i_low1) + 1)
    scoop = np.arange(i_low0, i_low1 + 1)
    i_deep = int(scoop[np.argmin(h_tip[scoop])])
    pk = int(np.nanargmax(np.where(np.isfinite(pen), pen, -1)))
    Bs = B[scoop]
    summ = dict(
        frames=n, duration_s=float(times[-1] - times[0]),
        entry_s=float(times[i_entry]), low_s=[float(times[i_low0]), float(times[i_low1])],
        curl_s=[float(times[i_c0]), float(times[i_c1])], curl_deg=float(drop), lift_s=float(times[i_lift]),
        depth_mm=float(h_tip[i_deep] * 1e3), depth_radius_mm=float(rad[i_deep] * 1e3),
        sweep_mm=float(np.linalg.norm(np.diff(tipB[sw], axis=0), axis=1).sum() * 1e3),
        low_path_mm=float(np.linalg.norm(np.diff(tipB[scoop], axis=0), axis=1).sum() * 1e3),
        tip_height_scoop_mm=dict(p5=float(np.percentile(h_tip[scoop], 5) * 1e3), median=float(np.median(h_tip[scoop]) * 1e3)),
        penetration_max_mm=float(np.nanmax(pen) * 1e3) if np.isfinite(pen).any() else None,
        penetration_at_s=float(times[pk]), penetration_phase=str(phase[pk]),
        tilt_scoop_deg=dict(min=float(tilt[scoop].min()), median=float(np.median(tilt[scoop])), max=float(tilt[scoop].max())),
        tilt_start_deg=float(tilt[0]), tilt_end_deg=float(tilt[-1]),
        low_rim_dir_B_deg=float(np.degrees(np.angle(np.mean(np.exp(1j * a[scoop]))))),
        low_rim_azimuth_world_deg=float(np.degrees(np.arctan2(low_w[scoop, 1].mean(), low_w[scoop, 0].mean()))),
        bowl_motion_scoop=dict(translation_mm=float(np.linalg.norm(Bs[:, :3, 3] - Bs[:, :3, 3].mean(0), axis=1).max() * 1e3),
                               rotation_deg=float(max(np.degrees(np.linalg.norm(Rot.from_matrix(Bs[0, :3, :3].T @ R).as_rotvec()))
                                                      for R in Bs[:, :3, :3]))),
    )
    sig = dict(tipB=tipB, h_tip=h_tip, pen=pen, face_up=face_up, tilt=tilt, phase=phase)
    idx = dict(entry=i_entry, low0=i_low0, low1=i_low1, c0=i_c0, c1=i_c1, lift=i_lift, deep=i_deep)
    check_phases(idx, n)
    return summ, sig, idx


def check_phases(idx, n):
    """Raise ValueError unless entry <= low0 <= deep <= low1 < lift and entry <= c0 <= c1 <= lift < n."""
    i = idx
    chains = [('entry', 'low0', 'deep', 'low1'), ('entry', 'c0', 'c1', 'lift')]
    bad = [f'{a} {i[a]} > {b} {i[b]}' for ch in chains for a, b in zip(ch, ch[1:]) if i[a] > i[b]]
    if i['low1'] >= i['lift']:
        bad.append(f"low1 {i['low1']} >= lift {i['lift']}")
    if not (0 <= i['entry'] and i['lift'] < n):
        bad.append('index outside the take')
    if bad:
        raise ValueError('phase order violated: ' + '; '.join(bad))


def crop(idx, times, before=CROP_BEFORE, after=CROP_AFTER):
    """Frame range [entry - before, lift + after] (inclusive indices)."""
    i0 = max(0, int(np.searchsorted(times, times[idx['entry']] - before)))
    i1 = min(len(times) - 1, int(np.searchsorted(times, times[idx['lift']] + after)))
    return i0, i1


def ingest_take(geo, tsv, calibration, pts=None):
    """One raw take -> (episode arrays, summary). calibration: dict with layout_H (5, 3), x (m), parts.

    Raises ValueError if the phases cannot be detected.
    """
    layout = np.asarray(calibration['layout_H'], float)
    x, parts = np.asarray(calibration['x'], float), list(calibration['parts'])
    pts = sm.subsample(geo.spoon_V) if pts is None else pts
    t, M, ok = sio.load_take(tsv)
    H, S, rb, rs = robust_poses(M, layout, geo.spoon_s, ok)
    rel_all = np.linalg.inv(H) @ S
    rel_cal_all = apply(rel_all, x, parts)
    try:
        summ, sig, idx = analyse(geo, t, H, H @ rel_cal_all)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f'phase detection failed: {type(e).__name__}: {e}') from e
    i0, i1 = crop(idx, t)
    k = slice(i0, i1 + 1)
    rel, rel_cal = rel_all[k], rel_cal_all[k]
    d_raw = sm.deepest_par(geo, rel, pts)[0]
    d_cal = sm.deepest_par(geo, rel_cal, pts)[0]
    tt = t[k] - t[i0]
    rel_idx = {n: int(v - i0) for n, v in idx.items()}
    scoop = slice(rel_idx['low0'], rel_idx['low1'] + 1)
    arrays = dict(t=tt, H=H[k], raw=rel, cal=rel_cal, depth_raw=d_raw, depth_cal=d_cal, phase=sig['phase'][k],
                  idx=np.array(json.dumps(rel_idx)), t_qtm0=np.array(t[i0]), crop=np.array([i0, i1]),
                  bowl_fit_res=rb[k], spoon_fit_res=rs[k], not_gap_filled=ok[k])
    summary = dict(
        duration_s=float(tt[-1]),
        phases_s={n: round(float(tt[v]), 2) for n, v in rel_idx.items()},
        bowl_fit_res_mm=dict(median=float(np.median(rb) * 1e3), frames_over_4mm=int((rb > BAD_FIT).sum())),
        gap_filled_bowl_frames=int((~ok[:, :5]).any(axis=1).sum()),
        raw=dict(penetration_max_mm=float(np.nanmax(d_raw) * 1e3),
                 clearance_scoop_median_mm=float(-np.nanmedian(d_raw[scoop]) * 1e3)),
        cal=dict(penetration_max_mm=float(np.nanmax(d_cal) * 1e3),
                 clearance_scoop_median_mm=float(-np.nanmedian(d_cal[scoop]) * 1e3),
                 frames_inside_gt2mm=int(np.nansum(d_cal > 0.002))),
        tilt_scoop_deg=summ['tilt_scoop_deg'], bowl_motion_scoop=summ['bowl_motion_scoop'], curl_deg=summ['curl_deg'],
        source=dict(tsv=str(tsv), sha256=sio.sha256(tsv), qtm_header=sio.read_header(tsv),
                    crop_qtm_s=[float(t[i0]), float(t[i1])]),
        calibration=dict(x=list(map(float, x)), parts=parts, spoon=geo.spoon),
    )
    return arrays, summary
