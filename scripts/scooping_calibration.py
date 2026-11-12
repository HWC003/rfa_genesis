"""Session calibration: measured bowl marker layout + spoon-vs-bowl registration, once per recording session.

1. learn_layout: generalised Procrustes of the 5 bowl markers over frames where all are measured, best-fitted
   (Kabsch) onto the USD template -> layout_H, the bowl markers in the holder frame as they really sit.
2. fit: in the calibration take the spoon tip is rubbed over the empty bowl, so in every contact frame the deepest
   spoon point should lie on the inner surface. Robust point-to-plane ICP (Gauss-Newton) of a constant correction
   rel_cal = C @ rel @ Ts (scooping_ingest.apply): t = bowl shift (holder frame), s = spoon shift ({S}), optional
   w = bowl rotation. Cross-validated on 5 time blocks. Gate: CV RMS below GATE_CV_RMS_MM.

Migrated from the diagnostics, behaviour unchanged:
  learn_layout      2026-10-19/soybean_calibration/tools/learn_layout.py
  icp, mad_band     2026-10-19/soybean_calibration/tools/calibrate.py
  frame selection   calibrate.py::main (tip inside the rim cylinder, every 4th frame)
The 2026-10-21 v3 rerun (spoon_mount_v3/tools/rerun_correction_v3.py) is fit(..., spoon='v3') with the frames
selected by the v2 mesh (select_spoon='v2'); select_spoon=None uses the fitted spoon for both.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import scooping_io as sio
import scooping_metrics as sm
from scooping_ingest import apply, pose, robust_poses

GATE_CV_RMS_MM = 1.0
CV_BLOCKS = 5


def learn_layout(tsvs, bowl_usd, iters=20):
    """Measured bowl marker layout in the holder frame (5, 3) and a report dict, from the given takes."""
    shapes = []
    for path in tsvs:
        _, M, ok = sio.load_take(path)
        shapes.append(M[ok[:, :5].all(1), :5])
    X = np.concatenate(shapes); X = X - X.mean(1, keepdims=True)
    mean = X[0].copy()
    for _ in range(iters):
        al = []
        for Y in X:
            R, _ = Rot.align_vectors(mean, Y); al.append(R.apply(Y))
        al = np.array(al); new = al.mean(0); new -= new.mean(0)
        if np.abs(new - mean).max() < 1e-7:
            break
        mean = new
    spread = np.linalg.norm(al - mean, axis=2)
    T, _ = pose(mean, bowl_usd)
    layout_H = mean @ T[:3, :3].T + T[:3, 3]
    diff = np.linalg.norm(layout_H - bowl_usd, axis=1) * 1e3
    rep = dict(frames=int(len(X)), names=sio.NAMES[:5],
               marker_noise_mm_p50=(np.median(spread, 0) * 1e3).round(2).tolist(),
               marker_noise_mm_p95=(np.percentile(spread, 95, 0) * 1e3).round(2).tolist(),
               usd_vs_measured_mm=diff.round(2).tolist(), usd_vs_measured_rms_mm=float(np.sqrt((diff ** 2).mean())))
    return layout_H, rep


def mad_band(r, centre):
    """Frames whose residual is within 3 scaled MADs (+ 0.1 mm) of `centre`, and the scaled MAD (m)."""
    ok = np.isfinite(r)
    mad = 1.4826 * np.median(np.abs(r[ok] - np.median(r[ok]))) + 1e-4
    return ok & (np.abs(r - centre) < 3 * mad), mad


def icp(geo, pts, rel, parts, frames, iters=8):
    """Robust point-to-plane fit of the correction parameters on the given frames. Returns x and the residuals."""
    O = sm.T_HB[:3, 3]
    x = np.zeros(3 * len(parts))
    for it in range(iters + 1):
        rc = apply(rel[frames], x, parts)
        d, p, c, n, _ = sm.deepest_par(geo, rc, pts)
        r = -d                                            # signed clearance of the deepest point (want 0)
        ok = np.isfinite(r)
        inl = ok if it == 0 else mad_band(r, np.median(r[ok]))[0]
        if it == iters or not parts:
            break
        J = []
        for name in parts:
            if name == 't':
                J.append(n[inl])
            elif name == 'w':
                J.append(np.cross(p[inl] - O, n[inl]))
            else:
                J.append(np.einsum('ki,kij->kj', n[inl], rc[inl, :3, :3]))
        dx = np.linalg.lstsq(np.concatenate(J, axis=1), -r[inl], rcond=None)[0]
        x = x + dx
        if np.abs(dx).max() < 2e-5:
            rc = apply(rel[frames], x, parts)
            r = -sm.deepest_par(geo, rc, pts)[0]
            ok = np.isfinite(r)
            inl = mad_band(r, np.median(r[ok]))[0]
            break
    return x, dict(residual=r, inlier=inl)


def stats(r, inl):
    r = r[inl] * 1e3
    return dict(rms_mm=float(np.sqrt(np.mean(r ** 2))), median_mm=float(np.median(r)),
                p5_mm=float(np.percentile(r, 5)), p95_mm=float(np.percentile(r, 95)), inliers=int(inl.sum()))


def calibration_frames(geo, rel, pts):
    """Frames with the dish tip inside the rim cylinder and a spoon point near the bowl; every 4th (40 ms at 100 Hz)."""
    d0 = sm.deepest_par(geo, rel, pts)[0]
    tipB = np.linalg.inv(sm.T_HB) @ rel @ np.r_[sm.TIP, 1]
    inside = np.isfinite(d0) & (tipB[:, 2] < sm.RIM_Z) & (np.hypot(tipB[:, 0], tipB[:, 1]) < sm.RIM_R)
    return np.nonzero(inside)[0][::4]


def cross_validate(geo, pts, rel, parts, frames):
    """Fit on 4 of CV_BLOCKS time blocks, measure the contact residual on the held-out one."""
    blocks = np.array_split(frames, CV_BLOCKS)
    cv = []
    for b in range(CV_BLOCKS):
        train = np.concatenate([blocks[i] for i in range(CV_BLOCKS) if i != b])
        xb, _ = icp(geo, pts, rel, parts, train, iters=6)
        r = -sm.deepest_par(geo, apply(rel[blocks[b]], xb, parts), pts)[0]
        ok = np.isfinite(r)
        mad = 1.4826 * np.median(np.abs(r[ok])) + 1e-4
        keep = ok & (np.abs(r) < max(3 * mad, 0.002))
        cv.append(dict(rms_mm=float(np.sqrt(np.mean(r[keep] ** 2)) * 1e3), median_mm=float(np.median(r[ok]) * 1e3),
                       x_mm=(xb * 1e3).round(2).tolist()))
    return dict(cv_rms_mm=float(np.mean([c['rms_mm'] for c in cv])),
                cv_median_mm=float(np.mean([c['median_mm'] for c in cv])),
                cv_param_spread_mm=np.std([c['x_mm'] for c in cv], axis=0).round(2).tolist() if parts else [],
                blocks=cv)


def fit(cal_tsv, layout_H, spoon='v3', parts=('t', 's'), select_spoon=None, cv=True):
    """Calibration of one session from its calibration take. Returns the calibration dict (JSON-able)."""
    parts = list(parts)
    geo = sm.DemoGeometry(spoon)
    pts = sm.subsample(geo.spoon_V)
    _, M, ok = sio.load_take(cal_tsv)
    H, S, rb, _ = robust_poses(M, np.asarray(layout_H), geo.spoon_s, ok)
    rel = np.linalg.inv(H) @ S
    if select_spoon and select_spoon != spoon:
        g2 = sm.DemoGeometry(select_spoon)
        frames = calibration_frames(g2, rel, sm.subsample(g2.spoon_V))
    else:
        frames = calibration_frames(geo, rel, pts)
    x, res = icp(geo, pts, rel, parts, frames)
    out = dict(layout_H=np.asarray(layout_H).tolist(), x=x.tolist(), x_mm=(x * 1e3).round(4).tolist(), parts=parts,
               spoon=spoon, frame_select_spoon=select_spoon or spoon, n_frames=int(len(frames)),
               fit=stats(res['residual'], res['inlier']), inlier_frac=float(res['inlier'].mean()),
               source=dict(tsv=str(cal_tsv), sha256=sio.sha256(cal_tsv)))
    if cv:
        out['cv'] = cross_validate(geo, pts, rel, parts, frames)
        out['gate'] = dict(cv_rms_max_mm=GATE_CV_RMS_MM, passed=out['cv']['cv_rms_mm'] < GATE_CV_RMS_MM)
    return out, dict(frames=frames, residual=res['residual'], inlier=res['inlier'])
