"""Motion edits on a demo's poses (no IK): what the robot is asked to do with the bowl and the spoon.

Frames (data/diagnostics/CONVENTIONS.md): holder frame H (bowl axis = holder +y, rim centre RIM_C, axis through
x = BOWL_AXIS_X, z = 0); spoon frame {S}; world z up. A demo is T(t) = bowl holder in the world and the spoon pose
relative to it. Edits:
  turn_and_place   yaw about the vertical through the first Bowl_center, then that point moved to the placement
  wall_margin      spoon pushed off the bowl surface (0.5 mm near the axis -> 3 mm on the wall) ("calwall")
  smooth_relative  spoon-in-bowl pose de-spiked (Hampel) and low-passed at 4 Hz
  smooth_bowl      bowl: flat demos one level pose; tilted demos one min-jerk tilt to the mean scoop pose, then held
                   (optional tip-back after the scoop)
  spin_to          bowl turned about its own axis to a given spin
  floor_lift       spoon raised along the bowl normal e (holder +y) where the planned clearance is below a margin
                   (correction-ladder rung for floor jams)

Migrated unchanged from the diagnostics:
  turn_and_place   2026-10-21/soybean_batch/tools/prepare.py::turned_markers + scooping_kinematics.mapped_targets
                   (on poses instead of markers)
  wall_margin      2026-10-21/soybean_batch/tools/wall_margin.py::correct_wall
  floor_lift       2026-11-06/bowl_smoothing_and_spin/tools/floor_lift.py::lift_profile
  the rest         2026-11-06/bowl_smoothing_and_spin/tools/replan_bowl.py and bowl_common.py
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp

import scooping_metrics as sm

RIM_C = np.r_[sm.BOWL_AXIS_X, sm.RIM_CENTRE[1], 0.]   # rim centre on the bowl axis, holder frame
REL_LOWPASS_HZ = 4.0      # spoon-in-bowl: the scoop is below 3 Hz, the demo's bowl shake above
FLAT_TILT_DEG = 5.0       # own tilt below this = flat demo (2026-10-17 soybean: flat 0.4-1.1 deg, tilted >= 15.5 deg)
WALL = dict(floor=0.0005, wall=0.003, r0=0.055, r1=0.065, rim_r=0.0735)


def pose_stack(p, R):
    T = np.repeat(np.eye(4)[None], len(p), 0); T[:, :3, :3] = R.as_matrix(); T[:, :3, 3] = p
    return T


# ---------------------------------------------------------------- placement / time ---
def turn_and_place(H, yaw_deg, point, pivot_local):
    """Holder poses turned by yaw_deg about the vertical through the first pivot sample (pivot_local in the holder
    frame: the Bowl_center marker), then translated so that sample sits at `point` (world, m)."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    piv = (H[0] @ np.r_[pivot_local, 1])[:3]
    Rz = np.eye(4); Rz[:2, :2] = [[c, -s], [s, c]]; Rz[:3, 3] = piv - Rz[:3, :3] @ piv
    out = Rz @ H
    out[:, :3, 3] += np.asarray(point, float) - piv
    return out


# ----------------------------------------------------------------- spoon margin ---
def _wall_need(r):
    w = WALL
    return w['floor'] + (w['wall'] - w['floor']) * np.clip((r - w['r0']) / (w['r1'] - w['r0']), 0, 1)


def wall_margin(geo, rel, iters=5, sigma=8):
    """Spoon {S} poses in the holder frame pushed along the surface normal until every spoon point keeps its margin
    (inner surface; closed mesh near the rim). Translation only, Gaussian-smoothed over time.
    geo: scooping_metrics.DemoGeometry. Returns (rel, shift (N, 3) m)."""
    g, verts, w = geo.g, geo.spoon_V, WALL
    pts = verts[np.unique(np.round(verts / 0.003).astype(int), axis=0, return_index=True)[1]]
    shift = np.zeros((len(rel), 3))
    for _ in range(iters):
        need = np.zeros((len(rel), 3))
        for k, T in enumerate(rel):
            v = pts @ T[:3, :3].T + T[:3, 3] + shift[k]
            r = np.hypot(v[:, 0] - sm.BOWL_AXIS_X, v[:, 2])
            req = []
            sel = (v[:, 1] - g.height(v[:, [0, 2]]) < w['wall'] + 0.01) & (r <= w['rim_r'])
            if sel.any():
                sd = geo.inner.depth(v[sel]); viol = sd + _wall_need(r[sel])
                if np.isfinite(viol).any():
                    j = int(np.nanargmax(viol))
                    if viol[j] > 0:
                        p = v[sel][j]; c = geo.inner.closest(p[None])[0]; n = c - p; nn = np.linalg.norm(n)
                        if nn > 1e-6:
                            n = n / nn if sd[j] > 0 else -n / nn
                            req.append(n * viol[j])
            rim = v[(r > w['rim_r']) & (r < 0.095) & (v[:, 1] > -0.04) & (v[:, 1] < 0.06)]
            if len(rim):
                sd = geo.bowl_q.signed_distance(rim); j = int(np.argmax(sd))
                if sd[j] > -w['wall']:
                    c = geo.bowl_q.on_surface(rim[j:j + 1])[0][0]; n = c - rim[j]; nn = np.linalg.norm(n)
                    if nn > 1e-6:
                        n = n / nn if sd[j] > 0 else -n / nn
                        req.append(n * (sd[j] + w['wall']))
            if req:
                need[k] = max(req, key=np.linalg.norm)
        sm_need = gaussian_filter1d(need, sigma, axis=0, mode='nearest')
        shift += np.where(np.linalg.norm(need, axis=1, keepdims=True) > np.linalg.norm(sm_need, axis=1, keepdims=True),
                          need, sm_need)
        shift = gaussian_filter1d(shift, sigma / 2, axis=0, mode='nearest')
    out = rel.copy(); out[:, :3, 3] += shift
    return out, shift


# ---------------------------------------------------------------- bowl measures ---
def rim_centre(T):
    return T[:, :3, :3] @ RIM_C + T[:, :3, 3]


def holder_from_rim(c, R):
    """Holder poses with rim centre c and rotation R."""
    return pose_stack(c - R.apply(RIM_C), R)


def tilt_deg(T):
    """Angle between the bowl axis (holder +y) and world +z (deg)."""
    return np.degrees(np.arccos(np.clip(T[..., 2, 1], -1, 1)))


def spin_deg(T):
    """Holder +x about the bowl axis from world +Y projected onto the rim plane (deg, unwrapped)."""
    a, x = T[..., :3, 1], T[..., :3, 0]
    r = np.array([0, 1., 0]) - (a @ np.array([0, 1., 0]))[..., None] * a
    r /= np.linalg.norm(r, axis=-1, keepdims=True)
    s = np.cross(a, r)
    return np.degrees(np.unwrap(np.arctan2((x * s).sum(-1), (x * r).sum(-1))))


def dish_height_mm(T, S):
    """Dish centre height above the rim along the bowl axis (mm); T holder, S spoon {S}, both world."""
    dish = np.einsum('nij,j->ni', S[:, :3, :3], sm.DISH_CENTRE) + S[:, :3, 3]
    local = np.einsum('nji,nj->ni', T[:, :3, :3], dish - T[:, :3, 3]) - sm.RIM_CENTRE
    return local[:, 1] * 1000


def own_tilt_deg(T, inside):
    """Tilt of the mean bowl axis over the frames with the dish below the rim (the demo's scooping tilt)."""
    a = T[inside, :3, 1].mean(0)
    return float(np.degrees(np.arccos(np.clip(a[2] / np.linalg.norm(a), -1, 1))))


# ------------------------------------------------------------------ bowl edits ---
def level(R):
    """R turned (minimally) so the bowl axis (holder +y) is world +z."""
    a = R.as_matrix()[:, 1]
    v = np.cross(a, [0, 0, 1.]); s = np.linalg.norm(v)
    if s < 1e-9:
        return R
    return Rotation.from_rotvec(v / s * np.arctan2(s, a[2])) * R


def min_jerk(u):
    u = np.clip(u, 0, 1)
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def smooth_bowl(t, T, inside, flat, tipback=False):
    """Bowl hold: (holder poses, tilt-end time).

    flat: one level pose (mean position and spin) for the whole trajectory. Tilted: one min-jerk tilt from the
    frame-0 pose to the HOLD pose (mean rim centre + rotation over the dish-below-rim frames), finishing when the
    dish first goes below the rim, then held. tipback: after the dish's last frame below the rim, one min-jerk move
    back to the demo's end pose (mean of the last 0.2 s).
    """
    c, R = rim_centre(T), Rotation.from_matrix(T[:, :3, :3])
    if flat:
        Rh = level(R.mean())
        return holder_from_rim(np.tile(c.mean(0), (len(t), 1)), Rotation.concatenate([Rh] * len(t))), 0.0
    ch, Rh = c[inside].mean(0), R[inside].mean()
    t_end = t[np.argmax(inside)]
    s = min_jerk(t / t_end)
    cs = c[0] + s[:, None] * (ch - c[0])
    Rs = Slerp([0, 1], Rotation.concatenate([R[0], Rh]))(s)
    if tipback:
        t_exit = t[len(t) - 1 - np.argmax(inside[::-1])]
        last = t > t[-1] - 0.2
        u = min_jerk((t - t_exit) / (t[-1] - t_exit))
        cs = cs + u[:, None] * (c[last].mean(0) - ch)
        Rs = Rotation.concatenate([Slerp([0, 1], Rotation.concatenate([r, R[last].mean()]))([ui])
                                   for r, ui in zip(Rs, u)])
    return holder_from_rim(cs, Rs), t_end


def about_axis(delta_deg):
    """Holder-frame rotation by delta about the bowl axis (holder +y through x = BOWL_AXIS_X, z = 0)."""
    M = np.repeat(np.eye(4)[None], len(delta_deg), 0)
    M[:, :3, :3] = Rotation.from_rotvec(np.radians(delta_deg)[:, None] * [0, 1., 0]).as_matrix()
    p0 = np.array([sm.BOWL_AXIS_X, 0, 0])
    M[:, :3, 3] = p0 - M[:, :3, :3] @ p0
    return M


def wrap(a):
    return (a + 180) % 360 - 180


def spin_to(T, target_deg):
    """Holder poses turned about the bowl axis so the spin is target_deg on every frame."""
    d = wrap(target_deg - spin_deg(T))
    for sign in (1, -1):
        out = T @ about_axis(sign * d)
        if np.abs(wrap(spin_deg(out) - target_deg)).max() < 0.1:
            return out
    raise RuntimeError('spin rotation did not reach the target')


# ----------------------------------------------------------------- spoon edits ---
def hampel(x, k=5, n_sigma=3.0):
    """Single-frame spikes replaced by the local median (time is kept; no frame is dropped)."""
    y = x.copy()
    for i in range(len(x)):
        w = x[max(0, i - k):i + k + 1]
        med = np.median(w, 0); mad = 1.4826 * np.median(np.abs(w - med), 0)
        bad = np.abs(x[i] - med) > n_sigma * np.maximum(mad, 1e-9)
        y[i, bad] = med[bad]
    return y


def smooth_relative(t, X, hz=REL_LOWPASS_HZ):
    """Pose stack (e.g. the spoon wrist in the holder frame) de-spiked and zero-phase low-passed at `hz`."""
    R = Rotation.from_matrix(X[:, :3, :3]); R0 = R.mean()
    x = hampel(np.c_[X[:, :3, 3], (R0.inv() * R).as_rotvec()])
    b, a = butter(2, hz / (0.5 / np.median(np.diff(t))))
    x = filtfilt(b, a, x, axis=0)
    return pose_stack(x[:, :3], R0 * Rotation.from_rotvec(x[:, 3:]))


# ---------------------------------------------------------------- floor lift ---
LIFT_ENV_S = 0.15          # running-max half window of the lift envelope (s)
LIFT_LOWPASS_HZ = 3.0


def clearance_mm(geom, T, R, spoon_tool):
    """Planned spoon-bowl clearance (ScoopGeometry.gap, mm) for holder poses T and right-wrist poses R."""
    return np.array([geom.gap(B, W @ spoon_tool) for B, W in zip(T, R)]) * 1000


def floor_lift(geom, t, T, R, spoon_tool, lift_mm, iters=6):
    """Right-wrist targets R raised along the bowl normal (holder +y of T) wherever the clearance is below lift_mm.

    needed(t) = max(0, lift_mm - clearance(t)); envelope = running max over +-LIFT_ENV_S, then a 3 Hz zero-phase
    low-pass (smooth ramps in and out); repeated until the clearance is >= lift_mm everywhere (the floor is not
    exactly normal to e everywhere). Returns (lift (mm, per frame), lifted R, clearance after (mm))."""
    dt = np.diff(t).mean()
    b, a = butter(2, LIFT_LOWPASS_HZ / (0.5 / dt))
    L = np.zeros(len(t))
    for _ in range(iters):
        Rl = R.copy(); Rl[:, :3, 3] += (L / 1000)[:, None] * T[:, :3, 1]
        c = clearance_mm(geom, T, Rl, spoon_tool)
        need = np.maximum(0, lift_mm - c)
        if need.max() < .05:
            break
        env = maximum_filter1d(need, size=2 * int(round(LIFT_ENV_S / dt)) + 1)
        L += np.maximum(filtfilt(b, a, env), 0) * 1.2
    return L, Rl, c
