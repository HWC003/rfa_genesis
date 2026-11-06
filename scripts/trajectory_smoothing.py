"""Start a joint trajectory from rest without changing its path.

The sb_robot_trajs joint CSVs start mid-motion (up to ~6 rad/s in the first 4 ms step; the spoon is entering
the bowl), which a robot standing still at the first row cannot follow. smooth_start retimes the first
`window` seconds of the CSV over `duration` seconds with a quintic time scaling s(tau):

    s(0) = 0, s'(0) = 0, s''(0) = 0        start at rest at the first row
    s(D) = window, s'(D) = 1, s''(D) = 0   join the CSV at t = window at its own speed

Every joint passes through exactly the same joint-space path as the CSV (same spoon poses during bowl entry),
only slower at the start. The rest of the CSV follows unchanged, shifted by duration - window seconds.
s stays monotonic for window < duration <= 2.5 * window.

Pure numpy: used by oa_real_scooping_replay.py (ROS) and oa_joint_replay.py (Genesis).
"""
import numpy as np


def smooth_start(times, q, window=0.3, duration=0.7):
    """Return (times, q) with the first `window` s retimed over `duration` s; window <= 0 returns the input."""
    times, q = np.asarray(times, float), np.asarray(q, float)
    if window <= 0:
        return times, q
    dt = float(np.median(np.diff(times)))
    window = round(window / dt) * dt          # keep both ends on the CSV's sample grid
    steps = round(duration / dt)
    duration = steps * dt
    if not 0 < window < times[-1] - times[0]:
        raise ValueError(f'blend window {window:.3f} s must lie inside the trajectory')
    if not window < duration <= 2.5 * window + 1e-9:
        raise ValueError(f'blend duration must be in ({window:.3f}, {2.5 * window:.3f}] s for a monotonic retiming')
    r = duration / window
    c3, c4, c5 = 10 - 4 * r, 7 * r - 15, 6 - 3 * r
    u = np.arange(steps + 1) / steps
    s = times[0] + window * (c3 * u**3 + c4 * u**4 + c5 * u**5)
    blend = np.stack([np.interp(s, times, q[:, j]) for j in range(q.shape[1])], axis=1)
    rest = times > times[0] + window + dt / 2
    new_times = np.r_[times[0] + u * duration, times[rest] + (duration - window)]
    return new_times, np.vstack([blend, q[rest]])


def peak_velocity(times, q):
    """Largest finite-difference joint speed (rad/s)."""
    return float(np.abs(np.diff(q, axis=0) / np.diff(times)[:, None]).max())
