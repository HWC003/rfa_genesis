"""Start a joint trajectory from rest without changing its path.

The sb_robot_trajs joint CSVs start mid-motion (up to ~6 rad/s in the first 4 ms step; the spoon is entering
the bowl), which a robot standing still at the first row cannot follow. smooth_start retimes the first
`window` seconds of the CSV over `duration` seconds with a quintic time scaling s(tau):

    s(0) = 0, s'(0) = 0, s''(0) = 0        start at rest at the first row
    s(D) = window, s'(D) = 1, s''(D) = 0   join the CSV at t = window at its own speed

Every joint passes through exactly the same joint-space path as the CSV (same spoon poses during bowl entry),
only slower at the start. The rest of the CSV follows unchanged, shifted by duration - window seconds.
s stays monotonic for window < duration <= 2.5 * window.

Where it is applied (since 2026-11-13, pipeline migration step 4):
  - scooping_pipeline.py plan writes the ramp into the plan CSV (session plan: start_ramp [window, duration]), so
    the CSV starts at rest and simulation and robot run the same file. Its demo_time_s column maps every row back to
    the demo (QTM) time, ramp included.
  - oa_real_scooping_replay.py (default --blend-window auto) and oa_pure_scooping.py --joints --blend-window: only for
    CSVs that start moving (sb_robot_trajs v1 / v2); a CSV that starts at rest (starts_at_rest) is replayed as is.

Pure numpy: used by scooping_pipeline.py, oa_real_scooping_replay.py (ROS) and oa_pure_scooping.py (Genesis).
"""
import numpy as np

REST_SPEED = 0.05          # rad/s: a first step slower than this counts as starting at rest (v1 / v2 start >= 0.17)
DEFAULT_WINDOW, DEFAULT_DURATION = 0.3, 0.7


def smooth_start(times, q, window=DEFAULT_WINDOW, duration=DEFAULT_DURATION, return_source=False):
    """Return (times, q) with the first `window` s retimed over `duration` s; window <= 0 returns the input.
    return_source: also return, per output row, the input time it shows (the time map, for demo_time_s)."""
    times, q = np.asarray(times, float), np.asarray(q, float)
    if window <= 0:
        return (times, q, times.copy()) if return_source else (times, q)
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
    if return_source:
        return new_times, np.vstack([blend, q[rest]]), np.r_[s, times[rest]]
    return new_times, np.vstack([blend, q[rest]])


def start_speed(times, q):
    """Largest joint speed over the first step (rad/s)."""
    q = np.asarray(q, float)
    return float(np.abs(q[1] - q[0]).max() / (times[1] - times[0]))


def starts_at_rest(times, q, tol=REST_SPEED):
    """True when the first step is slower than tol (rad/s): the CSV already has its start ramp."""
    return start_speed(times, q) < tol


def auto_window(times, q, window=None):
    """The blend window to use: `window` if given, else 0 for a CSV that starts at rest and DEFAULT_WINDOW otherwise."""
    if window is not None:
        return window
    return 0.0 if starts_at_rest(times, q) else DEFAULT_WINDOW


def peak_velocity(times, q):
    """Largest finite-difference joint speed (rad/s)."""
    return float(np.abs(np.diff(q, axis=0) / np.diff(times)[:, None]).max())
