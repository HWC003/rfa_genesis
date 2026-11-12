"""Plan: a demo's bowl and spoon poses -> both arms' joint trajectory (4 ms rows) + plan metrics.

    targets_from_episode   episode (scooping_ingest) -> spoon margin (cal | calwall) -> turn_and_place (yaw,
                           placement) -> time scale -> holder poses T(t) and right-wrist targets WR(t) (world)
    targets_from_markers   legacy front end: a marker CSV as written by 2026-10-21 prepare.py / wall_margin.py
                           (sim_ready_trajs), Kabsch-fitted exactly as scooping_kinematics.plan_trajectory does
    plan                   bowl = 'demo' | 'hold' (scooping_motion.smooth_bowl, spoon kept in the bowl frame and
                           low-passed); spin = True: bowl turned about its axis to SPIN_TARGET + delta, first delta
                           in SPIN_DELTAS that passes check_plan; IK with a settled frame 0 and the bowl-arm J2 rule
    sample                 joint splines at DT -> rows for the joint CSV (scooping_io.write_joint_csv)

bowl='hold', spin=True is the 2026-11-06 'smooth_spin' re-plan (sb_robot_trajs/v2).
Migrated from data/diagnostics/2026-11-06/bowl_smoothing_and_spin/tools/replan_bowl.py (main, choose_spin, plan_one,
evaluate) and bowl_common.py (kins, wrist targets, bowl measures); behaviour unchanged.
"""
import numpy as np
from scipy.spatial.transform import Rotation

import scooping_kinematics as sk
import scooping_metrics as sm
import scooping_motion as mo
import scooping_scene as ss

SPIN_TARGET = 180.0                                       # X_B toward the gripper, as close to world +Y as the tilt allows
SPIN_DELTAS = [0, 2, -2, 4, -4, 6, -6, 8, -8, 10, -10]
MIN_MARGIN_DEG = 5.0
BOWL_J2_MIN_DEG = -90.0                                    # bowl arm J2 below this = shoulder-up branch (2026-11-10)
CONTINUITY = 0.012
DT_OUT = ss.DT


class Planner:
    def __init__(self, spoon='v3', margin_deg=MIN_MARGIN_DEG, continuity=CONTINUITY):
        usd = ss.robot_usd(False, spoon)
        self.spoon = spoon
        self.tpl = ss.marker_templates(usd)
        self.holder_to_link = np.asarray(self.tpl['holder_to_link'])
        self.spoon_tool = np.asarray(self.tpl['spoon_tool'])
        self.kl = sk.ArmKinematics('left', margin=np.deg2rad(margin_deg), continuity=continuity)
        self.kr = sk.ArmKinematics('right', margin=np.deg2rad(margin_deg), continuity=continuity)
        self.geom = sm.ScoopGeometry(usd, spoon_tool=self.spoon_tool)
        self.margin_deg = margin_deg

    # ---------------------------------------------------------------- targets ---
    def targets_from_markers(self, markers):
        """(N, 9, 3) world markers -> holder poses T and right-wrist poses WR (world, 4x4)."""
        pos, quat = sk.pose_targets(markers, np.array(self.tpl['bowl']), np.array(self.tpl['spoon']))
        WL, WR = [mo.pose_stack(pos[:, i], Rotation.from_quat(quat[:, i][:, [1, 2, 3, 0]])) for i in range(2)]
        return WL @ self.holder_to_link, WR

    def targets_from_episode(self, ep, rel, yaw_deg, point, time_scale):
        """Episode arrays + the spoon-in-holder poses to use -> (times, holder poses T, right-wrist poses WR)."""
        T = mo.turn_and_place(ep['H'], yaw_deg, point, np.asarray(self.tpl['bowl_holder'])[0])
        WR = T @ rel @ np.linalg.inv(self.spoon_tool)
        return ep['t'] * time_scale, T, WR

    # ------------------------------------------------------------------- plan ---
    def _plan(self, kin, t, W, seed, j2=False):
        q = Rotation.from_matrix(W[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
        return sk.plan_arm(kin, t, W[:, :3, 3], q, seed, settle=True, j2_min_deg=BOWL_J2_MIN_DEG if j2 else None)

    def _check(self, kin, t, plan, j2=False):
        return sk.check_plan(kin, t, plan, self.margin_deg, BOWL_J2_MIN_DEG if j2 else None)

    def plan_bowl(self, t, T):
        return self._plan(self.kl, t, T @ np.linalg.inv(self.holder_to_link), ss.SEED_Q[ss.LEFT_ARM], j2=True)

    def choose_spin(self, t, T):
        """Constant spin SPIN_TARGET + delta, the first delta whose bowl-arm plan passes check_plan; if none passes,
        the one with the largest joint margin (flagged)."""
        tried = []
        for d in SPIN_DELTAS:
            Tn = mo.spin_to(T, SPIN_TARGET + d)
            p = self.plan_bowl(t, Tn)
            ok, m = self._check(self.kl, t, p, j2=True)
            tried.append((d, ok, m, Tn, p))
            if ok:
                return d, True, Tn, p, tried
        d, ok, m, Tn, p = max(tried, key=lambda x: x[2]['min_limit_margin_deg'])
        return d, False, Tn, p, tried

    def plan(self, t, T, WR, flat=None, bowl='hold', spin=True):
        """Both arms' plans for holder poses T and right-wrist poses WR. Returns (plans, bowl poses, info).
        flat: None = computed from the own tilt (scooping_motion.FLAT_TILT_DEG)."""
        X = np.linalg.inv(T) @ WR                              # right wrist in the holder frame
        inside = mo.dish_height_mm(T, WR @ self.spoon_tool) < 0
        own_tilt = mo.own_tilt_deg(T, inside)
        flat = own_tilt < mo.FLAT_TILT_DEG if flat is None else flat
        info = dict(own_tilt_deg=own_tilt, flat=bool(flat), bowl=bowl, spin=spin, tipback=False, t_tilt_s=0.0)
        seed_r = ss.SEED_Q[ss.RIGHT_ARM]
        if bowl == 'hold':
            Xs = mo.smooth_relative(t, X)
            Tb, info['t_tilt_s'] = mo.smooth_bowl(t, T, inside, flat)
            rplan = self._plan(self.kr, t, Tb @ Xs, seed_r)
            if sk.arm_metrics(self.kr, t, rplan)['failing_frames'] and not flat:      # spoon arm out of reach
                info['tipback'] = True
                Tb, info['t_tilt_s'] = mo.smooth_bowl(t, T, inside, False, tipback=True)
                rplan = self._plan(self.kr, t, Tb @ Xs, seed_r)
            dX = np.linalg.inv(X) @ Xs
            info['spoon_in_bowl_change_max_mm'] = float(1000 * np.linalg.norm(dX[:, :3, 3], axis=1).max())
        elif bowl == 'demo':
            Tb = T
            rplan = self._plan(self.kr, t, WR, seed_r)
        else:
            raise ValueError(f'bowl must be demo or hold, not {bowl}')
        if spin:
            d, ok, Tb, lplan, tried = self.choose_spin(t, Tb)
            info.update(spin_delta_deg=d, spin_ok=ok, spin_tried=' '.join(
                f"{x[0]:+d}:{'ok' if x[1] else 'x'}({x[2]['min_limit_margin_deg']:.1f})" for x in tried))
        else:
            lplan = self.plan_bowl(t, Tb)
        okl, ml = self._check(self.kl, t, lplan, j2=True)
        okr, mr = self._check(self.kr, t, rplan)
        info.update(plan_ok=bool(okl and okr), left=ml, right=mr, inside=inside)    # inside: array, not for JSON
        return [lplan, rplan], Tb, info

    # ---------------------------------------------------------------- outputs ---
    @staticmethod
    def sample(t, plans):
        """Rows at DT_OUT from 0 to the end: (tt, left q (n, 7), right q (n, 7))."""
        tt = np.arange(0, t[-1] + 1e-9, DT_OUT)
        return tt, plans[0]['spline'](tt), plans[1]['spline'](tt)

    def metrics(self, t, plans, inside=None):
        """Plan metrics (as 2026-11-06 replan_bowl.evaluate): IK, margins, speeds, bowl tilt / spin, clearance.
        inside: (N,) dish below the rim, for the tilt during the hold."""
        tt, ql, qr = self.sample(t, plans)
        Tb = np.array([self.kl.fk(q)[0] @ self.holder_to_link for q in ql])
        S = [self.kr.fk(q)[0] @ self.spoon_tool for q in qr[::5]]
        gap = np.array([self.geom.gap(B, s) for B, s in zip(Tb[::5], S)]) * 1000
        ml_, mr_ = (sk.arm_metrics(k, t, p) for k, p in zip((self.kl, self.kr), plans))
        marg = lambda k, q: np.degrees(np.minimum(q - k.low, k.high - q).min(0))  # noqa: E731
        ml, mr = marg(self.kl, ql), marg(self.kr, qr)
        vel = lambda q: np.abs(np.diff(q, axis=0)).max() / DT_OUT  # noqa: E731
        sp, tl = mo.spin_deg(Tb), mo.tilt_deg(Tb)
        out = dict(
            ik_ok=ml_['failing_frames'] == 0 and mr_['failing_frames'] == 0,
            ik_raw_err_mm_left=round(ml_['max_pos_err_mm'], 2), ik_raw_err_mm_right=round(mr_['max_pos_err_mm'], 2),
            max_raw_step_deg_left=round(ml_['max_raw_step_deg'], 1), max_raw_step_deg_right=round(mr_['max_raw_step_deg'], 1),
            min_margin_left_deg=f'{ml.min():.1f} (L{ml.argmin() + 1})', min_margin_right_deg=f'{mr.min():.1f} (R{mr.argmin() + 1})',
            left_j2_min_deg=round(float(np.degrees(ql[:, 1].min())), 1),
            peak_speed_left_rad_s=round(vel(ql), 2), peak_speed_right_rad_s=round(vel(qr), 2),
            tilt_start_end_deg=f'{tl[0]:.1f}/{tl[-1]:.1f}',
            spin_mean_range_deg=f'{mo.wrap(sp.mean()):.1f}/{np.ptp(sp):.1f}',
            xb_from_world_y_deg=round(float(np.abs(mo.wrap(sp - SPIN_TARGET)).max()), 1),
            min_clearance_mm=round(float(gap.min()), 1), duration_s=round(float(tt[-1]), 3), samples=int(len(tt)))
        if inside is not None:
            ins = np.interp(tt, t, inside.astype(float)) > .5
            out['tilt_start_hold_end_deg'] = f'{tl[0]:.1f}/{tl[ins].mean():.1f}/{tl[-1]:.1f}'
        return out
