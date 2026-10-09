"""Explainer figures for scripts/scooping_metrics.py (reporting only; no simulation).

Every measured value is computed by scooping_metrics itself, on two saved diagnostic runs (read-only):
  ON  = 2026-10-02 food_contact_models/runs/v9_sph3                     (SPH 3 mm, pressure push on: food hovers)
  OFF = 2026-10-06 settle_tilt_sph_push/runs/v9_sph3_nopush_direct216    (SPH 3 mm, push off: food touches)
Both replay the 34 deg wall scoop (data/trajectory/gen_trajs/gen_scoop_traj_34deg).

  python docs/scooping_metrics/make_figures.py      # writes 01-06_*.png next to this file
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import EllipseCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import scooping_metrics as sm  # noqa: E402

RUNS = {'ON': ROOT / 'data/diagnostics/2026-10-02/food_contact_models/runs/v9_sph3',
        'OFF': ROOT / 'data/diagnostics/2026-10-06/settle_tilt_sph_push/runs/v9_sph3_nopush_direct216'}
LABEL = {'ON': 'push on (food hovers)', 'OFF': 'push off (food touches)'}
R_P = .0015                                        # particle radius (m)
BLUE, ORANGE, AQUA, YELLOW, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#4a3aa7'
INK, MUTED, SHELL, GRID = '#0b0b0b', '#52514e', '#6b6a66', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'lines.linewidth': 2})
G = sm.ScoopGeometry()
MID = int(np.argmin(abs(G.gy)))                    # centre-line row of the dish grid (y_S = 0)
EXT = [G.gx[0] * 1e3, G.gx[-1] * 1e3, G.gy[0] * 1e3, G.gy[-1] * 1e3]


def load(run):
    d = RUNS[run]
    with (d / 'metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    m = {k: np.array([np.nan if r[k] in ('', 'nan') else float(r[k]) for r in rows]) for k in rows[0] if k != 'phase'}
    return np.load(d / 'snapshots.npz'), np.load(d / 'settled_particles.npz'), m, json.loads((d / 'summary.json').read_text())


def save(fig, name):
    fig.savefig(HERE / name, dpi=150); plt.close(fig); print(HERE / name)


def shell(ax, lw=0):
    ok = np.isfinite(G.top[:, MID])
    ax.fill_between(G.gx[ok] * 1e3, G.bot[ok, MID] * 1e3, G.top[ok, MID] * 1e3, color=SHELL, lw=lw, zorder=2)


def circles(ax, xy_mm, color, alpha=.9, z=3):
    if len(xy_mm):
        ax.add_collection(EllipseCollection(np.full(len(xy_mm), 2 * R_P * 1e3), np.full(len(xy_mm), 2 * R_P * 1e3),
                                            np.zeros(len(xy_mm)), units='xy', offsets=xy_mm, offset_transform=ax.transData,
                                            facecolors=color, edgecolors='k', linewidths=.25, alpha=alpha, zorder=z))


# ---------------------------------------------------------------------------------------------------------------
# 01  ScoopGeometry.__init__: the dish maps every food metric uses
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.15, 1]))
im = ax[0].imshow((G.top * 1e3).T, origin='lower', extent=EXT, cmap='Blues_r', aspect='equal')
fig.colorbar(im, ax=ax[0], shrink=.8, label='dish face height  top(x, y)  (mm, Z_S)')
X, Y = np.meshgrid(G.gx * 1e3, G.gy * 1e3, indexing='ij')
ax[0].contour(X, Y, G.inner.astype(float), levels=[.5], colors=[ORANGE], linewidths=2)
ax[0].contour(X, Y, G.hollow.astype(float), levels=[.5], colors=[INK], linewidths=1.5, linestyles='--')
ax[0].plot(sm.DISH_CENTRE[0] * 1e3, sm.DISH_CENTRE[1] * 1e3, 'x', color=INK, ms=8, mew=2)
ax[0].axhline(0, color=MUTED, lw=.8, ls=':')
ax[0].legend(handles=[Line2D([], [], color=ORANGE, label='inner: dish footprint, 2 mm in from the edge'),
                      Line2D([], [], color=INK, ls='--', lw=1.5, label='hollow: face below the lowest edge point'),
                      Line2D([], [], color=INK, marker='x', ls='', label='DISH_CENTRE'),
                      Line2D([], [], color=MUTED, ls=':', lw=.8, label='centre line (right panel)')],
             loc='upper left', fontsize=7.5, frameon=True, framealpha=.9)
ax[0].set(xlabel='X_S (mm)   handle <-  -> tip', ylabel='Y_S (mm)', title='Top view of the dish, 1 mm grid (spoon frame)')
ax[0].grid(False)
shell(ax[1])
ok = np.isfinite(G.top[:, MID])
edge = np.isfinite(G.top) & np.isfinite(G.bot); edge = edge & ~__import__('scipy.ndimage', fromlist=['x']).binary_erosion(edge)
ax[1].axhline(G.top[edge].min() * 1e3, color=INK, ls='--', lw=1.2)
ax[1].text(137, G.top[edge].min() * 1e3 + .4, 'lowest edge point: cells below it = hollow', fontsize=8, color=INK)
hol = G.hollow[:, MID]
ax[1].plot(G.gx[hol] * 1e3, G.top[hol, MID] * 1e3, color=INK, lw=4, solid_capstyle='butt', zorder=3)
ax[1].annotate('top(x, y): ray cast down (-Z_S)', (150, G.top[np.argmin(abs(G.gx - .150)), MID] * 1e3), (141, 2),
               arrowprops=dict(arrowstyle='->', color=MUTED), fontsize=8)
ax[1].annotate('bot(x, y): ray cast up (+Z_S)', (160, G.bot[np.argmin(abs(G.gx - .160)), MID] * 1e3), (140, -16),
               arrowprops=dict(arrowstyle='->', color=MUTED), fontsize=8)
ax[1].set(xlim=(135, 208), ylim=(-18, 6), aspect='equal', xlabel='X_S (mm)', ylabel='Z_S (mm)',
          title='Centre-line section (Y_S = 0): face top, underside bot')
ax[1].legend(handles=[Patch(color=SHELL, label='shell between bot and top'),
                      Line2D([], [], color=INK, lw=4, label='hollow cells on this line')], loc='lower right', fontsize=7.5)
fig.suptitle('01  ScoopGeometry.__init__ - precomputed dish maps (from the spoon collision mesh)', x=.01, ha='left', color=INK)
save(fig, '01_dish_maps.png')

# ---------------------------------------------------------------------------------------------------------------
# 02  dish_state: how each particle is classified against the spoon
f, _, _, s_off = load('OFF')
T_SUB = '7.600'                                    # dish level and submerged in the pool
P, R = f['P_' + T_SUB].astype(float), f['R_' + T_SUB]
st = G.dish_state(P, R)
L = st['L']
sl = abs(L[:, 1]) < .003
cls = [('below', st['below'], VIOLET, 'below: under the underside (here: pool under the dish)'),
       ('embedded', st['embedded'], ORANGE, 'embedded: inside the shell (> 0.5 mm under the face)'),
       ('on', st['on'] & ~st['embedded'] & ~st['below'], BLUE, 'on: footprint and -2 mm <= dz <= 20 mm'),
       ('above', st['above'] & ~st['on'], AQUA, 'above, but > 20 mm: not on the dish'),
       ('off', ~(st['on'] | st['above'] | st['embedded'] | st['below']), '#b9b8b3', 'outside the footprint (beyond the rim)')]
fig, ax = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
a = ax[0]
gx = G.gx[ok] * 1e3
a.fill_between(gx, (G.top[ok, MID] - .002) * 1e3, (G.top[ok, MID] + .020) * 1e3, color=BLUE, alpha=.08, lw=0, zorder=1)
a.plot(gx, (G.top[ok, MID] - .0005) * 1e3, color=ORANGE, lw=1, ls='--', zorder=4)
shell(a)
for _, m_, c, _ in cls:
    circles(a, L[sl & m_][:, [0, 2]] * 1e3, c)
a.set(xlim=(130, 212), ylim=(-30, 30), aspect='equal', xlabel='X_S (mm)', ylabel='Z_S (mm)',
      title=f'OFF run at t = {float(T_SUB):.1f} s (dish submerged), slice |Y_S| < 3 mm')
a.legend(handles=[Patch(color=c, label=f'{lab}  [{int((sl & m_).sum())} in slice]') for _, m_, c, lab in cls] +
         [Patch(color=BLUE, alpha=.18, label='"on" window: face -2 mm ... +20 mm'),
          Line2D([], [], color=ORANGE, ls='--', lw=1, label='face - 0.5 mm: "above" / "embedded" threshold')],
         loc='upper left', fontsize=7, framealpha=.95, ncol=1)
# zooms: one bottom-layer particle per run, at the end of the hold
for a, run in zip(ax[1:], ['ON', 'OFF']):
    f2, _, _, _ = load(run)
    P2, R2 = f2['final_particles'], f2['final_spoon']
    st2 = G.dish_state(P2, R2); L2 = st2['L']; sl2 = abs(L2[:, 1]) < .003
    cand = np.where(sl2 & st2['on'] & (abs(L2[:, 0] - .170) < .004))[0]
    k = cand[np.argmin(st2['dz'][cand])]
    x0, z0 = L2[k, 0] * 1e3, L2[k, 2] * 1e3
    face = G.top[st2['ix'][k], st2['iy'][k]] * 1e3
    circles(a, L2[sl2 & st2['on']][:, [0, 2]] * 1e3, BLUE if run == 'ON' else ORANGE, alpha=.35, z=3)
    shell(a); a.plot(gx, G.top[ok, MID] * 1e3, color=INK, lw=1, zorder=4)
    a.add_patch(plt.Circle((x0, z0), R_P * 1e3, fill=False, ec=INK, lw=2, zorder=6))
    a.plot(x0, z0, 'o', color=INK, ms=3, zorder=7)
    a.annotate('', (x0 + 2.4, face), (x0 + 2.4, z0), arrowprops=dict(arrowstyle='<->', color=INK, lw=1.3, shrinkA=0, shrinkB=0), zorder=7)
    a.text(x0 + 2.9, max(z0, face) + .3, f'dz = {st2["dz"][k] * 1e3:.2f} mm\ngap = dz - r = {(st2["dz"][k] - R_P) * 1e3:+.2f} mm',
           fontsize=8.5, zorder=7, bbox=dict(fc='white', ec='none', alpha=.85))
    a.set(xlim=(x0 - 9, x0 + 9), ylim=(face - 3.5, face + 8), aspect='equal', xlabel='X_S (mm)',
          title=f'{run}: {LABEL[run]}, end of hold\nlowest particle near the dish centre')
ax[2].text(.02, .02, 'negative gap: the drawn sphere overlaps the face;\nSPH particles are points, r is nominal',
           transform=ax[2].transAxes, fontsize=7.5, color=MUTED)
fig.suptitle('02  dish_state(P, R) - one boolean per particle, from dz = particle centre height above the dish face '
             '(spoon frame)', x=.01, ha='left', color=INK)
save(fig, '02_dish_state.png')

# ---------------------------------------------------------------------------------------------------------------
# 03  gap_stats: lowest particle per 2x2 mm column -> air gap, food contact fraction (A), contact area (B)
from scipy.ndimage import binary_closing
import trimesh
hc = G.hollow[::2, ::2]                            # legacy hollow, per 2x2 mm column
ROW = MID // 2                                     # the column row containing Y_S = 0
DISH = np.isfinite(G.top) & np.isfinite(G.bot)
pts = trimesh.sample.sample_surface_even(G.meshes['spoon'], 200000, seed=0)[0] * 1e3   # spoon outline, top view
SPX, SPY = np.arange(115, 210, .5), np.arange(-23, 23, .5)
SPM = np.zeros((len(SPX), len(SPY)), bool)
ii, jj = np.rint((pts[:, 0] - SPX[0]) / .5).astype(int), np.rint((pts[:, 1] - SPY[0]) / .5).astype(int)
k_ = (ii >= 0) & (ii < len(SPX)) & (jj >= 0) & (jj < len(SPY)); SPM[ii[k_], jj[k_]] = True
SPM = binary_closing(SPM, iterations=2)
fig = plt.figure(figsize=(15, 10.2), constrained_layout=True)
gsp = fig.add_gridspec(2, 2, height_ratios=[1.05, 1])
for j, run in enumerate(['ON', 'OFF']):
    f, _, _, _ = load(run)
    st = G.dish_state(f['final_particles'], f['final_spoon']); gs_ = G.gap_stats(st, R_P); L = st['L']
    idx = np.where(st['on'])[0]                    # the group-by-minimum of gap_stats, keeping particle indices
    cx, cy, dz = st['ix'][idx] // 2, st['iy'][idx] // 2, st['dz'][idx]
    key = cx * 1000 + cy; o = np.lexsort((dz, key)); key, cx, cy, dz, idx = key[o], cx[o], cy[o], dz[o], idx[o]
    first = np.r_[True, key[1:] != key[:-1]]
    gap = (dz[first] - R_P) * 1e3; ccx, ccy, low = cx[first], cy[first], idx[first]
    grid = np.full(hc.shape, np.nan); grid[ccx, ccy] = gap
    touch = grid < 1.0
    # ---- side view of the centre column row
    a = fig.add_subplot(gsp[0, j])
    for c in range(hc.shape[0] + 1):
        a.axvline(G.gx[0] * 1e3 + 2 * c - .5, color=GRID, lw=.8, zorder=0)
    shell(a)
    in_row = st['on'] & (st['iy'] // 2 == ROW)
    circles(a, L[in_row][:, [0, 2]] * 1e3, BLUE if run == 'ON' else ORANGE, alpha=.3)
    for cyy, k, g in zip(ccy, low, gap):
        if cyy != ROW:
            continue
        x0, z0 = L[k, 0] * 1e3, L[k, 2] * 1e3; face = G.top[st['ix'][k], st['iy'][k]] * 1e3
        a.add_patch(plt.Circle((x0, z0), R_P * 1e3, fill=False, ec=INK, lw=1.4, zorder=5))
        a.plot([x0, x0], [face, z0 - R_P * 1e3], color=ORANGE if g < 1.0 else MUTED, lw=2.5, zorder=6, solid_capstyle='butt')
    a.set(xlim=(139, 207), ylim=(-13, 12), aspect='equal', xlabel='X_S (mm)   handle <-  -> tip', ylabel='Z_S (mm)',
          title=f'{run}: {LABEL[run]} - side view of the 2 mm-wide column row at Y_S ~ 0')
    if j == 0:
        a.legend(handles=[Line2D([], [], color=GRID, lw=1, label='2 mm column boundaries'),
                          Line2D([], [], marker='o', ls='', mfc='none', mec=INK, ms=9, label='lowest particle in each column'),
                          Line2D([], [], color=MUTED, lw=2.5, label='its gap >= 1 mm: floating'),
                          Line2D([], [], color=ORANGE, lw=2.5, label='its gap < 1 mm: resting on the face')],
                 loc='upper left', fontsize=7.5, framealpha=.95)
    # ---- top view of all columns
    b = fig.add_subplot(gsp[1, j])
    ext = [G.gx[0] * 1e3 - .5, G.gx[0] * 1e3 + 2 * hc.shape[0] - .5, G.gy[0] * 1e3 - .5, G.gy[0] * 1e3 + 2 * hc.shape[1] - .5]
    im = b.imshow(grid.T, origin='lower', extent=ext, cmap='Blues', vmin=-1.5, vmax=6, aspect='equal')
    XX, YY = np.meshgrid(G.gx[::2] * 1e3 + .5, G.gy[::2] * 1e3 + .5, indexing='ij')
    b.plot(XX[touch], YY[touch], 'o', ms=3.5, mfc=ORANGE, mec='k', mew=.3)
    b.contour(XX, YY, hc.astype(float), levels=[.5], colors=[MUTED], linewidths=1, linestyles=':')
    b.add_patch(Rectangle((ext[0], G.gy[2 * ROW] * 1e3 - .5), ext[1] - ext[0], 2, fill=False, ec=MUTED, lw=1, ls=':'))
    b.contour(*np.meshgrid(SPX, SPY, indexing='ij'), SPM.astype(float), levels=[.5], colors=[SHELL], linewidths=2.5)
    b.contour(*np.meshgrid(G.gx * 1e3, G.gy * 1e3, indexing='ij'), DISH.astype(float), levels=[.5], colors=[VIOLET], linewidths=1.2)
    n_food, n_touch = int(np.isfinite(grid).sum()), int(touch.sum())
    b.set(xlabel='X_S (mm)   handle <-  -> tip', ylabel='Y_S (mm)', xlim=(118, 209), ylim=(-23, 23),
          title=f'{run}: top view, {n_food} columns with food (coloured squares)\n'
                f'gap median = {gs_["gap_med_mm"]:.2f} mm;   A food_contact_fraction = {n_touch} dots / {n_food} = '
                f'{gs_["food_contact_fraction"] * 100:.0f}%\nB contact_area = {n_touch} x 4 mm2 = {gs_["contact_area_mm2"]:.0f} mm2'
                f'   (legacy hollow_contact {gs_["hollow_contact"] * 100:.0f}%)')
    b.grid(False)
    if j == 0:
        b.legend(handles=[Patch(color='#6fa8dc', label='coloured square: a column with food (colour = its gap)'),
                          Line2D([], [], marker='o', ls='', mfc=ORANGE, mec='k', label='dot: gap < 1 mm, food resting on the face'),
                          Line2D([], [], color=SHELL, lw=2.5, label='spoon outline (collision mesh)'),
                          Line2D([], [], color=VIOLET, lw=1.2, label='rim (edge of the dish)'),
                          Line2D([], [], color=MUTED, lw=1, ls=':', label='legacy hollow (only for hollow_contact)')],
                 loc='upper left', fontsize=7.2, framealpha=.95)
fig.colorbar(im, ax=fig.axes[-1], shrink=.8, label='gap per column (mm)')
fig.suptitle('03  gap_stats(st, r): (1) split the dish into 2x2 mm columns  (2) keep the lowest on-dish particle per column  '
             '(3) gap = its centre height - face - r\n(4) gap median / p10 over the columns with food   '
             '(A) food_contact_fraction = columns with gap < 1 mm / columns with food   (B) contact_area = those columns x 4 mm2',
             x=.01, ha='left', color=INK, fontsize=10)
save(fig, '03_gap_stats.png')

# ---------------------------------------------------------------------------------------------------------------
# 04  in_bowl, bowl_gap, low_rim_z: the pool at t = 0 in the bowl's downhill section
fig, ax = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True, sharey=True)
for a, run in zip(ax, ['ON', 'OFF']):
    _, sp, _, _ = load(run)
    Pw, B = sp['world'], sp['bowl']
    g = B[:3, :3].T @ np.array([0, 0, -1.]); d2 = np.array([g[0], g[2]]); d2 /= np.linalg.norm(d2)   # downhill, holder xz
    Lh, rad, h = G.bowl_local(Pw, B)
    s_ = (Lh[:, 0] - sm.BOWL_AXIS_X) * d2[0] + Lh[:, 2] * d2[1]; lat = -(Lh[:, 0] - sm.BOWL_AXIS_X) * d2[1] + Lh[:, 2] * d2[0]
    ib = G.in_bowl(Pw, B); sel = abs(lat) < .004
    ss = np.linspace(-.085, .085, 341)
    prof = G.height(np.c_[sm.BOWL_AXIS_X + ss * d2[0], ss * d2[1]])
    prof = np.where(prof > -.1, prof, np.nan)
    a.fill_between(ss * 1e3, prof * 1e3, -45, color=SHELL, alpha=.35, lw=0)
    a.plot(ss * 1e3, prof * 1e3, color=SHELL, lw=1.5)
    a.add_patch(Rectangle((-70, -30), 140, 65, fill=False, ec=MUTED, ls=':', lw=1.2))
    a.plot([-48, 48], [-21.5, -21.5], color=AQUA, lw=5, solid_capstyle='butt', zorder=4)
    circles(a, np.c_[s_[sel & ib], Lh[sel & ib, 1]] * 1e3, BLUE, alpha=.8)
    circles(a, np.c_[s_[sel & ~ib], Lh[sel & ~ib, 1]] * 1e3, '#b9b8b3', alpha=.8)
    # spill level: points of this section plane whose world z equals low_rim_z
    zr = G.low_rim_z(B)
    SS, YY = np.meshgrid(np.linspace(-.09, .09, 200), np.linspace(-.04, .05, 120))
    pts = np.stack([sm.BOWL_AXIS_X + SS * d2[0], YY, SS * d2[1]], -1)
    a.contour(SS * 1e3, YY * 1e3, (pts @ B[:3, :3].T + B[:3, 3])[..., 2], levels=[zr], colors=[ORANGE], linewidths=2)
    up = B[:3, :3].T @ np.array([0, 0, 1.]); up2 = np.array([up[0] * d2[0] + up[2] * d2[1], up[1]])
    a.annotate('', (60 + 14 * up2[0], 30 + 14 * up2[1]), (60, 30), arrowprops=dict(arrowstyle='->', color=INK, lw=1.5))
    a.text(63, 26, 'world up', fontsize=8)
    bg = G.bowl_gap(Pw[ib], B, R_P)
    top_mm = (Pw[ib, 2].max() - zr) * 1e3
    a.set(xlim=(-90, 90), ylim=(-45, 50), aspect='equal', xlabel='s: along the downhill direction in the holder frame (mm)  uphill <-  -> downhill',
          title=f'{run}: {LABEL[run]}\nfloor air gap median {bg["bowl_gap_med_mm"]:.2f} mm, floor contact {bg["bowl_floor_contact"] * 100:.0f}%, '
          f'pool top {top_mm:+.1f} mm vs spill level')
ax[0].set_ylabel('holder y = bowl axis (mm)')
ax[0].legend(handles=[Patch(color=BLUE, label='in_bowl: r < 70 mm, -30 < y < 35 mm (dotted box)'),
                      Patch(color='#b9b8b3', label='not in bowl (spilled)'),
                      Line2D([], [], color=AQUA, lw=5, label='bowl_gap region: flat floor, r < 48 mm'),
                      Line2D([], [], color=ORANGE, label='spill level = low_rim_z (world horizontal)'),
                      Patch(color=SHELL, alpha=.5, label='bowl surface h(x, z) (height map)')],
             loc='upper left', fontsize=7.5, framealpha=.95)
fig.suptitle('04  in_bowl / bowl_gap / low_rim_z - pool at t = 0, section through the bowl axis (|lateral| < 4 mm); '
             'pool above the orange line = above-rim bulge', x=.01, ha='left', color=INK)
save(fig, '04_pool_metrics.png')

# ---------------------------------------------------------------------------------------------------------------
# 05  Tracker.update: compares each particle with its own state one check (20 ms) earlier
fig, ax = plt.subplots(1, 2, figsize=(15, 5.6), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.25, 1]))
a = ax[0]
a.axhspan(-.5, 4, color=BLUE, alpha=.07, lw=0); a.axhspan(-1.05, -.5, color=ORANGE, alpha=.35, lw=0)
a.axhspan(-9.5, -1.05, color=VIOLET, alpha=.07, lw=0)
a.axhline(0, color=INK, lw=1); a.axhline(-.5, color=ORANGE, lw=1, ls='--')
a.text(5.45, 3.3, 'above  (dz > -0.5 mm)', fontsize=8, color=BLUE, ha='right')
a.text(5.45, -.95, 'embedded: inside the 1 mm shell', fontsize=8, color=ORANGE, ha='right')
a.text(5.45, -9.1, 'below the underside', fontsize=8, color=VIOLET, ha='right')
a.text(5.45, .15, 'dish face (dz = 0)', fontsize=7.5, color=INK, ha='right')
cases = [(1.0, 2.0, 1.2, 'stays on\nthe dish', 'no flag', MUTED),
         (2.1, .6, -.8, 'grazes into\nthe shell', 'spoon_through +\nembedded_ever\n(even if it comes back)', ORANGE),
         (3.2, .4, -4.5, 'passes\nthrough', 'spoon_through', ORANGE),
         (4.3, -6.5, -6.0, 'pool particle\nunder the spoon', 'no flag:\nwas never "above"', MUTED)]
for x, d0, d1, what, out, c in cases:
    a.plot(x - .22, d0, 'o', ms=16, mfc='white', mec=INK, mew=1.5, zorder=4)
    a.plot(x + .22, d1, 'o', ms=16, mfc=c, mec=INK, mew=1, zorder=4)
    a.annotate('', (x + .22, d1), (x - .22, d0), arrowprops=dict(arrowstyle='->', color=INK, lw=1.3, shrinkA=9, shrinkB=9), zorder=5)
    a.text(x, 4.4, what, ha='center', va='bottom', fontsize=8.5)
    a.text(x, -10.3, out, ha='center', va='top', fontsize=8, color=c if c != MUTED else INK, weight='bold' if c != MUTED else None)
a.set(xlim=(.4, 5.5), ylim=(-13, 10), xticks=[], ylabel='particle centre height above the dish face, dz (mm)  (not to scale)',
      title='Spoon rule, four example particles: open = previous check, filled = this check (20 ms later)')
a.grid(False)
a.text(.45, 9.6, 'flag = was "above" last check  AND  is now embedded or below.  Flags never reset.\n'
       'Bowl rule (bowl_through) is the same with the bowl surface: above it -> more than 1.5 mm under it.',
       fontsize=8, color=INK, va='top')
# real run: when did the counters change? (metrics.csv, every 20 ms)
f, _, m, s = load('OFF')
t = m['t']; b = ax[1]
b.fill_between(t, m['embedded_now'], step='mid', color=ORANGE, alpha=.35, lw=0, label='embedded_now: particles inside the shell at this check')
b.step(t, m['spoon_through_total'], where='mid', color=ORANGE, lw=2.5, label='spoon_through_total (cumulative)')
b.step(t, m['bowl_through_total'], where='mid', color=VIOLET, lw=2, ls='--', label='bowl_through_total (cumulative)')
k = int(np.argmax(m['spoon_through_total'] > 0))
P_end, R_end = f['final_particles'], f['final_spoon']
st_end = G.dish_state(P_end, R_end); pid = int(np.where(f['spoon_through'])[0][0])
b.annotate(f'flag fires at t = {t[k]:.2f} s:\nparticle {pid} was above at t = {t[k - 1]:.2f} s,\nembedded/below at t = {t[k]:.2f} s.\n'
           f'At the end it is {"ON the dish (retained)" if f["hold_ids"][pid] else ("on the dish" if st_end["on"][pid] else "off the dish")}:\n'
           'a graze, not a leak -> the count is an upper bound', (t[k], 1), (t[k] + 1.5, 2.6),
           arrowprops=dict(arrowstyle='->', color=INK), fontsize=8, bbox=dict(fc='white', ec=GRID))
b.set(xlim=(t[0], t[-1]), ylim=(-.2, 4.5), xlabel='trajectory time t (s)', ylabel='particles',
      title=f'OFF run, real counters every 20 ms\nembedded_ever = {s["embedded_ever"]}, spoon_through = {s["spoon_through"]}, '
            f'bowl_through = {s["bowl_through"]}')
b.legend(fontsize=7.5, loc='upper left')
fig.suptitle('05  Tracker.update(st, bowl) - leakage = a particle changing side between two consecutive checks, per particle ID',
             x=.01, ha='left', color=INK)
save(fig, '05_tracker.png')

# ---------------------------------------------------------------------------------------------------------------
# 06  ScoopEvaluator: pool, acquisition, same-ID retention and the success test, over time
f, _, m, s = load('ON')
t = m['t']; t_end, hold = 13.8, 2.0
fig, ax = plt.subplots(4, 1, figsize=(12, 9.5), sharex=True, constrained_layout=True,
                       gridspec_kw=dict(height_ratios=[2.2, 1, 1, 1]))
for a in ax:
    a.axvspan(t[0], 0, color=MUTED, alpha=.08, lw=0); a.axvspan(t_end + .5, t[-1], color=AQUA, alpha=.12, lw=0)
    a.axvline(0, color=INK, lw=1)
a = ax[0]
a.plot(t, m['on_dish'], color=MUTED, lw=4, alpha=.5, label='on_dish: any particle on the dish (under acquired here)')
a.plot(t, m['acquired'], color=BLUE, label='acquired = on the dish AND in the t=0 pool')
a.hlines(s['retained_same_ids'], t_end + .5, t[-1], color=ORANGE, lw=4,
         label=f'retained_same_ids = {s["retained_same_ids"]}: on the dish at EVERY check in the window')
a.set(ylabel='particles', title='Food on the dish (OFF/ON definitions identical; ON run shown)')
a2 = a.twinx(); a2.plot(t, m['in_bowl'], color=YELLOW, lw=1.2, ls='--'); a2.set_ylabel('in_bowl', color=YELLOW)
a2.spines['right'].set_visible(True); a2.grid(False)
a.text(-1.9, a.get_ylim()[1] * .85, 'settle\n(t < 0)', fontsize=8, color=MUTED)
a.text(.15, a.get_ylim()[1] * .85, f't = 0: pool = the {s["initial_in_bowl"]}\nparticle IDs in_bowl now', fontsize=8)
a.text(t_end + .6, a.get_ylim()[1] * .85, 'retention window\nt_end + 0.5 s ... end', fontsize=8, color=AQUA)
a.legend(handles=a.get_legend_handles_labels()[0] + [Line2D([], [], color=YELLOW, ls='--', lw=1.2, label='in_bowl (right axis)')],
         loc='center left', fontsize=7.5, framealpha=.95)
ax[1].plot(t, m['clearance_mm'], color=BLUE); ax[1].axhline(0, color=ORANGE, ls='--', lw=1.2)
ax[1].set(ylabel='clearance (mm)', ylim=(-2, 25), title=f'gap(): spoon-bowl clearance >= 0 during the trajectory  (min {s["min_clearance_mm"]:.2f} mm)')
ax[2].plot(t, m['contact_peak_N'], color=BLUE); ax[2].axhline(10, color=ORANGE, ls='--', lw=1.2)
ax[2].set(ylabel='contact (N)', ylim=(-1, 12), title=f'spoon_bowl_contact(): peak < 10 N  (peak {s["peak_rigid_contact_N"]:.1f} N);  '
          f'bowl moved < 2 mm  (max {s["max_bowl_translation_mm"]:.3f} mm)')
ax[3].plot(t, (m['spoon_lowest_z'] - m['bowl_highest_z']) * 1e3, color=BLUE); ax[3].axhline(0, color=ORANGE, ls='--', lw=1.2)
ax[3].set(ylabel='spoon lowest -\nbowl highest (mm)', xlabel='trajectory time t (s)',
          title=f'spoon_lowest_z - bowl_highest_z > 0 at the end  ({s["final_spoon_above_bowl_mm"]:.1f} mm)  ->  '
          f'success = {s["success"]}')
fig.suptitle('06  ScoopEvaluator.update / summary - the success test over one run (ON)', x=.01, ha='left', color=INK)
save(fig, '06_evaluator_timeline.png')
