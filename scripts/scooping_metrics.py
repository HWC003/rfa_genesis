"""Scooping/acquisition metrics: spoon and bowl geometry, food-on-spoon measures and the scoop success test.

Pure NumPy/trimesh on particle positions and link poses; nothing here steps the simulation.
Migrated from the diagnostics (definitions unchanged):
  - collision meshes, bowl height map, spoon vertices, clearance: 2026-09-29 tools/geometry.py
  - in_bowl, on_spoon_10mm, gap, spoon_lowest_z, bowl_highest_z: 2026-09-29 tools/wallscoop_sim.py::Measure
  - dish_state, gap_stats, bowl_gap, low_rim_z, Tracker: 2026-10-06 tools/common.py::Geo / Tracker
    (gap_stats adds food_contact_fraction and contact_area_mm2, 2026-10-07; hollow_contact kept as a legacy field)
  - ScoopEvaluator: the inline evaluation and success test of 2026-10-06 tools/v9_bench.py

Frames: spoon frame {S} = right wrist link (+X_S handle -> tip, +Z_S out of the concave face);
holder frame = bowl holder (bowl opening along +Y, bowl axis through x = 0.1549 m, z = 0).
Terminology follows data/diagnostics/CONVENTIONS.md (air gap, through-spoon leakage, above-rim bulge).
"""
from pathlib import Path

import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation

import scooping_kinematics as sk

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / 'assets/openarm_bimanual/openarm_spoon_bowl_w_o_markers.usd'
HOLDER = sk.LEFT + '/bowl_holder'
SURFACE_CACHE = ROOT / 'data/cache/bowl_surface_grid.npz'   # derived from ASSET; delete to recompute

DISH_CENTRE = np.array([.175, 0, -.009])      # dish centre in the right wrist frame {S} (m)
BOWL_AXIS_X = .1549                           # bowl axis x in the holder frame (axis along +Y, z = 0)
RIM_CENTRE = np.array([.1549, .0191, 0])      # bowl opening centre at rim height, holder frame
DISH_GX = np.arange(.135, .2071, .001)        # 1 mm dish grid along X_S
DISH_GY = np.arange(-.021, .0211, .001)       # and Y_S


# ---------------------------------------------------------------- geometry ----

def _transform(stage, xf, path, frame):
    return np.array(xf.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
                    * xf.GetLocalToWorldTransform(stage.GetPrimAtPath(frame)).GetInverse()).T


def load_meshes(asset=ASSET):
    """Collision meshes from USD: 'spoon' in the right wrist frame, 'bowl' (incl. holder) in the holder frame.
    Also returns holder_to_link (holder frame in the left wrist frame)."""
    stage = Usd.Stage.Open(str(asset)); xf = UsdGeom.XformCache()
    meshes = {'bowl': [], 'spoon': []}
    for p in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        name = str(p.GetPath())
        if not p.IsA(UsdGeom.Mesh) or not p.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if HOLDER in name:
            key, frame = 'bowl', HOLDER
        elif sk.RIGHT in name and 'spoon_sim' in name:
            key, frame = 'spoon', sk.RIGHT
        else:
            continue
        m = UsdGeom.Mesh(p)
        v = np.array(m.GetPointsAttr().Get()); counts = np.array(m.GetFaceVertexCountsAttr().Get())
        indices = np.array(m.GetFaceVertexIndicesAttr().Get()); faces = []; offset = 0
        for n in counts:
            poly = indices[offset:offset + n]; offset += n
            faces.extend([[poly[0], poly[j], poly[j + 1]] for j in range(1, n - 1)])
        T = _transform(stage, xf, name, frame)
        meshes[key].append(trimesh.Trimesh(v @ T[:3, :3].T + T[:3, 3], faces, process=True))
    return {k: trimesh.util.concatenate(v) for k, v in meshes.items()}, _transform(stage, xf, HOLDER, sk.LEFT)


def bowl_height_map(mesh, cache=SURFACE_CACHE):
    """Upper surface of the bowl mesh along the holder +Y axis, as h(x, z) on a 1 mm grid (-0.2 where no hit)."""
    if cache is not None and Path(cache).exists():
        a = np.load(cache); x, z, h = a['x'], a['z'], a['h']
    else:
        x = np.arange(mesh.bounds[0, 0] - .002, mesh.bounds[1, 0] + .002, .001)
        z = np.arange(mesh.bounds[0, 2] - .002, mesh.bounds[1, 2] + .002, .001)
        X, Z = np.meshgrid(x, z, indexing='ij')
        origins = np.c_[X.ravel(), np.full(X.size, .2), Z.ravel()]; dirs = np.tile([0, -1, 0.], (len(origins), 1))
        h = np.full(len(origins), -.2)
        for start in range(0, len(origins), 1000):
            loc, idx, _ = mesh.ray.intersects_location(origins[start:start + 1000], dirs[start:start + 1000], multiple_hits=False)
            if len(idx):
                h[start + idx] = loc[:, 1]
        h = h.reshape(X.shape)
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True); np.savez(cache, x=x, z=z, h=h)
    return RegularGridInterpolator((x, z), h, bounds_error=False, fill_value=-.2)


def sample_vertices(mesh, spacing=.0015):
    v = np.r_[mesh.vertices, mesh.triangles_center]
    _, idx = np.unique(np.round(v / spacing).astype(int), axis=0, return_index=True)
    return v[idx]


def clearance(rel, verts, height):
    """Smallest height of spoon vertices above the bowl's upper surface; rel = spoon wrist pose in the holder frame."""
    p = verts @ rel[:3, :3].T + rel[:3, 3]
    h = height(p[:, [0, 2]])
    return float(np.min(p[:, 1] - h)), p


def pose(link):
    """4x4 world pose of a Genesis link."""
    T = np.eye(4)
    T[:3, 3] = link.get_pos().detach().cpu().numpy()
    T[:3, :3] = Rotation.from_quat(link.get_quat().detach().cpu().numpy()[[1, 2, 3, 0]]).as_matrix()
    return T


def spoon_bowl_contact(robot, left, right):
    """(count, largest individual, resultant) rigid contact force between the two wrist links (N)."""
    contacts = robot.get_contacts()
    forces = [f if a == left.idx else -f
              for a, b, f in zip(contacts['link_a'].tolist(), contacts['link_b'].tolist(),
                                 contacts['force_a'].detach().cpu().numpy())
              if {a, b} == {left.idx, right.idx}]
    if not forces:
        return 0, 0.0, 0.0
    return len(forces), max(float(np.linalg.norm(f)) for f in forces), float(np.linalg.norm(np.sum(forces, axis=0)))


class ScoopGeometry:
    """Spoon dish and bowl measurements from the collision meshes (no physics)."""

    def __init__(self, asset=ASSET, surface_cache=SURFACE_CACHE):
        self.meshes, self.holder_to_link = load_meshes(asset)
        self.verts = sample_vertices(self.meshes['spoon'])
        self.height = bowl_height_map(self.meshes['bowl'], surface_cache)
        self.gx, self.gy = DISH_GX, DISH_GY
        X, Y = np.meshgrid(self.gx, self.gy, indexing='ij')
        spoon = self.meshes['spoon']
        top = np.full(X.size, np.nan); bot = np.full(X.size, np.nan)
        hit, ix, _ = spoon.ray.intersects_location(np.c_[X.ravel(), Y.ravel(), np.full(X.size, .1)],
                                                   np.tile([0, 0, -1.], (X.size, 1)), multiple_hits=False)
        top[ix] = hit[:, 2]
        hit, ix, _ = spoon.ray.intersects_location(np.c_[X.ravel(), Y.ravel(), np.full(X.size, -.1)],
                                                   np.tile([0, 0, 1.], (X.size, 1)), multiple_hits=False)
        bot[ix] = hit[:, 2]
        self.top, self.bot = top.reshape(X.shape), bot.reshape(X.shape)   # dish face / underside height maps
        ok = np.isfinite(self.top) & np.isfinite(self.bot)
        self.inner = binary_erosion(ok, iterations=2)                     # dish footprint, 2 mm in from the edge
        edge = ok & ~binary_erosion(ok)
        self.hollow = ok & (self.top < self.top[edge].min())             # below the lowest rim point when level
        bv = self.meshes['bowl'].vertices
        self.bowl_rim_verts = bv[(np.hypot(bv[:, 0] - BOWL_AXIS_X, bv[:, 2]) < .076) & (bv[:, 1] < .025)]
        self._ring = None

    # -- success-test geometry (2026-09-29 Measure) --
    def on_spoon_10mm(self, P, R):
        """Legacy 2026-09-29 count: centre over the dish face, 1 mm below to 10 mm above it."""
        L = (P - R[:3, 3]) @ R[:3, :3]
        ix = np.rint((L[:, 0] - self.gx[0]) / .001).astype(int); iy = np.rint((L[:, 1] - self.gy[0]) / .001).astype(int)
        ok = (ix >= 0) & (ix < len(self.gx)) & (iy >= 0) & (iy < len(self.gy))
        z = np.full(len(P), np.nan); z[ok] = self.top[ix[ok], iy[ok]]
        dz = L[:, 2] - z
        return np.isfinite(dz) & (dz >= -.001) & (dz <= .010)

    @staticmethod
    def in_bowl(P, Bb):
        """Centre inside the bowl volume (holder frame): r < 70 mm from the axis, -30 < y < 35 mm."""
        L = (P - Bb[:3, 3]) @ Bb[:3, :3]
        return (np.hypot(L[:, 0] - BOWL_AXIS_X, L[:, 2]) < .07) & (L[:, 1] > -.03) & (L[:, 1] < .035)

    def gap(self, Bb, R):
        """Spoon-bowl clearance (m): spoon vertices above the bowl's upper surface."""
        return clearance(np.linalg.inv(Bb) @ R, self.verts, self.height)[0]

    def spoon_lowest_z(self, R):
        sw = self.verts @ R[:3, :3].T + R[:3, 3]
        return float(sw[self.verts[:, 0] > .135, 2].min())

    def bowl_highest_z(self, Bb):
        return float((self.bowl_rim_verts @ Bb[:3, :3].T + Bb[:3, 3])[:, 2].max())

    # -- food on the dish (2026-10-06 Geo) --
    def spoon_local(self, P, R):
        L = (P - R[:3, 3]) @ R[:3, :3]
        ix = np.rint((L[:, 0] - self.gx[0]) / .001).astype(int); iy = np.rint((L[:, 1] - self.gy[0]) / .001).astype(int)
        inside = (ix >= 0) & (ix < len(self.gx)) & (iy >= 0) & (iy < len(self.gy))
        ix, iy = np.where(inside, ix, 0), np.where(inside, iy, 0)
        top = np.where(inside, self.top[ix, iy], np.nan); bot = np.where(inside, self.bot[ix, iy], np.nan)
        return L, ix, iy, inside & self.inner[ix, iy], top, bot

    def dish_state(self, P, R, window=.020):
        """Per particle: on the dish (centre within `window` above the face), and side of the spoon shell."""
        L, ix, iy, inner, top, bot = self.spoon_local(P, R)
        dz = L[:, 2] - top
        on = inner & np.isfinite(dz) & (dz >= -.002) & (dz <= window)
        above = inner & (dz > -.0005)
        embedded = inner & (L[:, 2] < top - .0005) & (L[:, 2] > bot)
        below = inner & (L[:, 2] <= bot)
        return dict(L=L, ix=ix, iy=iy, dz=dz, on=on, above=above, embedded=embedded, below=below)

    def gap_stats(self, st, r, contact_mm=1.0):
        """Bottom layer of the food on the dish, per 2x2 mm dish column (lowest on-dish particle in each column).

        gap_*_mm: lowest centre height above the face minus the radius, over columns that contain food.
        food_contact_fraction: share of those food columns whose gap < contact_mm, i.e. how much of the food's
            footprint rests on the face rather than floating above it.
        contact_area_mm2: those touching columns x 4 mm2, anywhere on the dish.
        hollow_contact (legacy, 2026-10-02): touching columns inside the level-spoon hollow / all hollow columns.
        """
        on = st['on']
        if on.sum() == 0:
            return dict(gap_med_mm=np.nan, gap_p10_mm=np.nan, gap_p90_mm=np.nan, centre_min_mm=np.nan,
                        food_contact_fraction=0., contact_area_mm2=0., hollow_contact=0.)
        cx, cy, dz = st['ix'][on] // 2, st['iy'][on] // 2, st['dz'][on]
        key = cx * 1000 + cy
        order = np.lexsort((dz, key)); key, dz, cx, cy = key[order], dz[order], cx[order], cy[order]
        first = np.r_[True, key[1:] != key[:-1]]
        g = (dz[first] - r) * 1e3
        hc = self.hollow[::2, ::2]
        touch = np.zeros_like(hc)
        sel = g < contact_mm
        touch[np.clip(cx[first][sel], 0, hc.shape[0] - 1), np.clip(cy[first][sel], 0, hc.shape[1] - 1)] = True
        return dict(gap_med_mm=float(np.median(g)), gap_p10_mm=float(np.percentile(g, 10)), gap_p90_mm=float(np.percentile(g, 90)),
                    centre_min_mm=float(dz.min() * 1e3), food_contact_fraction=float(sel.mean()),
                    contact_area_mm2=float(sel.sum() * 4.0), hollow_contact=float((touch & hc).sum() / max(hc.sum(), 1)))

    # -- pool in the bowl (2026-10-06 Geo) --
    def bowl_local(self, P, Bb):
        L = (P - Bb[:3, 3]) @ Bb[:3, :3]
        rad = np.hypot(L[:, 0] - BOWL_AXIS_X, L[:, 2])
        return L, rad, self.height(L[:, [0, 2]])

    def low_rim_z(self, Bb):
        """World z of the lowest point of the bowl lip (spill level)."""
        if self._ring is None:
            rr = np.arange(.055, .08, .0005); ring = []
            for th in np.linspace(0, 2 * np.pi, 360, endpoint=False):
                p = np.c_[BOWL_AXIS_X + rr * np.cos(th), rr * np.sin(th)]
                h = self.height(p); k = np.argmax(h)
                ring.append([p[k, 0], h[k], p[k, 1]])
            self._ring = np.array(ring)
        return float((self._ring @ Bb[:3, :3].T + Bb[:3, 3])[:, 2].min())

    def bowl_gap(self, P, Bb, r, floor_r=.048):
        """Pool bottom-layer air gap on the flat bowl floor (r < 48 mm from the axis): per 2x2 mm column, lowest
        centre above the inner surface minus the particle radius; and the fraction of floor columns in contact."""
        L, rad, h = self.bowl_local(P, Bb)
        dy = L[:, 1] - h
        sel = (rad < floor_r) & (h > -.1) & (dy > -.002) & (dy < .02)
        if not sel.any():
            return dict(bowl_gap_med_mm=np.nan, bowl_gap_p10_mm=np.nan, bowl_floor_contact=0.)
        key = np.rint(L[sel, 0] / .002).astype(int) * 1000 + np.rint(L[sel, 2] / .002).astype(int)
        order = np.lexsort((dy[sel], key)); key, d = key[order], dy[sel][order]
        g = (d[np.r_[True, key[1:] != key[:-1]]] - r) * 1e3
        n_cols = np.pi * floor_r ** 2 / .002 ** 2
        return dict(bowl_gap_med_mm=float(np.median(g)), bowl_gap_p10_mm=float(np.percentile(g, 10)),
                    bowl_floor_contact=float(np.sum(g < 1.) / n_cols))


class Tracker:
    """Pass-through detection between consecutive checks (distinct particle IDs, cumulative).

    Spoon: a particle above the dish face that is next seen inside the shell (>0.5 mm under the face) or under it,
    within the dish footprint both times -- this includes grazes into the shell, so treat it as an upper bound.
    Bowl: a particle above the bowl's inner surface that is next seen >1.5 mm under it, inside r < 64 mm.
    """

    def __init__(self, n):
        self.spoon_through = np.zeros(n, bool); self.spoon_embedded_ever = np.zeros(n, bool)
        self.bowl_through = np.zeros(n, bool); self.prev = None

    def update(self, st, bowl):
        L, rad, h = bowl
        valid = (h > -.1) & (rad < .064)
        b_above = valid & (L[:, 1] > h - .0005)
        b_under = valid & (L[:, 1] < h - .0015)
        self.spoon_embedded_ever |= st['embedded']
        if self.prev is not None:
            s_above, pb_above = self.prev
            self.spoon_through |= s_above & (st['embedded'] | st['below'])
            self.bowl_through |= pb_above & b_under
        self.prev = (st['above'], b_above)


class ScoopEvaluator:
    """Per-check metrics and the scoop success test (2026-09-29 criteria, 2026-10-06 measures).

    Call update() every check (the diagnostics used every 5 steps = 20 ms), with t on the trajectory clock
    (t < 0 while the food settles, t = 0 at the first trajectory sample). The pool is the set of particles
    in the bowl at the first check with t >= 0. Success: run completed; the same pool particle IDs stay on
    the dish (centre <= 20 mm above the face) from t_end + 0.5 s to the end; the spoon's lowest point ends
    above the bowl's highest point; clearance >= 0 and bowl movement < 2 mm during the trajectory; peak
    spoon-bowl contact < 10 N.
    """

    def __init__(self, geom, n_particles, radius, t_end, hold, submerged_window=None):
        self.g, self.r, self.t_end, self.hold, self.sub = geom, radius, t_end, hold, submerged_window
        self.trk = Tracker(n_particles); self.n = n_particles
        self.pool = self.hold_ids = self.hold_ids10 = self.B_ref = None
        self.rows = []; self.peak_contact = 0.; self.last = None

    def contact(self, peak):
        """Record a per-step spoon-bowl contact peak (N); call every physics step."""
        self.peak_contact = max(self.peak_contact, peak)

    def update(self, t, P, vel, R, B, contact_peak=0.):
        g = self.g
        if self.B_ref is None:
            self.B_ref = B.copy()
        st = g.dish_state(P, R); self.trk.update(st, g.bowl_local(P, B))
        on, on10 = st['on'], g.on_spoon_10mm(P, R)
        ib = g.in_bowl(P, B)
        if t >= -1e-8 and self.pool is None:
            self.pool = ib.copy()
        acq = on & self.pool if self.pool is not None else on & False
        acq10 = on10 & self.pool if self.pool is not None else on10 & False
        if t >= self.t_end + .5:
            self.hold_ids = acq.copy() if self.hold_ids is None else self.hold_ids & acq
            self.hold_ids10 = acq10.copy() if self.hold_ids10 is None else self.hold_ids10 & acq10
        row = dict(t=round(t, 3), on_dish=int(on.sum()), acquired=int(acq.sum()), acquired_10mm=int(acq10.sum()),
                   **g.gap_stats(st, self.r), embedded_now=int(st['embedded'].sum()),
                   spoon_through_total=int(self.trk.spoon_through.sum()), bowl_through_total=int(self.trk.bowl_through.sum()),
                   in_bowl=int(ib.sum()), pool_top_z=float(P[ib, 2].max()) if ib.any() else np.nan,
                   pool_top_above_low_rim_mm=float((P[ib, 2].max() - g.low_rim_z(B)) * 1e3) if ib.any() else np.nan,
                   **g.bowl_gap(P[ib], B, self.r), max_speed_m_s=float(np.linalg.norm(vel, axis=1).max()),
                   contact_peak_N=float(contact_peak), clearance_mm=g.gap(B, R) * 1e3,
                   bowl_translation_mm=float(np.linalg.norm(B[:3, 3] - self.B_ref[:3, 3]) * 1e3),
                   face_up_cos=float(R[2, 2]), spoon_lowest_z=g.spoon_lowest_z(R), bowl_highest_z=g.bowl_highest_z(B))
        self.rows.append(row); self.last = (P, R, B)
        return row

    def summary(self, particle_volume_mL=None):
        rows = self.rows
        rr = [x for x in rows if x['t'] >= 0]
        sub = [x for x in rows if self.sub and self.sub[0] <= x['t'] <= self.sub[1]]
        start = next((x for x in rows if x['t'] >= 0), None)
        last = rows[-1]
        s = dict(completed=bool(last['t'] >= self.t_end + self.hold - .01), n_particles=self.n, radius_mm=self.r * 1e3,
                 initial_in_bowl=int(self.pool.sum()) if self.pool is not None else 0,
                 settled_pool_top_z=start['pool_top_z'] if start else None,
                 start_pool_top_above_low_rim_mm=start['pool_top_above_low_rim_mm'] if start else None,
                 start_bowl_gap_med_mm=start['bowl_gap_med_mm'] if start else None,
                 start_bowl_floor_contact=start['bowl_floor_contact'] if start else None,
                 submerged_window_s=list(self.sub) if self.sub else None,
                 submerged_food_contact_fraction=float(np.median([x['food_contact_fraction'] for x in sub])) if sub else None,
                 submerged_contact_area_mm2=float(np.median([x['contact_area_mm2'] for x in sub])) if sub else None,
                 submerged_hollow_contact=float(np.median([x['hollow_contact'] for x in sub])) if sub else None,
                 submerged_gap_med_mm=float(np.nanmedian([x['gap_med_mm'] for x in sub])) if sub else None,
                 max_acquired=max((x['acquired'] for x in rr), default=0),
                 retained_same_ids=int(self.hold_ids.sum()) if self.hold_ids is not None else 0,
                 retained_same_ids_10mm=int(self.hold_ids10.sum()) if self.hold_ids10 is not None else 0,
                 final_gap_med_mm=last['gap_med_mm'], final_gap_p10_mm=last['gap_p10_mm'],
                 final_centre_min_mm=last['centre_min_mm'], final_food_contact_fraction=last['food_contact_fraction'],
                 final_contact_area_mm2=last['contact_area_mm2'], final_hollow_contact=last['hollow_contact'],
                 embedded_max=max(x['embedded_now'] for x in rows), embedded_ever=int(self.trk.spoon_embedded_ever.sum()),
                 spoon_through=int(self.trk.spoon_through.sum()), bowl_through=int(self.trk.bowl_through.sum()),
                 final_in_bowl=last['in_bowl'], peak_rigid_contact_N=float(self.peak_contact),
                 min_clearance_mm=min((x['clearance_mm'] for x in rr), default=np.nan),
                 max_bowl_translation_mm=max((x['bowl_translation_mm'] for x in rr), default=np.nan),
                 max_speed_m_s=max(x['max_speed_m_s'] for x in rows),
                 final_spoon_above_bowl_mm=(last['spoon_lowest_z'] - last['bowl_highest_z']) * 1e3)
        if particle_volume_mL is not None:
            s['particle_volume_mL'] = particle_volume_mL
            s['retained_volume_mL'] = s['retained_same_ids'] * particle_volume_mL
        s['success'] = bool(s['completed'] and s['retained_same_ids'] > 0 and s['final_spoon_above_bowl_mm'] > 0
                            and s['min_clearance_mm'] >= 0 and s['max_bowl_translation_mm'] < 2
                            and s['peak_rigid_contact_N'] < 10)
        return s
