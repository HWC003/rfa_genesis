"""Genesis scene for the bimanual scooping replay: robot constants, marker templates, scene build, bowl weld
and offscreen rendering.

Shared by oa_pure_scooping.py. The scene is the one oa_pure_scooping.py has always built (robot USD with the
holder on the left wrist, table, optional SPH food); the bowl weld, food particle size/viscosity and the
offscreen camera are options migrated from data/diagnostics/2026-10-06/settle_tilt_sph_push/tools/common.py.
"""
from pathlib import Path

import genesis as gs
import numpy as np
from pxr import Usd, UsdGeom

import scooping_kinematics as sk
from scooping_metrics import DISH_CENTRE, RIM_CENTRE

ASSETS_DIR = Path(__file__).resolve().parent.parent / 'assets'
TABLE_USD = ASSETS_DIR / 'props' / 'table.usd'
DT = 0.004
SUBSTEPS = 60
KP = np.array([230, 230, 190, 190, 30, 30, 30, 30, 30, 230, 230, 190, 190, 30, 30, 30])
KV = np.array([2.7, 2.7, 2.2, 2.2, 1.5, 1.5, 1.5, 0.2, 0.2, 2.7, 2.7, 2.2, 2.2, 1.5, 1.5, 1.5])
FORCE_LIMITS = np.array([40, 40, 27, 27, 7, 7, 7, 10, 10, 40, 40, 27, 27, 7, 7, 7])
# Finger angle measured on the handle block in oa_full_sim (scoop_full run). Finger-to-wrist
# contact is filtered as adjacent, so a closed (0.0) target would pass through the block.
HELD_FINGER_POS = 0.20
# IK seed: the replay-start joints oa_full_sim reached (scoop_full run, first held row).
SEED_Q = np.array([0.3784, -0.3537, 0.0445, 1.6384, -0.3253, 0.3495, -0.8398, HELD_FINGER_POS, HELD_FINGER_POS,
                   0.2241, 0.5404, -0.8492, 0.9817, 1.2600, 0.7853, -0.4997])
# Fixed-base robot order: left arm 0:7, left fingers 7:9, right arm 9:16.
LEFT_ARM, LEFT_FINGER, RIGHT_ARM = np.arange(7), np.arange(7, 9), np.arange(9, 16)
SPH_BOUNDS = ((0.1, -0.5, 0.0), (1.0, 0.5, 0.9))
FOOD_COLOR = (0.68, 0.62, 0.49)

HOLDER_PATH = sk.LEFT + '/bowl_holder'
SPOON_MARKER_COLUMNS = {'spoon_marker_1': 'Spoon_front', 'spoon_marker_2': 'Spoon_back',
                        'spoon_marker_3': 'Spoon_left', 'spoon_marker_4': 'Spoon_right'}
# The rims share the CSV's names but not its numbering: the capture walks the rim
# in the opposite direction, so pairing them 1:1 warps the constellation by 37 mm
# and the IK cannot resolve an orientation. Reversing the rims fits it to 2.2 mm.
BOWL_MARKER_COLUMNS = {'Bowl_center': 'Bowl_center', 'Bowl_rim_1': 'Bowl_rim_4',
                       'Bowl_rim_2': 'Bowl_rim_3', 'Bowl_rim_3': 'Bowl_rim_2', 'Bowl_rim_4': 'Bowl_rim_1'}


def robot_usd(show_markers=False):
    return ASSETS_DIR / 'openarm_bimanual' / ('openarm_spoon_bowl_w_markers.usd' if show_markers
                                              else 'openarm_spoon_bowl_w_o_markers.usd')


def marker_templates(usd):
    """Marker offsets straight from USD: the spoon is rigid with the right wrist, the holder with the left.

    Spoon site Xforms are identity -- the marker sphere nested under each carries the offset -- so the child
    prim is resolved, not the site. Returns spoon markers (right wrist frame), bowl markers (left wrist frame),
    bowl markers (holder frame), holder_to_link and the rim centre in the left wrist frame.
    """
    stage = Usd.Stage.Open(str(usd)); xforms = UsdGeom.XformCache()

    def local_points(link_path, prim_paths):
        link_inv = xforms.GetLocalToWorldTransform(stage.GetPrimAtPath(link_path)).GetInverse()
        return [np.array((xforms.GetLocalToWorldTransform(stage.GetPrimAtPath(p)) * link_inv).ExtractTranslation())
                for p in prim_paths]

    holder_to_link = np.array(xforms.GetLocalToWorldTransform(stage.GetPrimAtPath(HOLDER_PATH))
                              * xforms.GetLocalToWorldTransform(stage.GetPrimAtPath(sk.LEFT)).GetInverse()).T
    if list(BOWL_MARKER_COLUMNS.values()) + list(SPOON_MARKER_COLUMNS.values()) != sk.BOWL_NAMES + sk.SPOON_NAMES:
        raise ValueError("Marker order differs from the planner's CSV order")
    return dict(
        spoon=local_points(sk.RIGHT, [f'{sk.RIGHT}/sites/{m}/{m}' for m in SPOON_MARKER_COLUMNS]),
        bowl=local_points(sk.LEFT, [f'{HOLDER_PATH}/sites/{s}' for s in BOWL_MARKER_COLUMNS]),
        bowl_holder=local_points(HOLDER_PATH, [f'{HOLDER_PATH}/sites/{s}' for s in BOWL_MARKER_COLUMNS]),
        holder_to_link=holder_to_link,
        rim_center_local=holder_to_link[:3, :3] @ RIM_CENTRE + holder_to_link[:3, 3])


def build_scene(usd, q0, food=None, backend='gpu', headless=False, weld_bowl=False, camera=False):
    """Plane, robot (fixed base, gravity-compensated), table and optional SPH food; robot set to q0.

    food: None or dict(pos, size, particle_size, mu, vis_mode). weld_bowl welds the left wrist (bowl) to the
    world at q0 (stationary bowl, as the diagnostics' v9 runs). camera adds an offscreen camera for
    render_spoon / render_bowl. Returns a dict: scene, robot, food, left, right, plane, cam.
    """
    gs.init(backend=gs.gpu if backend == 'gpu' else gs.cpu, seed=0)
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(camera_pos=(1.49, 0.68, 0.96), camera_lookat=(0.35, 0.10, 0.33),
                                                enable_gui=not headless),
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS),
        vis_options=gs.options.VisOptions(visualize_sph_boundary=True),
        show_viewer=not headless,
        profiling_options=gs.options.ProfilingOptions(show_FPS=not headless),
        sph_options=gs.options.SPHOptions(particle_size=food['particle_size'] if food else 0.006,
                                          lower_bound=SPH_BOUNDS[0], upper_bound=SPH_BOUNDS[1]),
    )
    plane = scene.add_entity(gs.morphs.Plane(), name='Plane')
    robot = scene.add_entity(
        gs.morphs.USD(file=str(usd), fixed=True),
        # Cancel robot self-weight so the position controller tracks the plan more closely.
        # The 1 mm SDF cell is the standalone holder's in oa_full_sim; the concave bowl now lives on the robot.
        material=gs.materials.Rigid(sdf_cell_size=0.001, sdf_min_res=64, sdf_max_res=256, gravity_compensation=1.0),
        name='OpenArm')
    scene.add_entity(gs.morphs.USD(file=str(TABLE_USD), pos=(0.6, 0, 0), euler=(0, 0, 90), fixed=True), name='Table')
    liquid = None
    if food is not None:
        liquid = scene.add_entity(material=gs.materials.SPH.Liquid(mu=food['mu']),
                                  morph=gs.morphs.Box(pos=tuple(food['pos']), size=tuple(food['size'])),
                                  surface=gs.surfaces.Default(color=FOOD_COLOR, vis_mode=food.get('vis_mode', 'recon')))
    cam = scene.add_camera(res=(1280, 960), pos=(0.6, -0.3, 0.7), lookat=(0.35, -0.1, 0.45), fov=30,
                           near=0.005, far=5.0, GUI=False) if camera else None
    scene.build()
    robot.set_dofs_kp(KP); robot.set_dofs_kv(KV); robot.set_dofs_force_range(-FORCE_LIMITS, FORCE_LIMITS)
    # Start at the replay's first pose before any physics step: at zero joints the
    # attached holder would sit below the floor.
    robot.set_qpos(q0); robot.control_dofs_position(q0)
    left, right = robot.get_link(sk.LEFT), robot.get_link(sk.RIGHT)
    if weld_bowl:
        scene.sim.rigid_solver.add_weld_constraint(plane.base_link.idx, left.idx)
    return dict(scene=scene, robot=robot, food=liquid, left=left, right=right, plane=plane, cam=cam)


def bowl_downhill(B):
    """World downhill direction of a tilted bowl (holder pose B), projected onto the world XY plane."""
    g = B[:3, :3].T @ np.array([0, 0, -1.])
    dh = np.array([g[0], 0, g[2]]); dh /= max(np.linalg.norm(dh), 1e-9)
    d = B[:3, :3] @ dh; d[2] = 0
    return d / max(np.linalg.norm(d), 1e-9)


def _shoot(cam, target, eye, path):
    from PIL import Image
    cam.set_pose(pos=tuple(eye), lookat=tuple(target))
    Image.fromarray(np.asarray(cam.render(rgb=True)[0])).save(path)


def render_spoon(cam, R, path, dist=.17, elev=40.):
    """Oblique view of the dish from the spoon's -Y_S side, elev deg above horizontal, at the dish centre."""
    c = R[:3, :3] @ DISH_CENTRE + R[:3, 3]
    x = R[:3, 0] - R[2, 0] * np.array([0, 0, 1.]); x /= np.linalg.norm(x)
    side = -np.cross([0, 0, 1.], x); e = np.deg2rad(elev)
    _shoot(cam, c, c + dist * (np.cos(e) * side + np.sin(e) * np.array([0, 0, 1.])), path)


def render_bowl(cam, B, path, dist=.30, elev=55.):
    """Oblique view into the bowl from its downhill side, at the rim centre (B = holder pose)."""
    c = B[:3, :3] @ RIM_CENTRE + B[:3, 3]; e = np.deg2rad(elev)
    _shoot(cam, c, c + dist * (np.cos(e) * bowl_downhill(B) + np.sin(e) * np.array([0, 0, 1.])), path)
