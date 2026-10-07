"""Genesis pick-and-place: raise, approach, grasp, lift, carry and place a liquid-filled holder.

Steps 1-5 are those of oa_full_sim.py (same robot USD, holder, placement and gains);
the scooping replay is replaced by a place-and-release. Uses IK and joint
interpolation, without RRT: physical collisions apply, but the motion does not plan
around obstacles.
"""

import argparse
import time
from pathlib import Path

import genesis as gs
import numpy as np
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--headless', action='store_true', help='Disable viewer/debug drawing and exit after playback')
parser.add_argument('--no-liquid', action='store_true', help='Rigid-only comparison; changes physics')
parser.add_argument('--backend', choices=['gpu', 'cpu'], default='gpu')
parser.add_argument('--place-offset', type=float, nargs=2, default=[0.0, -0.10], metavar=('DX', 'DY'),
                    help='Place location relative to the pick location, in world X/Y (m)')
parser.add_argument('--carry-height', type=float, default=0.10,
                    help='Lift above the table during the carry (m), as in oa_full_sim step 5')
args = parser.parse_args()
phase_timings = []
def report_phase(name, started):
    elapsed = time.perf_counter() - started
    phase_timings.append((name, elapsed))
    print(f'[Timing] {name}: {elapsed:.3f} s', flush=True)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
# Holder is a separate free body here; the robot USD carries only the spoon.
OPENARM_USD = ASSETS_DIR / "openarm_bimanual" / "openarm_spoon_w_o_markers.usd"
BOWL_USD = ASSETS_DIR / "props" / "Bowl_holder_w_o_marker.usd"

# Place the handle near the table edge for finger clearance; the bowl stays over the table.
# World placement. Dimensions below are measured in the unmarked holder STL frame (m).
TABLE_TOP_Z = 0.30
BOWL_CENTER_XY = np.array([0.43, 0.30]) # Circular opening center, in WORLD X/Y.
BOWL_AXIS_LOCAL_X = 0.1549              # Circular bowl's central axis along local X.
HOLDER_BOTTOM_LOCAL_Y = -0.0279         # Underside; local Y becomes world height.
BOWL_RIM_LOCAL_Y = 0.0191
BLOCK_MID_LOCAL_Y = 0.0077              # Midpoint of block's local Y bounds.
BOWL_POS = (float(BOWL_CENTER_XY[0] - BOWL_AXIS_LOCAL_X),
            float(BOWL_CENTER_XY[1]), TABLE_TOP_Z - HOLDER_BOTTOM_LOCAL_Y)
BOWL_EULER = (90, 0, 0)                 # Opening faces up: local +Y -> world +Z.
GRASP_HEIGHT_OFFSET = 0.025             # Jaws extend ~27 mm below the reference point.
# Grasp 20 mm from the free end, 25 mm above mid-height, on the width centerline.
BLOCK_GRASP_LOCAL = np.array([0.020, BLOCK_MID_LOCAL_Y + GRASP_HEIGHT_OFFSET, 0.0])
BOWL_RIM_Z = BOWL_POS[2] + BOWL_RIM_LOCAL_Y
LIQUID_SIZE = (0.06, 0.06, 0.06)
LIQUID_POS = (BOWL_CENTER_XY[0], BOWL_CENTER_XY[1], BOWL_RIM_Z + 0.03 + LIQUID_SIZE[2] / 2)
PLACE_OFFSET = np.array([*args.place_offset, 0.0])
PLACE_CLEARANCE = 0.005                 # Release this far above the resting height; the holder drops onto the table.

gs.init(backend=gs.gpu if args.backend == "gpu" else gs.cpu, seed=0)
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(1.49, 0.68, 0.96), camera_lookat=(0.35, 0.10, 0.33),
        enable_gui=not args.headless,
    ),
    sim_options=gs.options.SimOptions(dt=0.004, substeps=60),
    sph_options=gs.options.SPHOptions(
        particle_size=0.006, lower_bound=(0.1, -0.5, 0.0), upper_bound=(1.0, 0.5, 0.9),
    ),
    vis_options=gs.options.VisOptions(visualize_sph_boundary=True),
    show_viewer=not args.headless,
    profiling_options=gs.options.ProfilingOptions(show_FPS=not args.headless),
)
plane = scene.add_entity(gs.morphs.Plane(), name="Plane")
openarm = scene.add_entity(
    gs.morphs.USD(file=str(OPENARM_USD), fixed=True),
    # Cancel robot self-weight so the position controller tracks the grasp more closely.
    material=gs.materials.Rigid(sdf_min_res=64, sdf_max_res=256, gravity_compensation=1.0), name="OpenArm",
)
table = scene.add_entity(
    gs.morphs.USD(file=str(ASSETS_DIR / "props" / "table.usd"),
                  pos=(0.6, 0, 0), euler=(0, 0, 90), fixed=True),
    name="Table",
)
# Source components are watertight: preserve them instead of wrapping/merging them.
# fixed=False lets contact forces lift the holder; liquid coupling is enabled by default.
bowl = scene.add_entity(
    gs.morphs.USD(file=str(BOWL_USD),
                  pos=BOWL_POS, euler=BOWL_EULER, fixed=False, watertighten=None),
    material=gs.materials.Rigid(sdf_cell_size=0.001, sdf_min_res=64, sdf_max_res=256),
    surface=gs.surfaces.Default(color=(0.4, 0.8, 1.0)), name="Bowl",
)
if not args.no_liquid:
    liquid = scene.add_entity(
        material=gs.materials.SPH.Liquid(),
        morph=gs.morphs.Box(pos=LIQUID_POS, size=LIQUID_SIZE),
        surface=gs.surfaces.Default(color=(0.68, 0.62, 0.49), vis_mode="recon"),
    )
scene.build()

# Static visual marker for the place location (holder grasp point); not a physical entity.
place_marker_pos = np.array(BOWL_POS) + Rotation.from_euler('xyz', BOWL_EULER, degrees=True).apply(BLOCK_GRASP_LOCAL) + PLACE_OFFSET
if not args.headless:
    scene.draw_debug_sphere(pos=place_marker_pos, radius=0.01, color=(1, 0, 0, 1))

# Fixed-base robot order: left arm 0:7, left fingers 7:9, right arm 9:16.
left_arm = np.arange(7)
left_finger = np.arange(7, 9)
right_arm = np.arange(9, 16)
openarm.set_dofs_kp(np.array([230, 230, 190, 190, 30, 30, 30, 30, 30,
                             230, 230, 190, 190, 30, 30, 30]))
openarm.set_dofs_kv(np.array([2.7, 2.7, 2.2, 2.2, 1.5, 1.5, 1.5, 0.2, 0.2,
                             2.7, 2.7, 2.2, 2.2, 1.5, 1.5, 1.5]))
force_limits = np.array([40, 40, 27, 27, 7, 7, 7, 10, 10, 40, 40, 27, 27, 7, 7, 7])
openarm.set_dofs_force_range(-force_limits, force_limits)
openarm.control_dofs_position(np.zeros(16))

# Read the existing grasp point from USD; Genesis IK takes its parent link + offset.
LEFT_GRIPPER_PATH = "/openarm_spoon/openarm_left_base_link/openarm_left_ee_base_link"
left_gripper = openarm.get_link(LEFT_GRIPPER_PATH)
stage = Usd.Stage.Open(str(OPENARM_USD))
xforms = UsdGeom.XformCache()
grip_to_link = (
    xforms.GetLocalToWorldTransform(stage.GetPrimAtPath(LEFT_GRIPPER_PATH + "/sites/left_gripper_grip_point"))
    * xforms.GetLocalToWorldTransform(stage.GetPrimAtPath(LEFT_GRIPPER_PATH)).GetInverse()
)
grip_local = np.array(grip_to_link.ExtractTranslation())

# Genesis' default (relative=True) getter removes its internal inertia-frame
# alignment, giving the moving USD frame needed for these STL-local coordinates.
def bowl_grasp_pose():
    rotation = Rotation.from_quat(bowl.get_quat().detach().cpu().numpy()[[1, 2, 3, 0]])
    position = bowl.get_pos().detach().cpu().numpy() + rotation.apply(BLOCK_GRASP_LOCAL)
    return position, rotation

grip_debug = None
target_debug = None

def grip_world_position():
    quat = left_gripper.get_quat(relative=False).detach().cpu().numpy()
    rotation = Rotation.from_quat(quat[[1, 2, 3, 0]])  # wxyz -> xyzw
    return left_gripper.get_pos(relative=False).detach().cpu().numpy() + rotation.apply(grip_local)


def step_and_draw():
    global grip_debug, target_debug
    scene.step()
    if args.headless:
        return
    if target_debug is not None:
        scene.clear_debug_object(target_debug)
    position, rotation = bowl_grasp_pose()
    frame = np.eye(4)
    frame[:3, :3] = rotation.as_matrix()
    frame[:3, 3] = position
    target_debug = scene.draw_debug_frame(T=frame, axis_length=0.06, origin_size=0.006,
                                         axis_radius=0.0015, color=(1, 0.25, 0, 1))
    if grip_debug is not None:
        scene.clear_debug_object(grip_debug)
    grip_debug = scene.draw_debug_sphere(pos=grip_world_position(), radius=0.005,
                                        color=(0, 1, 0, 0.8))


def move_to(qpos, steps=500):
    """Ramp joint targets (default 2 s), then settle for 0.4 s. No planning."""
    start = openarm.get_qpos().detach().cpu().numpy()
    qpos = qpos.detach().cpu().numpy() if hasattr(qpos, "detach") else np.asarray(qpos)
    for alpha in np.linspace(0, 1, steps):
        openarm.control_dofs_position(start + alpha * (qpos - start))
        step_and_draw()
    for _ in range(100):
        step_and_draw()


def solve_grip(target, rotation, fingers):
    """Left-arm IK for the grip point; right arm held at zero."""
    qpos = openarm.inverse_kinematics_multilink(
        links=[left_gripper], local_points=[grip_local], dofs_idx_local=left_arm,
        poss=[target], quats=[rotation.as_quat()[[3, 0, 1, 2]]],
        max_solver_iters=200, damping=0.001, pos_tol=1e-4, rot_tol=1e-3,
    )
    qpos[7:] = 0.0
    qpos[left_finger] = fingers
    return qpos


print("[Step 1] Raise left arm", flush=True)
# 1. Raise the left arm. Explicitly zero all other joint targets after IK.
qpos = openarm.inverse_kinematics_multilink(
    links=[left_gripper], local_points=[grip_local], dofs_idx_local=left_arm,
    poss=[[0.20, 0.30, 0.50]], quats=[[0.7071, 0, -0.7071, 0]],
    max_solver_iters=200, damping=0.001, pos_tol=1e-4, rot_tol=1e-3,
)
qpos[7:] = 0.0
move_to(qpos)

print("[Step 2] Open fingers", flush=True)
# 2. Open the fingers; keep the same left-arm target and right arm at zero.
gripper_open_pos = 0.4
qpos[left_finger] = gripper_open_pos
qpos[right_arm] = 0.0
move_to(qpos, steps=100)

# Refresh the target after the bowl and liquid have settled on the table.
grasp_target, bowl_rotation = bowl_grasp_pose()

print("[Step 3] Approach attachment", flush=True)
# 3. Move the grasp point to the attachment, keeping the fingers OPEN.
# Columns: gripper +X -> holder +Y; +Y -> holder -Z; +Z -> holder -X.
gripper_in_holder = np.column_stack(([0, 1, 0], [0, 0, -1], [-1, 0, 0]))
grasp_rotation = bowl_rotation * Rotation.from_matrix(gripper_in_holder)
# First align outside the free end, then insert horizontally between the open fingers.
# Holder +X runs from the free end toward the bowl.
approach = bowl_rotation.apply([0.06, 0, 0])
for target in (grasp_target - approach, grasp_target):
    qpos = solve_grip(target, grasp_rotation, gripper_open_pos)
    move_to(qpos)

error_mm = 1000 * np.linalg.norm(grip_world_position() - grasp_target)
print(f"Grasp position error after settling: {error_mm:.2f} mm")

print("[Step 4] Close fingers", flush=True)
# 4. Grasp the holder by closing fingers; keep the same left-arm target and right arm at zero.
gripper_closed_pos = 0.0
qpos[left_finger] = gripper_closed_pos
qpos[right_arm] = 0.0
move_to(qpos, steps=100)

# Require contact on BOTH fingers. Closing the motor target alone does not attach a bowl.
contacts = openarm.get_contacts(with_entity=bowl)
touching_links = set(contacts["link_a"].tolist() + contacts["link_b"].tolist())
finger_links = {link.idx for link in openarm.links
                if link.name.endswith(("/openarm_left_ee_inner_finger", "/openarm_left_ee_outer_finger"))}
if len(finger_links) != 2 or not finger_links.issubset(touching_links):
    raise RuntimeError("Both fingers must contact the holder before lifting. Check the orange target and jaw clearance.")
bowl_start_z = bowl_grasp_pose()[0][2]

# Weld the bowl to the gripper so it can't slip or tilt in its jaws during the carry.
phase_start = time.perf_counter()
scene.sim.rigid_solver.add_weld_constraint(left_gripper.idx, bowl.base_link.idx)
report_phase("add_weld", phase_start)

print("[Step 5] Lift holder", flush=True)
# 5. Keep the fingers closed and lift the holder.
# The SPH liquid follows through physical coupling; it is not parented to the gripper.
phase_start = time.perf_counter()
lift = np.array([0, 0, args.carry_height])
qpos = solve_grip(grasp_target + lift, grasp_rotation, gripper_closed_pos)
move_to(qpos)
report_phase("lift_ik_and_motion", phase_start)
bowl_lift = bowl_grasp_pose()[0][2] - bowl_start_z
print(f"Measured holder lift: {1000 * bowl_lift:.1f} mm (commanded gripper lift: {1000 * args.carry_height:.0f} mm)")
if bowl_lift < 0.5 * args.carry_height:
    print("The holder did not follow the lift sufficiently; inspect the grasp for slipping.")

# The holder keeps the pick orientation; only its position changes.
place_target = grasp_target + PLACE_OFFSET

print(f"[Step 6] Carry to place location (offset {PLACE_OFFSET[:2].tolist()} m)", flush=True)
# 6. Traverse at carry height, then lower to just above the resting height.
phase_start = time.perf_counter()
for target in (place_target + lift, place_target + [0, 0, PLACE_CLEARANCE]):
    qpos = solve_grip(target, grasp_rotation, gripper_closed_pos)
    move_to(qpos)
report_phase("carry_and_lower", phase_start)

print("[Step 7] Release holder", flush=True)
# 7. Remove the weld first; the fingers alone cannot hold it, so it settles onto the table.
scene.sim.rigid_solver.delete_weld_constraint(left_gripper.idx, bowl.base_link.idx)
qpos[left_finger] = gripper_open_pos
move_to(qpos, steps=100)

print("[Step 8] Retreat", flush=True)
# 8. Withdraw along the insertion axis (reverse of step 3), then rise clear.
retreat = place_target - approach
for target in (retreat, retreat + lift):
    qpos = solve_grip(target, grasp_rotation, gripper_open_pos)
    move_to(qpos)

placed_position, placed_rotation = bowl_grasp_pose()
place_error = placed_position - place_target
tilt_deg = np.rad2deg((bowl_rotation.inv() * placed_rotation).magnitude())
print(f"Place error (holder grasp point): {1000 * np.linalg.norm(place_error[:2]):.1f} mm in XY, "
      f"{1000 * place_error[2]:.1f} mm in Z; orientation change {tilt_deg:.2f} deg")
contacts = bowl.get_contacts(with_entity=openarm)
if len(contacts["link_a"]):
    print("WARNING: the robot is still touching the holder after the retreat.")

# Hold the final (retreated) pose.
while not args.headless and scene.viewer.is_alive():
    step_and_draw()
