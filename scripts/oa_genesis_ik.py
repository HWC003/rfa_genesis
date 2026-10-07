import os
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin
from genesis.utils.path_planning import RRTConnect

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
OPENARM_USD = ASSETS_DIR / "openarm_bimanual" / "openarm_spoon_w_o_markers.usd"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAJ_CSV = DATA_DIR / "trajectories" / "green_bean0017.csv"

# Table mesh + its existing pose put the tabletop at world Z = 0.30 m.
# Left arm is on +Y. Center the bowl opening over the left side of the table.
TABLE_TOP_Z = 0.30
# Desired WORLD X/Y of the circular bowl opening's center, not the block or STL origin.
# Chosen to bring the attachment within the left wrist's horizontal-grasp joint limits.
BOWL_CENTER_XY = np.array([0.46, 0.25])
# Measurements in the unmarked holder STL's LOCAL frame, rounded to 4 decimal places (meters).
BOWL_AXIS_LOCAL_X = 0.1549  # Circular bowl's central axis: local X; its local Z is 0.
HOLDER_BOTTOM_LOCAL_Y = -0.0279  # Lowest vertex along local Y, i.e. underside when upright.
BOWL_RIM_LOCAL_Y = 0.0191  # Bowl rim height along local Y.
BLOCK_MID_LOCAL_Y = 0.0077  # Midpoint of block's local Y bounds (-0.0143, 0.0297).
# A +90 deg X rotation maps the opening's local +Y direction to world +Z.
# These placement formulas assume that upright orientation.
BOWL_POS = (
    float(BOWL_CENTER_XY[0] - BOWL_AXIS_LOCAL_X), #x: WORLD position of the holder origin
    float(BOWL_CENTER_XY[1]), #y
    TABLE_TOP_Z - HOLDER_BOTTOM_LOCAL_Y, #z: place the underside at tabletop height
)
BOWL_EULER = (90, 0, 0)
# Raise the grasp 15 mm on the block (local +Y becomes world +Z).
# The target remains 7 mm below the block's top face.
GRASP_HEIGHT_OFFSET = 0.015
# Local X=0.020 is a chosen grasp depth, 20 mm from the block's free end (X=0).
# Local Z=0 uses the nominal width centerline; the mesh-bounds midpoint is 0.3 mm off it.
BLOCK_GRASP_LOCAL = np.array([0.020, BLOCK_MID_LOCAL_Y + GRASP_HEIGHT_OFFSET, 0.0])
# Release liquid above the level bowl and let gravity fill it.
# The measured local Y rim height maps to world Z under BOWL_EULER.
BOWL_RIM_Z = BOWL_POS[2] + BOWL_RIM_LOCAL_Y
LIQUID_SIZE = (0.07, 0.07, 0.07)
LIQUID_POS = (*BOWL_CENTER_XY, BOWL_RIM_Z + 0.03 + LIQUID_SIZE[2] / 2)

gs.init(backend=gs.gpu)

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(1.49, 0.68, 0.96),
        camera_lookat=(0.35, 0.10, 0.33),
        enable_gui=True,
    ),
    sim_options=gs.options.SimOptions(
        dt=4e-3,
        substeps=60,  # keeps dt/substeps under the SPH stability limit
    ),
    sph_options=gs.options.SPHOptions(
        particle_size=0.006,  # trade-off: bowl wall thickness (~3mm) vs speed
        lower_bound=(0.1, -0.5, 0.0),  # tight bounds keep the SPH hash grid small
        upper_bound=(1.0, 0.5, 0.9),
    ),
    vis_options=gs.options.VisOptions(
        visualize_sph_boundary=True,
    ),
    show_viewer=True,
)

plane = scene.add_entity(
    gs.morphs.Plane(),
    name="Plane",
)

openarm = scene.add_entity(
    gs.morphs.USD(
        file=str(OPENARM_USD),
        pos=(0.0, 0.0, 0.0),
        euler=(0, 0, 0),
        fixed=True,
        # recompute_inertia=True, # Only needed if want to use openarm_bimanual.usd
    ),
    # The spoon USD collider preserves its concave mesh for Genesis SDF collision.
    # Match the bowl's grid limits to resolve the spoon's thin scoop surface.
    material=gs.materials.Rigid(
        sdf_min_res=64,
        sdf_max_res=256,
    ),
    name="OpenArm",
)

table = scene.add_entity(
    gs.morphs.USD(
        file=str(ASSETS_DIR / "props" / "table.usd"),
        pos=(0.6, 0.0, 0.0),
        euler=(0, 0, 90),
        fixed=True,
    ),
    name="Table",
)

bowl = scene.add_entity(
    gs.morphs.USD(
        file=str(ASSETS_DIR / "props" / "Bowl_holder_w_o_marker.usd"),
        pos=BOWL_POS,
        euler=BOWL_EULER,
        fixed=True,
    ),
    material=gs.materials.Rigid(  # finer SDF to resolve the ~3mm bowl wall
        sdf_min_res=64,
        sdf_max_res=256,
    ),
    surface=gs.surfaces.Default(
        color=(0.4, 0.8, 1.0),
    ),
    name="Bowl",
)

liquid = scene.add_entity(
    material=gs.materials.SPH.Liquid(),
    morph=gs.morphs.Box(
        pos=LIQUID_POS,
        size=LIQUID_SIZE,
    ),
    surface=gs.surfaces.Default(
        color=(0.68, 0.62, 0.49),  # dark beige
        vis_mode="recon",
    ),
)

scene.build()

left_arm = np.arange(7)  # 7 DOF for the left arm
left_finger = np.arange(7, 9)  # 2 DOF for the left fingers

right_arm = np.arange(9, 16)  # 7 DOF for the right arm

# Set control gains for the openarm DOFs. 
openarm.set_dofs_kp(
    np.array([230, 230, 190, 190, 30, 30, 30,  # left arm
              30, 30,  # left fingers
              230, 230, 190, 190, 30, 30, 30,  # right arm
              ])  
)

openarm.set_dofs_kv(
    np.array([2.7, 2.7, 2.2, 2.2, 1.5, 1.5, 1.5,  # left arm
              0.2, 0.2,  # left fingers
              2.7, 2.7, 2.2, 2.2, 1.5, 1.5, 1.5,  # right arm
              ])  
)

openarm.set_dofs_force_range(
    np.array([-40, -40, -27, -27, -7, -7, -7, 
              -10, -10, 
              -40, -40, -27, -27, -7, -7, -7, 
              ]),
    np.array([40, 40, 27, 27, 7, 7, 7, 
              10, 10, 
              40, 40, 27, 27, 7, 7, 7, 
              ]),
)

# Position control must hold inactive joints too; planner constraints alone do not
# keep an uncommanded arm from moving under gravity during scene.step().
openarm.control_dofs_position(openarm.get_qpos().clone())

LEFT_GRIPPER_PATH = "/openarm_spoon/openarm_left_base_link/openarm_left_ee_base_link"
LEFT_GRIP_POINT_PATH = LEFT_GRIPPER_PATH + "/sites/left_gripper_grip_point"
left_gripper = openarm.get_link(LEFT_GRIPPER_PATH)

# Genesis IK uses a link + local offset, not a USD Xform as a separate link.
robot_stage = Usd.Stage.Open(str(OPENARM_USD))
xforms = UsdGeom.XformCache()
grip_to_link = (
    xforms.GetLocalToWorldTransform(
        robot_stage.GetPrimAtPath(LEFT_GRIP_POINT_PATH)
        )
    * xforms.GetLocalToWorldTransform(
        robot_stage.GetPrimAtPath(LEFT_GRIPPER_PATH)
        ).GetInverse()
)
grip_local = np.array(grip_to_link.ExtractTranslation())


# This helper returns the current gripper position in world coordinates, using the Genesis link pose and the local offset.
def grip_world_position():
    quat = left_gripper.get_quat().detach().cpu().numpy()  # Genesis: w, x, y, z
    rotation = Rotation.from_quat(quat[[1, 2, 3, 0]])
    return left_gripper.get_pos().detach().cpu().numpy() + rotation.apply(grip_local)

# This helper recalculates the target visualization from the live bowl pose.
def bowl_grasp_frame():
    # Read the live rigid-link pose so the marker stays attached to the holder.
    link = bowl.links[0]
    quat = link.get_quat(relative=False).detach().cpu().numpy()
    rotation = Rotation.from_quat(quat[[1, 2, 3, 0]])
    frame = np.eye(4)
    frame[:3, :3] = rotation.as_matrix()
    frame[:3, 3] = link.get_pos(relative=False).detach().cpu().numpy() + rotation.apply(BLOCK_GRASP_LOCAL)
    return frame

target_frame_debug = None

# This helper updates the debug visualization of the gripper and target grasp frame.
def update_grip_debug():
    global control_point_debug, target_frame_debug

    # Updates and draws the current gripper position as green sphere
    scene.clear_debug_object(control_point_debug)
    control_point_debug = scene.draw_debug_sphere(
        pos=grip_world_position(), radius=0.005, color=(0.0, 1.0, 0.0, 0.8),
    )

    # Updates and draws the target grasp frame as an orange frame
    if target_frame_debug is not None:
        scene.clear_debug_object(target_frame_debug)
    # The target is inside the block; long axes protrude so it is still visible.
    target_frame_debug = scene.draw_debug_frame(
        T=bowl_grasp_frame(), axis_length=0.06, origin_size=0.006,
        axis_radius=0.0015, color=(1.0, 0.25, 0.0, 1.0),
    )


class SelectedArmPlanner(RRTConnect):
    """Move only the selected DOFs during this planning call.

    Genesis' original method samples every joint within its physical limits.
    IK's dofs_idx_local is not forwarded to RRT. Override this internal sampling
    hook (specific to the installed Genesis version), leaving collision checks intact.
    """

    def __init__(self, entity, active_dofs):
        super().__init__(entity)
        if entity.n_qs != entity.n_dofs:
            raise ValueError("This planner requires one qpos per DOF (fixed-base revolute robot).")
        self.active_dofs = np.asarray(active_dofs, dtype=int)
        if (self.active_dofs.ndim != 1 or self.active_dofs.size == 0
                or np.any(self.active_dofs < 0) or np.any(self.active_dofs >= entity.n_dofs)):
            raise ValueError("active_dofs must be a nonempty list of valid joint indices.")
        self.held_dofs = np.setdiff1d(np.arange(entity.n_dofs), self.active_dofs)

    def _get_dofs_limit(self, qpos_goal, envs_idx):
        lower, upper, _ = super()._get_dofs_limit(qpos_goal, envs_idx)
        lower, upper = lower.clone(), upper.clone()
        invalid = ((qpos_goal < lower) | (qpos_goal > upper))[:, self.active_dofs].any(dim=-1)
        # Physical joint limits are soft: a held joint can settle just outside a
        # bound. Permit up to 0.001 rad only for HELD joints, without teleporting
        # them or widening the active joints' limits. Reject larger violations.
        held_limit_tolerance = 0.001
        invalid |= ((qpos_goal < lower - held_limit_tolerance)
                    | (qpos_goal > upper + held_limit_tolerance))[:, self.held_dofs].any(dim=-1)
        # Equal sampling bounds freeze these joints in this planner instance only.
        lower[:, self.held_dofs] = qpos_goal[:, self.held_dofs]
        upper[:, self.held_dofs] = qpos_goal[:, self.held_dofs]
        return lower, upper, invalid


def plan_arm(qpos_goal, active_dofs):
    planner = SelectedArmPlanner(openarm, active_dofs)
    start = openarm.get_qpos().clone()
    goal = start.clone()
    goal[planner.active_dofs] = qpos_goal[planner.active_dofs]
    if not bool(goal.isfinite().all().item()):
        raise RuntimeError("IK returned non-finite joint positions; no path was commanded.")
    lower, upper = openarm.get_dofs_limit()
    active_outside = ((goal < lower) | (goal > upper))[planner.active_dofs]
    held_outside = ((goal < lower - 0.001) | (goal > upper + 0.001))[planner.held_dofs]
    if bool(active_outside.any().item()) or bool(held_outside.any().item()):
        raise RuntimeError(
            "Cannot plan: active IK goal or held joint outside limits. "
            f"Active violations: {planner.active_dofs[active_outside.cpu().numpy()].tolist()}; "
            f"held violations (>0.001 rad): {planner.held_dofs[held_outside.cpu().numpy()].tolist()}."
        )
    if planner.held_dofs.size:
        # Maintain the same inactive-joint pose assumed by collision checking.
        # Clip only the CONTROL TARGET for tiny soft-limit overshoots, not the state.
        held_targets = start[planner.held_dofs].clamp(lower[planner.held_dofs], upper[planner.held_dofs])
        openarm.control_dofs_position(held_targets, dofs_idx_local=planner.held_dofs)
    # Planner internals return an INVALID mask (opposite of plan_path's valid mask).
    for attempt in range(3):
        path, invalid = planner.plan(
            qpos_goal=goal, 
            qpos_start=start,
            max_nodes=6000, 
            timeout=15.0,
            # If shortcutting fails validation, retry the unsmoothed search path.
            smooth_path=(attempt == 0),
            resolution=0.05 if attempt == 0 else 0.025,
            num_waypoints=200 if "PYTEST_VERSION" not in os.environ else 10,
        )
        if not bool(invalid.any().item()):
            return path[:, 0, :]
        print(f"Planning attempt {attempt + 1}/3 failed for joints {planner.active_dofs.tolist()}.")
    # Never execute Genesis' zero-filled failure buffer.
    raise RuntimeError("No collision-checked path found after 3 attempts; no failed path was commanded. Joint hold targets preserved.")

control_point_debug = scene.draw_debug_sphere(
    pos=grip_world_position(),
    radius=0.005,
    color=(0.0, 1.0, 0.0, 0.8),
)
update_grip_debug()

######################################
# 1. Move Left arm upwards above table
######################################

target_pos_left = np.array([0.20, 0.30, 0.50])
target_quat_left = np.array([0.7071, 0.0, -0.7071, 0.0])   

qpos = openarm.inverse_kinematics_multilink(
    links=[left_gripper],
    local_points=[grip_local],
    dofs_idx_local=left_arm,
    poss=[target_pos_left],
    quats=[target_quat_left],
    max_solver_iters=200,
    damping=0.001,
    pos_tol=1e-4,
    rot_tol=1e-3,
)

path = plan_arm(qpos, active_dofs=left_arm)

# draw the planned path
path_debug_left = scene.draw_debug_path(path, openarm, link_idx=left_gripper.idx_local)

# execute the planned path
for waypoint in path:
    openarm.control_dofs_position(
        waypoint[left_arm],
        dofs_idx_local=left_arm,
    )
    scene.step()
    update_grip_debug()

# let the PD controller settle onto the final waypoint
for i in range(100 if "PYTEST_VERSION" not in os.environ else 1):
    scene.step()
    update_grip_debug()

# remove the drawn path
scene.clear_debug_object(path_debug_left)

#######################################
# 2. Open the gripper fingers
#######################################

gripper_open_pos = 0.4  # radians
# Left gripper's finger joints range over [0, 0.785]; 
# Right gripper is a mirrored assembly whose finger joints range over [-0.785, 0] 

# finger_dofs_idx = np.concatenate([left_finger, right_finger])
finger_dofs_idx = np.concatenate([left_finger])

# finger_targets = np.array([gripper_open_pos] * len(left_finger) + [-gripper_open_pos] * len(right_finger)) #[0.4, 0.4, -0.4, -0.4]
finger_targets = np.array([gripper_open_pos] * len(left_finger)) #[0.4, 0.4]

openarm.control_dofs_position(finger_targets, dofs_idx_local=finger_dofs_idx)
for i in range(100 if "PYTEST_VERSION" not in os.environ else 1):
    scene.step()
    update_grip_debug()

########################################
# 3. Move Left arm downwards to the bowl
########################################

# Grasp 20 mm from the block's free end, centered across its width, 15 mm higher.
# Its full length is 73 mm: targeting the length midpoint pushes the palm into it.
# Rotate its local offset as well as translating it to the scene position.
bowl_rotation = Rotation.from_euler("xyz", BOWL_EULER, degrees=True)
target_pos_left = bowl_grasp_frame()[:3, 3].copy()
# Columns express gripper +X, +Y, +Z axes in the HOLDER frame, respectively.
# Gripper -Z approaches along holder +X; jaw closing axis +Y follows holder -Z.
gripper_in_holder = np.column_stack((
    [0, 1, 0],   # gripper +X -> holder +Y (up when bowl is upright)
    [0, 0, -1],  # gripper +Y -> holder -Z (across the block)
    [-1, 0, 0],  # gripper +Z -> holder -X; therefore approach -Z -> holder +X
))
target_rotation = bowl_rotation * Rotation.from_matrix(gripper_in_holder)
target_quat_left = target_rotation.as_quat()[[3, 0, 1, 2]]

qpos = openarm.inverse_kinematics_multilink(
    links=[left_gripper],
    local_points=[grip_local],
    dofs_idx_local=left_arm,
    poss=[target_pos_left],
    quats=[target_quat_left],
    max_solver_iters=200,
    damping=0.001,
    pos_tol=1e-4,
    rot_tol=1e-3,
)

path = plan_arm(qpos, active_dofs=left_arm)

# draw the planned path
path_debug_left = scene.draw_debug_path(path, openarm, link_idx=left_gripper.idx_local)

# execute the planned path
for waypoint in path:
    openarm.control_dofs_position(waypoint[left_arm], dofs_idx_local=left_arm)
    openarm.control_dofs_position(finger_targets, dofs_idx_local=finger_dofs_idx)
    scene.step()
    update_grip_debug()


# let the PD controller settle onto the final waypoint
for i in range(100 if "PYTEST_VERSION" not in os.environ else 1):
    scene.step()
    update_grip_debug()

# remove the drawn path
scene.clear_debug_object(path_debug_left)


final_position_error_mm = 1000 * np.linalg.norm(grip_world_position() - target_pos_left)
final_quat = left_gripper.get_quat().detach().cpu().numpy()
final_rotation = Rotation.from_quat(final_quat[[1, 2, 3, 0]])
final_angle_error_deg = np.degrees((target_rotation.inv() * final_rotation).magnitude())
print(f"Grasp alignment after settling: {final_position_error_mm:.2f} mm, {final_angle_error_deg:.2f} deg")

is_test = "PYTEST_VERSION" in os.environ
horizon = 5 if is_test else None

# step() honors the GUI: it advances only while playing, and applies a pending Rebuild Scene first. The viewer
# paces the loop to real time (ViewerOptions.realtime_factor), so no manual sleep is needed here.
frame = 0
while scene.viewer.is_alive():
    scene.step()
    update_grip_debug()
    frame += 1
    if horizon is not None and frame >= horizon:
        break
