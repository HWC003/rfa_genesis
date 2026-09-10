import os
from pathlib import Path
import numpy as np

import genesis as gs
from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
OPENARM_USD = ASSETS_DIR / "openarm_bimanual" / "openarm_pedestal.usd"

gs.init(backend=gs.gpu)

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(1.49, 0.68, 0.96),
        camera_lookat=(0.14, -0.67, 0.33),
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
        file=str(ASSETS_DIR / "props" / "Bowl.usd"),
        pos=(0.535, 0.065, 0.5),
        euler=(90, 15, 0),
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
        pos=(0.6, 0.0, 0.65),
        size=(0.07, 0.07, 0.07),
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
right_finger = np.arange(16, 18)  # 2 DOF for the right fingers

# Set control gains for the openarm DOFs. 
openarm.set_dofs_kp(
    np.array([230, 230, 190, 190, 30, 30, 30,  # left arm
              30, 30,  # left fingers
              230, 230, 190, 190, 30, 30, 30,  # right arm
              30, 30])  # right fingers
)

openarm.set_dofs_kv(
    np.array([2.7, 2.7, 2.2, 2.2, 1.5, 1.5, 1.5,  # left arm
              0.2, 0.2,  # left fingers
              2.7, 2.7, 2.2, 2.2, 1.5, 1.5, 1.5,  # right arm
              0.2, 0.2])  # right fingers
)

openarm.set_dofs_force_range(
    np.array([-40, -40, -27, -27, -7, -7, -7, -10, -10, -40, -40, -27, -27, -7, -7, -7, -10, -10]),
    np.array([40, 40, 27, 27, 7, 7, 7, 10, 10, 40, 40, 27, 27, 7, 7, 7, 10, 10]),
)

left_ee = openarm.get_link("/openarm_pedestal_flat2/openarm_left_base_link/openarm_left_ee_base_link")
right_ee = openarm.get_link("/openarm_pedestal_flat2/openarm_right_base_link/openarm_right_ee_base_link")

target_pos_left = np.array([0.20, 0.35, 0.50])
target_pos_right = np.array([0.20, -0.35, 0.50])

target_quat_left = np.array([0.7071, 0.0, -0.7071, 0.0])   
target_quat_right = np.array([0.7071, 0.0, -0.7071, 0.0])

# Move to a pose
qpos = openarm.inverse_kinematics_multilink(
    links=[left_ee, right_ee],
    poss=[target_pos_left, target_pos_right],
    quats=[target_quat_left, target_quat_right],
)

path = openarm.plan_path(
    qpos_goal=qpos,
    num_waypoints=200 if "PYTEST_VERSION" not in os.environ else 10,
    # ee_link_name=left_ee,
    # with_entity=cube,
)

# draw the planned path
path_debug_left = scene.draw_debug_path(path, openarm, link_idx=left_ee.idx_local)
path_debug_right = scene.draw_debug_path(path, openarm, link_idx=right_ee.idx_local)

# execute the planned path
for waypoint in path:
    openarm.control_dofs_position(waypoint)
    scene.step()

# let the PD controller settle onto the final waypoint
for i in range(100 if "PYTEST_VERSION" not in os.environ else 1):
    scene.step()

gripper_open_pos = 0.4  # radians
# Left gripper's finger joints range over [0, 0.785]; 
# Right gripper is a mirrored assembly whose finger joints range over [-0.785, 0] 
finger_dofs_idx = np.concatenate([left_finger, right_finger])
finger_targets = np.array([gripper_open_pos] * len(left_finger) + [-gripper_open_pos] * len(right_finger)) #[0.4, 0.4, -0.4, -0.4]

openarm.control_dofs_position(finger_targets, dofs_idx_local=finger_dofs_idx)
for i in range(100 if "PYTEST_VERSION" not in os.environ else 1):
    scene.step()

# remove the drawn path
scene.clear_debug_object(path_debug_left)
scene.clear_debug_object(path_debug_right)

# Uncomment the following lines for Interactive GUI Debugging. 
# This will open a GUI window with an ImGui overlay, allowing you to interact with the scene and view debug information.

# plugin = next(p for p in scene.viewer.plugins if isinstance(p, ImGuiOverlayPlugin))

# def custom_panel(imgui):
#     imgui.text("Custom Demo Panel")
#     imgui.text("This panel was registered via register_panel()")

# plugin.register_panel(custom_panel)

is_test = "PYTEST_VERSION" in os.environ
horizon = 5 if is_test else None

# step() honors the GUI: it advances only while playing, and applies a pending Rebuild Scene first. The viewer
# paces the loop to real time (ViewerOptions.realtime_factor), so no manual sleep is needed here.
frame = 0
while scene.viewer.is_alive():
    scene.step()
    frame += 1
    if horizon is not None and frame >= horizon:
        break

