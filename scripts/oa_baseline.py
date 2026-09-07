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
    show_viewer=True,
)

plane = scene.add_entity(
    gs.morphs.Plane(),
    name="Plane",
)
openarm = scene.add_entity(
    gs.morphs.USD(
        file=str(OPENARM_USD),
        fixed=True,
        # recompute_inertia=True,
    ),
    name="OpenArm",
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

# Move to a pose
qpos = openarm.inverse_kinematics(
    link=left_ee,
    pos=np.array([0.2, 0.2, 0.2]),
    quat=np.array([0, 1, 0, 0]),
)

path = openarm.plan_path(
    qpos_goal=qpos,
    num_waypoints=200 if "PYTEST_VERSION" not in os.environ else 10,
)

# draw the planned path
path_debug = scene.draw_debug_path(path, openarm)

# execute the planned path
for waypoint in path:
    openarm.control_dofs_position(waypoint)
    scene.step()

# remove the drawn path
scene.clear_debug_object(path_debug)

# allow robot to reach the last waypoint
for i in range(100 if "PYTEST_VERSION" not in os.environ else 1):
    scene.step()

# for i in range(1000):
#     scene.step()


# Uncomment the following lines for Interactive GUI Debugging. 
# This will open a GUI window with an ImGui overlay, allowing you to interact with the scene and view debug information.

# plugin = next(p for p in scene.viewer.plugins if isinstance(p, ImGuiOverlayPlugin))

# def custom_panel(imgui):
#     imgui.text("Custom Demo Panel")
#     imgui.text("This panel was registered via register_panel()")


# plugin.register_panel(custom_panel)

# is_test = "PYTEST_VERSION" in os.environ
# horizon = 5 if is_test else None

# # step() honors the GUI: it advances only while playing, and applies a pending Rebuild Scene first. The viewer
# # paces the loop to real time (ViewerOptions.realtime_factor), so no manual sleep is needed here.
# frame = 0
# while scene.viewer.is_alive():
#     scene.step()
#     frame += 1
#     if horizon is not None and frame >= horizon:
#         break

