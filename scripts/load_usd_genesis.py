import os
from pathlib import Path
import numpy as np

import genesis as gs
from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
OPENARM_USD = ASSETS_DIR / "openarm_bimanual" / "openarm_spoon_bowl_w_o_markers.usd"

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
        pos=(0.0, 0.0, 0.0),
        euler=(0, 0, 0),
        fixed=True,
        # recompute_inertia=True, # Only needed if want to use openarm_bimanual.usd
    ),
    name="OpenArm",
)

scene.build()

is_test = "PYTEST_VERSION" in os.environ
horizon = 5 if is_test else None

frame = 0
while scene.viewer.is_alive():
    scene.step()
    frame += 1
    if horizon is not None and frame >= horizon:
        break