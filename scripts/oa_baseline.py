from pathlib import Path

import genesis as gs

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
OPENARM_USD = ASSETS_DIR / "openarm_bimanual" / "openarm_bimanual.usd"

gs.init(backend=gs.gpu)

scene = gs.Scene(
    show_viewer=True,
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)
openarm = scene.add_entity(
    gs.morphs.USD(
        file=str(OPENARM_USD),
        fixed=True,
        recompute_inertia=True,
    ),
)

scene.build()
for i in range(1000):
    scene.step()