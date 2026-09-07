"""
EXAMPLE USAGE:
/home/hauwen/isaac_ws/IsaacLab/isaaclab.sh -p /home/hauwen/oa_ros2_ws/src/rfa_genesis/scripts/convert_usd.py \
/home/hauwen/oa_ros2_ws/src/rfa_genesis/assets/openarm_bimanual/configuration/openarm_pedestal_base.usd \
/home/hauwen/oa_ros2_ws/src/rfa_genesis/assets/openarm_bimanual/configuration/openarm_pedestal_base.usda
"""

import argparse
import sys
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


# Parse the script arguments before SimulationApp processes sys.argv.
args = parse_args()
args.input = args.input.expanduser().resolve()
args.output = args.output.expanduser().resolve()

if not args.input.is_file():
    raise FileNotFoundError(f"Input USD does not exist: {args.input}")
if not args.output.parent.is_dir():
    raise FileNotFoundError(f"Output directory does not exist: {args.output.parent}")

print(f"Input:  {args.input}", flush=True)
print(f"Output: {args.output}", flush=True)

# Prevent Kit/SimulationApp from interpreting this script's file arguments.
sys.argv = [sys.argv[0]]

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

try:
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(str(args.input))

    if layer is None:
        raise RuntimeError(f"Could not open USD layer: {args.input}")

    if not layer.Export(str(args.output)):
        raise RuntimeError(f"Could not export USD layer: {args.output}")

    print(f"Converted {args.input} -> {args.output}", flush=True)
finally:
    app.close()
