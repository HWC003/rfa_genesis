"""
Post-process feeding CSV files recorded on Qualysis to match the same coordinate frame and units as MuJoCo for simulation replay and retargeting
"""

import csv
import numpy as np
import os

SPOON_MARKERS = ["Spoon_front", "Spoon_back", "Spoon_left", "Spoon_right"]

# SPOON_MARKERS = ["Spoon_front", "Spoon_back", "Spoon_center"]  #OLD CONFIG TO REMOVE
BOWL_MARKERS = ["Bowl_center", "Bowl_rim_1", "Bowl_rim_2", "Bowl_rim_3", "Bowl_rim_4"]
# Output column order (formerly the site order of airbot_gripper_feeding_v2.xml).
# Recorded markers not listed here (Handle_*, Spoon_center) are dropped.
MARKER_ORDER = BOWL_MARKERS + ["Spoon_front", "Spoon_back", "Spoon_right", "Spoon_left"]

scooping_point = [0.30, 0.0, 0.40]  # Translate the trajectories so that they can be reached by robot arm
# scooping_point = [0.5, -0.05, -0.1]  # Translate the trajectories so that they can be reached by robot arm
# e.g. Scoop_center starting point will be at (0.5, -0.05, -0.1) in the MuJoCo world frame, which is reachable by the robot arm

def map_marker_positions(points_mm, bowl_index, point=None):
    """Qualisys mm -> world metres, anchored at the first bowl-center sample.

    Changing point applies only a common translation: orientations and all
    relative spoon/bowl motion remain unchanged.
    """
    mapped = np.asarray(points_mm, dtype=float)[..., [1, 0, 2]] / 1000.0
    mapped[..., 0] *= -1
    return mapped - mapped[0, bowl_index].copy() + np.asarray(scooping_point if point is None else point)


## Utility functions for loading and processing CSV data

def ReadCSV(file):
    import pandas as pd
    header_index = None
    with open(file, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.startswith("Frame,Time,"):
                header_index = index
                break

    if header_index is None:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(file, skiprows=header_index)
    return df

def postprocess_csv(input_csv, output_csv, marker_names=MARKER_ORDER):
    timestamps_list = []
    keypoints3d_list = []
    df = ReadCSV(input_csv)
    
    print(f"Marker order: {marker_names}")

    # Keep only the listed markers, in the listed order
    ordered_columns = []
    for base_col in ["Frame", "Time"]:
        if base_col in df.columns:
            ordered_columns.append(base_col)
    for name in marker_names:
        for suffix in [' X', ' Y', ' Z']:
            col_name = name + suffix
            if col_name in df.columns:
                ordered_columns.append(col_name)
    print(f"Ordered columns: {ordered_columns}")

    # Check if all expected columns are present
    missing_cols = set(ordered_columns) - set(df.columns)
    if missing_cols:
        print(f"Warning: Missing columns in CSV: {missing_cols}")

    df_ordered = df[ordered_columns]

    print(f"Original CSV top 5 rows:\n{df.head()}")

    # Convert from Qualysis coordinate frame to MuJoCo (x' = -y, y' = x, z' = z)
    x_cols = [col for col in df_ordered.columns if col.endswith(" X")]
    y_cols = [col for col in df_ordered.columns if col.endswith(" Y")]
    z_cols = [col for col in df_ordered.columns if col.endswith(" Z")]

    # Same mapping is reused by the fast kinematic candidate screen.
    values = np.stack([df_ordered[x_cols].to_numpy(), df_ordered[y_cols].to_numpy(),
                       df_ordered[z_cols].to_numpy()], axis=-1)
    if "Bowl_center X" not in x_cols:
        raise ValueError("Bowl_center is required to anchor the scooping-point transform")
    mapped = map_marker_positions(values, x_cols.index("Bowl_center X"))
    df_ordered = df_ordered.copy()
    df_ordered[x_cols], df_ordered[y_cols], df_ordered[z_cols] = mapped[..., 0], mapped[..., 1], mapped[..., 2]

    print(f"Post-processed CSV top 5 rows:\n{df_ordered.head()}")

    df_ordered.to_csv(output_csv, index=False)

    print(f"Post-processed CSV saved to {output_csv}")

    ordered_marker_names = []
    for name in marker_names:
        if f"{name} X" in df_ordered.columns:
            ordered_marker_names.append(name)

    if "Time" in df_ordered.columns:
        timestamps = df_ordered["Time"].to_numpy()
    elif "Frame" in df_ordered.columns:
        timestamps = df_ordered["Frame"].to_numpy()
    else:
        timestamps = np.arange(len(df_ordered), dtype=float)

    points_cols = []
    for name in ordered_marker_names:
        points_cols.extend([f"{name} X", f"{name} Y", f"{name} Z"])

    if points_cols:
        points_flat = df_ordered[points_cols].to_numpy()
        cleaned_points = points_flat.reshape(len(df_ordered), len(ordered_marker_names), 3)
    else:
        cleaned_points = np.empty((len(df_ordered), 0, 3), dtype=float)

    dataset = (timestamps, cleaned_points)
    
    print("timestamps shape:", timestamps.shape)
    print("cleaned_points shape:", cleaned_points.shape)
    print("ordered_marker_names len:", len(ordered_marker_names))
    print("first 3 marker names:", ordered_marker_names[:3])

    return dataset, ordered_marker_names

def _origin_axis_length(df, marker_names, default_length=0.1):
    coords = []
    for name in marker_names:
        cols = [name + " X", name + " Y", name + " Z"]
        if all(col in df.columns for col in cols):
            coords.append(df[cols].to_numpy())
    if not coords:
        return default_length

    points = np.vstack(coords)
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    span = max(maxs - mins)
    if not np.isfinite(span) or span <= 0:
        return default_length
    return max(default_length, 0.1 * float(span))


def _plot_origin_axes(ax, length):
    ax.plot([0, length], [0, 0], [0, 0], color="red", linewidth=2, label="x-axis")
    ax.plot([0, 0], [0, length], [0, 0], color="green", linewidth=2, label="y-axis")
    ax.plot([0, 0], [0, 0], [0, length], color="blue", linewidth=2, label="z-axis")


def _frame_label(df, row):
    """'F<frame> <time>s' from the CSV's Frame/Time columns (row position if absent)."""
    frame = df["Frame"].iloc[row] if "Frame" in df.columns else row
    label = f"F{int(frame)}"
    if "Time" in df.columns:
        label += f" {df['Time'].iloc[row]:.2f}s"
    return label


# Markers whose paths get frame labels in plot_markers; labelling all nine clutters the plot.
FRAME_LABEL_MARKERS = ["Spoon_front", "Bowl_center"]


def plot_markers(df, marker_names, label_step=100):
    """Marker paths. Every label_step rows, FRAME_LABEL_MARKERS get a dot and frame label."""
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for name in marker_names:
        x = df[name + ' X']
        y = df[name + ' Y']
        z = df[name + ' Z']
        line = ax.plot(x, y, z, label=name)[0]
        color = line.get_color()
        ax.scatter(x.iloc[0], y.iloc[0], z.iloc[0], color=color, marker='o', s=100)
        ax.scatter(x.iloc[-1], y.iloc[-1], z.iloc[-1], color=color, marker='o', s=100)
        ax.text(x.iloc[0], y.iloc[0], z.iloc[0], 'start ' + _frame_label(df, 0), color=color)
        ax.text(x.iloc[-1], y.iloc[-1], z.iloc[-1], 'end ' + _frame_label(df, len(df) - 1), color=color)
        if name in FRAME_LABEL_MARKERS:
            for row in range(label_step, len(df) - 1, label_step):
                ax.scatter(x.iloc[row], y.iloc[row], z.iloc[row], color=color, s=15)
                ax.text(x.iloc[row], y.iloc[row], z.iloc[row], _frame_label(df, row), color=color, fontsize=7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    axis_len = _origin_axis_length(df, marker_names)
    _plot_origin_axes(ax, axis_len)
    ax.scatter(0, 0, 0, color='black', marker='*', s=120, label='origin')
    ax.text(0, 0, 0, 'origin', color='black')
    ax.set_title(f'Marker paths (frame labels every {label_step} rows on {", ".join(FRAME_LABEL_MARKERS)})')
    ax.legend()
    plt.show()

def plot_markers_with_connections(df, marker_names, step=50):
    """Spoon/bowl snapshots every step rows, each labelled with its frame at the spoon."""
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    available_markers = set(marker_names)
    spoon_markers = [m for m in SPOON_MARKERS if m in available_markers and m != "Spoon_center"]
    bowl_markers = [m for m in BOWL_MARKERS if m in available_markers and m != "Bowl_center"]

    colors = {
        "spoon_markers": "tab:blue",
        "spoon_loop": "tab:orange",
        "bowl_markers": "tab:green",
        "bowl_loop": "tab:red",
    }

    frame_indices = np.arange(0, len(df), step)
    for idx in frame_indices:
        for name in spoon_markers:
            x = df.at[idx, name + ' X']
            y = df.at[idx, name + ' Y']
            z = df.at[idx, name + ' Z']
            ax.scatter(x, y, z, s=25, color=colors["spoon_markers"])

        for name in bowl_markers:
            x = df.at[idx, name + ' X']
            y = df.at[idx, name + ' Y']
            z = df.at[idx, name + ' Z']
            ax.scatter(x, y, z, s=25, color=colors["bowl_markers"])

        spoon_loop = [
            "Spoon_left",
            "Spoon_front",
            "Spoon_right",
            "Spoon_back",
            "Spoon_left",
        ]
        if all(name in spoon_markers for name in spoon_loop):
            xs, ys, zs = [], [], []
            for name in spoon_loop:
                xs.append(df.at[idx, name + ' X'])
                ys.append(df.at[idx, name + ' Y'])
                zs.append(df.at[idx, name + ' Z'])
            ax.plot(xs, ys, zs, color=colors["spoon_loop"], linewidth=1)

        bowl_loop = [
            "Bowl_rim_1",
            "Bowl_rim_2",
            "Bowl_rim_3",
            "Bowl_rim_4",
            "Bowl_rim_1",
        ]
        if all(name in bowl_markers for name in bowl_loop):
            xs, ys, zs = [], [], []
            for name in bowl_loop:
                xs.append(df.at[idx, name + ' X'])
                ys.append(df.at[idx, name + ' Y'])
                zs.append(df.at[idx, name + ' Z'])
            ax.plot(xs, ys, zs, color=colors["bowl_loop"], linewidth=1)

        # Frame label for this snapshot, at the spoon markers' centroid.
        if spoon_markers:
            centroid = np.mean([[df.at[idx, name + axis] for axis in (' X', ' Y', ' Z')]
                                for name in spoon_markers], axis=0)
            ax.text(*centroid, _frame_label(df, idx), color="black", fontsize=7)

    ax.set_title(f'Spoon/bowl snapshots every {step} rows (labels: frame, time)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    axis_len = _origin_axis_length(df, marker_names)
    _plot_origin_axes(ax, axis_len)
    ax.scatter(0, 0, 0, color='black', marker='*', s=120, label='origin')
    ax.text(0, 0, 0, 'origin', color='black')
    plt.show()

def marker_distance_stats(df, marker_names):
    import pandas as pd
    available_markers = set(marker_names)

    spoon_markers = [m for m in SPOON_MARKERS if m in available_markers]
    bowl_markers = [m for m in BOWL_MARKERS if m in available_markers]

    def marker_positions(name):
        cols = [name + " X", name + " Y", name + " Z"]
        if not all(col in df.columns for col in cols):
            return None
        return df[cols].to_numpy()

    def pair_stats(markers, group_label):
        rows = []
        for i in range(len(markers)):
            for j in range(i + 1, len(markers)):
                a = markers[i]
                b = markers[j]
                a_pos = marker_positions(a)
                b_pos = marker_positions(b)
                if a_pos is None or b_pos is None:
                    continue
                diffs = a_pos - b_pos
                distances = np.linalg.norm(diffs, axis=1)
                rows.append({
                    "group": group_label,
                    "marker_a": a,
                    "marker_b": b,
                    "mean_distance": float(np.mean(distances)),
                    "std_distance": float(np.std(distances)),
                })
        return rows

    rows = []
    rows.extend(pair_stats(spoon_markers, "spoon"))
    rows.extend(pair_stats(bowl_markers, "bowl"))

    stats_df = pd.DataFrame(rows)
    print(stats_df.to_string(index=False))
    return stats_df

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_csv = os.path.join(project_root, "data", "trajectory", "green_bean0017_cropped.csv")
    output_csv = os.path.join(project_root, "data", "trajectory", "gb_0017_gen_frame_cropped.csv")

    postprocess_csv(input_csv, output_csv)
    original_df = ReadCSV(input_csv)
    # plot_markers(original_df, SPOON_MARKERS + BOWL_MARKERS)
    
    # Plot the processed markers
    postprocessed_df = ReadCSV(output_csv)
    plot_markers(postprocessed_df, SPOON_MARKERS + BOWL_MARKERS)
    plot_markers_with_connections(postprocessed_df, SPOON_MARKERS + BOWL_MARKERS)
    marker_distance_stats(postprocessed_df, SPOON_MARKERS + BOWL_MARKERS)


if __name__ == "__main__":
    main()