import os
import math
import numpy as np
import pandas as pd
import pyvista as pv
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor

PAGE_SIZE = 20


# =========================
# Utils
# =========================
def list_csv_files(directory):
    return sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(".csv")
    ])


def paginated_select(items, title="Select item"):
    if not items:
        print("No items found")
        return None

    page = 0
    max_page = math.ceil(len(items) / PAGE_SIZE) - 1

    while True:
        print(f"\n{title} (page {page + 1}/{max_page + 1})\n")

        start = page * PAGE_SIZE
        end = start + PAGE_SIZE
        page_items = items[start:end]

        idx = 1
        if page > 0:
            print("0) << previous page")

        for item in page_items:
            print(f"{idx}) {item}")
            idx += 1

        if page < max_page:
            print(f"{idx}) >> next page")

        try:
            choice = int(input("\nEnter number: "))
        except ValueError:
            continue

        if page > 0 and choice == 0:
            page -= 1
            continue

        if page < max_page and choice == idx:
            page += 1
            continue

        real_index = start + choice - 1
        if 0 <= real_index < len(items):
            return items[real_index]


def normalize(values):
    vmin = values.min()
    vmax = values.max()
    if abs(vmax - vmin) < 1e-9:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


# =========================
# Main
# =========================
def main():
    directory = os.getcwd()

    # 1️⃣ CSV выбор
    files = list_csv_files(directory)
    file_name = paginated_select(files, "Select CSV file")
    if file_name is None:
        return

    df = pd.read_csv(os.path.join(directory, file_name))
    print(f"Loaded {len(df):,} points")

    # 2️⃣ Каналы
    channels = ["height"] + list(df.columns)[3:]
    channel = paginated_select(channels, "Select channel for visualization")
    if channel is None:
        return

    channel_index = channels.index(channel)

    # =========================
    # PyVista setup
    # =========================
    points = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    cloud = pv.PolyData(points)

    plotter = pv.Plotter()
    plotter.set_background("black")

    actor = None
    scalar_bar_actor = None  # Single scalar bar that we'll reuse

    # =========================
    # Channel logic
    # =========================
    def apply_channel(index):
        nonlocal actor, scalar_bar_actor

        ch = channels[index]

        if ch == "height":
            values = df["z"].to_numpy(dtype=np.float32)
        else:
            values = df[ch].to_numpy(dtype=np.float32)

        print(
            f"[{index + 1}/{len(channels)}] {ch} | "
            f"min={values.min():.3f}, "
            f"max={values.max():.3f}, "
            f"var={values.var():.3f}, "
            f"std={values.std():.3f}"
        )

        values = normalize(values)
        cloud[ch] = values

        # Remove old actor
        if actor is not None:
            plotter.remove_actor(actor)

        # Add new actor with updated scalars
        actor = plotter.add_points(
            cloud,
            scalars=ch,
            point_size=2,
            render_points_as_spheres=False,
            cmap="viridis",
            show_scalar_bar=False,
        )

        # Create or update scalar bar using VTK directly
        if scalar_bar_actor is None:
            # Create new scalar bar
            scalar_bar_actor = vtkScalarBarActor()
            scalar_bar_actor.SetLookupTable(actor.mapper.lookup_table)

            # Position and size
            scalar_bar_actor.SetPosition(0.87, 0.05)
            scalar_bar_actor.SetWidth(0.1)
            scalar_bar_actor.SetHeight(0.9)

            # Number of labels
            scalar_bar_actor.SetNumberOfLabels(5)

            # Font sizes
            scalar_bar_actor.GetTitleTextProperty().SetFontSize(12)
            scalar_bar_actor.GetLabelTextProperty().SetFontSize(10)

            # White text
            scalar_bar_actor.GetTitleTextProperty().SetColor(1, 1, 1)
            scalar_bar_actor.GetLabelTextProperty().SetColor(1, 1, 1)

            # Add to renderer
            plotter.renderer.AddActor2D(scalar_bar_actor)
        else:
            # Update existing scalar bar
            scalar_bar_actor.SetLookupTable(actor.mapper.lookup_table)

        # Update title
        scalar_bar_actor.SetTitle(ch)

        plotter.render()

    # =========================
    # Hotkeys
    # =========================
    def prev_channel():
        nonlocal channel_index
        channel_index = (channel_index - 1) % len(channels)
        apply_channel(channel_index)

    def next_channel():
        nonlocal channel_index
        channel_index = (channel_index + 1) % len(channels)
        apply_channel(channel_index)

    # Arrow keys
    plotter.add_key_event("Left", prev_channel)
    plotter.add_key_event("Right", next_channel)

    # Letters - try both lowercase and uppercase
    plotter.add_key_event("a", prev_channel)
    plotter.add_key_event("d", next_channel)
    plotter.add_key_event("A", prev_channel)
    plotter.add_key_event("D", next_channel)

    # =========================
    # Init
    # =========================
    apply_channel(channel_index)

    plotter.add_text(
        "<- / a  : previous channel\n"
        "-> / d  : next channel",
        position="upper_left",
        font_size=10,
        color="white"
    )

    plotter.camera_position = "xy"
    plotter.camera.zoom(1.5)

    plotter.show()


if __name__ == "__main__":
    main()