import os
import math
import numpy as np
import pandas as pd
import open3d as o3d

PAGE_SIZE = 20


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
        # os.system("clear")
        print(f"{title} (page {page + 1}/{max_page + 1})\n")

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
    min_v = np.min(values)
    max_v = np.max(values)
    if max_v - min_v < 1e-9:
        return np.zeros_like(values)
    return (values - min_v) / (max_v - min_v)


def hsv_to_rgb(hsv):
    h, s, v = hsv
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6

    if i == 0:
        return (v, t, p)
    if i == 1:
        return (q, v, p)
    if i == 2:
        return (p, v, t)
    if i == 3:
        return (p, q, v)
    if i == 4:
        return (t, p, v)
    return (v, p, q)


def values_to_hsv_colors(values, no_color=False):
    hsv = np.zeros((len(values), 3))
    if no_color:
        hsv[:, 0] = 0      # Hue
        hsv[:, 1] = 0
        hsv[:, 2] = 0.7
    else:
        hsv[:, 0] = values*0.8      # Hue
        hsv[:, 1] = 0.9
        hsv[:, 2] = 0.9

    rgb = np.array([hsv_to_rgb(h) for h in hsv])
    return o3d.utility.Vector3dVector(rgb)


def visualize(df, channel, do_normalize):
    required = {"x", "y", "z"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain x, y, z columns")

    points = df[["x", "y", "z"]].to_numpy(dtype=np.float32)

    # ---- height processing ----
    if channel == "height":
        values = df["z"].to_numpy(dtype=np.float32)
    else:
        values = df[channel].to_numpy(dtype=np.float32)

    # --- statistics ---
    mmin=values.min()
    mmax=values.max()
    print(f"'{channel}': min:{values.min():.3f}, max:{values.max():.3f}, var:{values.var():.3f}, std:{values.std():.3f}")

    if do_normalize:
        values = normalize(values)

    values = np.clip(values, 0.0, 1.0)
    colors = values_to_hsv_colors(values, no_color=(mmin==mmax))
    # if mmin==mmax:
    #     print("No color")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = colors

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"Open3D – colored by {channel}"
    )


def main():
    directory = os.getcwd()

    # 1️⃣ выбор CSV
    files = list_csv_files(directory)
    file_name = paginated_select(files, title="Select CSV file")
    if file_name is None:
        return

    df = pd.read_csv(os.path.join(directory, file_name))

    # 2️⃣ формируем список каналов (height первым)
    channels = ["height"] + list(df.columns)[3:]

    channel = paginated_select(
        channels,
        title="Select channel for visualization"
    )
    if channel is None:
        return

    # 3️⃣ нормализация
    # norm = input("\nNormalize values? (0 = no, 1 = yes): ").strip()
    do_normalize = 1#norm == "1"

    # 4️⃣ визуализация
    visualize(df, channel, do_normalize)

    # input("")

if __name__ == "__main__":
    main()