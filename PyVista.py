import pandas as pd
import numpy as np
import pyvista as pv

# =========================
# 1. Загрузка CSV
# =========================
csv_path = "asifa_2024_07_04_1720100310_0100016000_0_1720100310_0199880000_0.csv"
df = pd.read_csv(csv_path)

print(f"Loaded {len(df):,} points")

# =========================
# 2. Базовая фильтрация шума (опционально)
# =========================
df = df[
    (df["noise_flag"] == 0) &
    (df["segmentation_level_ghost"] < 0.5)
]

print(f"After filtering: {len(df):,} points")

# =========================
# 3. Создание Point Cloud
# =========================
points = df[["x", "y", "z"]].to_numpy(dtype=np.float32)

cloud = pv.PolyData(points)

# =========================
# 4. Добавление ВСЕХ числовых атрибутов
# =========================
exclude = {"x", "y", "z"}
for col in df.columns:
    if col not in exclude:
        if np.issubdtype(df[col].dtype, np.number):
            cloud[col] = df[col].to_numpy()

print(f"Scalars attached: {list(cloud.array_names)}")

# =========================
# 5. Viewer
# =========================
plotter = pv.Plotter()
plotter.set_background("black")

# Выбери один scalar по умолчанию
default_scalar = (
    "intensity_calibrated"
    if "intensity_calibrated" in cloud.array_names
    else cloud.array_names[0]
)

actor = plotter.add_points(
    cloud,
    scalars=default_scalar,
    cmap="viridis",
    point_size=2,
    render_points_as_spheres=False,
)

plotter.add_scalar_bar(
    title=default_scalar,
    n_labels=5
)

# =========================
# 6. Камера (адекватная для лидара)
# =========================
plotter.camera_position = "xy"
plotter.camera.zoom(1.5)

# =========================
# 7. Подсказка
# =========================
plotter.add_text(
    "PyVista LiDAR Viewer\n"
    "Use mouse to rotate / zoom\n"
    "Change scalar in code to visualize other fields",
    position="upper_left",
    font_size=10
)

plotter.show()
