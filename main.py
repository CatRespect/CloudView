import open3d as o3d
import pandas as pd
import numpy as np

# загружаем CSV
df = pd.read_csv("asifa_2024_07_04_1720100310_0100016000_0_1720100310_0199880000_0.csv")

# берём координаты
points = df[['x', 'y', 'z']].values

# создаём облако точек
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)


intensity = df['intensity_calibrated'].to_numpy()

# нормализация 0–1
intensity_norm = (intensity - intensity.min()) / (np.ptp(intensity) + 1e-6)

colors = np.stack([intensity_norm]*3, axis=1)

pcd.colors = o3d.utility.Vector3dVector(colors)

# визуализация
o3d.visualization.draw_geometries([pcd])
