import open3d as o3d
import numpy as np
import pandas as pd

# === 設定檔案路徑 ===
csv_path = "demo_files/sunrgbd_results/pointcloud_20250604_175004.csv"
txt_path = "demo_files/sunrgbd_results/detection_summary.txt"

# === 讀取 CSV 點雲資料（含 RGB）===
df = pd.read_csv(csv_path)
points = df[['x', 'y', 'z']].values
colors = df[['R', 'G', 'B']].values / 255.0  # 正規化為 0~1

# === 建立 Open3D 點雲物件 ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# === 讀取偵測框並建立 OrientedBoundingBox ===
boxes = []
with open(txt_path, 'r') as f:
    header = f.readline().strip().split('\t')
    has_size = 'length' in header and 'width' in header and 'height' in header

    for line in f:
        vals = line.strip().split('\t')
        x, y, z, lx, ly, lz, class_name, score = vals
        extent = np.array([0.8, 0.8, 0.8])  # fallback size
        
        score = float(score)
        if score < 0.95:
            continue  # 只保留高信心框
        
        center = np.array([float(x), float(y), float(z)])
        bbox = o3d.geometry.OrientedBoundingBox(center, np.eye(3), extent)
        bbox.color = [1, 0, 0]  # 紅色框
        boxes.append(bbox)

# === 顯示點雲與框 ===
print(f"共繪製 {len(boxes)} 個信心 > 0.95 的偵測框")
o3d.visualization.draw_geometries([pcd] + boxes)
