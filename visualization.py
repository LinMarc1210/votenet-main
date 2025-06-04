import open3d as o3d
import numpy as np

# === 設定路徑 ===
ply_path = "demo_files/input_pc_sunrgbd.ply"
txt_path = "demo_files/sunrgbd_results/detection_summary.txt"

# === 讀取點雲 ===
pcd = o3d.io.read_point_cloud(ply_path)

# === 讀取偵測框 ===
boxes = []
with open(txt_path, 'r') as f:
    next(f)  # skip header
    for line in f:
        x, y, z, class_name, score = line.strip().split('\t')
        score = float(score)
        if score < 0.95:
            continue  # 只要信心 > 0.9 的框
        center = np.array([float(x), float(y), float(z)])
        extent = np.array([0.8, 0.8, 0.8])  # 假設框大小固定，你可以依類別客製
        bbox = o3d.geometry.OrientedBoundingBox(center, np.eye(3), extent)
        bbox.color = [1, 0, 0]  # 紅色
        boxes.append(bbox)

# === 顯示 ===
print(f"共繪製 {len(boxes)} 個信心 > 0.9 的偵測框")
o3d.visualization.draw_geometries([pcd] + boxes)
