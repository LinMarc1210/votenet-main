import pandas as pd
import numpy as np

# === 讀入你的點雲 CSV ===
df_points = pd.read_csv("demo_files/sunrgbd_results/pointcloud_20250604_175004.csv")  # 請換成你的檔名
points = df_points[['x', 'y', 'z']].values
num_points = len(points)

# === 讀入 detection_summary.txt ===
df_boxes = pd.read_csv("demo_files/sunrgbd_results/detection_summary.txt", sep='\t')

# 初始化每個點的標籤與最高分數
labels = np.full(num_points, '', dtype=object)
scores = np.zeros(num_points)  # 用來記錄最高分數

# === 檢查每個框，標記落在其中的點 ===
for _, row in df_boxes.iterrows():
    cx, cy, cz = row['x'], row['y'], row['z']
    lx, ly, lz = row['length'], row['width'], row['height']
    # 替換 object_label 為 FDS 可能用到的 class name
    label = row['class_name']
    if 'desk' in label:
        label = 'dining table'
    score = row['score']

    # 判斷點是否在框內
    in_x = (points[:, 0] >= cx - lx / 2) & (points[:, 0] <= cx + lx / 2)
    in_y = (points[:, 1] >= cy - ly / 2) & (points[:, 1] <= cy + ly / 2)
    in_z = (points[:, 2] >= cz - lz / 2) & (points[:, 2] <= cz + lz / 2)
    inside = in_x & in_y & in_z

    # 若新框的信心分數比原本高，就更新標籤
    update_mask = inside & (score > scores)
    labels[update_mask] = label
    scores[update_mask] = score

# === 輸出結果 ===
df_points['object_label'] = labels
df_points.to_csv("pointcloud_20250329_193301_with_objects.csv", index=False)
print("✅ 完成：點雲已依照信心分數最高框進行分類 → pointcloud_20250329_193301_with_objects.csv")
