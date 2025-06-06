import pandas as pd
import numpy as np

# === 讀入你的點雲 CSV ===
df_points = pd.read_csv("demo_files/sunrgbd_results/pointcloud_20250604_175004.csv")  # 請換成你的檔名
points = df_points[['x', 'y', 'z']].values
num_points = len(points)

# === 讀入 detection_summary.txt ===
df_boxes = pd.read_csv("demo_files/sunrgbd_results/detection_summary.txt", sep='\t')

# 初始化每個點的標籤、分數與物件編號（object_num 預設為空）
labels = np.full(num_points, '', dtype=object)
scores = np.zeros(num_points)
object_nums = np.full(num_points, np.nan)

# === 檢查每個框，僅處理分數 ≥ 0.95 的 ===
object_id = 1  # 物件編號從 1 開始
for _, row in df_boxes.iterrows():
    score = row['score']
    if score < 0.95:
        continue

    cx, cy, cz = row['x'], row['y'], row['z']
    lx, ly, lz = row['length'], row['width'], row['height']
    label = row['class_name']
    if 'desk' in label:
        label = 'dining table'

    # 判斷點是否在框內
    in_x = (points[:, 0] >= cx - lx / 2) & (points[:, 0] <= cx + lx / 2)
    in_y = (points[:, 1] >= cy - ly / 2) & (points[:, 1] <= cy + ly / 2)
    in_z = (points[:, 2] >= cz - lz / 2) & (points[:, 2] <= cz + lz / 2)
    inside = in_x & in_y & in_z

    # 更新標籤與 object_num
    update_mask = inside & (score > scores)
    labels[update_mask] = label
    scores[update_mask] = score
    object_nums[update_mask] = object_id

    object_id += 1  # 下一個有效框遞增

# === 輸出結果 ===
df_points['object_label'] = labels

# 將 object_nums 中非 NaN 的轉為整數
df_points['object_num'] = pd.Series(object_nums).dropna().astype(int)
df_points.loc[np.isnan(object_nums), 'object_num'] = np.nan  # 保留空值

df_points.to_csv("pointcloud_20250329_193301_with_objects.csv", index=False)
print("✅ 完成：已標記 score ≥ 0.95 的物件，並指派整數 object_num → pointcloud_20250329_193301_with_objects.csv")
