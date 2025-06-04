from plyfile import PlyData
import open3d as o3d
import pandas as pd
import os

def convert_ply_to_txt(ply_path, txt_path=None):
    # 讀取 .ply 檔案
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex'].data

    # 自動產出輸出檔名
    if txt_path is None:
        txt_path = os.path.splitext(ply_path)[0] + ".txt"

    # 擷取欄位名
    fields = vertices.dtype.names

    # 寫入 .txt
    with open(txt_path, 'w') as f:
        f.write('\t'.join(fields) + '\n')  # 寫欄位標頭
        for v in vertices:
            values = [str(v[field]) for field in fields]
            f.write('\t'.join(values) + '\n')

    print(f"✅ 轉換完成：{txt_path}")

def convert_csv_to_ply(csv_path, ply_path=None):
    df = pd.read_csv(csv_path)
    # 建立點雲
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(df[["x", "y", "z"]].values)

    # # 加入顏色（正規化到 0~1）
    # if {"R", "G", "B"}.issubset(df.columns):
    #     colors = df[["R", "G", "B"]].values / 255.0
    #     pcd.colors = o3d.utility.Vector3dVector(colors)

    # 輸出 PLY 檔案
    if ply_path is None:
        ply_path = os.path.splitext(csv_path)[0] + ".ply"
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"轉換完成！已輸出 {ply_path}")

    return ply_path
        

# 📦 範例用法
ply_path = convert_csv_to_ply(os.path.join(os.path.dirname(__file__), "demo_files/sunrgbd_results/pointcloud_20250604_175004.csv"))
convert_ply_to_txt(ply_path)
# convert_ply_to_txt(os.path.join(os.path.dirname(__file__), "demo_files/sunrgbd_results/000000_pred_confident_nms_bbox.ply"))
