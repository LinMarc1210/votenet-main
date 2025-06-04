from plyfile import PlyData
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

# 📦 範例用法
convert_ply_to_txt(os.path.join(os.path.dirname(__file__), "demo_files/sunrgbd_results/000000_pred_confident_nms_bbox.ply"))
