import cv2
import numpy as np
import open3d as o3d
import os
import realsenselib as rslib

Cam = rslib.Cam_worker()  # width = 640, height = 480
golden_path = "golden_sample.pcd"


def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("X: ", x, " , Y: ", y)
        print("depth: ", depth_image[y, x])
        
        
def rs_to_pointcloud(color, depth, intrin):
    h, w = depth.shape
    fx, fy = intrin.fx, intrin.fy
    cx, cy = intrin.ppx, intrin.ppy
    depth_scale = 0.001  # 將 depth 單位從 mm 轉為 m

    # 生成像素座標網格
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth * depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # 建立點雲與對應顏色
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = color.reshape(-1, 3).astype(np.float32) / 255.0

    # 過濾：z > 0 且 z < 1.2 公尺
    z_flat = z.reshape(-1)
    valid = (z_flat > 0) & (z_flat < 1.2)
    points = points[valid]
    colors = colors[valid]

    # 建立 Open3D PointCloud 物件
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def visualize_pcds(pcd1, pcd2=None):
    if pcd2:
        pcd1.paint_uniform_color([0, 1, 0])  # Green: golden
        pcd2.paint_uniform_color([1, 0, 0])  # Red: current
        o3d.visualization.draw_geometries([pcd1, pcd2])
    else:
        o3d.visualization.draw_geometries([pcd1])

def compare_pcd_distance(pcd1, pcd2):
    # 使用 open3d 距離計算
    dists = pcd1.compute_point_cloud_distance(pcd2)
    if len(dists) == 0:
        return float('inf')
    return np.mean(dists)

golden_pcd = None

while True:
    color_image, depth_image = Cam.take_pic()
    cv2.namedWindow('RealSense - Pose Detection', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('RealSense - Pose Detection', draw_circle)
    cv2.imshow('RealSense - Pose Detection', color_image)
    
    print("🔍 拍攝並與 golden sample 比對")
    current_pcd = rs_to_pointcloud(color_image, depth_image, Cam.depth_intrin)

    if golden_pcd is None:
        if os.path.exists(golden_path):
            golden_pcd = o3d.io.read_point_cloud(golden_path)
            print("📂 從檔案載入 golden sample")
        else:
            print("⚠️ 尚未建立 golden sample！請先按下 a 建立")
            continue

    dist = compare_pcd_distance(golden_pcd, current_pcd)
    print(f"📏 平均點雲距離: {dist:.4f} m")
    k = cv2.waitKey(1)
    if k == -1:
        continue

    try:
        key = chr(k)
    except:
        continue

    if key in ['q', 'Q']:
        print(Cam.depth_intrin)
        break

    elif key in ['a', 'A']:
        print("📸 拍攝點雲並儲存為 golden sample")
        pcd = rs_to_pointcloud(color_image, depth_image, Cam.depth_intrin)
        visualize_pcds(pcd)
        o3d.io.write_point_cloud(golden_path, pcd)
        golden_pcd = pcd
        print(f"✅ 儲存成功：{golden_path}")

    elif key in ['b', 'B']:

        visualize_pcds(golden_pcd, current_pcd)

cv2.destroyAllWindows()
