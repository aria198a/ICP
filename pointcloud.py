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
    z = depth.astype(np.float32) * depth_scale
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
    try:
        if pcd2:
            pcd1.paint_uniform_color([0, 1, 0])  # Green: golden
            pcd2.paint_uniform_color([1, 0, 0])  # Red: current
            o3d.visualization.draw_geometries([pcd1, pcd2])
        else:
            o3d.visualization.draw_geometries([pcd1])
    except Exception as e:
        print("[Open3D] 視窗建立失敗（可能沒有 OpenGL/顯示環境）。略過 3D 顯示。", e)


def compare_pcd_distance(pcd1, pcd2):
    # 使用 open3d 距離計算
    dists = pcd1.compute_point_cloud_distance(pcd2)
    if len(dists) == 0:
        return float('inf')
    return float(np.mean(dists))

def icp(source_pcd, target_pcd, max_corr=0.02, max_iter=50):
    """
    用 Open3D 做最基本的 ICP 配準（source -> target）
    回傳: T(4x4), fitness, rmse
    """
    result = o3d.pipelines.registration.registration_icp(
        source=source_pcd, target=target_pcd,
        max_correspondence_distance=max_corr,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    return result.transformation, float(result.fitness), float(result.inlier_rmse)

def transform_to_readable(T):
    """
    T: 4x4 變換矩陣
    回傳: (yaw_deg, pitch_deg, roll_deg, t_mm_xyz)
    ZYX (yaw-pitch-roll)
    """
    R = T[:3, :3]
    t = T[:3, 3]

    # ZYX: R = Rz * Ry * Rx
    # pitch = -asin(R[2,0]); roll = atan2(R[2,1], R[2,2]); yaw = atan2(R[1,0], R[0,0])
    # 處理數值誤差
    r20 = np.clip(R[2, 0], -1.0, 1.0)
    pitch = -np.arcsin(r20)

    # 檢查接近萬向節鎖（gimbal lock）
    if np.isclose(abs(r20), 1.0, atol=1e-8):
        # pitch ~ ±90°，此時 roll 設 0，yaw 從另一元素取得
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])

    yaw_deg = float(np.degrees(yaw))
    pitch_deg = float(np.degrees(pitch))
    roll_deg = float(np.degrees(roll))

    t_mm = (float(t[0] * 1000.0), float(t[1] * 1000.0), float(t[2] * 1000.0))
    return yaw_deg, pitch_deg, roll_deg, t_mm

golden_pcd = None

while True:
    color_image, depth_image = Cam.take_pic()
    cv2.namedWindow('RealSense - Pose Detection', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('RealSense - Pose Detection', draw_circle)
    cv2.imshow('RealSense - Pose Detection', color_image)

    #print("🔍 拍攝並與 golden sample 比對")
    current_pcd = rs_to_pointcloud(color_image, depth_image, Cam.depth_intrin)

    if golden_pcd is None:
        if os.path.exists(golden_path):
            golden_pcd = o3d.io.read_point_cloud(golden_path)
            print("📂 從檔案載入 golden sample")
        else:
            print("⚠️ 尚未建立 golden sample！請先按下 a 建立")
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
            continue

    # 原本平均距離（未對齊）
    dist = compare_pcd_distance(golden_pcd, current_pcd)
    # print(f"📏 平均點雲距離: {dist:.4f} m")

    # ======（新增）做 ICP，並把 T 轉成好讀格式 ======
    T, fitness, rmse = icp(current_pcd, golden_pcd, max_corr=0.02, max_iter=50)
    yaw_deg, pitch_deg, roll_deg, t_mm = transform_to_readable(T)
    #print(f"🔧 ICP: fitness={fitness:.3f}, rmse={rmse*1000:.2f} mm")
    np.set_printoptions(suppress=True, precision=6, floatmode='fixed')
    #print("轉換矩陣:\n", T)
    print(f"Euler ZYX (deg)  yaw={yaw_deg:+.4f}, pitch={pitch_deg:+.4f}, roll={roll_deg:+.4f}")
    print(f"t (mm)           [{t_mm[0]:+.3f}, {t_mm[1]:+.3f}, {t_mm[2]:+.3f}]")

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
