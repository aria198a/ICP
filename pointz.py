import cv2 
import numpy as np
import open3d as o3d
import os
import realsenselib as rslib

Cam = rslib.Cam_worker()  # width = 640, height = 480
golden_path = "golden_sample.pcd"

# === 新增：Z 範圍控制（cm） ===
TB_MIN_CM, TB_MAX_CM = 5, 300   # 可自行調整 5cm ~ 300cm
INIT_Z_MIN_CM, INIT_Z_MAX_CM = 10, 120  # 預設 0.10m ~ 1.20m

def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        depth_img = param["depth"]
        if 0 <= y < depth_img.shape[0] and 0 <= x < depth_img.shape[1]:
            print("X: ", x, " , Y: ", y)
            print("depth(mm): ", int(depth_img[y, x]))

def rs_to_pointcloud(color, depth, intrin, z_min_m, z_max_m):
    h, w = depth.shape
    fx, fy = intrin.fx, intrin.fy
    cx, cy = intrin.ppx, intrin.ppy
    depth_scale = 0.001  # mm -> m

    # 生成像素座標網格
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32) * depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # 建立點雲與對應顏色
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = color.reshape(-1, 3).astype(np.float32) / 255.0

    # 用 Trackbar 的 z_min/z_max 篩選
    z_flat = z.reshape(-1)
    valid = (z_flat > z_min_m) & (z_flat < z_max_m)
    points = points[valid]
    colors = colors[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def visualize_pcds(pcd1, pcd2=None):
    if pcd2:
        # 避免覆蓋原色，複製一份再上色
        p1 = o3d.geometry.PointCloud(pcd1)
        p2 = o3d.geometry.PointCloud(pcd2)
        p1.paint_uniform_color([0, 1, 0])  # Green: golden
        p2.paint_uniform_color([1, 0, 0])  # Red: current
        o3d.visualization.draw_geometries([p1, p2])
    else:
        o3d.visualization.draw_geometries([pcd1])

def compare_pcd_distance(pcd1, pcd2):
    dists = pcd1.compute_point_cloud_distance(pcd2)
    if len(dists) == 0:
        return float('inf')
    return float(np.mean(dists))

# ======（新增）ICP 與位姿轉換 ======
def icp(source_pcd, target_pcd, max_corr=0.02, max_iter=50):
    """
    用 ICP 將 source 對齊到 target。
    回傳：T(4x4), fitness, rmse
    """
    if len(source_pcd.points) == 0 or len(target_pcd.points) == 0:
        return np.eye(4), 0.0, float('inf')
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
    將 4x4 轉換矩陣轉為 Euler ZYX (deg) 與 t(mm)。
    """
    R = T[:3, :3]
    t = T[:3, 3]

    r20 = np.clip(R[2, 0], -1.0, 1.0)
    pitch = -np.arcsin(r20)
    if np.isclose(abs(r20), 1.0, atol=1e-8):
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])

    yaw_deg = float(np.degrees(yaw))
    pitch_deg = float(np.degrees(pitch))
    roll_deg = float(np.degrees(roll))
    t_mm = (float(t[0]*1000.0), float(t[1]*1000.0), float(t[2]*1000.0))
    return yaw_deg, pitch_deg, roll_deg, t_mm
# ==================================

golden_pcd = None

# === Z 控制視窗與滑桿 ===
cv2.namedWindow('RealSense - Pose Detection', cv2.WINDOW_NORMAL)
cv2.namedWindow('Z Controls', cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('Z min (cm)', 'Z Controls', INIT_Z_MIN_CM, TB_MAX_CM, lambda v: None)
cv2.createTrackbar('Z max (cm)', 'Z Controls', INIT_Z_MAX_CM, TB_MAX_CM, lambda v: None)

while True:
    color_image, depth_image = Cam.take_pic()

    # 滑桿讀值（cm -> m），並保證有最小間距 1cm
    zmin_cm = max(TB_MIN_CM, min(TB_MAX_CM, cv2.getTrackbarPos('Z min (cm)', 'Z Controls')))
    zmax_cm = max(TB_MIN_CM, min(TB_MAX_CM, cv2.getTrackbarPos('Z max (cm)', 'Z Controls')))
    if zmin_cm >= zmax_cm - 1:
        zmax_cm = zmin_cm + 1
        cv2.setTrackbarPos('Z max (cm)', 'Z Controls', zmax_cm)
    zmin_m, zmax_m = zmin_cm / 100.0, zmax_cm / 100.0

    cv2.setMouseCallback('RealSense - Pose Detection', draw_circle, {"depth": depth_image})
    # 疊字顯示目前 Z 區間
    overlay = color_image.copy()
    txt = f"Z range: {zmin_m:.2f} ~ {zmax_m:.2f} m"
    cv2.rectangle(overlay, (0,0), (overlay.shape[1], 30), (0,0,0), -1)
    cv2.putText(overlay, txt, (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)
    cv2.imshow('RealSense - Pose Detection', overlay)

    # 依 Z 篩選產生 current_pcd
    print("🔍 拍攝並與 golden sample 比對")
    current_pcd = rs_to_pointcloud(color_image, depth_image, Cam.depth_intrin, zmin_m, zmax_m)

    # golden 管理
    if golden_pcd is None:
        if os.path.exists(golden_path):
            golden_pcd = o3d.io.read_point_cloud(golden_path)
            print("📂 從檔案載入 golden sample")
        else:
            print("⚠️ 尚未建立 golden sample！請先按下 a 建立")
            k = cv2.waitKey(1)
            if k != -1:
                try:
                    key = chr(k)
                except:
                    key = ''
                if key in ['a', 'A']:
                    pcd = rs_to_pointcloud(color_image, depth_image, Cam.depth_intrin, zmin_m, zmax_m)
                    visualize_pcds(pcd)
                    o3d.io.write_point_cloud(golden_path, pcd)
                    golden_pcd = pcd
                    print(f"✅ 儲存成功：{golden_path}")
                elif key in ['q', 'Q']:
                    print(Cam.depth_intrin)
                    break
            continue

    # ======（新增）做 ICP，並把 T 轉成好讀格式 ======
    if len(current_pcd.points) > 0 and len(golden_pcd.points) > 0:
        T, fitness, rmse = icp(current_pcd, golden_pcd, max_corr=0.02, max_iter=50)
        yaw_deg, pitch_deg, roll_deg, t_mm = transform_to_readable(T)
        np.set_printoptions(suppress=True, precision=6, floatmode='fixed')
        print(f"Euler ZYX (deg)  yaw={yaw_deg:+.4f}, pitch={pitch_deg:+.4f}, roll={roll_deg:+.4f}")
        print(f"t (mm)           [{t_mm[0]:+.3f}, {t_mm[1]:+.3f}, {t_mm[2]:+.3f}]")
        print(f"ICP fitness={fitness:.3f}, RMSE={rmse:.3f}")
    else:
        print(f"[Z=({zmin_m:.2f}~{zmax_m:.2f} m)] current 或 golden 無有效點，略過 ICP。")

    # 同時保留你原本的平均點雲距離指標
    dist = compare_pcd_distance(golden_pcd, current_pcd)
    print(f"📏 Z=({zmin_m:.2f}~{zmax_m:.2f} m) | 有效點={len(current_pcd.points)} | 平均點雲距離: {dist:.4f} m")

    # 鍵盤控制
    k = cv2.waitKey(1)
    if k == -1:
        continue

    try:
        key = chr(k)
    except:
        key = ''

    if key in ['q', 'Q']:
        print(Cam.depth_intrin)
        break

    elif key in ['a', 'A']:
        print("📸 拍攝點雲並儲存為 golden sample")
        pcd = rs_to_pointcloud(color_image, depth_image, Cam.depth_intrin, zmin_m, zmax_m)
        visualize_pcds(pcd)
        o3d.io.write_point_cloud(golden_path, pcd)
        golden_pcd = pcd
        print(f"✅ 儲存成功：{golden_path}")

    elif key in ['b', 'B']:
        visualize_pcds(golden_pcd, current_pcd)

cv2.destroyAllWindows()
