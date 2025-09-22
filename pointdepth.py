import cv2
import numpy as np
import open3d as o3d
import os
import realsenselib as rslib

# =========================
# 基本設定
# =========================
Cam = rslib.Cam_worker()  # width = 640, height = 480
golden_path = "golden_sample.pcd"

# 只做 Z 篩選（m）
Z_MIN, Z_MAX = 0.2, 0.8
_Z_PRESETS = [(0.2, 0.8), (0.1, 0.6), (0.3, 1.0)]
_Z_PRESET_IDX = 0

# Trackbar 以「公分」為單位比較直覺
# 給定可調範圍 5 cm ~ 300 cm（可自行調大）
TB_MIN_CM, TB_MAX_CM = 5, 300

def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        color_img, depth_img = param["color"], param["depth"]
        print("X:", x, "Y:", y)
        try:
            print("depth(mm):", int(depth_img[y, x]))
        except Exception:
            pass

def depth_to_points(color, depth, intrin, z_min_m, z_max_m):
    """
    僅用 Z 範圍保留點雲（無 ROI）
    color: HxWx3 BGR (uint8)
    depth: HxW (uint16, mm)
    intrin: 需有 fx, fy, ppx, ppy
    回傳: (Open3D PointCloud, 有效點數)
    """
    h, w = depth.shape
    fx, fy = intrin.fx, intrin.fy
    cx, cy = intrin.ppx, intrin.ppy

    depth_m = depth.astype(np.float32) * 0.001  # mm → m

    # 只用 Z 篩選
    mask = (depth_m > z_min_m) & (depth_m < z_max_m)
    if not np.any(mask):
        return o3d.geometry.PointCloud(), 0

    ys, xs = np.where(mask)
    zs = depth_m[ys, xs]

    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    pts = np.stack([X, Y, zs], axis=1)  # (N,3)

    cols = color[ys, xs, :].astype(np.float32) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd, pts.shape[0]

def visualize_pcds(pcd1, pcd2=None):
    try:
        if pcd2 is not None and len(pcd2.points) > 0:
            pcd1_c = o3d.geometry.PointCloud(pcd1)  # 避免原物件被上色覆蓋
            pcd2_c = o3d.geometry.PointCloud(pcd2)
            pcd1_c.paint_uniform_color([0, 1, 0])  # 綠: golden
            pcd2_c.paint_uniform_color([1, 0, 0])  # 紅: current
            o3d.visualization.draw_geometries([pcd1_c, pcd2_c])
        else:
            o3d.visualization.draw_geometries([pcd1])
    except Exception as e:
        print("[Open3D] 視窗建立失敗：", e)

def icp(source_pcd, target_pcd, max_corr=0.02, max_iter=50):
    result = o3d.pipelines.registration.registration_icp(
        source=source_pcd, target=target_pcd,
        max_correspondence_distance=max_corr,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    return result.transformation, float(result.fitness), float(result.inlier_rmse)

def transform_to_readable(T):
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
    t_mm = (float(t[0] * 1000.0), float(t[1] * 1000.0), float(t[2] * 1000.0))
    return yaw_deg, pitch_deg, roll_deg, t_mm

def cm_to_m(v_cm):
    return max(TB_MIN_CM, min(TB_MAX_CM, v_cm)) / 100.0

def ensure_order(zmin_m, zmax_m, gap_m=0.01):
    """確保 zmin < zmax，並維持至少 gap_m 的間距"""
    if zmin_m >= zmax_m - gap_m:
        zmax_m = zmin_m + gap_m
    return zmin_m, zmax_m

# =========================
# 介面初始化
# =========================
cv2.namedWindow('RealSense - Pose Detection', cv2.WINDOW_NORMAL)
cv2.namedWindow('Z Controls', cv2.WINDOW_AUTOSIZE)

# 用公分初始化 Trackbar
cv2.createTrackbar('Z min (cm)', 'Z Controls', int(Z_MIN*100), TB_MAX_CM, lambda v: None)
cv2.createTrackbar('Z max (cm)', 'Z Controls', int(Z_MAX*100), TB_MAX_CM, lambda v: None)

golden_pcd = None
print("操作提示：a=存 golden, b=顯示對比, z=切換 Z 預設, q=離開（滑鼠雙擊影像列印深度mm）")

while True:
    color_image, depth_image = Cam.take_pic()  # color: HxWx3 (BGR), depth: HxW (mm)

    # 從 Trackbar 讀 Z 範圍（cm → m）
    zmin_m = cm_to_m(cv2.getTrackbarPos('Z min (cm)', 'Z Controls'))
    zmax_m = cm_to_m(cv2.getTrackbarPos('Z max (cm)', 'Z Controls'))
    zmin_m, zmax_m = ensure_order(zmin_m, zmax_m, gap_m=0.01)  # 至少 1 cm 間距

    # 影像上方疊字顯示目前 Z 範圍 & golden 狀態
    overlay = color_image.copy()
    status = f"Z range: {zmin_m:.2f} ~ {zmax_m:.2f} m | golden: {'YES' if golden_pcd else 'NO'}"
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(overlay, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    # 顯示
    param = {"color": color_image, "depth": depth_image}
    cv2.setMouseCallback('RealSense - Pose Detection', draw_circle, param)
    cv2.imshow('RealSense - Pose Detection', overlay)

    # 產生 current 點雲（僅 Z 篩選）
    current_pcd, n_valid = depth_to_points(color_image, depth_image, Cam.depth_intrin, zmin_m, zmax_m)

    # 若尚未有 golden，先處理建立流程
    if golden_pcd is None:
        if os.path.exists(golden_path):
            golden_pcd = o3d.io.read_point_cloud(golden_path)
            print("📂 從檔案載入 golden sample")
        else:
            # 沒 golden 就不做 ICP，只提示
            if n_valid == 0:
                print(f"[Z=({zmin_m:.2f}~{zmax_m:.2f}m)] 有效點=0 | 請調整 Z 或建立 golden (a)")
            # 等待鍵盤
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('a'):
                # 以目前 Z 範圍建立 golden
                pcd_g, n_g = depth_to_points(color_image, depth_image, Cam.depth_intrin, zmin_m, zmax_m)
                if n_g > 0:
                    try:
                        visualize_pcds(pcd_g)
                    except Exception:
                        pass
                    o3d.io.write_point_cloud(golden_path, pcd_g)
                    golden_pcd = pcd_g
                    print(f"✅ 儲存成功：{golden_path}（有效點 {n_g}）")
                else:
                    print("❌ 有效點太少，無法建立 golden。請調整 Z 範圍。")
            elif k == ord('z'):
                # 切換 Z 預設
                _Z_PRESET_IDX = (_Z_PRESET_IDX + 1) % len(_Z_PRESETS)
                Z_MIN, Z_MAX = _Z_PRESETS[_Z_PRESET_IDX]
                # 同步到 Trackbar（cm）
                cv2.setTrackbarPos('Z min (cm)', 'Z Controls', int(Z_MIN*100))
                cv2.setTrackbarPos('Z max (cm)', 'Z Controls', int(Z_MAX*100))
                print(f"🔁 切換 Z 範圍：{Z_MIN:.2f} ~ {Z_MAX:.2f} m")
            continue

    # 有 golden 才做 ICP
    if len(current_pcd.points) > 0 and len(golden_pcd.points) > 0:
        T, fitness, rmse = icp(current_pcd, golden_pcd)
        yaw_deg, pitch_deg, roll_deg, t_mm = transform_to_readable(T)
        print(
            f"[Z=({zmin_m:.2f}~{zmax_m:.2f}m) 有效點={len(current_pcd.points):5d}] "
            f"Euler ZYX (deg)  yaw={yaw_deg:+.4f}, pitch={pitch_deg:+.4f}, roll={roll_deg:+.4f} | "
            f"t (mm) [{t_mm[0]:+.3f}, {t_mm[1]:+.3f}, {t_mm[2]:+.3f}] | "
            f"ICP fitness={fitness:.3f}, RMSE={rmse:.3f}"
        )
    else:
        print(f"[Z=({zmin_m:.2f}~{zmax_m:.2f}m)] 有效點={n_valid:5d} | 點太少，略過 ICP。")

    # 鍵盤控制
    k = cv2.waitKey(1) & 0xFF
    if k == 255:  # 無按鍵
        continue
    if k == ord('q'):
        break
    elif k == ord('a'):
        pcd_g, n_g = depth_to_points(color_image, depth_image, Cam.depth_intrin, zmin_m, zmax_m)
        if n_g > 0:
            try:
                visualize_pcds(pcd_g)
            except Exception:
                pass
            o3d.io.write_point_cloud(golden_path, pcd_g)
            golden_pcd = pcd_g
            print(f"✅ 儲存成功：{golden_path}（有效點 {n_g}）")
        else:
            print("❌ 有效點太少，無法建立 golden。請調整 Z 範圍。")
    elif k == ord('b'):
        if golden_pcd is not None and len(current_pcd.points) > 0:
            visualize_pcds(golden_pcd, current_pcd)
        else:
            print("❌ 尚無 golden 或 current 無效點。")
    elif k == ord('z'):
        _Z_PRESET_IDX = (_Z_PRESET_IDX + 1) % len(_Z_PRESETS)
        Z_MIN, Z_MAX = _Z_PRESETS[_Z_PRESET_IDX]
        cv2.setTrackbarPos('Z min (cm)', 'Z Controls', int(Z_MIN*100))
        cv2.setTrackbarPos('Z max (cm)', 'Z Controls', int(Z_MAX*100))
        print(f"🔁 切換 Z 範圍：{Z_MIN:.2f} ~ {Z_MAX:.2f} m")

cv2.destroyAllWindows()
