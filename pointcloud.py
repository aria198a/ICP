import math
import os
import argparse
import numpy as np
import open3d as o3d

# ---------------------------- helpers ---------------------------------

def print_cloud_stats(pcd: o3d.geometry.PointCloud, tag: str):
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        print(f"[Stats] {tag}: 0 points")
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    meds = np.median(pts, axis=0)
    print(f"[Stats] {tag}: N={len(pts)}  "
          f"X:[{mins[0]:.4f},{maxs[0]:.4f}]  "
          f"Y:[{mins[1]:.4f},{maxs[1]:.4f}]  "
          f"Z:[{mins[2]:.4f},{maxs[2]:.4f}]  Z_med={meds[2]:.4f}")

def ensure_normals(pcd: o3d.geometry.PointCloud, voxel: float):
    if len(pcd.points) == 0:
        return
    if not pcd.has_normals():
        radius = max(1e-3, voxel * 2.0)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50)
        )
        # OK for relative alignment
        pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))

def ensure_proper_rotation(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0:
        Vt[-1, :] *= -1
        R_proj = U @ Vt
    return R_proj

def rotmat_to_axis_angle(R: np.ndarray):
    R = ensure_proper_rotation(R)
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = math.acos(tr)
    if abs(theta) < 1e-12:
        return np.array([1.0, 0.0, 0.0]), 0.0
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * math.sin(theta))
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return axis, theta

def rotmat_to_ypr_zyx(R: np.ndarray):
    """Intrinsic ZYX: returns yaw(Z), pitch(Y), roll(X) in radians."""
    R = ensure_proper_rotation(R)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    if sy >= 1e-8:
        yaw   = math.atan2(R[1, 0], R[0, 0])
        pitch = math.atan2(-R[2, 0], sy)
        roll  = math.atan2(R[2, 1], R[2, 2])
    else:
        yaw   = math.atan2(-R[0, 1], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        roll  = 0.0
    return yaw, pitch, roll

# ----------------------------- class ----------------------------------

class ICPFITTER:
    def __init__(self,
                 src_path=None,
                 tgt_path=None,
                 use_icp=True,
                 default_region="source"):
        """
        src_path: 你的 SOURCE 點雲
        tgt_path: 你的 TARGET 點雲
        default_region: 預設使用 'source' 或 'target'
        """

        
        if src_path is None:
            src_path = os.path.abspath(
                os.path.join(os.getcwd(), "golden", "3D_golden.ply")
            )
        if tgt_path is None:
            tgt_path = os.path.abspath(
                os.path.join(os.getcwd(), "golden(yaw+1)", "3D_golden(yaw+1).ply")
            )

        # 統一的路徑表（新 key）
        self.paths = {
            "source": src_path,   # 你的基準（正視）
            "target": tgt_path,   # 你的旋轉後
        }

        self.use_icp = use_icp
        self.default_region = default_region

        # 防呆：檔案不存在就拋錯，早點發現路徑問題
        for name, p in {"source": src_path, "target": tgt_path}.items():
            if not os.path.isfile(p):
                raise FileNotFoundError(f"[ICPFITTER] {name} 檔案找不到：{p}")

    # ----------------- new: compute ΔT from two files ------------------

    def delta_T_from_files(self, source_path, target_path,
                           voxel=0.003, icp_mode="point_to_plane",
                           max_corr=None, print_stats=True):
        """
        Compute ΔT so that ΔT @ SOURCE ≈ TARGET.
        Returns (T, info)
        info: axis, angle_deg, yaw_deg, pitch_deg, roll_deg, t_cm, fitness, rmse_mm
        """
        src = o3d.io.read_point_cloud(source_path)
        tgt = o3d.io.read_point_cloud(target_path)
        if src.is_empty() or tgt.is_empty():
            raise RuntimeError("讀入的點雲為空，請檢查路徑/檔案。")

        if print_stats:
            print_cloud_stats(src, "SRC(raw)")
            print_cloud_stats(tgt, "TGT(raw)")

        # Downsample
        src_d = src.voxel_down_sample(voxel)
        tgt_d = tgt.voxel_down_sample(voxel)

        if icp_mode == "point_to_plane":
            ensure_normals(src_d, voxel)
            ensure_normals(tgt_d, voxel)
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

        if max_corr is None:
            max_corr = voxel * 8.0  # e.g., 3mm voxel -> 24mm

        init = np.eye(4)
        reg = o3d.pipelines.registration.registration_icp(
            src_d, tgt_d, max_corr, init, estimation,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)
        )

        T = reg.transformation
        R = T[:3, :3]
        t = T[:3, 3]

        axis, theta = rotmat_to_axis_angle(R)
        yaw, pitch, roll = rotmat_to_ypr_zyx(R)

        info = {
            "axis": axis.tolist(),
            "angle_deg": math.degrees(theta),
            "yaw_deg": math.degrees(yaw),
            "pitch_deg": math.degrees(pitch),
            "roll_deg": math.degrees(roll),
            "t_cm": (t * 100.0).tolist(),
            "fitness": reg.fitness,
            "rmse_mm": reg.inlier_rmse * 1000.0,
        }
        return T, info

    # ------------------ the rest (from your example) -------------------

    def save_golden(self, color_image, depth_image, depth_intrin, region="screw"):
        """用當下影像建立點雲並存成該區域的 golden 檔"""
        pcd = self.rs_to_pointcloud(color_image, depth_image, depth_intrin)
        if pcd is None or len(pcd.points) == 0:
            print("⚠️ 無法從當前影像建立點雲，未儲存 golden")
            return False

        path = self._region_path(region)
        ok = o3d.io.write_point_cloud(path, pcd)
        if ok:
            self.golden[region] = pcd
            print(f"✅ 已儲存 {region} golden: {path} (點數={len(pcd.points)})")
            return True
        else:
            print(f"❌ 儲存失敗: {path}")
            return False

    def icp_fit(self, color_image, depth_image, depth_intrin, region=None):
        region = (region or self.default_region).lower()
        g = self.golden.get(region, None)
        if g is None:
            print(f"⚠️ {region} golden 不存在，請先 save {region}")
            return None

        cur = self.rs_to_pointcloud(color_image, depth_image, depth_intrin)
        if cur is None or len(cur.points) == 0:
            print("⚠️ 無法建立當前點雲")
            return None

        # 下採樣與法線（穩定 ICP）
        g_d = g.voxel_down_sample(0.003)
        c_d = cur.voxel_down_sample(0.003)
        ensure_normals(g_d, 0.003)
        ensure_normals(c_d, 0.003)

        if self.use_icp:
            threshold = 0.05  # 5 cm
            init = np.eye(4)
            result = o3d.pipelines.registration.registration_icp(
                c_d, g_d, threshold, init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            return float(result.inlier_rmse)
        else:
            d = np.mean(g_d.compute_point_cloud_distance(c_d))
            return float(d)

    # ---- internal tools ----

    def _region_path(self, region):
        region = region.lower()
        if region not in self.paths:
            raise ValueError(f"未知區域: {region}，應為 'screw' 或 'battery'")
        return self.paths[region]

    def rs_to_pointcloud(self, color, depth, intrin):
        """
        color: HxWx3 uint8 (BGR/RGB 皆可)
        depth: HxW uint16 (深度 raw)
        intrin: RealSense intrinsics 物件 (含 fx, fy, ppx, ppy)
        """
        if color is None or depth is None or intrin is None:
            return None
        h, w = depth.shape
        fx, fy = intrin.fx, intrin.fy
        cx, cy = intrin.ppx, intrin.ppy

        depth_scale = 0.001  # D435 常見
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.astype(np.float32) * depth_scale
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        colors = color.reshape(-1, 3).astype(np.float32) / 255.0

        valid = (z.reshape(-1) > 0) & (z.reshape(-1) < 1.5)
        points = points[valid]
        colors = colors[valid]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

# ----------------------------- CLI ------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compute ΔT so that ΔT @ SOURCE ≈ TARGET (from two files)."
    )
    ap.add_argument("--source", default="golden/3D_golden.ply",
                    help="SOURCE point cloud path (default: golden/3D_golden.ply)")
    ap.add_argument("--target", default="golden(yaw+1)/3D_golden(yaw+1).ply",
                    help="TARGET point cloud path (default: golden(yaw+1)/3D_golden(yaw+1).ply)")
    ap.add_argument("--voxel", type=float, default=0.003,
                    help="Voxel downsample size in meters (default: 0.003)")
    ap.add_argument("--icp_mode", choices=["point_to_plane", "point_to_point"],
                    default="point_to_plane", help="ICP metric (default: point_to_plane)")
    ap.add_argument("--max_corr", type=float, default=None,
                    help="Max correspondence distance in meters (default: 8*voxel)")
    ap.add_argument("--no_stats", action="store_true", help="Do not print raw stats")
    args = ap.parse_args()

    fitter = ICPFITTER()
    T, info = fitter.delta_T_from_files(
        source_path=args.source,
        target_path=args.target,
        voxel=args.voxel,
        icp_mode=args.icp_mode,
        max_corr=args.max_corr,
        print_stats=not args.no_stats,
    )

    np.set_printoptions(precision=6, suppress=True)
    print("\n=== ΔT (SOURCE → TARGET) ===")
    print(T)
    print(f"\n主角度（axis-angle）≈ {info['angle_deg']:.6f}°")
    print(f"軸  = {info['axis']}")
    print(f"Euler ZYX (deg) = "
          f"[yaw {info['yaw_deg']:.6f}, pitch {info['pitch_deg']:.6f}, roll {info['roll_deg']:.6f}]")
    print(f"t   = {info['t_cm']} cm")
    print(f"ICP: fitness={info['fitness']:.6f}, RMSE={info['rmse_mm']:.3f} mm")


if __name__ == "__main__":
    main()