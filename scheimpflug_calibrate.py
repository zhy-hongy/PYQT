#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
沙姆（Scheimpflug）相机标定 —— 两步法

流程：
  第一步：OpenCV 针孔模型标定，得到内参 K、畸变系数及每张图的外参初值；
  第二步：引入沙姆倾斜参数 (tau, rho, d) 与增强畸变模型，用光束法平差（Bundle Adjustment）
          做非线性最小二乘联合优化。

依赖：opencv-python, numpy, scipy
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

# ==================== 配置（请按实际修改）====================
CONFIG = {
  # 棋盘格内角点 (列, 行)
  "chessboard_size": (8, 11),
  # 方格边长 mm
  "square_size_mm": 1.5,
  # 标定图目录或通配符
  "images_path": r"C:\Users\1\Pictures\Camera Roll\*.jpg",
  # 像元尺寸 mm（用于将像素焦距换算为物理焦距 c）
  "pixel_size_x_mm": 0.004,
  "pixel_size_y_mm": 0.004,
  # 沙姆角初值（度）；未知时务必为 0，勿盲目设大角度
  "tau_init_deg": 0.0,
  "rho_init_deg": 0.0,
  # 第二步畸变模型: none | k1 | k2 | full | full_tilt
  "distortion_model": "full",
  # 分阶段 BA：先针孔对齐，再估计倾斜，最后联合畸变
  "ba_multistage": True,
  "show_corners": False,
  "save_result": True,
  "result_path": "calibration_data/scheimpflug_result.npz",
}
# ============================================================


# ---------------------------------------------------------------------------
# 几何与投影（沙姆倾斜模型）
# ---------------------------------------------------------------------------
def rodrigues_to_R(rvec: np.ndarray) -> np.ndarray:
  theta = np.linalg.norm(rvec)
  if theta < 1e-10:
    return np.eye(3)
  k = rvec / theta
  K = np.array(
    [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float64
  )
  return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def tilt_homography(tau: float, rho: float, d: float) -> np.ndarray:
  """倾斜变换单应矩阵 H（透视镜头）。"""
  ct, st = np.cos(tau), np.sin(tau)
  cr, sr = np.cos(rho), np.sin(rho)
  R_t = np.array(
    [
      [cr * cr * (1 - ct) + ct, cr * sr * (1 - ct), sr * st],
      [cr * sr * (1 - ct), sr * sr * (1 - ct) + ct, -cr * st],
      [-sr * st, cr * st, ct],
    ],
    dtype=np.float64,
  )
  r11, r12, r13 = R_t[0]
  r21, r22, r23 = R_t[1]
  r31, r32, r33 = R_t[2]
  return np.array(
    [
      [(r11*r33 - r13*r31)/r33, (r21*r33 - r23*r31)/r33, 0],
            [(r12*r33 - r13*r32)/r33, (r22*r33 - r23*r32)/r33, 0],
            [r13/d                  , r23/d                 , r33]
    ],
    dtype=np.float64,
  )


def project_scheimpflug(
  P_w: np.ndarray,
  rvec: np.ndarray,
  tvec: np.ndarray,
  *,
  cx_mm: float,
  cy_mm: float,
  d: float,
  tau: float,
  rho: float,
  k1: float,
  k2: float,
  k3: float,
  p1: float,
  p2: float,
  alpha1: float,
  alpha2: float,
  sx: float,
  sy: float,
  cx: float,
  cy: float,
) -> np.ndarray:
  """世界点 -> 像平面像素坐标（传感器平面坐标为毫米）。"""
  R = rodrigues_to_R(rvec)
  P_c = (R @ P_w.T).T + tvec
  Zc = np.maximum(P_c[:, 2], 1e-8)
  xu = cx_mm * P_c[:, 0] / Zc
  yu = cy_mm * P_c[:, 1] / Zc
  N = P_w.shape[0]
  H = tilt_homography(tau, rho, d)
  homo = H @ np.vstack((xu, yu, np.ones(N)))
  xt = homo[0] / homo[2]
  yt = homo[1] / homo[2]
  r2 = xt * xt + yt * yt
  r4, r6 = r2 * r2, r2 * r2 * r2
  radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
  xd = xt * radial + 2 * p1 * xt * yt + p2 * (r2 + 2 * xt * xt) + alpha1 * tau * xt
  yd = (
    yt * radial
    + p1 * (r2 + 2 * yt * yt)
    + 2 * p2 * xt * yt
    + alpha2 * tau * yt
  )
  u = xd / sx + cx
  v = yd / sy + cy
  return np.column_stack((u, v))


def convert_opencv_dist_to_sensor(dist: np.ndarray, c_ref_mm: float) -> Tuple[float, float, float, float, float]:
  """
  OpenCV 畸变定义在归一化像平面 (X/Z, Y/Z) 上；
  沙姆模型在传感器毫米坐标 (cx_mm*X/Z, …) 上，需按 c^n 缩放系数。
  """
  dist = np.asarray(dist, dtype=np.float64).ravel()
  k1 = float(dist[0]) / (c_ref_mm**2) if len(dist) > 0 else 0.0
  k2 = float(dist[1]) / (c_ref_mm**4) if len(dist) > 1 else 0.0
  p1 = float(dist[2]) / (c_ref_mm**2) if len(dist) > 2 else 0.0
  p2 = float(dist[3]) / (c_ref_mm**2) if len(dist) > 3 else 0.0
  k3 = float(dist[4]) / (c_ref_mm**6) if len(dist) > 4 else 0.0
  return k1, k2, k3, p1, p2


def _proj_kwargs_from_intrinsic(I: Dict) -> Dict:
  return dict(
    cx_mm=I["cx_mm"],
    cy_mm=I["cy_mm"],
    d=I["d"],
    tau=I["tau"],
    rho=I["rho"],
    k1=I["k1"],
    k2=I["k2"],
    k3=I["k3"],
    p1=I["p1"],
    p2=I["p2"],
    alpha1=I.get("alpha1", 0.0),
    alpha2=I.get("alpha2", 0.0),
    sx=I["sx"],
    sy=I["sy"],
    cx=I["cx"],
    cy=I["cy"],
  )


def compute_reproj_rmse(
  objpoints: List[np.ndarray],
  imgpoints: List[np.ndarray],
  intrinsic: Dict,
  extrinsics: List[Tuple[np.ndarray, np.ndarray]],
) -> float:
  kw = _proj_kwargs_from_intrinsic(intrinsic)
  errs = []
  for (objp, imgp), (rvec, tvec) in zip(zip(objpoints, imgpoints), extrinsics):
    obs = np.asarray(imgp, dtype=np.float64).reshape(-1, 2)
    pred = project_scheimpflug(objp.astype(np.float64), rvec, tvec, **kw)
    errs.append(np.linalg.norm(pred - obs, axis=1))
  if not errs:
    return float("inf")
  e = np.concatenate(errs)
  return float(np.sqrt(np.mean(e * e)))


def estimate_extrinsics_solvepnp(
  objpoints: List[np.ndarray],
  imgpoints: List[np.ndarray],
  K: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
  """用针孔 K（零畸变）为每张图估计外参，作为沙姆 BA 初值。"""
  dist0 = np.zeros(5, np.float64)
  out = []
  for objp, imgp in zip(objpoints, imgpoints):
    ok, rvec, tvec = cv2.solvePnP(
      objp, imgp, K, dist0, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
      rvec = np.zeros(3, np.float64)
      tvec = np.array([0.0, 0.0, 100.0], np.float64)
    out.append((rvec.flatten(), tvec.flatten()))
  return out


# ---------------------------------------------------------------------------
# 数据采集
# ---------------------------------------------------------------------------
def collect_corners(
  images_glob: str,
  pattern_size: Tuple[int, int],
  square_size_mm: float,
  show: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], Tuple[int, int]]:
  paths = sorted(set(glob.glob(images_glob)))
  if not paths:
    raise FileNotFoundError(f"未找到图片: {images_glob}")

  cols, rows = pattern_size
  # OpenCV calibrateCamera 要求 Point3f / Point2f（float32）
  objp = np.zeros((cols * rows, 3), np.float32)
  objp[:, :2] = (
    np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32) * square_size_mm
  )

  objpoints: List[np.ndarray] = []
  imgpoints: List[np.ndarray] = []
  img_size: Optional[Tuple[int, int]] = None
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

  for path in paths:
    img = cv2.imread(path)
    if img is None:
      continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_size is None:
      img_size = (gray.shape[1], gray.shape[0])

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ok:
      print(f"  跳过（未检测到角点）: {os.path.basename(path)}")
      continue

    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    objpoints.append(objp.copy())
    imgpoints.append(corners.astype(np.float32))  # shape (N, 1, 2)
    print(f"  有效: {os.path.basename(path)}")

    if show:
      vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
      cv2.drawChessboardCorners(vis, pattern_size, corners, ok)
      cv2.imshow("corners", vis)
      if cv2.waitKey(300) == 27:
        show = False

  cv2.destroyAllWindows()
  if len(objpoints) < 3:
    raise ValueError(f"有效视图不足（{len(objpoints)}），至少需要 3 张。")
  assert img_size is not None
  return objpoints, imgpoints, img_size


# ---------------------------------------------------------------------------
# 第一步：传统针孔标定（OpenCV）
# ---------------------------------------------------------------------------
def step1_pinhole_calibration(
  objpoints: List[np.ndarray],
  imgpoints: List[np.ndarray],
  image_size: Tuple[int, int],
) -> Dict:
  print("\n" + "=" * 55)
  print("第一步：OpenCV 针孔模型标定（初值）")
  print("=" * 55)

  rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
  )
  dist = dist.ravel()
  print(f"针孔重投影 RMS: {rms:.4f} px")
  print("K =\n", K)
  print("dist =", dist)

  return {
    "rms_pinhole": float(rms),
    "K": K,
    "dist": dist,
    "rvecs": [r.flatten() for r in rvecs],
    "tvecs": [t.flatten() for t in tvecs],
  }


def pinhole_to_scheimpflug_init(
  pinhole: Dict,
  objpoints: List[np.ndarray],
  imgpoints: List[np.ndarray],
  image_size: Tuple[int, int],
  sx: float,
  sy: float,
  tau_init_deg: float,
  rho_init_deg: float,
) -> Tuple[Dict, List[Tuple[np.ndarray, np.ndarray]]]:
  """将针孔标定结果映射为沙姆模型初值（含畸变域换算与外参重估）。"""
  K = pinhole["K"]
  dist = pinhole["dist"]
  fx, fy = K[0, 0], K[1, 1]
  cx, cy = K[0, 2], K[1, 2]

  cx_mm = float(fx * sx)
  cy_mm = float(fy * sy)
  c_mean = 0.5 * (cx_mm + cy_mm)
  tau0 = np.radians(tau_init_deg)
  rho0 = np.radians(rho_init_deg)
  d0 = c_mean / max(np.cos(tau0), 1e-6)

  k1, k2, k3, p1, p2 = convert_opencv_dist_to_sensor(dist, c_mean)

  intrinsic = {
    "cx_mm": cx_mm,
    "cy_mm": cy_mm,
    "c": c_mean,
    "d": d0,
    "tau": tau0,
    "rho": rho0,
    "cx": float(cx),
    "cy": float(cy),
    "k1": k1,
    "k2": k2,
    "k3": k3,
    "p1": p1,
    "p2": p2,
    "alpha1": 0.0,
    "alpha2": 0.0,
    "sx": sx,
    "sy": sy,
    "image_size": image_size,
    "K_pinhole": K,
  }
  extrinsics = estimate_extrinsics_solvepnp(objpoints, imgpoints, K)
  rmse0 = compute_reproj_rmse(objpoints, imgpoints, intrinsic, extrinsics)
  print(f"沙姆初值重投影（换算畸变后）: {rmse0:.4f} px")
  if rmse0 > 5.0 and abs(tau_init_deg) > 1.0:
    print("  提示：初值误差较大，已将 tau/rho 重置为 0 再估计外参。")
    intrinsic["tau"] = 0.0
    intrinsic["rho"] = 0.0
    intrinsic["d"] = c_mean
    rmse0 = compute_reproj_rmse(objpoints, imgpoints, intrinsic, extrinsics)
    print(f"  重置 tau=0 后: {rmse0:.4f} px")
  return intrinsic, extrinsics


# ---------------------------------------------------------------------------
# 第二步：Bundle Adjustment（scipy TRF + soft_l1）
# ---------------------------------------------------------------------------
def _parse_intrinsic_block(params: np.ndarray, model: str):
  """返回 (cx_mm, cy_mm, d, tau, rho, cx, cy, k1..a2, int_len)。"""
  z = 0.0
  if model == "none":
    cx_mm, cy_mm, d, tau, rho, cx, cy = params[:7]
    return cx_mm, cy_mm, d, tau, rho, cx, cy, z, z, z, z, z, z, z, 7
  if model == "k1":
    cx_mm, cy_mm, d, tau, rho, cx, cy, k1 = params[:8]
    return cx_mm, cy_mm, d, tau, rho, cx, cy, k1, z, z, z, z, z, z, 8
  if model == "k2":
    cx_mm, cy_mm, d, tau, rho, cx, cy, k1, k2 = params[:9]
    return cx_mm, cy_mm, d, tau, rho, cx, cy, k1, k2, z, z, z, z, z, 9
  if model == "full":
    cx_mm, cy_mm, d, tau, rho, cx, cy, k1, k2, p1, p2 = params[:11]
    return cx_mm, cy_mm, d, tau, rho, cx, cy, k1, k2, z, p1, p2, z, z, 11
  cx_mm, cy_mm, d, tau, rho, cx, cy, k1, k2, k3, p1, p2, a1, a2 = params[:14]
  return cx_mm, cy_mm, d, tau, rho, cx, cy, k1, k2, k3, p1, p2, a1, a2, 14


def ba_residuals(
  params: np.ndarray,
  objpoints: List[np.ndarray],
  imgpoints: List[np.ndarray],
  sx: float,
  sy: float,
  model: str,
) -> np.ndarray:
  vals = _parse_intrinsic_block(params, model)
  cx_mm, cy_mm, d, tau, rho, cx, cy = vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6]
  k1, k2, k3, p1, p2, a1, a2 = vals[7], vals[8], vals[9], vals[10], vals[11], vals[12], vals[13]
  int_len = vals[14]
  res = []
  for i, (objp, imgp) in enumerate(zip(objpoints, imgpoints)):
    obs = np.asarray(imgp, dtype=np.float64).reshape(-1, 2)
    base = int_len + i * 6
    pred = project_scheimpflug(
      objp.astype(np.float64),
      params[base : base + 3],
      params[base + 3 : base + 6],
      cx_mm=cx_mm,
      cy_mm=cy_mm,
      d=d,
      tau=tau,
      rho=rho,
      k1=k1,
      k2=k2,
      k3=k3,
      p1=p1,
      p2=p2,
      alpha1=a1,
      alpha2=a2,
      sx=sx,
      sy=sy,
      cx=cx,
      cy=cy,
    )
    res.append((pred - obs).ravel())
  return np.concatenate(res)


def _intrinsic_x0_lo_hi(
  I: Dict,
  w: int,
  h: int,
  model: str,
  *,
  fix_tau_rho: bool = False,
  tau_span_deg: float = 25.0,
) -> Tuple[List[float], List[float], List[float]]:
  cx_mm, cy_mm = I["cx_mm"], I["cy_mm"]
  c_lo, c_hi = 0.5 * min(cx_mm, cy_mm), 2.5 * max(cx_mm, cy_mm)
  tau_c, rho_c = I["tau"], I["rho"]
  if fix_tau_rho:
    tau_lo, tau_hi = tau_c - 1e-6, tau_c + 1e-6
    rho_lo, rho_hi = rho_c - 1e-6, rho_c + 1e-6
  else:
    span = np.radians(tau_span_deg)
    tau_lo, tau_hi = tau_c - span, tau_c + span
    rho_lo, rho_hi = rho_c - span, rho_c + span

  base = [cx_mm, cy_mm, I["d"], tau_c, rho_c, I["cx"], I["cy"]]
  lo = [c_lo, c_lo, 0.5 * min(cx_mm, cy_mm), tau_lo, rho_lo, 0, 0]
  hi = [c_hi, c_hi, 3.0 * max(cx_mm, cy_mm), tau_hi, tau_hi, w - 1, h - 1]

  if model == "none":
    pass
  elif model == "k1":
    base.append(I["k1"])
    lo.append(-1.0)
    hi.append(1.0)
  elif model == "k2":
    base.extend([I["k1"], I["k2"]])
    lo.extend([-1.0, -1.0])
    hi.extend([1.0, 1.0])
  elif model == "full":
    base.extend([I["k1"], I["k2"], I["p1"], I["p2"]])
    lo.extend([-1.0, -1.0, -0.5, -0.5])
    hi.extend([1.0, 1.0, 0.5, 0.5])
  else:
    base.extend([I["k1"], I["k2"], I["k3"], I["p1"], I["p2"], I["alpha1"], I["alpha2"]])
    lo.extend([-1.0, -1.0, -1.0, -0.5, -0.5, -5.0, -5.0])
    hi.extend([1.0, 1.0, 1.0, 0.5, 0.5, 5.0, 5.0])
  return base, lo, hi


def _params_to_intrinsic_extrinsics(
  x: np.ndarray,
  model: str,
  n_views: int,
  template: Dict,
) -> Tuple[Dict, List[Tuple[np.ndarray, np.ndarray]]]:
  v = _parse_intrinsic_block(x, model)
  cx_mm, cy_mm, d, tau, rho, cx, cy = v[0], v[1], v[2], v[3], v[4], v[5], v[6]
  int_len = v[14]
  I = dict(template)
  I.update(
    {
      "cx_mm": float(cx_mm),
      "cy_mm": float(cy_mm),
      "c": 0.5 * (float(cx_mm) + float(cy_mm)),
      "d": float(d),
      "tau": float(tau),
      "rho": float(rho),
      "cx": float(cx),
      "cy": float(cy),
      "k1": float(v[7]),
      "k2": float(v[8]),
      "k3": float(v[9]),
      "p1": float(v[10]),
      "p2": float(v[11]),
      "alpha1": float(v[12]),
      "alpha2": float(v[13]),
    }
  )
  ext = []
  for i in range(n_views):
    base = int_len + i * 6
    ext.append((x[base : base + 3].copy(), x[base + 3 : base + 6].copy()))
  return I, ext


def _run_ba_stage(
  objpoints: List[np.ndarray],
  imgpoints: List[np.ndarray],
  intrinsic: Dict,
  extrinsics: List[Tuple[np.ndarray, np.ndarray]],
  model: str,
  *,
  fix_tau_rho: bool = False,
  tau_span_deg: float = 25.0,
  verbose: int = 1,
) -> Tuple[Dict, List[Tuple[np.ndarray, np.ndarray]], object]:
  w, h = intrinsic["image_size"]
  sx, sy = intrinsic["sx"], intrinsic["sy"]
  x0, lo, hi = _intrinsic_x0_lo_hi(
    intrinsic, w, h, model, fix_tau_rho=fix_tau_rho, tau_span_deg=tau_span_deg
  )
  for rvec, tvec in extrinsics:
    x0.extend(rvec.tolist())
    x0.extend(tvec.tolist())
    lo.extend([-np.pi] * 3 + [-1e4, -1e4, 1.0])
    hi.extend([np.pi] * 3 + [1e4, 1e4, 1e4])
  x0 = np.clip(np.array(x0, np.float64), lo, hi)
  result = least_squares(
    ba_residuals,
    x0,
    bounds=(lo, hi),
    args=(objpoints, imgpoints, sx, sy, model),
    method="trf",
    loss="linear",
    max_nfev=2000,
    ftol=1e-10,
    xtol=1e-10,
    verbose=verbose,
  )
  I_out, ext_out = _params_to_intrinsic_extrinsics(
    result.x, model, len(objpoints), intrinsic
  )
  if model == "full_tilt":
    vals = _parse_intrinsic_block(result.x, model)
    I_out["alpha1"] = float(vals[12])
    I_out["alpha2"] = float(vals[13])
  return I_out, ext_out, result


def step2_bundle_adjustment(
  objpoints: List[np.ndarray],
  imgpoints: List[np.ndarray],
  intrinsic: Dict,
  extrinsics: List[Tuple[np.ndarray, np.ndarray]],
  distortion_model: str = "full",
  multistage: bool = True,
) -> Dict:
  print("\n" + "=" * 55)
  print("第二步：沙姆参数 + 光束法平差（Bundle Adjustment）")
  print("=" * 55)

  I, ext = intrinsic, extrinsics
  rmse0 = compute_reproj_rmse(objpoints, imgpoints, I, ext)
  print(f"BA 前 RMSE: {rmse0:.4f} px")

  stages: List[Tuple[str, str, bool, float]] = []
  if multistage:
    stages = [
      ("2a 对齐针孔", "none", True, 0.0),
      ("2b 估计倾斜", "none", False, 20.0),
      ("2c 径向畸变", "k1", False, 20.0),
      ("2d 联合优化", distortion_model, False, 25.0),
    ]
  else:
    stages = [("联合优化", distortion_model, False, 25.0)]

  result = None
  for label, model, fix_tau, tau_span in stages:
    print(f"\n--- 子阶段 {label} (模型={model}) ---")
    I, ext, result = _run_ba_stage(
      objpoints,
      imgpoints,
      I,
      ext,
      model,
      fix_tau_rho=fix_tau,
      tau_span_deg=tau_span,
      verbose=1,
    )
    rmse = compute_reproj_rmse(objpoints, imgpoints, I, ext)
    print(f"  本子阶段 RMSE: {rmse:.4f} px")

  assert result is not None
  rmse = compute_reproj_rmse(objpoints, imgpoints, I, ext)
  final_model = stages[-1][1]
  int_len = _parse_intrinsic_block(result.x, final_model)[14]

  out = {
    "cx_mm": I["cx_mm"],
    "cy_mm": I["cy_mm"],
    "c": I["c"],
    "d": I["d"],
    "tau": I["tau"],
    "rho": I["rho"],
    "cx": I["cx"],
    "cy": I["cy"],
    "k1": I["k1"],
    "k2": I["k2"],
    "k3": I["k3"],
    "p1": I["p1"],
    "p2": I["p2"],
    "alpha1": I.get("alpha1", 0.0),
    "alpha2": I.get("alpha2", 0.0),
    "sx": I["sx"],
    "sy": I["sy"],
    "image_size": intrinsic["image_size"],
    "rmse": rmse,
    "distortion_model": distortion_model,
    "optim_success": result.success,
    "raw_x": result.x,
    "rvecs": [],
    "tvecs": [],
  }
  for i in range(len(objpoints)):
    base = int_len + i * 6
    out["rvecs"].append(result.x[base : base + 3].copy())
    out["tvecs"].append(result.x[base + 3 : base + 6].copy())

  print(f"\n沙姆模型最终重投影 RMSE: {rmse:.4f} px")
  print(f"  焦距 cx_mm = {out['cx_mm']:.4f} mm, cy_mm = {out['cy_mm']:.4f} mm")
  print(f"  主距 d = {out['d']:.4f} mm")
  print(f"  倾斜角 tau = {np.degrees(out['tau']):.3f}°, 旋转角 rho = {np.degrees(out['rho']):.3f}°")
  print(f"  主点 cx = {out['cx']:.2f}, cy = {out['cy']:.2f} px")
  print(f"  畸变 k1={out['k1']:.6f}, k2={out['k2']:.6f}, p1={out['p1']:.6f}, p2={out['p2']:.6f}")
  return out


def save_calibration(result: Dict, path: str, pinhole_rms: float) -> None:
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  np.savez(
    path,
    cx_mm=result["cx_mm"],
    cy_mm=result["cy_mm"],
    c=result["c"],
    d=result["d"],
    tau=result["tau"],
    rho=result["rho"],
    cx=result["cx"],
    cy=result["cy"],
    k1=result["k1"],
    k2=result["k2"],
    k3=result["k3"],
    p1=result["p1"],
    p2=result["p2"],
    alpha1=result["alpha1"],
    alpha2=result["alpha2"],
    sx=result["sx"],
    sy=result["sy"],
    rmse=result["rmse"],
    pinhole_rms=pinhole_rms,
    raw_x=result["raw_x"],
  )
  txt = path.replace(".npz", ".txt")
  with open(txt, "w", encoding="utf-8") as f:
    f.write("沙姆相机标定结果（两步法）\n")
    f.write(f"针孔初值 RMS: {pinhole_rms:.6f} px\n")
    f.write(f"沙姆 BA RMS: {result['rmse']:.6f} px\n")
    f.write(f"c={result['c']}, d={result['d']}\n")
    f.write(f"tau(deg)={np.degrees(result['tau'])}, rho(deg)={np.degrees(result['rho'])}\n")
    f.write(f"cx={result['cx']}, cy={result['cy']}\n")
    f.write(
      f"k1={result['k1']}, k2={result['k2']}, k3={result['k3']}, "
      f"p1={result['p1']}, p2={result['p2']}\n"
    )
  print(f"结果已保存: {path}\n         {txt}")


def calibrate_two_step(config: Dict) -> Optional[Dict]:
  print("=" * 55)
  print("沙姆相机标定 —— 两步法")
  print("=" * 55)
  print(f"棋盘格: {config['chessboard_size']}, 方格: {config['square_size_mm']} mm")

  objpoints, imgpoints, img_size = collect_corners(
    config["images_path"],
    tuple(config["chessboard_size"]),
    config["square_size_mm"],
    show=config.get("show_corners", False),
  )
  print(f"有效视图: {len(objpoints)}")

  pinhole = step1_pinhole_calibration(objpoints, imgpoints, img_size)
  intrinsic, extrinsics = pinhole_to_scheimpflug_init(
    pinhole,
    objpoints,
    imgpoints,
    img_size,
    config["pixel_size_x_mm"],
    config["pixel_size_y_mm"],
    config.get("tau_init_deg", 0.0),
    config.get("rho_init_deg", 0.0),
  )

  scheimpflug = step2_bundle_adjustment(
    objpoints,
    imgpoints,
    intrinsic,
    extrinsics,
    distortion_model=config.get("distortion_model", "full"),
    multistage=config.get("ba_multistage", True),
  )

  if config.get("save_result", True):
    save_calibration(
      scheimpflug,
      config["result_path"],
      pinhole["rms_pinhole"],
    )

  return {"pinhole": pinhole, "scheimpflug": scheimpflug}


if __name__ == "__main__":
  res = calibrate_two_step(CONFIG)
  if res:
    print("\n标定完成。")
    print("提示：RMSE < 0.5 px 通常较好；若 tau 与机械倾角相差很大，请检查像元尺寸与棋盘格尺寸。")
