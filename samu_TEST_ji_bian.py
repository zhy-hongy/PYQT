import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

print("=== 沙姆相机投影模型（顺序：先单应倾斜，再在实际平面上畸变） ===")

# ==========================================
# 传感器与图像尺寸参数
# ==========================================
sx_mm = 0.004       # 像素水平尺寸 (mm/pixel) → 每个像素在传感器上的物理宽度，例如 4μm
sy_mm = 0.004       # 像素垂直尺寸 (mm/pixel) → 每个像素的物理高度
cx = 640.0          # 图像主点水平坐标 (pixel) → 光轴与传感器交点（通常位于图像中心）
cy = 360.0          # 图像主点垂直坐标 (pixel) → 同样为中心点（1280×720 的一半）

# ==========================================
# 沙姆相机内参（倾斜模型）
# ==========================================
c_mm = 8.0          # 主距 (mm) → 透镜光学中心到虚拟参考平面（与光轴垂直）的垂直距离
                    # 类似于传统相机的焦距，但这里是“垂直距离”，用于虚拟针孔投影
d_mm = 8.2          # 单应分母参数 (mm) → 控制倾斜传感器平面沿光轴方向的位置偏移，
                    # 与主距 c 共同决定沙姆几何关系。在理想沙姆条件下 d = c / cos(tau)
tau_deg = 10.0      # 传感器倾斜角 (度) → 传感器平面相对于光轴（或者虚拟平面）的倾斜角度
tau = np.radians(tau_deg)   # 弧度表示
rho_deg = 0.0       # 方位角 (度) → 倾斜平面的旋转方向（绕光轴旋转），0° 表示沿水平方向倾斜
rho = np.radians(rho_deg)

# 畸变系数（施加在实际传感器平面）
k1, k2, k3 = 0.01, 0.001, 0.0
p1, p2 = 0.0005, 0.0005
# 注意：alpha1, alpha2 通常为0，因为已经倾斜了；若需要也可保留
alpha1, alpha2 = 0.0, 0.0


# ==========================================
# 外参：世界坐标系 → 相机坐标系变换
# ==========================================
# 以下参数定义了物体平面（棋盘格所在平面 Yw=0）相对于相机的位姿
phi_deg = 180.0 - 22.67   # 物平面绕世界 Xw 轴的旋转角度 (度)
phi = np.radians(phi_deg) # 弧度
tx, ty, tz = 20.0, 0.0, 218.55   # 平移向量 (mm) → 相机光心在世界坐标系中的位置
                                 # tx,ty 为水平/垂直偏移，tz 为沿 Z 轴距离（物距）
psi_deg = -8.0            # 相机绕自身 Yc 轴的旋转角度 (度) → 使相机光轴在世界水平面内偏转
psi = np.radians(psi_deg)

# 物平面网格
X_min, X_max, step_X = -20, 20, 5
Z_min, Z_max, step_Z = 0, 20, 5
X_grid = np.arange(X_min, X_max + step_X, step_X)
Z_grid = np.arange(Z_min, Z_max + step_Z, step_Z)
nX, nZ = len(X_grid), len(Z_grid)

print(f"倾斜角 τ = {tau_deg:.1f}°, 方位角 ρ = {rho_deg:.1f}°")
print(f"主距 c = {c_mm:.2f} mm, 单应参数 d = {d_mm:.2f} mm")
print(f"畸变施加位置: 实际传感器平面 (xt, yt)")
print(f"畸变系数: k1={k1}, k2={k2}, k3={k3}, p1={p1}, p2={p2}")

output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 2. 辅助函数（标定代码中的函数）
# ==========================================
def compute_tilt_homography(tau, rho, d, lens_type='perspective'):
    c_tau, s_tau = np.cos(tau), np.sin(tau)
    c_rho, s_rho = np.cos(rho), np.sin(rho)
    Rt = np.array([
        [c_rho**2*(1-c_tau) + c_tau, c_rho*s_rho*(1-c_tau),     s_rho*s_tau],
        [c_rho*s_rho*(1-c_tau),      s_rho**2*(1-c_tau) + c_tau, -c_rho*s_tau],
        [-s_rho*s_tau,               c_rho*s_tau,                c_tau]
    ])
    r11, r12, r13 = Rt[0,0], Rt[0,1], Rt[0,2]
    r21, r22, r23 = Rt[1,0], Rt[1,1], Rt[1,2]
    r31, r32, r33 = Rt[2,0], Rt[2,1], Rt[2,2]
    if lens_type == 'telecentric':
        delta = r11*r22 - r12*r21
        H = np.array([[r22/delta, -r12/delta, 0],
                      [-r21/delta, r11/delta, 0],
                      [0, 0, 1]])
    else:
        H = np.array([
            [r11*r33 - r13*r31, r21*r33 - r23*r31, 0],
            [r12*r33 - r13*r32, r22*r33 - r23*r32, 0],
            [r13/d,             r23/d,             r33]
        ])
    return H

def apply_distortion_on_xy(x, y, k1, k2, k3, p1, p2):
    """在任意平面坐标上施加径向+切向畸变"""
    r2 = x*x + y*y
    r4 = r2*r2
    r6 = r2*r4
    radial = 1.0 + k1*r2 + k2*r4 + k3*r6
    x_rad = x * radial
    y_rad = y * radial
    x_tan = 2.0*p1*x*y + p2*(r2 + 2.0*x*x)
    y_tan = p1*(r2 + 2.0*y*y) + 2.0*p2*x*y
    xd = x_rad + x_tan
    yd = y_rad + y_tan
    return xd, yd

# ==========================================
# 3. 生成网格点并计算投影
# ==========================================
X_mesh, Z_mesh = np.meshgrid(X_grid, Z_grid)
X_vec = X_mesh.flatten()
Z_vec = Z_mesh.flatten()
nPts = len(X_vec)

uv_distorted = np.zeros((nPts, 2))
uv_ideal = np.zeros((nPts, 2))
xt_all = np.zeros(nPts)
yt_all = np.zeros(nPts)

for i in range(nPts):
    X, Z = X_vec[i], Z_vec[i]
    # 世界 -> 相机坐标系
    Xc_raw = X + tx
    Yc_raw = -np.sin(phi) * Z + ty
    Zc_raw =  np.cos(phi) * Z + tz
    Xc = np.cos(psi) * Xc_raw + np.sin(psi) * Zc_raw
    Yc = Yc_raw
    Zc = -np.sin(psi) * Xc_raw + np.cos(psi) * Zc_raw
    Zc = max(Zc, 1e-8)

    # 步骤1：虚拟平面理想投影
    xu = c_mm * Xc / Zc
    yu = c_mm * Yc / Zc

    # 步骤2：倾斜单应（得到实际传感器平面上的理想点）
    H = compute_tilt_homography(tau, rho, d_mm, lens_type='perspective')
    pts_homo = H @ np.array([[xu], [yu], [1.0]])
    xt_ideal = pts_homo[0, 0] / pts_homo[2, 0]
    yt_ideal = pts_homo[1, 0] / pts_homo[2, 0]

    # 步骤3：在实际传感器平面上施加畸变
    if k1==0 and k2==0 and k3==0 and p1==0 and p2==0:
        xt_dist, yt_dist = xt_ideal, yt_ideal
    else:
        xt_dist, yt_dist = apply_distortion_on_xy(xt_ideal, yt_ideal, k1, k2, k3, p1, p2)

    xt_all[i] = xt_dist
    yt_all[i] = yt_dist

    # 像素坐标
    uv_distorted[i, 0] = xt_dist / sx_mm + cx
    uv_distorted[i, 1] = yt_dist / sy_mm + cy

    # 理想投影（无畸变）用于对比
    uv_ideal[i, 0] = xt_ideal / sx_mm + cx
    uv_ideal[i, 1] = yt_ideal / sy_mm + cy

uv_distorted_grid = uv_distorted.reshape(nZ, nX, 2)
uv_ideal_grid = uv_ideal.reshape(nZ, nX, 2)
xt_grid = xt_all.reshape(nZ, nX)
yt_grid = yt_all.reshape(nZ, nX)

# ==========================================
# 4. 绘图（与之前类似）
# ==========================================
fig1, axs = plt.subplots(1, 2, figsize=(12, 5))
fig1.suptitle(f'物平面 | 倾斜 τ={tau_deg:.1f}°, ρ={rho_deg:.1f}°')
ax = axs[0]
for i in range(nZ): ax.plot(X_grid, Z_grid[i]*np.ones(nX), 'k-', lw=1.5)
for j in range(nX): ax.plot(X_grid[j]*np.ones(nZ), Z_grid, 'k-', lw=1.5)
ax.scatter(X_vec, Z_vec, color='r', s=20)
ax.set_title('物平面棋盘格')
ax.set_xlabel('X (mm)'); ax.set_ylabel('Z (mm)')
ax.axis('equal'); ax.grid(True)

ax = axs[1]
for i in range(nZ): ax.plot(xt_grid[i, :], yt_grid[i, :], 'g-', lw=1.5)
for j in range(nX): ax.plot(xt_grid[:, j], yt_grid[:, j], 'g-', lw=1.5)
ax.scatter(xt_all, yt_all, color='r', s=20)
ax.set_title('畸变后传感器物理坐标 (xt, yt)')
ax.set_xlabel('xt (mm)'); ax.set_ylabel('yt (mm)')
ax.axis('equal'); ax.grid(True)
plt.tight_layout()

# 图2：像素平面理想 vs 畸变
plt.figure(figsize=(10, 8))
for i in range(nZ):
    plt.plot(uv_ideal_grid[i, :, 0], uv_ideal_grid[i, :, 1], 'b--', lw=1, label='理想 (无畸变)' if i==0 else "")
for j in range(nX):
    plt.plot(uv_ideal_grid[:, j, 0], uv_ideal_grid[:, j, 1], 'b--', lw=1)
for i in range(nZ):
    plt.plot(uv_distorted_grid[i, :, 0], uv_distorted_grid[i, :, 1], 'r-', lw=1.5, label='畸变后' if i==0 else "")
for j in range(nX):
    plt.plot(uv_distorted_grid[:, j, 0], uv_distorted_grid[:, j, 1], 'r-', lw=1.5)
plt.gca().invert_yaxis()
plt.title(f'像素平面对比 (虚线:理想, 实线:畸变)\n畸变施加在实际传感器平面')
plt.xlabel('u (pixel)'); plt.ylabel('v (pixel)')
plt.xlim(0, 1280); plt.ylim(720, 0)
plt.axis('equal'); plt.grid(True); plt.legend()
plt.tight_layout()

# 图3：棋盘格图像
fig3 = plt.figure(figsize=(8, 6), facecolor='white')
ax3 = fig3.add_axes([0, 0, 1, 1])
ax3.set_facecolor('white')
ax3.set_xlim(0, 1280)
ax3.set_ylim(720, 0)

for i in range(nZ - 1):
    for j in range(nX - 1):
        p1 = uv_distorted_grid[i, j]
        p2 = uv_distorted_grid[i, j+1]
        p3 = uv_distorted_grid[i+1, j+1]
        p4 = uv_distorted_grid[i+1, j]
        verts = [p1, p2, p3, p4]
        color = 'white' if (i + j) % 2 == 0 else 'black'
        poly = plt.Polygon(verts, closed=True, facecolor=color, edgecolor='black', linewidth=0.8)
        ax3.add_patch(poly)

ax3.axis('off')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(output_dir, f"scheimpflug_chessboard_{timestamp}.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='white')
print(f"已保存棋盘格图像: {save_path}")

plt.show()
print("\n程序运行完成。顺序：虚拟投影 → 单应倾斜 → 实际平面畸变 → 像素")