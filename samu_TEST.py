import numpy as np
import matplotlib.pyplot as plt

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False            # 用来正常显示负号

print("=== 1. 开始执行：沙姆相机 vs 针孔相机投影对比 ===")

# ==========================================
# 1. 参数定义
# ==========================================
f_mm = 8.0               # 焦距 (mm)
sx_um = 4.0              # 像素尺寸 (μm/pixel)
sy_um = 4.0
cx = 640.0               # 主点 (pixel)
cy = 360.0
theta_deg = 10.0         # 沙姆角 (度)
theta = np.radians(theta_deg)

# 外参：物平面位姿
phi_deg = 180.0 - 23.67  # 物平面绕世界 Xw 轴旋转角度 (度)
phi = np.radians(phi_deg)
tx, ty, tz = 0.0, 0.0, 105.72  # 平移 (mm)

# 棋盘格参数（物平面局部坐标 X, Z，单位 mm）
X_min, X_max, step_X = -20, 20, 5
Z_min, Z_max, step_Z = 0, 20, 5
X_grid = np.arange(X_min, X_max + step_X, step_X)
Z_grid = np.arange(Z_min, Z_max + step_Z, step_Z)
nX, nZ = len(X_grid), len(Z_grid)

print(f"物平面旋转角 φ = {phi_deg:.1f}°, 沙姆角 θ = {theta_deg:.1f}°")
print(f"网格点数: {nX} × {nZ}")

# ==========================================
# 2. 构建变换矩阵（用于矩阵形式投影）
# ==========================================
# 2.1 物平面提升矩阵 M_plane (4×3)
M_plane = np.array([
    [1, 0, 0],
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
], dtype=float)

# 2.2 外参矩阵 E (4×4)
R = np.array([
    [1, 0, 0],
    [0, np.cos(phi), -np.sin(phi)],
    [0, np.sin(phi),  np.cos(phi)]
])
t = np.array([[tx], [ty], [tz]])
E = np.block([[R, t], [np.zeros((1, 3)), 1]])

# 2.3 物理投影矩阵 (3×4)
P_scheimpflug = np.array([
    [f_mm, 0, 0, 0],
    [0, f_mm / np.cos(theta), 0, 0],
    [0, -np.sin(theta), np.cos(theta), 0]
])
P_pinhole = np.array([
    [f_mm, 0, 0, 0],
    [0, f_mm, 0, 0],
    [0, 0, 1, 0]
])

# 2.4 像素转换矩阵 K_pixel (3×3)
K_pixel = np.array([
    [1000.0 / sx_um, 0, cx],
    [0, 1000.0 / sy_um, cy],
    [0, 0, 1]
])

# 2.5 整体单应性矩阵 H (3×3)
H_scheimpflug = K_pixel @ P_scheimpflug @ E @ M_plane
H_pinhole = K_pixel @ P_pinhole @ E @ M_plane

print("\n=== 沙姆投影单应性矩阵 H_scheimpflug ===")
print(H_scheimpflug)

# ==========================================
# 3. 生成物平面网格点（齐次坐标形式）
# ==========================================
X_mesh, Z_mesh = np.meshgrid(X_grid, Z_grid)
X_vec = X_mesh.flatten()
Z_vec = Z_mesh.flatten()
nPts = len(X_vec)
plane_homo = np.vstack([X_vec, Z_vec, np.ones(nPts)])  # 3×nPts

# ==========================================
# 4. 方法一：显式公式计算像素坐标（逐点）
# ==========================================
uv_s_explicit = np.zeros((nPts, 2))
uv_p_explicit = np.zeros((nPts, 2))
xp_s_all, yp_s_all = np.zeros(nPts), np.zeros(nPts)
xp_p_all, yp_p_all = np.zeros(nPts), np.zeros(nPts)

for i in range(nPts):
    X, Z = X_vec[i], Z_vec[i]
    
    # 外参变换
    Xc = X + tx
    Yc = -np.sin(phi) * Z + ty
    Zc =  np.cos(phi) * Z + tz
    
    # 沙姆投影
    D = np.cos(theta) * Zc - np.sin(theta) * Yc
    xp_s = f_mm * Xc / D
    yp_s = (f_mm / np.cos(theta)) * Yc / D
    uv_s_explicit[i, 0] = (1000.0 / sx_um) * xp_s + cx
    uv_s_explicit[i, 1] = (1000.0 / sy_um) * yp_s + cy
    xp_s_all[i], yp_s_all[i] = xp_s, yp_s
    
    # 针孔投影
    xp_p = f_mm * Xc / Zc
    yp_p = f_mm * Yc / Zc
    uv_p_explicit[i, 0] = (1000.0 / sx_um) * xp_p + cx
    uv_p_explicit[i, 1] = (1000.0 / sy_um) * yp_p + cy
    xp_p_all[i], yp_p_all[i] = xp_p, yp_p

# ==========================================
# 5. 方法二：矩阵形式计算像素坐标（整体单应性）
# ==========================================
pixel_homo_s = H_scheimpflug @ plane_homo
uv_s_matrix = (pixel_homo_s[0:2, :] / pixel_homo_s[2, :]).T

pixel_homo_p = H_pinhole @ plane_homo
uv_p_matrix = (pixel_homo_p[0:2, :] / pixel_homo_p[2, :]).T

# ==========================================
# 6. 一致性验证
# ==========================================
max_diff_s = np.max(np.abs(uv_s_explicit - uv_s_matrix))
max_diff_p = np.max(np.abs(uv_p_explicit - uv_p_matrix))

print("\n=== 一致性验证（显式公式 vs 矩阵形式） ===")
print(f"沙姆投影最大差异: {max_diff_s:.3e} pixel")
print(f"针孔投影最大差异: {max_diff_p:.3e} pixel")

# ==========================================
# 7. 可视化
# ==========================================
# 变形恢复网格格式用于绘图
xp_s_grid = xp_s_all.reshape((nZ, nX))
yp_s_grid = yp_s_all.reshape((nZ, nX))
xp_p_grid = xp_p_all.reshape((nZ, nX))
yp_p_grid = yp_p_all.reshape((nZ, nX))

uv_s_reshaped = uv_s_explicit.reshape((nZ, nX, 2))
uv_p_reshaped = uv_p_explicit.reshape((nZ, nX, 2))

# 绘图 1：物平面与物理像平面对比
fig1, axs = plt.subplots(1, 3, figsize=(15, 5))
fig1.suptitle(f'物平面旋转 φ = {phi_deg:.1f}°, 沙姆角 θ = {theta_deg:.1f}°', fontsize=14)

# 子图1：物平面
ax = axs[0]
for i in range(nZ): ax.plot(X_grid, Z_grid[i]*np.ones(nX), 'k-', lw=1.5)
for j in range(nX): ax.plot(X_grid[j]*np.ones(nZ), Z_grid, 'k-', lw=1.5)
ax.scatter(X_vec, Z_vec, color='r', s=20)
ax.set_title('物平面棋盘格 (Yw = 0 平面)')
ax.set_xlabel('X (mm)'); ax.set_ylabel('Z (mm)')
ax.axis('equal'); ax.grid(True)

# 子图2：针孔投影物理像平面
ax = axs[1]
for i in range(nZ): ax.plot(xp_p_grid[i, :], yp_p_grid[i, :], 'b-', lw=1.5)
for j in range(nX): ax.plot(xp_p_grid[:, j], yp_p_grid[:, j], 'b-', lw=1.5)
ax.scatter(xp_p_all, yp_p_all, color='r', s=20)
ax.set_title('针孔投影物理像平面 (xp, yp)')
ax.set_xlabel('xp (mm)'); ax.set_ylabel('yp (mm)')
ax.axis('equal'); ax.grid(True)

# 子图3：沙姆投影物理像平面
ax = axs[2]
for i in range(nZ): ax.plot(xp_s_grid[i, :], yp_s_grid[i, :], 'g-', lw=1.5)
for j in range(nX): ax.plot(xp_s_grid[:, j], yp_s_grid[:, j], 'g-', lw=1.5)
ax.scatter(xp_s_all, yp_s_all, color='r', s=20)
ax.set_title('沙姆投影物理像平面 (xp, yp)')
ax.set_xlabel('xp (mm)'); ax.set_ylabel('yp (mm)')
ax.axis('equal'); ax.grid(True)


# 绘图 2：像素平面叠加对比
plt.figure(figsize=(8, 6))
# 绘制沙姆投影网格
for i in range(nZ):
    plt.plot(uv_s_reshaped[i, :, 0], uv_s_reshaped[i, :, 1], 'b-', lw=1, label='沙姆' if i==0 else "")
for j in range(nX):
    plt.plot(uv_s_reshaped[:, j, 0], uv_s_reshaped[:, j, 1], 'b-', lw=1)
# 绘制针孔投影网格
for i in range(nZ):
    plt.plot(uv_p_reshaped[i, :, 0], uv_p_reshaped[i, :, 1], 'r--', lw=1, label='针孔' if i==0 else "")
for j in range(nX):
    plt.plot(uv_p_reshaped[:, j, 0], uv_p_reshaped[:, j, 1], 'r--', lw=1)

plt.gca().invert_yaxis()  # 像素坐标系 V 轴向下
plt.title(f'投影差异叠加 (蓝色实线-沙姆, 红色虚线-针孔)\nφ={phi_deg:.1f}°, θ={theta_deg:.1f}°')
plt.xlabel('u (pixel)'); plt.ylabel('v (pixel)')
plt.xlim(0, 1280); plt.ylim(720, 0)
plt.axis('equal'); plt.grid(True); plt.legend()


# ==========================================
# 8. 第二部分：纯几何视角三路验证（无外参）
# ==========================================
print("\n" + "="*40)
print("=== 2. 开始执行：沙姆投影公式几何验证（纯相机系） ===")
f_mm = 50.0

# 生成相机系下测试点
grid_range, grid_step = 100, 50
Xc_vals = np.arange(-grid_range, grid_range + grid_step, grid_step)
Yc_vals = np.arange(-grid_range, grid_range + grid_step, grid_step) + 100
Xc_mesh, Yc_mesh = np.meshgrid(Xc_vals, Yc_vals)
Xc_vec2 = Xc_mesh.flatten()
Yc_vec2 = Yc_mesh.flatten()
Zc_vec2 = 800.0 * np.ones_like(Xc_vec2)
nPts2 = len(Xc_vec2)

# 方法一：显式公式
D_formula = np.cos(theta) * Zc_vec2 - np.sin(theta) * Yc_vec2
xp_formula = f_mm * Xc_vec2 / D_formula
yp_formula = (f_mm / np.cos(theta)) * Yc_vec2 / D_formula

# 方法二：纯空间几何直线方程求交点
t_geom = f_mm / D_formula
Qx = t_geom * Xc_vec2
Qy = t_geom * Yc_vec2
Qz = t_geom * Zc_vec2

P0 = np.array([0, 0, f_mm / np.cos(theta)]) # 芯片中心（主点）
ex = np.array([1, 0, 0])
ey = np.array([0, np.cos(theta), np.sin(theta)])

xp_geom, yp_geom = np.zeros(nPts2), np.zeros(nPts2)
for i in range(nPts2):
    Q = np.array([Qx[i], Qy[i], Qz[i]])
    delta = Q - P0
    xp_geom[i] = np.dot(delta, ex)
    yp_geom[i] = np.dot(delta, ey)

# 方法三：齐次矩阵形式
camera_homo2 = np.vstack([Xc_vec2, Yc_vec2, Zc_vec2, np.ones(nPts2)])
image_homo2 = P_scheimpflug @ camera_homo2
xp_matrix2 = image_homo2[0, :] / image_homo2[2, :]
yp_matrix2 = image_homo2[1, :] / image_homo2[2, :]

# 打印三种方法最大差异
diff_fg_x = np.max(np.abs(xp_formula - xp_geom))
diff_fg_y = np.max(np.abs(yp_formula - yp_geom))
diff_fm_x = np.max(np.abs(xp_formula - xp_matrix2))
diff_fm_y = np.max(np.abs(yp_formula - yp_matrix2))

print(f"| 显式公式 vs 几何法   | xp差异: {diff_fg_x:.3e} mm | yp差异: {diff_fg_y:.3e} mm |")
print(f"| 显式公式 vs 矩阵形式 | xp差异: {diff_fm_x:.3e} mm | yp差异: {diff_fm_y:.3e} mm |")

plt.show()

# ==========================================
# 9. 新增：传入图片并验证投影正确性
# ==========================================
print("\n=== 3. 验证投影正确性：图像 → 物平面反向映射 ===")

# ------------------------------
# 9.1 构造逆单应性矩阵
# ------------------------------
H_scheimpflug_inv = np.linalg.inv(H_scheimpflug)   # 用于像素 → 物平面
print("沙姆相机逆单应性矩阵 H_inv:")
print(H_scheimpflug_inv)

def pixel_to_object(u, v, H_inv):
    """将像素坐标 (u,v) 映射到物平面 (X, Z)"""
    homo_pixel = np.array([u, v, 1.0])
    homo_obj = H_inv @ homo_pixel
    X = homo_obj[0] / homo_obj[2]
    Z = homo_obj[1] / homo_obj[2]
    return X, Z

# ------------------------------
# 9.2 生成模拟图像（或读取外部图片）
# ------------------------------
# 方法A：利用已有的棋盘格投影生成“模拟图像”（像素网格图）
img_h, img_w = 720, 1280  # 与前面一致
sim_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

# 将物平面棋盘格投影到像素平面，得到每个网格点的像素坐标
for i in range(nZ):
    for j in range(nX):
        u = int(uv_s_reshaped[i, j, 0])
        v = int(uv_s_reshaped[i, j, 1])
        if 0 <= u < img_w and 0 <= v < img_h:
            sim_img[v, u] = [255, 255, 255]   # 白点标记网格交点

# 也可以绘制网格线（略）
# 这里只演示：用交点白点作为特征点

# ------------------------------
# 9.3 反向映射：从模拟图像的白点恢复物平面坐标
# ------------------------------
recovered_points = []
original_points = []  # 原始理论物点 (X, Z)
for i in range(nZ):
    for j in range(nX):
        u = int(uv_s_reshaped[i, j, 0])
        v = int(uv_s_reshaped[i, j, 1])
        if 0 <= u < img_w and 0 <= v < img_h:
            X_rec, Z_rec = pixel_to_object(u, v, H_scheimpflug_inv)
            recovered_points.append((X_rec, Z_rec))
            original_points.append((X_grid[j], Z_grid[i]))

recovered_points = np.array(recovered_points)
original_points = np.array(original_points)

# 计算重投影误差（在物平面上的欧氏距离）
errors = np.linalg.norm(recovered_points - original_points, axis=1)
max_err = np.max(errors)
mean_err = np.mean(errors)
print(f"\n反向映射误差统计（物平面，单位 mm）：")
print(f"  最大误差: {max_err:.3e} mm")
print(f"  平均误差: {mean_err:.3e} mm")

# ------------------------------
# 9.4 可视化反向映射结果
# ------------------------------
plt.figure(figsize=(10, 5))

# 子图1：模拟图像（白点为棋盘格交点）
plt.subplot(1, 2, 1)
plt.imshow(sim_img, cmap='gray')
plt.title('模拟图像（沙姆投影）\n白色交点 = 棋盘格投影位置')
plt.xlabel('u (pixel)')
plt.ylabel('v (pixel)')
plt.gca().invert_yaxis()

# 子图2：反向映射恢复的物平面点 vs 理论棋盘格
plt.subplot(1, 2, 2)
plt.scatter(original_points[:, 0], original_points[:, 1], 
            c='b', marker='o', s=30, label='理论物点')
plt.scatter(recovered_points[:, 0], recovered_points[:, 1], 
            c='r', marker='x', s=30, label='反向映射恢复点')
plt.title('物平面点对比（理论 vs 恢复）')
plt.xlabel('X (mm)')
plt.ylabel('Z (mm)')
plt.axis('equal')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print("\n程序运行完成。")