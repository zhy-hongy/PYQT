import numpy as np
import matplotlib.pyplot as plt

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

print("=== 沙姆相机模型对比：文章5步骤完整模型 vs 简易模型 ===\n")

# ==========================================
# 1. 参数定义（统一）
# ==========================================
f_mm = 8.0               # 焦距 (mm)
sx_um = 4.0              # 像素尺寸 (μm/pixel)
sy_um = 4.0
cx = 640.0               # 主点 (pixel)
cy = 360.0

# 沙姆角（传感器倾斜角）—— 用于两种模型
theta_deg = 10.0
theta = np.radians(theta_deg)

# 外参：物平面位姿（世界坐标系中的物平面绕X轴旋转 phi 角）
phi_deg = 180.0 - 23.67
phi = np.radians(phi_deg)
tx, ty, tz = 0.0, 0.0, 105.72   # 平移 (mm)

# 棋盘格参数（物平面局部坐标 X, Z，单位 mm，Yw=0）
X_min, X_max, step_X = -20, 20, 5
Z_min, Z_max, step_Z = 0, 20, 5
X_grid = np.arange(X_min, X_max + step_X, step_X)
Z_grid = np.arange(Z_min, Z_max + step_Z, step_Z)
nX, nZ = len(X_grid), len(Z_grid)

print(f"物平面旋转角 φ = {np.degrees(phi):.1f}°, 沙姆角 θ = {theta_deg:.1f}°")
print(f"网格点数: {nX} × {nZ}\n")

# ==========================================
# 2. 定义两种沙姆投影模型
# ==========================================

# ------------------------------
# 模型A：简易沙姆模型（原代码中方法）
# ------------------------------
def scheimpflug_simple(Xw, Yw, Zw, R, t, f, theta, sx_um, sy_um, cx, cy):
    """
    输入世界点 (N,3) ，外参 R(3,3), t(3,)
    输出像素坐标 (N,2)
    """
    N = Xw.shape[0]
    Xc = R[0,0]*Xw + R[0,1]*Yw + R[0,2]*Zw + t[0]
    Yc = R[1,0]*Xw + R[1,1]*Yw + R[1,2]*Zw + t[1]
    Zc = R[2,0]*Xw + R[2,1]*Yw + R[2,2]*Zw + t[2]

    D = np.cos(theta) * Zc - np.sin(theta) * Yc
    xp = f * Xc / D
    yp = (f / np.cos(theta)) * Yc / D

    u = xp / (sx_um/1000.0) + cx
    v = yp / (sy_um/1000.0) + cy
    return u, v

# ------------------------------
# 模型B：文章完整5步骤模型
# ------------------------------
def scheimpflug_full(Xw, Yw, Zw, R, t, f,
                     K1, K2, P1, P2,
                     tau, rho, d,
                     sx_um, sy_um, cx, cy):
    """
    完整模型，含畸变、倾斜单应性
    """
    N = Xw.shape[0]
    # 步骤1：世界 → 相机
    Xc = R[0,0]*Xw + R[0,1]*Yw + R[0,2]*Zw + t[0]
    Yc = R[1,0]*Xw + R[1,1]*Yw + R[1,2]*Zw + t[1]
    Zc = R[2,0]*Xw + R[2,1]*Yw + R[2,2]*Zw + t[2]

    # 步骤2：虚拟未倾斜平面针孔投影
    xu = f * Xc / Zc
    yu = f * Yc / Zc

    # 步骤3：畸变
    r2 = xu*xu + yu*yu
    r4 = r2 * r2
    radial = 1 + K1*r2 + K2*r4
    xd = xu * radial + (2*P1*xu*yu + P2*(r2 + 2*xu*xu))
    yd = yu * radial + (P1*(r2 + 2*yu*yu) + 2*P2*xu*yu)

    # 步骤4：倾斜单应性
    ct, st = np.cos(tau), np.sin(tau)
    cp, sp = np.cos(rho), np.sin(rho)
    R_t = np.array([
        [cp*cp*(1-ct) + ct,      cp*sp*(1-ct),          sp*st],
        [cp*sp*(1-ct),           sp*sp*(1-ct) + ct,    -cp*st],
        [-sp*st,                 cp*st,                 ct]
    ])
    r11, r12, r13 = R_t[0,0], R_t[0,1], R_t[0,2]
    r21, r22, r23 = R_t[1,0], R_t[1,1], R_t[1,2]
    r31, r32, r33 = R_t[2,0], R_t[2,1], R_t[2,2]

    Hp = np.array([
        [r11*r33 - r13*r31,   r21*r33 - r23*r31,   0],
        [r12*r33 - r13*r32,   r22*r33 - r23*r32,   0],
        [r13/d,               r23/d,               r33]
    ])

    pts_h = np.column_stack([xd, yd, np.ones(N)]).T
    pts_t_h = Hp @ pts_h
    xt = pts_t_h[0,:] / pts_t_h[2,:]
    yt = pts_t_h[1,:] / pts_t_h[2,:]

    # 步骤5：像素转换
    u = xt / (sx_um/1000.0) + cx
    v = yt / (sy_um/1000.0) + cy
    return u, v

# ==========================================
# 3. 生成世界坐标点（物平面 Yw=0）
# ==========================================
X_mesh, Z_mesh = np.meshgrid(X_grid, Z_grid)
Xw = X_mesh.flatten()
Zw = Z_mesh.flatten()
Yw = np.zeros_like(Xw)
nPts = len(Xw)

# 外参矩阵 R, t
R = np.array([
    [1, 0, 0],
    [0, np.cos(phi), -np.sin(phi)],
    [0, np.sin(phi),  np.cos(phi)]
])
t_vec = np.array([tx, ty, tz])

# ==========================================
# 4. 配置两种模型的参数（尽量可比）
# ==========================================
# 简易模型参数
# 直接使用 f, theta, R, t

# 完整模型参数
# 为了突出倾斜建模差异，先设畸变为0，d取有限值（如50mm），倾斜角tau = theta，方向角rho=0
K1, K2, P1, P2 = 0.0, 0.0, 0.0, 0.0    # 先无畸变，便于观察倾斜单应性差异
tau = theta          # 传感器倾斜角与简易模型一致
rho = 0.0            # 绕X轴倾斜（与简易模型一致）
d_mm = 100000000.0          # 出瞳到传感器距离 (mm) —— 有限值，与简易模型无限远出瞳形成对比

# 调用两种模型
u_simple, v_simple = scheimpflug_simple(Xw, Yw, Zw, R, t_vec, f_mm, theta,
                                        sx_um, sy_um, cx, cy)
u_full, v_full = scheimpflug_full(Xw, Yw, Zw, R, t_vec, f_mm,
                                  K1, K2, P1, P2,
                                  tau, rho, d_mm,
                                  sx_um, sy_um, cx, cy)

# ==========================================
# 5. 可视化对比
# ==========================================
# 转换为网格形状
u_simple_grid = u_simple.reshape(nZ, nX)
v_simple_grid = v_simple.reshape(nZ, nX)
u_full_grid = u_full.reshape(nZ, nX)
v_full_grid = v_full.reshape(nZ, nX)

plt.figure(figsize=(12, 5))

# 子图1：像素平面叠加对比
plt.subplot(1, 2, 1)
for i in range(nZ):
    plt.plot(u_simple_grid[i,:], v_simple_grid[i,:], 'b-', lw=1, label='简易模型' if i==0 else "")
for j in range(nX):
    plt.plot(u_simple_grid[:,j], v_simple_grid[:,j], 'b-', lw=1)
for i in range(nZ):
    plt.plot(u_full_grid[i,:], v_full_grid[i,:], 'r--', lw=1, label='文章完整模型' if i==0 else "")
for j in range(nX):
    plt.plot(u_full_grid[:,j], v_full_grid[:,j], 'r--', lw=1)
plt.gca().invert_yaxis()
plt.title(f'两种沙姆模型像素投影对比\n简易(蓝实线) vs 完整(红虚线)  θ={theta_deg}°, d={d_mm}mm')
plt.xlabel('u (pixel)'); plt.ylabel('v (pixel)')
plt.xlim(0,1280); plt.ylim(720,0)
plt.axis('equal'); plt.grid(True); plt.legend()

# 子图2：物理像平面坐标差异（xp, yp）—— 提取中间变量需要重新计算简易模型物理坐标
# 简易物理坐标计算
Xc_s = R[0,0]*Xw + R[0,1]*Yw + R[0,2]*Zw + t_vec[0]
Yc_s = R[1,0]*Xw + R[1,1]*Yw + R[1,2]*Zw + t_vec[1]
Zc_s = R[2,0]*Xw + R[2,1]*Yw + R[2,2]*Zw + t_vec[2]
D_s = np.cos(theta) * Zc_s - np.sin(theta) * Yc_s
xp_simple = f_mm * Xc_s / D_s
yp_simple = (f_mm / np.cos(theta)) * Yc_s / D_s

# 完整模型物理坐标（即倾斜传感器平面上的坐标，步骤4输出）
# 需要从完整模型内部提取，这里重新调用一次获取 xt, yt（不转像素）
def get_full_physical(Xw, Yw, Zw, R, t, f, K1, K2, P1, P2, tau, rho, d):
    N = Xw.shape[0]
    Xc = R[0,0]*Xw + R[0,1]*Yw + R[0,2]*Zw + t[0]
    Yc = R[1,0]*Xw + R[1,1]*Yw + R[1,2]*Zw + t[1]
    Zc = R[2,0]*Xw + R[2,1]*Yw + R[2,2]*Zw + t[2]
    xu = f * Xc / Zc
    yu = f * Yc / Zc
    r2 = xu*xu + yu*yu
    r4 = r2*r2
    radial = 1 + K1*r2 + K2*r4
    xd = xu * radial + (2*P1*xu*yu + P2*(r2 + 2*xu*xu))
    yd = yu * radial + (P1*(r2 + 2*yu*yu) + 2*P2*xu*yu)
    ct, st = np.cos(tau), np.sin(tau)
    cp, sp = np.cos(rho), np.sin(rho)
    R_t = np.array([
        [cp*cp*(1-ct) + ct,      cp*sp*(1-ct),          sp*st],
        [cp*sp*(1-ct),           sp*sp*(1-ct) + ct,    -cp*st],
        [-sp*st,                 cp*st,                 ct]
    ])
    r13, r23, r33 = R_t[0,2], R_t[1,2], R_t[2,2]
    Hp = np.array([
        [R_t[0,0]*r33 - r13*R_t[2,0],   R_t[1,0]*r33 - r23*R_t[2,0],   0],
        [R_t[0,1]*r33 - r13*R_t[2,1],   R_t[1,1]*r33 - r23*R_t[2,1],   0],
        [r13/d,                         r23/d,                         r33]
    ])
    pts_h = np.column_stack([xd, yd, np.ones(N)]).T
    pts_t_h = Hp @ pts_h
    xt = pts_t_h[0,:] / pts_t_h[2,:]
    yt = pts_t_h[1,:] / pts_t_h[2,:]
    return xt, yt

xt_full, yt_full = get_full_physical(Xw, Yw, Zw, R, t_vec, f_mm,
                                     K1, K2, P1, P2, tau, rho, d_mm)

xp_simple_grid = xp_simple.reshape(nZ, nX)
yp_simple_grid = yp_simple.reshape(nZ, nX)
xt_full_grid = xt_full.reshape(nZ, nX)
yt_full_grid = yt_full.reshape(nZ, nX)

plt.subplot(1, 2, 2)
for i in range(nZ):
    plt.plot(xp_simple_grid[i,:], yp_simple_grid[i,:], 'b-', lw=1, label='简易模型' if i==0 else "")
for j in range(nX):
    plt.plot(xp_simple_grid[:,j], yp_simple_grid[:,j], 'b-', lw=1)
for i in range(nZ):
    plt.plot(xt_full_grid[i,:], yt_full_grid[i,:], 'r--', lw=1, label='完整模型' if i==0 else "")
for j in range(nX):
    plt.plot(xt_full_grid[:,j], yt_full_grid[:,j], 'r--', lw=1)
plt.title(f'物理像平面坐标对比\n简易(xp,yp)  vs  完整(xt,yt)')
plt.xlabel('x (mm)'); plt.ylabel('y (mm)')
plt.axis('equal'); plt.grid(True); plt.legend()

plt.tight_layout()
plt.show()

# ==========================================
# 6. 定量差异统计
# ==========================================
# 像素坐标差异
diff_u = u_full - u_simple
diff_v = v_full - v_simple
max_diff_u = np.max(np.abs(diff_u))
max_diff_v = np.max(np.abs(diff_v))
rmse = np.sqrt(np.mean(diff_u**2 + diff_v**2))

print("\n=== 两种模型投影差异统计（像素） ===")
print(f"最大 |Δu| = {max_diff_u:.3f} pixel")
print(f"最大 |Δv| = {max_diff_v:.3f} pixel")
print(f"RMSE = {rmse:.3f} pixel")

# 物理坐标差异
diff_xp = xt_full - xp_simple
diff_yp = yt_full - yp_simple
max_dx = np.max(np.abs(diff_xp))
max_dy = np.max(np.abs(diff_yp))
print(f"\n物理像平面坐标差异 (mm):")
print(f"最大 |Δx| = {max_dx:.3e} mm")
print(f"最大 |Δy| = {max_dy:.3e} mm")

# ==========================================
# 7. 可选：开启畸变效果演示
# ==========================================
print("\n--- 附加演示：完整模型开启畸变效果（径向 K1=0.05）---")
K1_w = 0.05
u_full_dist, v_full_dist = scheimpflug_full(Xw, Yw, Zw, R, t_vec, f_mm,
                                            K1_w, K2, P1, P2,
                                            tau, rho, d_mm,
                                            sx_um, sy_um, cx, cy)

plt.figure(figsize=(8,6))
u_simple_grid = u_simple.reshape(nZ, nX)
v_simple_grid = v_simple.reshape(nZ, nX)
u_full_dist_grid = u_full_dist.reshape(nZ, nX)
v_full_dist_grid = v_full_dist.reshape(nZ, nX)

for i in range(nZ):
    plt.plot(u_simple_grid[i,:], v_simple_grid[i,:], 'b-', lw=1, label='简易模型' if i==0 else "")
for j in range(nX):
    plt.plot(u_simple_grid[:,j], v_simple_grid[:,j], 'b-', lw=1)
for i in range(nZ):
    plt.plot(u_full_dist_grid[i,:], v_full_dist_grid[i,:], 'r--', lw=1, label='完整模型+畸变' if i==0 else "")
for j in range(nX):
    plt.plot(u_full_dist_grid[:,j], v_full_dist_grid[:,j], 'r--', lw=1)
plt.gca().invert_yaxis()
plt.title(f'完整模型开启畸变(K1={K1_w})后对比\n简易模型(蓝) vs 完整含畸变(红虚线)')
plt.xlabel('u (pixel)'); plt.ylabel('v (pixel)')
plt.xlim(0,1280); plt.ylim(720,0)
plt.axis('equal'); plt.grid(True); plt.legend()
plt.show()

print("\n程序运行完成。")