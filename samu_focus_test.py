import numpy as np
import matplotlib.pyplot as plt
import cv2  # 用于绘制 CMOS 图像上的粗线圆环

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

print("=== 沙姆相机：CMOS 上的长方形、同心黑线圆、中心虚线 ===")

# ==========================================
# 1. 参数定义（保持不变）
# ==========================================
f_mm = 8.0
sx_um = 4.0
sy_um = 4.0
cx = 640.0
cy = 512.0
theta_deg = 9.3
theta = np.radians(theta_deg)

phi_deg = 180.0 - 23.67
phi = np.radians(phi_deg)
tx, ty, tz = 0.0, 0.0, 105.72

print(f"物平面旋转角 φ = {phi_deg:.1f}°, 沙姆角 θ = {theta_deg:.1f}°")

#==========================================
# 1. 用户可调参数
# 同心圆半径（三个圆）
radii = [30, 60, 100]   # 外径可以根据需要调整，这里示例 r=6,10,14
thickness_pixel = 1   # 线宽 1 个像素（也可改为 5）


# ==========================================
# 2. 构建变换矩阵
# ==========================================
M_plane = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

R = np.array([[1, 0, 0],
              [0, np.cos(phi), -np.sin(phi)],
              [0, np.sin(phi),  np.cos(phi)]])
t = np.array([[tx], [ty], [tz]])
E = np.block([[R, t], [np.zeros((1, 3)), 1]])

P_scheimpflug = np.array([[f_mm, 0, 0, 0],
                          [0, f_mm / np.cos(theta), 0, 0],
                          [0, -np.sin(theta), np.cos(theta), 0]])

K_pixel = np.array([[1000.0 / sx_um, 0, cx],
                    [0, 1000.0 / sy_um, cy],
                    [0, 0, 1]])

H = K_pixel @ P_scheimpflug @ E @ M_plane
H_inv = np.linalg.inv(H)

print("单应性矩阵 H:")
print(H)
print("逆矩阵 H_inv:")
print(H_inv)

def pixel_to_object(u, v):
    u_arr = np.asarray(u, dtype=float).flatten()
    v_arr = np.asarray(v, dtype=float).flatten()
    n = len(u_arr)
    homo_pixel = np.vstack([u_arr, v_arr, np.ones(n)])
    homo_obj = H_inv @ homo_pixel
    X = homo_obj[0, :] / homo_obj[2, :]
    Z = homo_obj[1, :] / homo_obj[2, :]
    if np.isscalar(u) and np.isscalar(v):
        return X[0], Z[0]
    else:
        return X.reshape(np.shape(u)), Z.reshape(np.shape(v))

# ==========================================
# 3. 图像尺寸及绘图参数
# ==========================================
img_h, img_w = 1024, 1280
margin = 100
offset = 150

# 改为正方形（居中放置）
square_side = min(img_w, img_h) - 2 * margin
x1 = (img_w - square_side) // 2
y1 = (img_h - square_side) // 2
x2 = x1 + square_side
y2 = y1 + square_side

# 四个圆角位置
circle_centers = [
    (x1 + offset, y1 + offset),
    (x2 - offset, y1 + offset),
    (x2 - offset, y2 - offset),
    (x1 + offset, y2 - offset)
]

# ==========================================
# 4. 生成 CMOS 图像
# ==========================================
cmos_img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)

# 绘制正方形边框
cv2.rectangle(cmos_img, (x1, y1), (x2, y2), (0,0,0), thickness=1)

# 绘制四个位置的三个同心圆环
for (cu, cv) in circle_centers:
    for r in radii:
        cv2.circle(cmos_img, (cu, cv), r, (0,0,0), thickness=thickness_pixel)
# 绘制中心虚线（水平+垂直）
mid_u = (x1 + x2) // 2
mid_v = (y1 + y2) // 2
dash_len = 10
gap_len = 10
for u in range(x1, x2+1):
    if (u - x1) % (dash_len + gap_len) < dash_len:
        cmos_img[mid_v, u] = [0, 0, 0]
for v in range(y1, y2+1):
    if (v - y1) % (dash_len + gap_len) < dash_len:
        cmos_img[v, mid_u] = [0, 0, 0]

# 显示 CMOS 图像
plt.figure(figsize=(10, 6))
plt.imshow(cmos_img)
plt.title(f'CMOS 上的长方形、同心黑线圆 (线宽{thickness_pixel}px) 和中心虚线')
plt.xlabel('u (pixel)')
plt.ylabel('v (pixel)')
plt.gca().invert_yaxis()
plt.show()

# ==========================================
# 5. 物平面映射（带坐标轴，绘制黑色椭圆线）
# ==========================================
corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
X_corners, Z_corners = pixel_to_object(corners[:,0], corners[:,1])

side_lengths = []
for i in range(4):
    j = (i+1) % 4
    dx = X_corners[j] - X_corners[i]
    dz = Z_corners[j] - Z_corners[i]
    length = np.sqrt(dx**2 + dz**2)
    side_lengths.append(length)

mid_top = ((X_corners[0]+X_corners[1])/2, (Z_corners[0]+Z_corners[1])/2)
mid_bottom = ((X_corners[2]+X_corners[3])/2, (Z_corners[2]+Z_corners[3])/2)
height = np.sqrt((mid_bottom[0]-mid_top[0])**2 + (mid_bottom[1]-mid_top[1])**2)

print("\n=== 映射后四边形信息 ===")
print(f"四条边长 (mm): {side_lengths[0]:.3f}, {side_lengths[1]:.3f}, {side_lengths[2]:.3f}, {side_lengths[3]:.3f}")
print(f"高度 (上下边中点距离): {height:.3f} mm")

# 生成用于绘制椭圆的圆周点（足够平滑）
theta_sample = np.linspace(0, 2*np.pi, 1000)

plt.figure(figsize=(8, 6))
# 绘制矩形边界
plt.plot(X_corners[[0,1,2,3,0]], Z_corners[[0,1,2,3,0]], 'b-', linewidth=2, label='矩形边界')

# 绘制同心椭圆（黑色线，线宽选择 2.5，视觉上较粗）
for (cu, cv) in circle_centers:
    for r in radii:
        u_circle = cu + r * np.cos(theta_sample)
        v_circle = cv + r * np.sin(theta_sample)
        Xe, Ze = pixel_to_object(u_circle, v_circle)
        plt.plot(Xe, Ze, 'k-', linewidth=2.5)   # 黑色椭圆线

# 绘制中心虚线投影
num_samples = 200
u_hline = np.linspace(x1, x2, num_samples)
v_hline = np.full_like(u_hline, mid_v)
X_hline, Z_hline = pixel_to_object(u_hline, v_hline)
plt.plot(X_hline, Z_hline, 'g--', linewidth=1.5, label='水平中线投影')

v_vline = np.linspace(y1, y2, num_samples)
u_vline = np.full_like(v_vline, mid_u)
X_vline, Z_vline = pixel_to_object(u_vline, v_vline)
plt.plot(X_vline, Z_vline, 'g--', linewidth=1.5, label='垂直中线投影')

plt.title('CMOS 同心黑线圆及中心虚线投影到物平面')
plt.xlabel('X (mm)')
plt.ylabel('Z (mm)')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()

# ==========================================
# 6. 无坐标轴、无文字的物平面图像（纯图形，用于展示）
# ==========================================
plt.figure(figsize=(8, 6), facecolor='white')
plt.plot(X_corners[[0,1,2,3,0]], Z_corners[[0,1,2,3,0]], 'b-', linewidth=2)

for (cu, cv) in circle_centers:
    for r in radii:
        u_circle = cu + r * np.cos(theta_sample)
        v_circle = cv + r * np.sin(theta_sample)
        Xe, Ze = pixel_to_object(u_circle, v_circle)
        plt.plot(Xe, Ze, 'k-', linewidth=2.5)

plt.plot(X_hline, Z_hline, 'g--', linewidth=1.5)
plt.plot(X_vline, Z_vline, 'g--', linewidth=1.5)

plt.axis('off')
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.gca().set_aspect('equal')
plt.show()

# ==========================================
# 7. 生成真实物理尺寸的 A4 纯图形（1:1 毫米，黑色椭圆线）
# ==========================================
print("\n=== 生成真实尺寸（1:1 毫米）的物平面图形，固定在 A4 纸张 ===")

# 收集所有数据点计算范围
all_X = np.concatenate([X_corners, X_hline, X_vline])
all_Z = np.concatenate([Z_corners, Z_hline, Z_vline])
for (cu, cv) in circle_centers:
    for r in radii:
        u_circle = cu + r * np.cos(theta_sample)
        v_circle = cv + r * np.sin(theta_sample)
        Xe, Ze = pixel_to_object(u_circle, v_circle)
        all_X = np.concatenate([all_X, Xe])
        all_Z = np.concatenate([all_Z, Ze])

X_min, X_max = np.min(all_X), np.max(all_X)
Z_min, Z_max = np.min(all_Z), np.max(all_Z)
width_mm = X_max - X_min
height_mm = Z_max - Z_min
print(f"数据范围: X: [{X_min:.1f}, {X_max:.1f}] mm, 宽度 = {width_mm:.1f} mm")
print(f"           Z: [{Z_min:.1f}, {Z_max:.1f}] mm, 高度 = {height_mm:.1f} mm")

# A4 纸张尺寸
a4_width_mm = 210
a4_height_mm = 297
a4_width_inch = a4_width_mm / 25.4
a4_height_inch = a4_height_mm / 25.4

rel_w = width_mm / a4_width_mm
rel_h = height_mm / a4_height_mm
if rel_w > 1 or rel_h > 1:
    print("警告：图形尺寸超出 A4 纸张，将缩放以完整显示，但会破坏 1:1 比例。")
    scale = min(1/rel_w, 1/rel_h)
    rel_w *= scale
    rel_h *= scale
    width_mm *= scale
    height_mm *= scale
    print(f"缩放因子 = {scale:.3f}")
else:
    scale = 1.0

left = (1 - rel_w) / 2
bottom = (1 - rel_h) / 2

fig = plt.figure(figsize=(a4_width_inch, a4_height_inch), facecolor='white')
ax = fig.add_axes([left, bottom, rel_w, rel_h])

# 绘制内容
ax.plot(X_corners[[0,1,2,3,0]], Z_corners[[0,1,2,3,0]], 'b-', linewidth=2)

for (cu, cv) in circle_centers:
    for r in radii:
        u_circle = cu + r * np.cos(theta_sample)
        v_circle = cv + r * np.sin(theta_sample)
        Xe, Ze = pixel_to_object(u_circle, v_circle)
        ax.plot(Xe, Ze, 'k-', linewidth=2.5)

ax.plot(X_hline, Z_hline, 'k-', linewidth=1.5)
ax.plot(X_vline, Z_vline, 'k-', linewidth=1.5)

ax.set_xlim(X_min, X_max)
ax.set_ylim(Z_min, Z_max)
ax.set_aspect('equal')
ax.axis('off')

output_pdf = "object_plane_A4_true_scale.pdf"
plt.savefig(output_pdf, dpi=300, bbox_inches=None, pad_inches=0)
print(f"已保存 A4 纸张、真实尺寸图形到文件: {output_pdf}")
print("打印时请选择『实际大小』，图形周围的空白即为纸张留白。")
print(f"图形实际打印尺寸 = {width_mm:.1f} mm × {height_mm:.1f} mm")
plt.close(fig)

print("\n程序运行完成。")