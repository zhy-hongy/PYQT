import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

print("=== 沙姆相机：六边形排列实心圆图案（物平面显示实体椭圆） ===")

# ==========================================
# 1. 相机成像参数
# ==========================================
f_mm = 8.0
sx_um = 4.0
sy_um = 4.0
cx = 640.0
cy = 516.0
theta_deg = 9.3
theta = np.radians(theta_deg)
phi_deg = 180.0 - 23.67
phi = np.radians(phi_deg)
tx, ty, tz = 0.0, 0.0, 105.72

print(f"物平面旋转角 φ = {phi_deg:.1f}°, 沙姆角 θ = {theta_deg:.1f}°")

# ==========================================
# 2. 单应性矩阵
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

def pixel_to_object(u, v):
    """批量转换像素坐标 -> 物平面坐标 (X, Z)"""
    u_arr = np.asarray(u, dtype=float).flatten()
    v_arr = np.asarray(v, dtype=float).flatten()
    n = len(u_arr)
    homo_pixel = np.vstack([u_arr, v_arr, np.ones(n)])
    homo_obj = H_inv @ homo_pixel
    X = homo_obj[0, :] / homo_obj[2, :]
    Z = homo_obj[1, :] / homo_obj[2, :]
    if np.isscalar(u) and np.isscalar(v):
        return X[0], Z[0]
    return X.reshape(np.shape(u)), Z.reshape(np.shape(v))

# ==========================================
# 3. CMOS 图像尺寸
# ==========================================
img_h, img_w = 1032, 1280

# ==========================================
# 4. 生成 CMOS 上的实心圆阵列（仅用于预览，不用于物平面绘图）
# ==========================================
# 用户可调参数 -------------------------------------------------
dot_diameter_px = 50      # 圆直径 (像素)
dot_radius_px = dot_diameter_px // 2   # 半径
spacing_px = 100          # 圆心距 (像素)
rows = 10                 # 行数
cols = 10                 # 列数

# 六边形排列（隔行偏移）
v_spacing = int(round(spacing_px * np.sqrt(3) / 2))
total_width = (cols - 1) * spacing_px + dot_diameter_px
total_height = (rows - 1) * v_spacing + dot_diameter_px
offset_x = (img_w - total_width) // 2
offset_y = (img_h - total_height) // 2

# 生成用于预览的 CMOS 图像（白色背景，黑色圆）
cmos_img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)

# 存储每个圆的圆心和半径（用于后续物平面映射）
circle_params = []   # 每个元素为 (cx_dot, cy_dot, radius_px)

for row in range(rows):
    x_offset = offset_x + (spacing_px // 2 if row % 2 == 1 else 0)
    y = offset_y + row * v_spacing
    for col in range(cols):
        cx_dot = x_offset + col * spacing_px
        cy_dot = y
        # 记录圆心和半径
        circle_params.append((cx_dot, cy_dot, dot_radius_px))
        # 仅用于预览：绘制实心圆（所有内部点）
        for dx in range(-dot_radius_px, dot_radius_px + 1):
            for dy in range(-dot_radius_px, dot_radius_px + 1):
                u = cx_dot + dx
                v = cy_dot + dy
                if (0 <= u < img_w and 0 <= v < img_h and
                    dx*dx + dy*dy <= dot_radius_px * dot_radius_px):
                    cmos_img[v, u] = [0, 0, 0]

# 显示 CMOS 预览图
plt.figure(figsize=(10, 6))
plt.imshow(cmos_img)
plt.title(f'CMOS 实心圆阵列（直径{dot_diameter_px}px, 圆心距{spacing_px}px, {rows}x{cols}）')
plt.xlabel('u (pixel)')
plt.ylabel('v (pixel)')
plt.gca().invert_yaxis()
plt.show()

# ==========================================
# 辅助函数：将单个圆映射为椭圆多边形（边界采样 + 填充）
# ==========================================
def get_ellipse_polygon(cx, cy, radius_px, num_points=300):
    """
    对给定的圆（像素坐标）进行边界密集采样，映射到物平面，
    返回椭圆多边形的 (X, Z) 坐标数组，可直接用于 fill()
    """
    angles = np.linspace(0, 2*np.pi, num_points)
    u_circle = cx + radius_px * np.cos(angles)
    v_circle = cy + radius_px * np.sin(angles)
    Xe, Ze = pixel_to_object(u_circle, v_circle)
    return Xe, Ze

# ==========================================
# 5. 物平面绘图（带坐标轴，使用 fill 实心填充）
# ==========================================
plt.figure(figsize=(8, 6))
for (cx_dot, cy_dot, r) in circle_params:
    Xe, Ze = get_ellipse_polygon(cx_dot, cy_dot, r, num_points=300)
    plt.fill(Xe, Ze, 'k', linewidth=0)   # 填充黑色，无边框

plt.title('物平面上的实体椭圆（光滑填充）')
plt.xlabel('X (mm)')
plt.ylabel('Z (mm)')
plt.axis('equal')
plt.grid(True)
plt.show()

# ==========================================
# 6. 无坐标轴的纯图形（同样使用 fill）
# ==========================================
plt.figure(figsize=(8, 6), facecolor='white')
for (cx_dot, cy_dot, r) in circle_params:
    Xe, Ze = get_ellipse_polygon(cx_dot, cy_dot, r, num_points=300)
    plt.fill(Xe, Ze, 'k', linewidth=0)

plt.axis('off')
plt.gca().set_aspect('equal')
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.show()

# ==========================================
# 7. 生成真实尺寸 A4 PDF（1:1 毫米）
# ==========================================
print("\n=== 生成真实尺寸（1:1 毫米）的物平面实体椭圆 PDF ===")

# 收集所有椭圆多边形的顶点，用于计算数据范围
all_X = []
all_Z = []
for (cx_dot, cy_dot, r) in circle_params:
    Xe, Ze = get_ellipse_polygon(cx_dot, cy_dot, r, num_points=300)
    all_X.extend(Xe)
    all_Z.extend(Ze)

X_min, X_max = np.min(all_X), np.max(all_X)
Z_min, Z_max = np.min(all_Z), np.max(all_Z)
width_mm = X_max - X_min
height_mm = Z_max - Z_min

print(f"数据范围: X: [{X_min:.1f}, {X_max:.1f}] mm, 宽度 = {width_mm:.1f} mm")
print(f"           Z: [{Z_min:.1f}, {Z_max:.1f}] mm, 高度 = {height_mm:.1f} mm")

# A4 纸张尺寸
a4_width_mm, a4_height_mm = 210, 297
a4_width_inch = a4_width_mm / 25.4
a4_height_inch = a4_height_mm / 25.4

rel_w = width_mm / a4_width_mm
rel_h = height_mm / a4_height_mm
scale = 1.0
if rel_w > 1 or rel_h > 1:
    scale = min(1/rel_w, 1/rel_h)
    rel_w *= scale
    rel_h *= scale
    width_mm *= scale
    height_mm *= scale
    print(f"警告：图形超出 A4，缩放因子 {scale:.3f}，打印尺寸将不是 1:1")
else:
    print("图形未超出 A4，将按 1:1 毫米实际尺寸打印")

left = (1 - rel_w) / 2
bottom = (1 - rel_h) / 2

fig = plt.figure(figsize=(a4_width_inch, a4_height_inch), facecolor='white')
ax = fig.add_axes([left, bottom, rel_w, rel_h])

# 绘制所有椭圆（填充）
for (cx_dot, cy_dot, r) in circle_params:
    Xe, Ze = get_ellipse_polygon(cx_dot, cy_dot, r, num_points=300)
    ax.fill(Xe, Ze, 'k', linewidth=0)

ax.set_xlim(X_min, X_max)
ax.set_ylim(Z_min, Z_max)
ax.set_aspect('equal')
ax.axis('off')

output_pdf = "./object_plane_实体椭圆_A4_光滑.pdf"
plt.savefig(output_pdf, dpi=300, bbox_inches=None, pad_inches=0)
print(f"已保存 A4 真实尺寸 PDF: {output_pdf}")
print(f"图形实际打印尺寸 = {width_mm:.1f} mm × {height_mm:.1f} mm")
plt.close(fig)

print("\n程序运行完成。")