import cv2
import numpy as np

# =========================
# 十字检测（CMOS对准核心）
# =========================
def detect_cross_center(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi/180,
        threshold=80,
        minLineLength=50,
        maxLineGap=8
    )

    if lines is None:
        return None, None

    h_lines = []
    v_lines = []

    for x1,y1,x2,y2 in lines[:,0]:
        if abs(y2 - y1) < abs(x2 - x1):
            h_lines.append((x1,y1,x2,y2))
        else:
            v_lines.append((x1,y1,x2,y2))

    if len(h_lines)==0 or len(v_lines)==0:
        return None, None

    h = np.mean(h_lines, axis=0)
    v = np.mean(v_lines, axis=0)

    def line_eq(p):
        x1,y1,x2,y2 = p
        A = y2 - y1
        B = x1 - x2
        C = A*x1 + B*y1
        return A,B,C

    A1,B1,C1 = line_eq(h)
    A2,B2,C2 = line_eq(v)

    D = A1*B2 - A2*B1
    if abs(D) < 1e-6:
        return None, None

    x = (C1*B2 - C2*B1) / D
    y = (A1*C2 - A2*C1) / D

    return (x,y), (h,v)


# =========================
# PCA计算旋转误差
# =========================
def compute_orientation(lines):

    pts = []

    for x1,y1,x2,y2 in lines:
        pts.append([x1,y1])
        pts.append([x2,y2])

    pts = np.array(pts, dtype=np.float32)

    mean, eigenvectors = cv2.PCACompute(pts, mean=None)

    vx, vy = eigenvectors[0]

    angle = np.degrees(np.arctan2(vy, vx))

    return angle


# =========================
# 对焦检测（4角）
# =========================
def focus_analysis(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    regions = {
        "TL": gray[0:h//3, 0:w//3],
        "TR": gray[0:h//3, 2*w//3:w],
        "BL": gray[2*h//3:h, 0:w//3],
        "BR": gray[2*h//3:h, 2*w//3:w]
    }

    def score(r):
        return cv2.Laplacian(r, cv2.CV_64F).var()

    return {k: score(v) for k,v in regions.items()}