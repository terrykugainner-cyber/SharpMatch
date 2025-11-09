import cv2
import numpy as np

def draw_box(img, x, y, w, h, text=None, thickness=2):
    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    # 確保座標和尺寸是 Python 原生整數類型（OpenCV 要求）
    x, y, w, h = int(x), int(y), int(w), int(h)
    cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), thickness)
    if text:
        text_y = int(max(0, y-5))
        cv2.putText(vis, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(vis, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return vis

def colorize_scoremap(score):
    s = score.copy()
    s = np.clip(s, 0, 1)
    s8 = (s*255).astype(np.uint8)
    cm = cv2.applyColorMap(s8, cv2.COLORMAP_JET)
    return cm

def draw_oriented_points(img, points, dirs, step=1, color=(0, 255, 0), arrow_length=5):
    """
    在圖像上繪製帶方向的點
    
    參數:
    - img: 輸入圖像
    - points: 點的位置陣列 (N, 2) 或 (N,) 包含 [x, y] 座標
    - dirs: 方向角度陣列 (N,) 弧度制
    - step: 採樣步長，每隔 step 個點繪製一個
    - color: 箭頭顏色 (B, G, R)
    - arrow_length: 箭頭長度
    """
    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    
    # 確保 points 是二維數組
    points = np.asarray(points)
    if points.ndim == 1 and len(points) == 2:
        points = points.reshape(1, 2)
    
    dirs = np.asarray(dirs)
    
    # 按步長採樣點並繪製
    for i in range(0, len(points), step):
        px, py = int(points[i][0]), int(points[i][1])
        
        # 確保座標在圖像範圍內
        if px < 0 or py < 0 or px >= vis.shape[1] or py >= vis.shape[0]:
            continue
        
        # 繪製點
        cv2.circle(vis, (px, py), 1, color, -1)
        
        # 將角度轉換為方向向量
        angle = dirs[i]
        dx = np.cos(angle) * arrow_length
        dy = np.sin(angle) * arrow_length
        
        # 計算箭頭終點
        end_x = int(px + dx)
        end_y = int(py + dy)
        
        # 繪製方向箭頭
        cv2.arrowedLine(vis, (px, py), (end_x, end_y), color, 1, tipLength=0.3)
    
    return vis

def draw_polygon(img, pts, color=(0,255,0), thickness=2):
    import numpy as np, cv2
    vis=img.copy();
    if vis.ndim==2: vis=cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    pts=np.array(pts, np.int32).reshape(-1,1,2)
    cv2.polylines(vis, [pts], True, color, thickness, cv2.LINE_AA)
    return vis
