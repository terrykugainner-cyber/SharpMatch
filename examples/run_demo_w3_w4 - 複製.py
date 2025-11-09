import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2, numpy as np, matplotlib.pyplot as plt
from sharp.shape_model import create
from sharp.shape_match import chamfer_coarse, topk_peaks
from sharp.visualize import draw_oriented_points
from sharp.edges import canny_edges
from sharp.distance import distance_from_edges, normalize_img

img = cv2.imread('data/demo/scene.png', cv2.IMREAD_GRAYSCALE)
"""
# a template ROI roughly around the circle/text
#roi = (290,210,60,60)
roi = (1105,1065,160,160)

model = create(img, roi, edge_threshold=(60,140), sampling_step=2, use_polarity=False)
"""
img = cv2.imread('data/demo/scene.png', cv2.IMREAD_GRAYSCALE)

# 直接讀取模板圖像文件
tpl_img = cv2.imread('data/demo/template.png', cv2.IMREAD_GRAYSCALE)
if tpl_img is None:
    raise SystemExit('無法載入模板圖像: data/demo/template.png')

# 將整個模板圖像作為 ROI（從模板圖像本身提取，ROI 就是整個圖像）
tpl_h, tpl_w = tpl_img.shape[:2]
roi = (0, 0, tpl_w, tpl_h)

# 從模板圖像創建模型（使用整個圖像作為 ROI）
model = create(tpl_img, roi, edge_threshold=(60,140), sampling_step=2, use_polarity=False)

# visualize model oriented points
tpl = model['template']

# 取得模板的寬度和高度
H, W = tpl.shape[:2]
# 計算中心點（以圖像座標系，原點在左上角）
center_x = W // 2
center_y = H // 2
print(f'模板尺寸: W={W}, H={H}', f'模板中心: ({center_x}, {center_y})')


vis_tpl = draw_oriented_points(tpl, model['points'], model['dirs'], step=max(1,int(len(model['points'])/800)+1))
cv2.imwrite('examples/out_w4_model_oriented.png', vis_tpl)

# compute distance transform on scene edges
edges = canny_edges(img, 60, 140)
dt = distance_from_edges(edges)
plt.figure(figsize=(6,4))
plt.title('Distance Transform (scene)')
plt.imshow(normalize_img(dt), cmap='gray')
plt.axis('off')
plt.tight_layout(); plt.savefig('examples/out_w3_dt.png'); plt.close()

# coarse chamfer match on full image
scoremap, (ox,oy) = chamfer_coarse(model, img, roi=None)
# visualize scoremap
# Since roi=None, ox=oy=0, so scoremap already matches dt size
score_full_vis = (scoremap*255).astype(np.uint8)
score_full_vis = cv2.applyColorMap(score_full_vis, cv2.COLORMAP_JET)
cv2.imwrite('examples/out_w3_chamfer_heat.png', score_full_vis)

# pick top-1 peak (only best match)
peaks = topk_peaks(scoremap, K=1, min_dist=10)
th, tw = tpl.shape[:2]
vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for (px,py,sc) in peaks:
    if(sc < 0.99):
        continue    
    # scoremap是從conv[tpl_h-1:, tpl_w-1:]提取的
    # scoremap[px,py]對應conv[tpl_h-1+py, tpl_w-1+px]
    # 在cv2.filter2D中，預設錨點在kernel中心(-1,-1)
    # 所以conv[tpl_h-1+py, tpl_w-1+px]表示模板中心在dt[tpl_h-1+py, tpl_w-1+px]
    # 模板左上角在dt中的位置需要減去模板中心到左上角的偏移
    # 轉換到原圖座標（考慮crop的偏移），模板左上角位置為：
    x = px + ox + tw - 1 - tw//2
    y = py + oy + th - 1 - th//2

    print(f'[MATCH] x={x}, y={y}, score={sc:.2f}')
    # 繪製模板框
    cv2.rectangle(vis, (x, y), (x+tw, y+th), (0,255,0), 2)
    cv2.putText(vis, f'{sc:.2f}', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(vis, f'{sc:.2f}', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
cv2.imwrite('examples/out_w3_topk.png', vis)
print('[DONE] W3~W4 demo generated: out_w4_model_oriented.png, out_w3_dt.png, out_w3_chamfer_heat.png, out_w3_topk.png')
