import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2, numpy as np, matplotlib.pyplot as plt
from sharp.pyramid import build_pyramid
from sharp.gradients import gradient_mag_dir
from sharp.edges import canny_edges
from sharp.ncc import ncc_scoremap, pyramid_ncc_search
from sharp.visualize import draw_box, colorize_scoremap
from sharp.shape_match import topk_peaks



img = cv2.imread('data/demo/scene.png', cv2.IMREAD_GRAYSCALE)
tpl = cv2.imread('data/demo/template.png', cv2.IMREAD_GRAYSCALE)

# 1) Pyramid
pyr = build_pyramid(img, num_levels=4, scale=0.5)
plt.figure()
for i, pi in enumerate(pyr):
    plt.subplot(1,4,i+1)
    plt.title(f'P{i} {pi.shape[1]}x{pi.shape[0]}')
    plt.imshow(pi, cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.savefig('examples/out_pyramid.png'); plt.close()

# 2) Gradients & Edges
mag, ang = gradient_mag_dir(img)
edg = canny_edges(img, 60, 140)
plt.figure(figsize=(12,3))
plt.subplot(1,3,1); plt.title('Image'); plt.imshow(img, cmap='gray'); plt.axis('off')
plt.subplot(1,3,2); plt.title('Grad Mag'); plt.imshow(mag, cmap='gray'); plt.axis('off')
plt.subplot(1,3,3); plt.title('Canny'); plt.imshow(edg, cmap='gray'); plt.axis('off')
plt.tight_layout(); plt.savefig('examples/out_grad_edges.png'); plt.close()

# 3) NCC heatmap
score = ncc_scoremap(tpl, img)
cm = colorize_scoremap(score)
cv2.imwrite('examples/out_ncc_heatmap.png', cm)

# 4) Pyramid search for single best match (demonstration)
x_best, y_best, sc_best = pyramid_ncc_search(tpl, img, roi=None, num_levels=4, scale=0.5)
print(f'[BEST MATCH from pyramid] x={x_best}, y={y_best}, score={sc_best:.3f}')

# 5) Extract multiple peaks from scoremap (multi-match)
K = 150  # Number of matches to find
min_dist = max(tpl.shape[0], tpl.shape[1]) // 2  # Minimum distance between matches
peaks = topk_peaks(score, K=K, min_dist=min_dist)

# 取得模板的寬度和高度
H, W = tpl.shape[:2]
vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# 計算中心點（以圖像座標系，原點在左上角）
center_x = W // 2
center_y = H // 2
print(f'模板尺寸: W={W}, H={H}', f'模板中心: ({center_x}, {center_y})')

# Draw all matches
for i, (px, py, sc_peak) in enumerate(peaks):
    if(sc_peak < 0.85):
        continue
    vis = draw_box(vis, px, py, W, H, text=f'#{i+1} {sc_peak:.3f}')
    print(f'[MATCH {i+1}] x={px}, y={py}, score={sc_peak:.3f}')

cv2.imwrite('examples/out_match.png', vis)
print(f'[DONE] Found {len(peaks)} matches, saved to examples/out_match.png')
