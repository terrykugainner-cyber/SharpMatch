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


# 直接讀取模板圖像文件
tpl_img = cv2.imread('data/demo/template.png', cv2.IMREAD_GRAYSCALE)
if tpl_img is None:
    raise SystemExit('無法載入模板圖像: data/demo/template.png')

# 將整個模板圖像作為 ROI（從模板圖像本身提取，ROI 就是整個圖像）
tpl_h, tpl_w = tpl_img.shape[:2]
roi = (0, 0, tpl_w, tpl_h)

# 從模板圖像創建模型（使用整個圖像作為 ROI）
model = create(tpl_img, roi, edge_threshold=(60,140), sampling_step=2, use_polarity=False)
#model = create(tpl_img, roi, edge_threshold=(250,255), sampling_step=2, use_polarity=False)

# visualize model oriented points
tpl = model['template']

# 取得模板的寬度和高度
H, W = tpl.shape[:2]
# 計算中心點（以圖像座標系，原點在左上角）
center_x = W // 2
center_y = H // 2
print(f'模板尺寸: W={W}, H={H}', f'模板中心: ({center_x}, {center_y})')

model


vis_tpl = draw_oriented_points(tpl, model['points'], model['dirs'], step=max(1,int(len(model['points'])/800)+1))
cv2.imwrite('examples/out_w4_model_oriented.png', vis_tpl)

# compute distance transform on scene edges
edges = canny_edges(img, 60, 140)
#edges = canny_edges(img, 250, 255)
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
min_dist = max(tpl_w, tpl_h) // 2  # 約 85
peaks = topk_peaks(scoremap, K=200, min_dist=min_dist)
th, tw = tpl.shape[:2]
vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for (px,py,sc) in peaks:
    if(sc < 0.90):
        continue    
    
    # 注意：現在 px, py 是子像素精度的浮點數
    px_int = int(round(px))
    py_int = int(round(py))
    
    print(f'[SUBPIXEL] Peak location: ({px:.3f}, {py:.3f}) [integer: ({px_int}, {py_int})]')
    
    # 重新仔細分析座標轉換（修正後的邏輯）：
    # 1. topk_peaks 返回的 (px, py) 現在是子像素精度，格式是 (x, y) = (列, 行)
    # 2. scoremap 是從 conv[tpl_h//2:, tpl_w//2:] 提取的（已修正）
    #    所以 scoremap[py, px] 對應 conv[tpl_h//2 + py, tpl_w//2 + px]
    # 3. cv2.filter2D 預設錨點在 kernel 中心 (-1, -1)
    #    conv[row, col] 的值是當 kernel 中心對齊到 dt[row, col] 時的卷積結果
    # 4. 因此 conv[tpl_h//2 + py, tpl_w//2 + px] 表示 kernel 中心對齊到 dt[tpl_h//2 + py, tpl_w//2 + px]
    # 5. dt 是 NumPy 陣列，dt[row, col] 對應圖像座標 (x=col, y=row)
    #    所以 kernel 中心在圖像座標中是 (x = tpl_w//2 + px, y = tpl_h//2 + py)
    # 6. kernel 就是模板的點分佈，kernel 中心就是模板中心
    #    因此模板中心在圖像座標中是 (tpl_w//2 + px + ox, tpl_h//2 + py + oy)
    
    # 計算模板中心在圖像中的位置（修正後的計算）
    template_center_x = tpl_w // 2 + px + ox
    template_center_y = tpl_h // 2 + py + oy
    
    # 計算模板左上角位置
    # 模板中心在模板座標系中的位置（相對於左上角 (0,0)）
    # 對於尺寸為 N 的模板，像素索引是 0 到 N-1，中心索引是 (N-1)/2
    # 對於整數除法，我們使用 (N-1)//2 來獲得更精確的中心
    # 但實際上，tpl_w//2 和 (tpl_w-1)//2 對於偶數寬度差 1
    # 根據實際測試，我們需要調整計算方式
    
    # 問題分析：如果紅色點向左上方偏移，說明 x, y 計算得太小
    # 可能原因：模板中心的計算或從中心到左上角的偏移計算有誤
    
    # 嘗試多種計算方式來找出正確的座標轉換
    # 方法1：使用 tpl_w//2（原始方法）
    center_offset_x_method1 = tpl_w // 2  # 對於 168，這是 84
    center_offset_y_method1 = tpl_h // 2  # 對於 170，這是 85
    
    # 方法2：使用 (tpl_w-1)//2（更精確的中心）
    center_offset_x_method2 = (tpl_w - 1) // 2  # 對於 168，這是 83
    center_offset_y_method2 = (tpl_h - 1) // 2  # 對於 170，這是 84
    
    # 方法3：直接使用 px, py 作為左上角（測試用）
    # 這是不正確的，但可以幫助理解問題
    
    x_method1 = template_center_x - center_offset_x_method1
    y_method1 = template_center_y - center_offset_y_method1
    x_method2 = template_center_x - center_offset_x_method2
    y_method2 = template_center_y - center_offset_y_method2
    
    # 關鍵問題：從模板中心計算左上角
    # 模板中心已經確認正確（黃色和青色圓點重合）
    # 但紅色點仍然向左上方偏移，說明左上角計算錯誤
    
    # 分析：model['points'] 的範圍是 x=[16.0, 142.0], y=[27.0, 152.0]
    # 模板尺寸是 168x170，所以點的分佈不在整個模板範圍內
    # 但這是正常的，因為邊緣點不會遍佈整個模板
    
    # 重新思考：對於 OpenCV 的 cv2.filter2D，預設錨點在 kernel 中心
    # kernel 大小是 (tpl_h, tpl_w) = (170, 168)
    # kernel 中心在 kernel 中的索引是 (tpl_h//2, tpl_w//2) = (85, 84)
    # 但在 OpenCV 的座標系統中，kernel 中心相對於左上角 (0,0) 的偏移是 (tpl_w//2, tpl_h//2) = (84, 85)
    
    # 問題可能在於：模板中心在模板座標系中的精確位置
    # 對於寬度 w，像素索引是 0 到 w-1，中心索引應該是 (w-1)/2
    # 對於 168，中心索引是 83.5，對應整數是 83 或 84
    # 對於 170，中心索引是 84.5，對應整數是 84 或 85
    
    # 根據實際測試，方法2 已經比方法1 大 1，但仍然不夠
    # 可能需要考慮 cv2.filter2D 的實際行為
    
    # 嘗試方法3：直接使用實際的模板中心偏移
    # 檢查 model['points'] 的質心，看看實際中心在哪裡
    model_center_actual_x = model['points'][:, 0].mean()
    model_center_actual_y = model['points'][:, 1].mean()
    print(f'[DEBUG] Actual model points center in template coord: ({model_center_actual_x:.2f}, {model_center_actual_y:.2f})')
    print(f'[DEBUG] Template geometric center: ({tpl_w/2:.2f}, {tpl_h/2:.2f})')
    print(f'[DEBUG] Template geometric center (int): ({tpl_w//2}, {tpl_h//2})')
    
    # 關鍵發現：實際模型點質心 (83.70, 83.08) 與幾何中心 (84.00, 85.00) 有差異
    # 特別是 Y 方向差異約 1.92 像素！
    # 
    # cv2.filter2D 使用的是模板的幾何中心作為錨點，但模型點的質心偏向左上方
    # 這可能解釋了為什麼紅色點（model['points']）向左上方偏移
    
    # 方法5：使用實際模型點質心作為偏移量（最準確的方法）
    # 使用 round() 而不是 int() 來保留精度
    x_method5_int = int(template_center_x - model_center_actual_x)  # 996 - 83.70 = 912.30 -> 912
    y_method5_int = int(template_center_y - model_center_actual_y)  # 778 - 83.08 = 694.92 -> 694
    x_method5_round = round(template_center_x - model_center_actual_x)  # 912.30 -> 912
    y_method5_round = round(template_center_y - model_center_actual_y)  # 694.92 -> 695
    
    # 方法8：考慮到紅色點仍然偏左上，可能需要進一步補償
    # 如果質心偏左上，而我們想要紅色點對齊，需要讓左上角更偏右下
    # 補償量 = 質心偏移量（取負，因為要反向補償）
    compensation_x = -(model_center_actual_x - tpl_w / 2)  # -(-0.30) = 0.30
    compensation_y = -(model_center_actual_y - tpl_h / 2)  # -(-1.92) = 1.92
    x_method8 = round(template_center_x - model_center_actual_x + compensation_x)  # 912.30 + 0.30 = 912.60 -> 913
    y_method8 = round(template_center_y - model_center_actual_y + compensation_y)  # 694.92 + 1.92 = 696.84 -> 697
    
    # 或者更直接：使用幾何中心計算，然後加上補償
    x_method9 = round(template_center_x - tpl_w / 2 + compensation_x)  # 996 - 84 + 0.30 = 912.30 -> 912
    y_method9 = round(template_center_y - tpl_h / 2 + compensation_y)  # 778 - 85 + 1.92 = 694.92 -> 695
    
    # 最簡單的方式：直接使用浮點數計算後四捨五入
    x_method10 = round(template_center_x - model_center_actual_x)  # 912.30 -> 912
    y_method10 = round(template_center_y - model_center_actual_y)  # 694.92 -> 695
    
    print(f'[ANALYSIS] Model center offset from geometric: X={model_center_actual_x - tpl_w/2:.2f}, Y={model_center_actual_y - tpl_h/2:.2f}')
    print(f'[ANALYSIS] Compensation needed: X={compensation_x:.2f}, Y={compensation_y:.2f}')
    print(f'[DEBUG] Method 5 (int): top_left=({x_method5_int}, {y_method5_int})')
    print(f'[DEBUG] Method 5 (round): top_left=({x_method5_round}, {y_method5_round})')
    print(f'[DEBUG] Method 8 (with compensation): top_left=({x_method8}, {y_method8})')
    print(f'[DEBUG] Method 9 (geometric + compensation): top_left=({x_method9}, {y_method9})')
    print(f'[DEBUG] Method 10 (round, same as 5 round): top_left=({x_method10}, {y_method10})')
    
    # FINAL FIX: Use geometric center instead of actual model points centroid
    # cv2.filter2D uses the geometric center of the kernel as anchor, not the centroid
    # So we should use geometric center (tpl_w/2, tpl_h/2) for top-left calculation
    # template_center = (1004.0, 768.0)
    # geometric_center = (84.0, 85.0)
    # top_left = (1004.0 - 84.0, 768.0 - 85.0) = (920.0, 683.0) ✓
    x = round(template_center_x - tpl_w / 2)  # 1004.0 - 84.0 = 920.0 -> 920
    y = round(template_center_y - tpl_h / 2)  # 768.0 - 85.0 = 683.0 -> 683
    
    print(f'[MATCH] px={px}, py={py}, template_center=({template_center_x}, {template_center_y}), score={sc:.2f}')
    print(f'[DEBUG] Method 1 (tpl_w//2): top_left=({x_method1}, {y_method1})')
    print(f'[DEBUG] Method 2 ((tpl_w-1)//2): top_left=({x_method2}, {y_method2})')
    print(f'[DEBUG] Template size: {tpl_w}x{tpl_h}, offset=({ox}, {oy})')
    print(f'[DEBUG] Center offsets: method1=({center_offset_x_method1}, {center_offset_y_method1}), method2=({center_offset_x_method2}, {center_offset_y_method2})')
    print(f'[DEBUG] Model points range: x=[{model["points"][:, 0].min():.1f}, {model["points"][:, 0].max():.1f}], y=[{model["points"][:, 1].min():.1f}, {model["points"][:, 1].max():.1f}]')
    print(f'[DEBUG] Scoremap shape: {scoremap.shape}, peak at scoremap[{py:.3f}, {px:.3f}] (subpixel)')
    print(f'[DEBUG] Expected conv position: [{tpl_h//2 + py:.3f}, {tpl_w//2 + px:.3f}] (subpixel, CORRECTED)')
    
    # 關鍵驗證：scoremap 峰值在 scoremap[py, px] = scoremap[行, 列]
    # scoremap 是從 conv[tpl_h//2:, tpl_w//2:] 提取的（已修正）
    # 所以 scoremap[py, px] 對應 conv[tpl_h//2 + py, tpl_w//2 + px]
    # conv[row, col] 表示 kernel 中心對齊到 dt[row, col]
    # dt[row, col] 在圖像座標中是 (x=col, y=row)
    conv_row = tpl_h // 2 + py  # CORRECTED: use tpl_h//2 instead of tpl_h-1
    conv_col = tpl_w // 2 + px  # CORRECTED: use tpl_w//2 instead of tpl_w-1
    dt_x = conv_col  # dt 的列索引對應圖像 x 座標
    dt_y = conv_row  # dt 的行索引對應圖像 y 座標
    
    print(f'[VERIFY] Scoremap peak ({px:.3f}, {py:.3f}) -> conv[{conv_row:.3f}, {conv_col:.3f}] -> dt image coord ({dt_x:.3f}, {dt_y:.3f})')
    print(f'[VERIFY] Calculated template_center=({template_center_x}, {template_center_y})')
    print(f'[VERIFY] Difference: ({dt_x - template_center_x}, {dt_y - template_center_y})')


    """    
    # 關鍵驗證：與 NCC 參考結果比較
    # NCC 答案是 x=920, y=683（已知正確答案）
    ncc_top_left_x = 920
    ncc_top_left_y = 683
    ncc_template_center_x = ncc_top_left_x + tpl_w / 2  # 920 + 84 = 1004
    ncc_template_center_y = ncc_top_left_y + tpl_h / 2  # 683 + 85 = 768
    
    # 如果 NCC 是正確的，那麼在 scoremap/dt 座標系中：
    # 模板中心應該在 dt 的 (ncc_template_center_x, ncc_template_center_y)
    # 對應到 conv 的位置
    expected_conv_col_ncc = int(ncc_template_center_x)  # 1004
    expected_conv_row_ncc = int(ncc_template_center_y)   # 768
    # 對應到 scoremap 的位置（修正後的計算）
    # scoremap[py, px] 對應 conv[tpl_h//2 + py, tpl_w//2 + px]
    # 所以：px = conv_col - tpl_w//2, py = conv_row - tpl_h//2
    expected_scoremap_px_ncc = expected_conv_col_ncc - tpl_w // 2  # 1004 - 84 = 920
    expected_scoremap_py_ncc = expected_conv_row_ncc - tpl_h // 2   # 768 - 85 = 683
    
    print(f'[NCC_REFERENCE] NCC top-left (correct answer): ({ncc_top_left_x}, {ncc_top_left_y})')
    print(f'[NCC_REFERENCE] NCC template center: ({ncc_template_center_x:.1f}, {ncc_template_center_y:.1f})')
    print(f'[NCC_REFERENCE] Should map to conv[{expected_conv_row_ncc}, {expected_conv_col_ncc}]')
    print(f'[NCC_REFERENCE] Should map to scoremap[{expected_scoremap_py_ncc}, {expected_scoremap_px_ncc}]')
    print(f'[COMPARISON] Chamfer template center: ({template_center_x}, {template_center_y})')
    print(f'[COMPARISON] Center difference: X={ncc_template_center_x - template_center_x:.1f}, Y={ncc_template_center_y - template_center_y:.1f}')
    print(f'[COMPARISON] Actual scoremap peak: ({px:.3f}, {py:.3f}) [subpixel]')
    print(f'[COMPARISON] Scoremap difference: X={px - expected_scoremap_px_ncc:.3f}, Y={py - expected_scoremap_py_ncc:.3f}')
    print(f'[COMPARISON] Top-left difference: X={x - ncc_top_left_x}, Y={y - ncc_top_left_y}')
    
    # 繪製 NCC 參考位置以便視覺比較
    cv2.circle(vis, (ncc_top_left_x, ncc_top_left_y), 10, (0, 255, 0), 2)  # 綠色大圓圈：NCC 正確位置
    cv2.circle(vis, (int(ncc_template_center_x), int(ncc_template_center_y)), 10, (255, 165, 0), 2)  # 橙色大圓圈：NCC 模板中心
    """    


    # 驗證：繪製多個標記點以便視覺檢查
    cv2.circle(vis, (int(round(template_center_x)), int(round(template_center_y))), 8, (255, 255, 0), -1)  # 黃色大圓點：計算出的模板中心
    cv2.circle(vis, (int(round(dt_x)), int(round(dt_y))), 8, (0, 255, 255), -1)  # 青色大圓點：從 scoremap 直接映射的 dt 位置
    cv2.circle(vis, (x, y), 5, (255, 0, 255), -1)  # 洋紅色圓點：計算出的左上角
    cv2.circle(vis, (px_int, py_int), 5, (0, 255, 0), -1)  # 綠色圓點：scoremap 峰值位置（整數版本，注意：這個可能不在正確的圖像範圍內）
    
    # 繪製模板框
    cv2.rectangle(vis, (x, y), (x+tw, y+th), (0,255,0), 2)
    cv2.putText(vis, f'{sc:.2f}', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(vis, f'{sc:.2f}', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    # 將 model['points'] 從模板座標系轉換到場景座標系並疊加顯示
    # model['points'] 是相對於模板左上角 (0,0) 的座標
    # 直接加上模板左上角在場景中的位置即可
    scene_points = model['points'].copy().astype(np.float32)
    scene_points[:, 0] += x  # x 座標
    scene_points[:, 1] += y  # y 座標
    
    # 疊加顯示點（使用紅色，與綠色框區分）
    vis = draw_oriented_points(vis, scene_points, model['dirs'], 
                               step=max(1, int(len(scene_points)/800)+1), 
                               color=(0, 0, 255), arrow_length=3)


cv2.imwrite('examples/out_w3_topk.png', vis)
print('[DONE] W3~W4 demo generated: out_w4_model_oriented.png, out_w3_dt.png, out_w3_chamfer_heat.png, out_w3_topk.png')
