import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from sharp.ncc import (grid_search_theta_scale_ncc, pyramid_ncc_search_rotated,
                       multi_match_ncc_from_peaks)
from sharp.shape_match import chamfer_coarse, topk_peaks
from sharp.shape_model import create

def draw_rotated_box(img, center_x, center_y, w, h, theta, scale, color):
    """繪製旋轉和縮放的矩形框"""
    # 計算實際尺寸
    actual_w = int(w * scale)
    actual_h = int(h * scale)
    
    # 計算四個角點（相對於中心）
    corners = np.array([
        [-actual_w/2, -actual_h/2],
        [actual_w/2, -actual_h/2],
        [actual_w/2, actual_h/2],
        [-actual_w/2, actual_h/2]
    ], dtype=np.float32)
    
    # 旋轉
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    corners = corners @ R.T
    
    # 平移到中心位置
    corners[:, 0] += center_x
    corners[:, 1] += center_y
    
    # 繪製
    pts = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
    cv2.circle(img, (int(center_x), int(center_y)), 5, color, -1)

def main():
    # 載入場景圖像
    img = cv2.imread('data/demo/scene.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit('無法載入場景圖像: data/demo/scene.png')
    
    # 載入模板圖像
    tpl_img = cv2.imread('data/demo/template.png', cv2.IMREAD_GRAYSCALE)
    if tpl_img is None:
        raise SystemExit('無法載入模板圖像: data/demo/template.png')
    
    H, W = img.shape[:2]
    tpl_h, tpl_w = tpl_img.shape[:2]
    
    print("[步驟 1] 使用 Chamfer 進行粗匹配...")
    # 可選：先用 Chamfer 找到初始位置
    model = create(tpl_img, (0, 0, tpl_w, tpl_h), 
                   edge_threshold=(60, 140), sampling_step=2, use_polarity=False)
    
    scoremap, (ox, oy) = chamfer_coarse(model, img, roi=None)
    peaks = topk_peaks(scoremap, K=1, min_dist=10)
    tx, ty, _ = peaks[0]
    
    # 計算模板中心位置
    template_center_x = tpl_w // 2 + tx + ox
    template_center_y = tpl_h // 2 + ty + oy
    
    print(f"[步驟 2] Chamfer 初始位置: ({template_center_x:.1f}, {template_center_y:.1f})")
    
    # 準備角度和縮放網格
    #ang_grid = np.deg2rad(np.linspace(-180, 180, 37))  # 每 10 度一個步長
    #scales = np.linspace(0.7, 1.4, 15) # 每 0.1 縮放一個步長
    ang_grid = np.deg2rad(np.linspace(-20, 20, 21)) # 每 1 度一個步長
    scales = np.linspace(0.8, 1.2, 11) # 每 0.1 縮放一個步長

    
    # 方法 0: 多匹配方法（從 Chamfer 峰值開始）
    print("\n[方法 0] 多匹配 NCC（從 Chamfer 峰值，排除重複定位）...")
    min_dist = max(tpl_w, tpl_h)  # 最小距離為整個模板尺寸（更嚴格，避免重複定位）
    multi_matches = multi_match_ncc_from_peaks(
        tpl_img, img, scoremap, (ox, oy), 
        K=10, 
        min_dist_px=min_dist,  # 峰值間最小距離
        ang_grid=ang_grid, 
        scales=scales, 
        topk_per_position=2,  # 減少每個位置的候選數，避免過多重複
        min_angle_diff=np.deg2rad(10),  # 最小角度差異 10 度（更寬鬆）
        min_scale_diff=0.1,  # 最小縮放差異（更寬鬆）
        score_threshold=0.75,  # Chamfer 分數閾值
        use_position_priority=True,  # 啟用位置優先的 NMS，避免同一物體多個框
        fast_mode=True,  # 啟用快速模式
        use_parallel=False,  # 啟用並行化計算（可設為 False 關閉）
        num_workers=None,  # 自動檢測 CPU 核心數（可手動指定，如 4 表示使用 4 個進程）
        refine_local=False,  # 局部精細優化（快速版本）
        refine_threshold=0.8,  # 只對分數 >= 0.8 的匹配進行精細優化
        refine_iterative=False,  # 啟用迭代優化（最精確，但較慢）
        refine_iter_max_iters=5,  # 最大迭代次數
        refine_iter_tol=1e-3  # 收斂容差
    )
    
    print(f"找到 {len(multi_matches)} 個不重複的匹配:")
    for i, (sc, x, y, th, s) in enumerate(multi_matches):
        print(f"  {i+1}. 分數={sc:.4f}, 位置=({x:.1f}, {y:.1f}), "
              f"角度={np.rad2deg(th):.1f}°, 縮放={s:.3f}")
    
    # 方法 1: 使用單層 NCC 網格搜索（單一位置）
    print("\n[方法 1] NCC 網格搜索（單一位置，旋轉+縮放）...")
    candidates = grid_search_theta_scale_ncc(
        tpl_img, img, template_center_x, template_center_y,
        ang_grid, scales, topk=5
    )
    
    print(f"前 5 個候選:")
    for i, (sc, x, y, th, s) in enumerate(candidates):
        print(f"  {i+1}. 分數={sc:.4f}, 位置=({x:.1f}, {y:.1f}), "
              f"角度={np.rad2deg(th):.1f}°, 縮放={s:.3f}")
    
    best_score, best_x, best_y, best_theta, best_scale = candidates[0]
    
    # 方法 2: 使用金字塔 NCC 搜索
    print("\n[方法 2] 金字塔 NCC 搜索（旋轉+縮放）...")
    result = pyramid_ncc_search_rotated(
        tpl_img, img, template_center_x, template_center_y,
        ang_range=(-np.pi, np.pi),
        scale_range=(0.8, 1.2),  #scale_range=(0.7, 1.4),
        num_levels=4,       
        ang_steps=21,  #ang_steps=37        
        scale_steps=11  #scale_steps=15
    )
    
    pyr_x, pyr_y, pyr_theta, pyr_scale, pyr_score = result
    print(f"最佳結果: 分數={pyr_score:.4f}, 位置=({pyr_x:.1f}, {pyr_y:.1f}), "
          f"角度={np.rad2deg(pyr_theta):.1f}°, 縮放={pyr_scale:.3f}")
    
    # 視覺化結果
    print("\n[步驟 3] 生成視覺化結果...")
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 方法 0 結果（多匹配，使用不同顏色）
    colors = [
        (0, 255, 0),    # 綠色
        (255, 0, 0),    # 藍色
        (0, 0, 255),    # 紅色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 洋紅色
        (0, 255, 255),  # 黃色
        (128, 0, 128),  # 紫色
        (255, 165, 0),  # 橙色
        (0, 128, 128),  # 深青色
        (128, 128, 0)   # 橄欖色
    ]
    
    for i, (sc, x, y, th, s) in enumerate(multi_matches):
        if(sc < 0.75):
            continue    

        color = colors[i % len(colors)]
        draw_rotated_box(vis, x, y, tpl_w, tpl_h, th, s, color)
        # 在匹配位置附近顯示編號和分數
        cv2.putText(vis, f'#{i+1}:{sc:.2f},{np.rad2deg(th):.1f}deg,{s:.3f}x',     
                    (int(x) - 30, int(y) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    cv2.putText(vis, f'Multi-Match: {len(multi_matches)} found', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 方法 1 結果（單一最佳匹配，使用較粗的線條）
    draw_rotated_box(vis, best_x, best_y, tpl_w, tpl_h, best_theta, best_scale, (255, 255, 255))
    cv2.putText(vis, f'Single Best: {best_score:.3f}', 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 方法 2 結果（金字塔，使用較粗的線條）
    draw_rotated_box(vis, pyr_x, pyr_y, tpl_w, tpl_h, pyr_theta, pyr_scale, (255, 200, 0))
    cv2.putText(vis, f'Pyramid NCC: {pyr_score:.3f}', 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    
    # Chamfer 初始位置（黃色圓圈）
    cv2.circle(vis, (int(template_center_x), int(template_center_y)), 10, (0, 255, 255), 2)
    cv2.putText(vis, 'Chamfer Init', 
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imwrite('examples/out_ncc_rot_scale.png', vis)
    print("[完成] 結果已保存到: examples/out_ncc_rot_scale.png")

if __name__ == "__main__":
    main()

