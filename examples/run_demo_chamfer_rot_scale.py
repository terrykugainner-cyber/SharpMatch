import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import time
from sharp.shape_match import chamfer_coarse, topk_peaks
from sharp.shape_model import create
from sharp.ncc import pyramid_ncc_search_rotated

def draw_rotated_box(img, center_x, center_y, w, h, theta, scale, color):
    """繪製旋轉和縮放的矩形框"""
    actual_w = int(w * scale)
    actual_h = int(h * scale)
    
    corners = np.array([
        [-actual_w/2, -actual_h/2],
        [actual_w/2, -actual_h/2],
        [actual_w/2, actual_h/2],
        [-actual_w/2, actual_h/2]
    ], dtype=np.float32)
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    corners = corners @ R.T
    corners[:, 0] += center_x
    corners[:, 1] += center_y
    
    pts = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
    cv2.circle(img, (int(center_x), int(center_y)), 5, color, -1)

def main():
    # 載入圖像
    img = cv2.imread('data/demo/scene.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit('無法載入場景圖像: data/demo/scene.png')
    
    tpl_img = cv2.imread('data/demo/template.png', cv2.IMREAD_GRAYSCALE)
    if tpl_img is None:
        raise SystemExit('無法載入模板圖像: data/demo/template.png')
    
    tpl_h, tpl_w = tpl_img.shape[:2]
    H, W = img.shape[:2]
    
    print("=" * 60)
    print("Chamfer + NCC 金字塔匹配")
    print("=" * 60)
    
    # ========== 階段1: Chamfer 粗定位（僅平移） ==========
    print("\n[階段1] Chamfer 粗定位（僅平移）...")
    start_time = time.time()
    
    model = create(tpl_img, (0, 0, tpl_w, tpl_h), 
                   edge_threshold=(60, 140), sampling_step=2, use_polarity=False)
    
    scoremap, (ox, oy) = chamfer_coarse(model, img, roi=None)
    
    # 提取候選位置
    peaks = topk_peaks(scoremap, K=10, min_dist=50)
    
    candidate_positions = []
    for px, py, score in peaks:
        # 計算模板中心位置
        center_x = tpl_w // 2 + px + ox
        center_y = tpl_h // 2 + py + oy
        candidate_positions.append((center_x, center_y, score))
    
    chamfer_time = time.time() - start_time
    print(f"  找到 {len(candidate_positions)} 個候選位置")
    print(f"  耗時: {chamfer_time:.3f} 秒")
    
    if len(candidate_positions) == 0:
        print("  警告: 未找到任何候選位置")
        return
    
    # 顯示候選位置
    for i, (cx, cy, score) in enumerate(candidate_positions[:5]):
        print(f"  候選{i+1}: 位置=({cx:.1f}, {cy:.1f}), 分數={score:.4f}")
    
    # ========== 階段2: NCC 金字塔細定位（角度+縮放） ==========
    print("\n[階段2] NCC 金字塔細定位（角度+縮放）...")
    start_time = time.time()
    
    # 搜索參數
    ang_range = (-np.deg2rad(15), np.deg2rad(15))  # ±15度
    scale_range = (0.8, 1.2)  # 縮放範圍
    num_pyramid_levels = 4  # 金字塔層數
    ang_steps = 17  # 角度搜索步數
    scale_steps = 7  # 縮放搜索步數
    
    print(f"  角度範圍: {np.rad2deg(ang_range[0]):.1f}° ~ {np.rad2deg(ang_range[1]):.1f}°")
    print(f"  縮放範圍: {scale_range[0]:.2f} ~ {scale_range[1]:.2f}")
    print(f"  金字塔層數: {num_pyramid_levels}")
    
    # 對每個候選位置進行 NCC 金字塔搜索
    all_results = []
    for i, (center_x, center_y, chamfer_score) in enumerate(candidate_positions):
        print(f"  處理候選 {i+1}/{len(candidate_positions)}: ({center_x:.1f}, {center_y:.1f})...")
        
        try:
            # 使用 NCC 金字塔進行精細搜索
            x, y, theta, scale, ncc_score = pyramid_ncc_search_rotated(
                tpl_img, img, center_x, center_y,
                ang_range=ang_range,
                scale_range=scale_range,
                num_levels=num_pyramid_levels,
                scale=0.5,  # 金字塔縮放因子
                mask=None,
                roi=None,
                ang_steps=ang_steps,
                scale_steps=scale_steps
            )
            
            all_results.append({
                'x': x,
                'y': y,
                'theta': theta,
                'scale': scale,
                'ncc_score': ncc_score,
                'chamfer_score': chamfer_score,
                'combined_score': ncc_score * 0.7 + chamfer_score * 0.3  # 組合分數
            })
            
            print(f"    → NCC: 分數={ncc_score:.4f}, 角度={np.rad2deg(theta):.1f}°, 縮放={scale:.3f}")
            
        except Exception as e:
            print(f"    → 錯誤: {e}")
            continue
    
    ncc_time = time.time() - start_time
    print(f"\n  完成 {len(all_results)} 個候選位置的細定位")
    print(f"  耗時: {ncc_time:.3f} 秒")
    
    if len(all_results) == 0:
        print("  警告: 未找到任何有效匹配")
        return
    
    # 按組合分數排序
    all_results.sort(key=lambda r: r['combined_score'], reverse=True)
    
    # 顯示前5個結果
    print(f"\n  前 5 個匹配結果:")
    for i, r in enumerate(all_results[:5]):
        print(f"    {i+1}. 分數={r['combined_score']:.4f} (NCC={r['ncc_score']:.4f}), "
              f"位置=({r['x']:.1f}, {r['y']:.1f}), "
              f"角度={np.rad2deg(r['theta']):.1f}°, 縮放={r['scale']:.3f}")
    
    # ========== 視覺化結果 ==========
    print("\n[階段3] 生成視覺化結果...")
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    colors = [
        (0, 255, 0),    # 綠色 - 最佳
        (255, 0, 0),    # 藍色
        (0, 0, 255),    # 紅色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 洋紅色
        (0, 255, 255),  # 黃色
        (128, 0, 128),  # 紫色
        (255, 165, 0),  # 橙色
    ]
    
    # 繪製所有匹配結果
    for i, r in enumerate(all_results):
        if(r["combined_score"] < 0.8):
            continue

        color = colors[i % len(colors)]
        draw_rotated_box(vis, r['x'], r['y'], tpl_w, tpl_h, r['theta'], r['scale'], color)
        
        label = f'#{i+1}:{r["combined_score"]:.2f}, {np.rad2deg(r["theta"]):.1f}deg, {r["scale"]:.3f}x'
        cv2.putText(vis, label, 
                   (int(r['x']) - 30, int(r['y']) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 標註信息
    cv2.putText(vis, f'Found {len(all_results)} objects', 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if len(all_results) > 0:
        best = all_results[0]
        cv2.putText(vis, f'Best: {best["combined_score"]:.3f} @ {np.rad2deg(best["theta"]):.1f}deg, {best["scale"]:.3f}x', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 保存結果
    output_path = 'examples/out_chamfer_ncc_pyramid.png'
    cv2.imwrite(output_path, vis)
    print(f"  結果已保存到: {output_path}")
    
    # 總結
    print("\n" + "=" * 60)
    print("匹配總結")
    print("=" * 60)
    print(f"✓ Chamfer 粗定位: {len(candidate_positions)} 個候選位置 ({chamfer_time:.3f}秒)")
    print(f"✓ NCC 細定位: {len(all_results)} 個有效匹配 ({ncc_time:.3f}秒)")
    print(f"✓ 總耗時: {chamfer_time + ncc_time:.3f} 秒")
    
    if len(all_results) > 0:
        best = all_results[0]
        print(f"\n  最佳匹配:")
        print(f"    位置: ({best['x']:.1f}, {best['y']:.1f})")
        print(f"    角度: {np.rad2deg(best['theta']):.1f}°")
        print(f"    縮放: {best['scale']:.3f}")
        print(f"    NCC 分數: {best['ncc_score']:.4f}")
        print(f"    組合分數: {best['combined_score']:.4f}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
