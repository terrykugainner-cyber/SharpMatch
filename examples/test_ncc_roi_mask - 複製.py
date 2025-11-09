"""
測試框架：NCC with ROI and Mask 功能測試

此測試程式演示和驗證：
1. ROI 搜尋功能 - 限制搜索區域
2. Mask 支援功能 - 模板遮罩（ZNCC with mask）
3. 組合使用 ROI + Mask
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sharp.ncc import ncc_scoremap, pyramid_ncc_search
from sharp.visualize import draw_box, colorize_scoremap

def test_basic_ncc():
    """測試 1: 基本 NCC（無 ROI，無 mask）"""
    print("\n" + "="*60)
    print("測試 1: 基本 NCC（無 ROI，無 mask）")
    print("="*60)
    
    img = cv2.imread('data/demo/scene.png', cv2.IMREAD_GRAYSCALE)
    tpl = cv2.imread('data/demo/template.png', cv2.IMREAD_GRAYSCALE)
    
    if img is None or tpl is None:
        print("錯誤：無法載入測試圖像")
        return None
    
    # 基本 NCC
    score = ncc_scoremap(tpl, img)
    x, y, sc = pyramid_ncc_search(tpl, img)
    
    print(f"匹配結果: x={x}, y={y}, score={sc:.4f}")
    
    # 可視化
    vis = draw_box(img, x, y, tpl.shape[1], tpl.shape[0], text=f'score={sc:.3f}')
    cv2.imwrite('examples/test_ncc_basic.png', vis)
    print("結果已保存: examples/test_ncc_basic.png")
    
    return x, y, sc

def test_roi_search():
    """測試 2: ROI 搜尋功能"""
    print("\n" + "="*60)
    print("測試 2: ROI 搜尋功能")
    print("="*60)
    
    img = cv2.imread('data/demo/scene.png', cv2.IMREAD_GRAYSCALE)
    tpl = cv2.imread('data/demo/template.png', cv2.IMREAD_GRAYSCALE)
    
    if img is None or tpl is None:
        print("錯誤：無法載入測試圖像")
        return None
    
    H, W = img.shape[:2]
    tpl_h, tpl_w = tpl.shape[:2]
    
    # 定義 ROI：只在圖像中心區域搜索
    roi = (W//4, H//4, W//2, H//2)
    print(f"ROI: (x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]})")
    
    # ROI NCC
    score_roi = ncc_scoremap(tpl, img, roi=roi)
    x_roi, y_roi, sc_roi = pyramid_ncc_search(tpl, img, roi=roi)
    
    print(f"ROI 匹配結果: x={x_roi}, y={y_roi}, score={sc_roi:.4f}")
    
    # 可視化：繪製 ROI 和匹配結果
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # 繪製 ROI 框（藍色）
    x1, y1, w, h = roi
    cv2.rectangle(vis, (x1, y1), (x1+w, y1+h), (255, 0, 0), 2)
    cv2.putText(vis, 'ROI', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    # 繪製匹配結果（綠色）
    vis = draw_box(vis, x_roi, y_roi, tpl_w, tpl_h, text=f'score={sc_roi:.3f}')
    cv2.imwrite('examples/test_ncc_roi.png', vis)
    print("結果已保存: examples/test_ncc_roi.png")
    
    return x_roi, y_roi, sc_roi

def test_mask_support():
    """測試 3: Mask 支援功能"""
    print("\n" + "="*60)
    print("測試 3: Mask 支援功能（ZNCC with mask）")
    print("="*60)
    
    img = cv2.imread('data/demo/scene.png', cv2.IMREAD_GRAYSCALE)
    tpl = cv2.imread('data/demo/template.png', cv2.IMREAD_GRAYSCALE)
    
    if img is None or tpl is None:
        print("錯誤：無法載入測試圖像")
        return None
    
    tpl_h, tpl_w = tpl.shape[:2]
    
    # 創建多種 mask 進行測試
    masks = {}
    
    # Mask 1: 只使用中心圓形區域
    mask1 = np.zeros((tpl_h, tpl_w), dtype=np.uint8)
    cv2.circle(mask1, (tpl_w//2, tpl_h//2), min(tpl_w, tpl_h)//3, 255, -1)
    masks['center_circle'] = mask1
    
    # Mask 2: 只使用中心矩形區域（排除邊緣）
    mask2 = np.zeros((tpl_h, tpl_w), dtype=np.uint8)
    border = 5
    cv2.rectangle(mask2, (border, border), (tpl_w-border, tpl_h-border), 255, -1)
    masks['center_rect'] = mask2
    
    # Mask 3: 只使用模板中的亮區域（假設文字是亮的）
    _, mask3 = cv2.threshold(tpl, 200, 255, cv2.THRESH_BINARY)
    masks['bright_pixels'] = mask3
    
    results = {}
    for mask_name, mask in masks.items():
        print(f"\n測試 mask: {mask_name}")
        print(f"  Mask 有效像素數: {np.count_nonzero(mask)} / {tpl_w * tpl_h}")
        
        # 使用 mask 的 NCC
        score_mask = ncc_scoremap(tpl, img, mask=mask)
        x_mask, y_mask, sc_mask = pyramid_ncc_search(tpl, img, mask=mask)
        
        print(f"  匹配結果: x={x_mask}, y={y_mask}, score={sc_mask:.4f}")
        results[mask_name] = (x_mask, y_mask, sc_mask)
        
        # 可視化：顯示 mask 和匹配結果
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(tpl, cmap='gray')
        axes[0].set_title('Template')
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title(f'Mask ({mask_name})')
        axes[1].axis('off')
        
        vis = draw_box(img.copy(), x_mask, y_mask, tpl_w, tpl_h, text=f'score={sc_mask:.3f}')
        axes[2].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'Match Result (score={sc_mask:.3f})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'examples/test_ncc_mask_{mask_name}.png', dpi=150)
        plt.close()
        print(f"  結果已保存: examples/test_ncc_mask_{mask_name}.png")
    
    return results

def test_roi_and_mask():
    """測試 4: ROI + Mask 組合使用"""
    print("\n" + "="*60)
    print("測試 4: ROI + Mask 組合使用")
    print("="*60)
    
    img = cv2.imread('data/demo/scene.png', cv2.IMREAD_GRAYSCALE)
    tpl = cv2.imread('data/demo/template.png', cv2.IMREAD_GRAYSCALE)
    
    if img is None or tpl is None:
        print("錯誤：無法載入測試圖像")
        return None
    
    H, W = img.shape[:2]
    tpl_h, tpl_w = tpl.shape[:2]
    
    # 定義 ROI
    roi = (W//4, H//4, W//2, H//2)
    
    # 創建 mask（中心圓形）
    mask = np.zeros((tpl_h, tpl_w), dtype=np.uint8)
    cv2.circle(mask, (tpl_w//2, tpl_h//2), min(tpl_w, tpl_h)//3, 255, -1)
    
    print(f"ROI: (x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]})")
    print(f"Mask 有效像素數: {np.count_nonzero(mask)} / {tpl_w * tpl_h}")
    
    # 使用 ROI + Mask 的 NCC
    score_combo = ncc_scoremap(tpl, img, roi=roi, mask=mask)
    x_combo, y_combo, sc_combo = pyramid_ncc_search(tpl, img, roi=roi, mask=mask)
    
    print(f"ROI+Mask 匹配結果: x={x_combo}, y={y_combo}, score={sc_combo:.4f}")
    
    # 可視化
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # 繪製 ROI 框（藍色）
    x1, y1, w, h = roi
    cv2.rectangle(vis, (x1, y1), (x1+w, y1+h), (255, 0, 0), 2)
    cv2.putText(vis, 'ROI', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    # 繪製匹配結果（綠色）
    vis = draw_box(vis, x_combo, y_combo, tpl_w, tpl_h, text=f'score={sc_combo:.3f}')
    
    # 在模板區域疊加 mask 預覽
    tpl_vis = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)
    mask_vis = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    tpl_vis = cv2.addWeighted(tpl_vis, 0.7, mask_vis, 0.3, 0)
    
    # 組合顯示
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(tpl_vis, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Template with Mask')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Match Result (ROI+Mask, score={sc_combo:.3f})')
    axes[1].axis('off')
    
    score_vis = colorize_scoremap(score_combo)
    axes[2].imshow(cv2.cvtColor(score_vis, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Score Map (ROI+Mask)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('examples/test_ncc_roi_mask_combo.png', dpi=150)
    plt.close()
    print("結果已保存: examples/test_ncc_roi_mask_combo.png")
    
    return x_combo, y_combo, sc_combo

def test_performance_comparison():
    """測試 5: 性能比較（無 ROI vs 有 ROI）"""
    print("\n" + "="*60)
    print("測試 5: 性能比較（無 ROI vs 有 ROI）")
    print("="*60)
    
    img = cv2.imread('data/demo/scene.png', cv2.IMREAD_GRAYSCALE)
    tpl = cv2.imread('data/demo/template.png', cv2.IMREAD_GRAYSCALE)
    
    if img is None or tpl is None:
        print("錯誤：無法載入測試圖像")
        return None
    
    import time
    
    H, W = img.shape[:2]
    roi = (W//4, H//4, W//2, H//2)
    
    # 測試無 ROI
    print("\n測試 1: 無 ROI（全圖搜索）")
    start = time.time()
    x1, y1, sc1 = pyramid_ncc_search(tpl, img, roi=None)
    time1 = time.time() - start
    print(f"  結果: x={x1}, y={y1}, score={sc1:.4f}")
    print(f"  時間: {time1*1000:.2f} ms")
    
    # 測試有 ROI
    print("\n測試 2: 有 ROI（限制搜索區域）")
    start = time.time()
    x2, y2, sc2 = pyramid_ncc_search(tpl, img, roi=roi)
    time2 = time.time() - start
    print(f"  結果: x={x2}, y={y2}, score={sc2:.4f}")
    print(f"  時間: {time2*1000:.2f} ms")
    
    speedup = time1 / time2 if time2 > 0 else float('inf')
    print(f"\n速度提升: {speedup:.2f}x")
    print(f"ROI 面積比例: {roi[2]*roi[3]/(W*H):.2%}")

def main():
    """執行所有測試"""
    print("\n" + "="*60)
    print("NCC with ROI and Mask 測試框架")
    print("="*60)
    
    # 檢查測試圖像是否存在
    if not os.path.exists('data/demo/scene.png') or not os.path.exists('data/demo/template.png'):
        print("錯誤：測試圖像不存在")
        print("請確保 data/demo/scene.png 和 data/demo/template.png 存在")
        return
    
    try:
        # 執行所有測試
        test_basic_ncc()
        test_roi_search()
        test_mask_support()
        test_roi_and_mask()
        test_performance_comparison()
        
        print("\n" + "="*60)
        print("所有測試完成！")
        print("="*60)
        print("\n生成的測試結果文件：")
        print("  - examples/test_ncc_basic.png")
        print("  - examples/test_ncc_roi.png")
        print("  - examples/test_ncc_mask_center_circle.png")
        print("  - examples/test_ncc_mask_center_rect.png")
        print("  - examples/test_ncc_mask_bright_pixels.png")
        print("  - examples/test_ncc_roi_mask_combo.png")
        
    except Exception as e:
        print(f"\n測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
