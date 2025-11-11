# SharpMatch - 電腦視覺匹配工具包

SharpMatch 是一個專為電腦視覺應用設計的 Python 工具包，提供多種圖像匹配和物體檢測功能。本專案整合了特徵檢測、仿射變換、NCC（歸一化互相關聯）和 Chamfer 距離等多種技術，適用於工業檢測、機器視覺和自動化應用。

## 主要功能

### 1. 仿射匹配 (Affine Matching)
- 使用 SIFT/ORB/AKAZE 特徵檢測器
- Lowe's Ratio Test 篩選匹配
- RANSAC 估計仿射變換矩陣
- 支援單物件和多物件檢測
- 自動診斷特徵點和匹配數量不足的問題

### 2. 形狀匹配 (Shape Matching)
- Chamfer 距離粗定位
- 邊緣檢測和距離變換
- 支援多尺度金字塔搜索
- 適合於輪廓明顯的物體檢測

### 3. NCC 匹配 (Normalized Cross-Correlation)
- 零均值歸一化互相關聯 (ZNCC)
- 支援旋轉和縮放搜索
- 金字塔加速的粗到細搜索
- 遮罩支援，可限制匹配區域

### 4. 反扭轉功能 (Warping)
- 根據仿射變換將場景影像反扭轉到模板大小
- 生成與模板尺寸相同的裁切區塊
- 支援多種插值方法和邊界處理
- 輸出有效區域遮罩，方便後續處理

### 5. 透明疊圖輸出 (Alpha Overlays)
- 生成 RGBA 透明背景的疊圖
- 包含匹配線段和重投影框
- 適合於前端 UI 和簡報疊加
- 支援自定義顏色和線條樣式

## 安裝方法

### 從源碼安裝
```bash
git clone https://github.com/your-repo/SharpMatch.git
cd SharpMatch
pip install -e .
```

### 使用 pip 安裝
```bash
pip install sharp-match
```

## 快速開始

### 單物件仿射匹配
```python
from sharp.affine_match import affine_match, visualize_matches
import cv2

# 載入圖像
template = cv2.imread('template.png', cv2.IMREAD_GRAYSCALE)
scene = cv2.imread('scene.png', cv2.IMREAD_GRAYSCALE)

# 執行匹配
result = affine_match(template, scene)

# 可視化結果
vis = visualize_matches(template, scene, result)
cv2.imwrite('result.png', vis)
```

### 多物件檢測
```python
from sharp.affine_match import multi_affine_match_hybrid, visualize_multi_matches

# 執行多物件檢測
results = multi_affine_match_hybrid(
    template, scene,
    chamfer_k=20,
    min_inlier_ratio=0.4,
    min_matches=8,
    min_inliers=6,
    scale_range=(0.7, 1.3),
    min_chamfer_score=0.9,
    post_min_inlier_ratio=0.5,
    post_min_matches=10,
    post_min_inliers=8,
    post_scale_range=(0.75, 1.25),
    post_min_chamfer_score=0.92
)

# 可視化結果
vis = visualize_multi_matches(template, scene, results)
cv2.imwrite('multi_result.png', vis)
```

> 提示：`multi_affine_match_hybrid` 支援 Stage 2 / Stage 3 兩層品質門檻，可根據專案需求調整內點比率、內點數、匹配數、縮放範圍與 Chamfer 分數，避免品質不佳的定位結果混入。

### 反扭轉場景區塊
```python
from sharp.affine_match import warp_scene_to_template

# 反扭轉場景區塊到模板大小
warped_scene, mask = warp_scene_to_template(
    template, scene, result['affine_matrix'],
    output_mask=True
)

cv2.imwrite('warped.png', warped_scene)
cv2.imwrite('mask.png', mask)
```

### 生成透明疊圖
```python
from sharp.affine_match import create_matches_overlay, create_reprojection_overlay

# 創建匹配線段的透明疊圖
matches_overlay = create_matches_overlay(template, scene, result, show_all=False)
cv2.imwrite('matches_overlay.png', matches_overlay)

# 創建重投影框的透明疊圖
reproject_overlay = create_reprojection_overlay(template, scene, result)
cv2.imwrite('reproject_overlay.png', reproject_overlay)
```

## 命令行工具

### 運行仿射匹配演示
```bash
# 基本演示
python examples/run_demo_affine.py

# 包含透明疊圖輸出
python examples/run_demo_affine.py --save-alpha-overlays
```

### 運行 NCC 匹配演示
```bash
python examples/run_demo_ncc_rot_scale.py
```

### 運行 Chamfer 匹配演示
```bash
python examples/run_demo_chamfer_rot_scale.py
```

## API 參考

### affine_match 函數
```python
result = affine_match(
    template,                    # 模板圖像
    scene,                      # 場景圖像
    detector_type='auto',         # 檢測器類型: 'auto', 'sift', 'orb', 'akaze'
    matcher_type='bf',           # 匹配器類型: 'bf', 'flann'
    ratio_threshold=0.75,         # Lowe's Ratio Test 閾值
    max_keypoints=3000,          # 最大特徵點數量
    ransac_reproj_thresh=3.0,    # RANSAC 重投影閾值
    ransac_max_iters=2000,        # RANSAC 最大迭代次數
    ransac_confidence=0.99,       # RANSAC 置信度
    roi=None                      # 感興趣區域 (x, y, w, h)
)
```

### warp_scene_to_template 函數
```python
warped_scene, mask = warp_scene_to_template(
    template,                    # 模板圖像
    scene,                      # 場景圖像
    affine_matrix,               # 仿射變換矩陣
    interpolation=cv2.INTER_LINEAR, # 插值方法
    border_mode=cv2.BORDER_CONSTANT, # 邊界處理模式
    border_value=0,              # 邊界填充值
    output_mask=False             # 是否輸出遮罩
)
```

## 配置文件

專案使用 YAML 配置文件來存儲默認參數：

```yaml
# configs/affine.yaml
detector_type: auto
matcher_type: flann
ratio_threshold: 0.75
max_keypoints: 3000
ransac_reproj_thresh: 3.0
ransac_max_iters: 2000
ransac_confidence: 0.99
```

## 輸出文件說明

### 匹配結果
- `out_affine_matches.png` - 匹配點可視化
- `out_affine_reproject.png` - 重投影可視化
- `out_affine_multi.png` - 多物件檢測結果
- `out_affine_warp_comparison.png` - 反扭轉比較圖

### 反扭轉輸出
- `outputs/warped_to_template.png` - 反扭轉後的場景區塊
- `outputs/warped_mask.png` - 有效區域遮罩

### 透明疊圖
- `overlays/matches_overlay.png` - 匹配線段透明疊圖
- `overlays/reproject_overlay.png` - 重投影框透明疊圖

### 數據文件
- `out_affine.json` - 匹配結果 JSON
- `out_affine_multi.json` - 多物件檢測結果 JSON
- `summary.csv` - 匹配統計摘要

## 故障排除

### 常見問題

1. **找不到模組錯誤**
   ```
   ModuleNotFoundError: No module named 'sharp'
   ```
   **解決方案**:
   - 設置 PYTHONPATH 環境變數
   - 在專案根目錄執行命令
   - 使用 `pip install -e .` 安裝開發版本

2. **特徵點數量不足**
   ```
   [診斷] 特徵/匹配數量可能偏低，建議調整：
   - 通過 ratio test 的匹配僅 3 個 (< 15)
   ```
   **解決方案**:
   - 增加 `max_keypoints` 參數
   - 放寬 `ratio_threshold` 參數
   - 改用 SIFT 檢測器
   - 提升圖像對比度和清晰度

3. **匹配失敗**
   ```
   ValueError: 無法估計仿射矩陣
   ```
   **解決方案**:
   - 檢查圖像對應關係
   - 降低 RANSAC 閾值
   - 確保圖像有足夠的重疊區域

## 貢獻指南

歡迎提交 Issue 和 Pull Request！請確保：

1. 代碼符合 PEP 8 風格
2. 添加適當的單元測試
3. 更新相關文檔
4. 提供清晰的提交訊息

## 許可證

本專案採用 MIT 許可證。詳見 LICENSE 文件。

## 聯繫方式

- 問題和功能請求：[GitHub Issues](https://github.com/your-repo/SharpMatch/issues)
- 技術討論：[GitHub Discussions](https://github.com/your-repo/SharpMatch/discussions)

## 更新日誌

### v1.0.0 (2025-11-11)
- 初始版本發布
- 實現仿射匹配功能
- 添加 NCC 和 Chamfer 匹配
- 支援多物件檢測
- 新增反扭轉功能
- 添加透明疊圖輸出
- 實現診斷和建議系統
