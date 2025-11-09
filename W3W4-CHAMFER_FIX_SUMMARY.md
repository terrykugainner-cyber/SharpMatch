# Chamfer 匹配算法修正總結

## 問題描述
Chamfer 匹配結果與 NCC（Normalized Cross-Correlation）參考答案存在固定偏差：
- **偏差量**：X = -8 像素，Y = 10 像素
- **表現**：距離和在錯誤位置比正確位置小約 2.18 倍
- **影響**：匹配結果不準確，模型點無法正確對齊

## 根本原因

### 核心問題：Kernel 翻轉導致的座標映射錯誤

**問題點**：代碼中使用 `kernel[::-1, ::-1]` 翻轉 kernel 後進行卷積
```python
# 錯誤的方式
flipped_kernel = kernel[::-1, ::-1]
conv = cv2.filter2D(dt, cv2.CV_32F, flipped_kernel, anchor=(-1, -1))
```

**影響**：
- 翻轉 kernel 會改變卷積運算時的座標對應關係
- 導致距離和計算對應到錯誤的 DT 位置
- 在正確位置的距離和（1835.04）比錯誤位置（841.64）大約 2.18 倍

## 關鍵修正

### 1. 移除 Kernel 翻轉（核心修正）
```python
# 修正後：使用原始 kernel，不翻轉
anchor_x_explicit = tpl_w // 2  # 84
anchor_y_explicit = tpl_h // 2  # 85
conv = cv2.filter2D(dt, cv2.CV_32F, kernel, 
                   anchor=(anchor_x_explicit, anchor_y_explicit),
                   borderType=cv2.BORDER_CONSTANT)
```

**原理**：
- `cv2.filter2D` 使用未翻轉 kernel 時執行相關運算（correlation）
- 相關運算是 Chamfer 匹配的正確操作
- 明確設置 anchor 為幾何中心，確保座標映射正確

**效果**：距離和在正確位置變為最小值，偏差從 8-10 像素降至 0

### 2. Scoremap 提取偏移量修正
```python
# 修正前（錯誤）
out = conv[tpl_h-1:, tpl_w-1:]

# 修正後（正確）
anchor_offset_y = tpl_h // 2  # 85
anchor_offset_x = tpl_w // 2   # 84
out = conv[anchor_offset_y:, anchor_offset_x:]
```

**原理**：
- anchor 在 kernel 中心時，第一個有效位置是 `dt[tpl_h//2, tpl_w//2]`
- scoremap[0, 0] 應對應 conv[tpl_h//2, tpl_w//2]

### 3. Top-left 座標計算修正
```python
# 修正前：使用實際質心（不正確）
x = round(template_center_x - model_center_actual_x)  # 使用 (83.70, 83.08)
y = round(template_center_y - model_center_actual_y)

# 修正後：使用幾何中心（正確）
x = round(template_center_x - tpl_w / 2)  # 使用 (84.0, 85.0)
y = round(template_center_y - tpl_h / 2)
```

**原理**：
- `cv2.filter2D` 的 anchor 基於 kernel 的幾何中心，而非點集質心
- 因此計算 top-left 時應使用幾何中心偏移

**效果**：Y 方向偏差從 2 像素降至 0

## 輔助改進

### 1. 子像素峰值檢測
- 實現 `refine_peak_subpixel()` 函數
- 使用加權質心法將整數像素位置細化為子像素精度
- 提升峰值檢測精度（約 0.1 像素）

### 2. Kernel 構建改進
- 使用實際權重（`model["weights"]`）而非固定 1.0
- Kernel radius 設為 0（單像素精度），避免取樣錯誤 DT 值
- 權重累積使用 sum 而非 max

### 3. DT 精度提升
- 將 `maskSize` 從 3 改為 5
- 使用 5x5 mask 替代 3x3，提高距離計算精度

### 4. 邊界處理改進
- `borderType` 從 `BORDER_REPLICATE` 改為 `BORDER_CONSTANT`
- 邊界使用 0 值，避免邊界效應影響計算

## 修正效果

### 修正前
- Scoremap 峰值：`(829, 609)`
- 模板中心：`(996, 778)`
- 與 NCC 差異：X = -8, Y = 10
- 距離和比率：2.18（錯誤位置更小）

### 修正後
- Scoremap 峰值：`(920, 683)` ✓（與 NCC 完全一致）
- 模板中心：`(1004, 768)` ✓（與 NCC 完全一致）
- 與 NCC 差異：X = 0, Y = 0 ✓
- 距離和比率：1.00 ✓（正確位置為最小值）

## 重要教訓

1. **卷積 vs 相關運算**：
   - Chamfer 匹配需要相關運算，不需要翻轉 kernel
   - OpenCV 的 `filter2D` 使用未翻轉 kernel 時自動執行相關運算

2. **Anchor 點的重要性**：
   - Anchor 設置必須與 scoremap 提取邏輯一致
   - 應使用幾何中心而非質心

3. **診斷工具的重要性**：
   - 加入距離和診斷幫助快速定位問題
   - 比較正確位置與錯誤位置的距離和差異

4. **座標系統的一致性**：
   - 確保從 scoremap → conv → dt → 圖像的座標轉換鏈正確
   - 每個步驟都需要驗證座標對應關係

## 修正文件清單

1. **h:\SharpMatch\sharp\shape_match.py**
   - 移除 kernel 翻轉
   - 修正 scoremap 提取偏移
   - 改進 kernel 構建（權重、radius）
   - 添加診斷代碼

2. **h:\SharpMatch\sharp\distance.py**
   - DT 精度從 maskSize=3 提升至 5

3. **h:\SharpMatch\examples\run_demo_w3_w4.py**
   - 修正 top-left 計算（使用幾何中心）
   - 更新座標轉換邏輯
   - 添加子像素精度支持

## 後續優化建議

1. **清理診斷代碼**（可選）：
   - 保留關鍵診斷信息便於未來調試
   - 移除詳細的逐步輸出

2. **性能優化**（可選）：
   - DT maskSize=5 比 maskSize=3 慢約 2 倍，可考慮作為可選參數
   - Kernel radius 保持為 0（單像素）以確保精度

3. **文檔更新**：
   - 更新函數文檔，說明 anchor 和座標映射的關係
   - 添加使用示例

---

**修正日期**：2024年  
**核心問題**：Kernel 翻轉導致的座標映射錯誤  
**修正效果**：偏差從 8-10 像素降至 0 像素，完全匹配 NCC 參考結果
