# Sharp Matching (Python) — W1~W2 + ROI/Mask NCC + Affine Matching

## 安裝
```bash
pip install -r requirements.txt
```

## 快速開始

### 基礎功能演示
```bash
python examples/run_demo_w1_w2.py
python examples/run_demo_ncc_mask_roi.py
```
輸出檔位於 `examples/` 目錄。

### 仿射匹配（Affine Matching）

#### CLI 使用
```bash
python tools/affine_candidate.py --template data/demo/template.png --image data/demo/scene.png --detector auto --ratio 0.75 --outdir outputs
```

更多 CLI 選項：
```bash
# 使用指定檢測器
python tools/affine_candidate.py --template tpl.png --image img.png --detector sift --ratio 0.7

# 使用指定匹配器（bf 或 flann）
python tools/affine_candidate.py --template tpl.png --image img.png --matcher flann
# - bf: Brute Force 匹配（預設），精確但較慢，適合小數據集
# - flann: FLANN 近似最近鄰匹配，快速但為近似匹配，適合大數據集
#   - SIFT/AKAZE 使用 FLANN KDTree 索引
#   - ORB 使用 FLANN LSH 索引

# 使用配置文件
python tools/affine_candidate.py --config configs/affine.yaml --template tpl.png --image img.png

# 顯示詳細統計
python tools/affine_candidate.py --template tpl.png --image img.png --debug

# 設置品質門檻（當內點數 < 30 或內點比率 < 0.25 時會失敗退出）
python tools/affine_candidate.py --template tpl.png --image img.png --min-inliers 30 --min-inlier-ratio 0.25
```

#### 品質門檻參數
- `--min-inliers <數量>`: 最小內點數量門檻（預設: 30）
- `--min-inlier-ratio <比率>`: 最小內點比率門檻（預設: 0.25）

當匹配結果的內點數或內點比率低於門檻時，程式會：
- 以退出碼 2 退出（品質不達標）
- 輸出當前結果和建議改進方案
- 不生成可視化結果

退出碼說明：
- `0`: 成功
- `1`: 一般錯誤（如圖像載入失敗、匹配異常等）
- `2`: 品質不達標（內點數或內點比率未達門檻）

#### 程式使用
```python
from tools.affine_candidate import run_affine_matching

# 可 import 調用
result = run_affine_matching(
    template_path='data/demo/template.png',
    image_path='data/demo/scene.png',
    detector_type='auto',
    matcher_type='bf',         # 匹配器類型：'bf' 或 'flann'（預設 'bf'）
    ratio=0.75,
    max_keypoints=3000,
    min_inliers=30,           # 最小內點數量門檻
    min_inlier_ratio=0.25,    # 最小內點比率門檻
    output_dir='outputs'
)

# 返回值：
# 0: 成功
# 1: 一般錯誤
# 2: 品質不達標
```

#### 測試演示
```bash
python examples/run_demo_affine.py
```
該腳本會：
1. 執行單元測試（驗證 inliers >= 20、不崩潰等）
2. 執行多物件檢測演示
3. 生成可視化結果

## 測試
```bash
pytest -q
```
