import cv2, numpy as np
from .distance import distance_from_edges
from .edges import canny_edges

def build_pyramid(image, num_levels=3, scale_factor=2.0):
    """構建圖像金字塔。
    
    Args:
        image: 輸入圖像（灰度或 BGR）
        num_levels: 金字塔層數（包括原始圖像）
        scale_factor: 每層之間的縮放因子（默認 2.0，即每次縮小一半）
    
    Returns:
        List of images from finest (original) to coarsest (smallest)
    """
    pyramid = [image]
    current = image
    
    for i in range(1, num_levels):
        h, w = current.shape[:2]
        new_w = int(w / scale_factor)
        new_h = int(h / scale_factor)
        # 確保至少保留一些像素
        if new_w < 10 or new_h < 10:
            break
        current = cv2.resize(current, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pyramid.append(current)
    
    return pyramid

def chamfer_coarse_pyramid(model, image, roi=None, num_levels=3, 
                          use_weights=True, kernel_radius=0, 
                          search_window_scale=1.5):
    """使用金字塔加速的 Chamfer 粗匹配（僅平移）。
    
    從最粗層開始搜索，逐步細化到原始解析度。
    這可以大幅減少計算量，特別是在大圖像上。
    
    Args:
        model: dict from shape_model.create
        image: gray/BGR
        roi: optional (x,y,w,h) search window (在原始圖像座標系中)
        num_levels: 金字塔層數（默認 3）
        use_weights: 是否使用模型權重（默認 True）
        kernel_radius: kernel 半徑（默認 0）
        search_window_scale: 在細化層時的搜索窗口縮放因子（默認 1.5，擴大搜索範圍以補償低解析度誤差）
    
    Returns:
        scoremap (float32, larger is better), (offx, offy)
    """
    # 構建圖像金字塔
    img_pyramid = build_pyramid(image, num_levels)
    
    # 構建模板金字塔
    tpl_img = model["template"]
    tpl_pyramid = build_pyramid(tpl_img, num_levels)
    
    # 從最粗層開始，逐步細化
    best_candidate = None  # 存儲最佳候選位置 (center_x, center_y, score)
    
    # 從最粗層（最小圖像）到最細層（原始圖像）
    for level in range(len(img_pyramid) - 1, -1, -1):
        level_img = img_pyramid[level]
        level_tpl = tpl_pyramid[level]
        
        # 計算當前層相對於原始圖像的縮放因子
        # level 0 是原始圖像，level (num_levels-1) 是最粗層
        # 所以 scale_down = 2^(num_levels-1-level)
        scale_down = 2.0 ** (len(img_pyramid) - 1 - level)
        
        # 在非最粗層，需要根據上一層結果調整搜索區域
        if level < len(img_pyramid) - 1 and best_candidate is not None:
            # 從上一層的最佳候選位置開始，在當前層進行細化搜索
            # 將候選位置映射到當前層
            level_cx = best_candidate[0] / scale_down
            level_cy = best_candidate[1] / scale_down
            
            # 計算搜索窗口大小（基於模板大小和搜索窗口縮放因子）
            level_tpl_h, level_tpl_w = level_tpl.shape[:2]
            search_radius = int(max(level_tpl_w, level_tpl_h) * search_window_scale)
            
            # 計算搜索區域邊界
            x0 = max(0, int(level_cx - search_radius))
            y0 = max(0, int(level_cy - search_radius))
            x1 = min(level_img.shape[1], int(level_cx + search_radius))
            y1 = min(level_img.shape[0], int(level_cy + search_radius))
            
            if x1 > x0 and y1 > y0:
                level_roi = (x0, y0, x1 - x0, y1 - y0)
            else:
                level_roi = None
        else:
            # 最粗層：全圖搜索
            level_roi = None
        
        # 為當前層創建模型
        level_tpl_h, level_tpl_w = level_tpl.shape[:2]
        original_tpl_h, original_tpl_w = tpl_img.shape[:2]
        
        # 計算點座標的縮放因子
        scale_x = level_tpl_w / original_tpl_w
        scale_y = level_tpl_h / original_tpl_h
        
        # 創建當前層的模型（深度複製以避免修改原始模型）
        level_model = {}
        level_model.update(model)  # 複製所有屬性
        level_model["template"] = level_tpl
        # 縮放模型點
        level_points = model["points"].copy()
        level_points[:, 0] *= scale_x
        level_points[:, 1] *= scale_y
        level_model["points"] = level_points
        
        # 在當前層執行 Chamfer 匹配
        if level_roi is not None:
            # 調整 ROI 以確保在圖像範圍內
            x0, y0, w, h = level_roi
            x0 = max(0, x0)
            y0 = max(0, y0)
            w = min(w, level_img.shape[1] - x0)
            h = min(h, level_img.shape[0] - y0)
            level_roi = (x0, y0, w, h) if w > 0 and h > 0 else None
        
        level_scoremap, (level_ox, level_oy) = chamfer_coarse(
            level_model, level_img, roi=level_roi, 
            use_weights=use_weights, kernel_radius=kernel_radius
        )
        
        # 從 scoremap 中提取最佳候選位置
        level_peaks = topk_peaks(level_scoremap, K=1, min_dist=int(max(level_tpl_w, level_tpl_h) * 0.3))
        
        if len(level_peaks) > 0:
            px, py, score = level_peaks[0]
            # 當前層座標
            level_x = px + level_ox
            level_y = py + level_oy
            # 加上模板中心偏移
            level_center_x = level_x + level_tpl_w // 2
            level_center_y = level_y + level_tpl_h // 2
            
            # 映射回原始圖像座標系
            orig_center_x = level_center_x * scale_down
            orig_center_y = level_center_y * scale_down
            
            best_candidate = (orig_center_x, orig_center_y, score)
        
        # 如果是最後一層（原始解析度），進行最終匹配
        if level == 0:
            if best_candidate is not None:
                # 在最佳位置周圍創建搜索區域進行精細匹配
                best_cx, best_cy, _ = best_candidate
                original_tpl_h, original_tpl_w = tpl_img.shape[:2]
                
                # 計算搜索半徑：至少需要模板大小 + 一些邊緣空間
                # 考慮金字塔可能帶來的誤差，使用更大的搜索半徑
                search_radius = int(max(original_tpl_w, original_tpl_h) * 0.5)  # 50% 的模板 size
                # 確保搜索半徑至少為模板大小的一半，以容納可能的誤差
                min_radius = max(original_tpl_w, original_tpl_h) // 2
                search_radius = max(search_radius, min_radius)
                
                x0 = max(0, int(best_cx - search_radius))
                y0 = max(0, int(best_cy - search_radius))
                x1 = min(image.shape[1], int(best_cx + search_radius))
                y1 = min(image.shape[0], int(best_cy + search_radius))
                
                # 計算實際 ROI 尺寸
                roi_w = x1 - x0
                roi_h = y1 - y0
                
                # 確保 ROI 足夠大以容納模板（至少比模板大一些）
                if roi_w > original_tpl_w and roi_h > original_tpl_h:
                    final_roi = (x0, y0, roi_w, roi_h)
                    try:
                        final_scoremap, (final_ox, final_oy) = chamfer_coarse(
                            model, image, roi=final_roi,
                            use_weights=use_weights, kernel_radius=kernel_radius
                        )
                        
                        # 將 scoremap 擴展到全圖尺寸
                        full_scoremap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
                        full_scoremap[final_oy:final_oy + final_scoremap.shape[0],
                                      final_ox:final_ox + final_scoremap.shape[1]] = final_scoremap
                        
                        return full_scoremap, (0, 0)
                    except ValueError:
                        # 如果 ROI 仍然有問題，回退到全圖搜索
                        pass
            
            # 如果沒有候選位置或 ROI 計算失敗，回退到全圖搜索
            return chamfer_coarse(model, image, roi=roi, 
                                 use_weights=use_weights, kernel_radius=kernel_radius)
    
    # 如果所有層都失敗，回退到標準方法
    return chamfer_coarse(model, image, roi=roi, 
                         use_weights=use_weights, kernel_radius=kernel_radius)

def chamfer_coarse(model, image, roi=None, use_weights=True, kernel_radius=0):
    """Translation-only chamfer matching (coarse) using distance transform.
    Score is negative sum of distances (the lower the better), we output a positive similarity map by inversion.
    Args:
        model: dict from shape_model.create
        image: gray/BGR
        roi: optional (x,y,w,h) search window
        use_weights: if True, use model weights in kernel; if False, use uniform weights (default: True)
        kernel_radius: radius for kernel points (0=single pixel, 1=3x3, 2=5x5) (default: 1)
    Returns:
        scoremap (float32, larger is better), (offx, offy)
    """
    if image.ndim == 3:
        g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        g = image
    # compute edges and distance in the search area
    if roi is not None:
        x,y,w,h = roi
        crop = g[y:y+h, x:x+w]
        offx, offy = x, y
    else:
        crop = g; offx = offy = 0

    low, high = model.get("edge_threshold",(60,140))
    edg = canny_edges(crop, low, high)
    # Use improved DT precision (maskSize=5 instead of 3)
    # This should give more accurate distance values and better peak detection
    dt = distance_from_edges(edg, mask_size=5)

    # build a sparse template mask from model points
    # Improved version: use actual weights and better point placement
    tpl_h, tpl_w = model["template"].shape[:2]
    kernel = np.zeros((tpl_h, tpl_w), np.float32)
    pts = model["points"].astype(np.float32)  # Keep float for precise positioning
    weights = model["weights"] if use_weights else np.ones(len(pts), dtype=np.float32)
    
    # Improved kernel building: use actual weights and precise point placement
    # CRITICAL: Use radius=0 (single pixel) for precise matching
    # Larger radius can cause errors in distance sum calculation
    for (x0, y0), w in zip(pts, weights):
        # Use precise pixel coordinates (no rounding for now, but clip)
        xi = int(round(x0))
        yi = int(round(y0))
        xi = np.clip(xi, 0, tpl_w-1)
        yi = np.clip(yi, 0, tpl_h-1)
        
        # CRITICAL FIX: Use single pixel (radius=0) for accurate distance sum
        # Using radius > 0 can cause the kernel to sample wrong DT values
        # This is likely causing the 2x distance sum error
        if kernel_radius == 0:
            # Single pixel: sum weights for overlapping points
            # This is the most accurate for Chamfer matching
            kernel[yi, xi] += float(w)
        else:
            # If radius is used, be careful: it should only add to robustness
            # but might introduce errors. For now, prefer radius=0
            temp_kernel = np.zeros((tpl_h, tpl_w), np.float32)
            cv2.circle(temp_kernel, (xi, yi), kernel_radius, float(w), -1)
            kernel = kernel + temp_kernel  # Sum weights for overlapping regions

    # Correlate DT with kernel: lower sum distance => better match
    # We invert and normalize to [0,1] as a similarity
    
    # CRITICAL FIX: The kernel flip might be causing coordinate misalignment
    # For Chamfer matching, we need to compute: sum over all kernel points of DT values
    # When kernel center is at dt[r, c], a kernel point at (i, j) should sample dt[r + (i - anchor_y), c + (j - anchor_x)]
    # 
    # OpenCV's filter2D performs correlation (not convolution) when kernel is NOT flipped
    # But standard Chamfer matching uses correlation with flipped kernel
    # However, the coordinate mapping after flip might be wrong
    #
    # Let's try: use unflipped kernel with explicit anchor to ensure correct mapping
    
    # DIAGNOSTIC: Check kernel statistics
    kernel_nonzero = np.count_nonzero(kernel)
    print(f'[KERNEL_DIAG] Kernel size: {tpl_h}x{tpl_w}, non-zero points: {kernel_nonzero}')
    print(f'[KERNEL_DIAG] Kernel weight sum: {kernel.sum():.2f}')
    print(f'[KERNEL_DIAG] Kernel center (geometric): ({tpl_w//2}, {tpl_h//2})')
    
    # CRITICAL: Try unflipped kernel with explicit anchor
    # The anchor should be at kernel center: (tpl_w//2, tpl_h//2) = (84, 85)
    # When anchor aligns to dt[r, c], kernel[i,j] samples dt[r + (i - 85), c + (j - 84)]
    anchor_x_explicit = tpl_w // 2  # 84
    anchor_y_explicit = tpl_h // 2  # 85
    
    # Use unflipped kernel - this gives correlation (which is what we want)
    # filter2D with unflipped kernel performs correlation
    conv = cv2.filter2D(dt, cv2.CV_32F, kernel, 
                       anchor=(anchor_x_explicit, anchor_y_explicit),
                       borderType=cv2.BORDER_CONSTANT)
    
    # CRITICAL FIX: Correct scoremap extraction offset
    # When anchor is at kernel center (tpl_h//2, tpl_w//2):
    # - conv[row, col] means kernel center aligned to dt[row, col]
    # - The first valid position is when anchor can be at dt[tpl_h//2, tpl_w//2]
    # - So scoremap[0, 0] should correspond to conv[tpl_h//2, tpl_w//2]
    # - Therefore: scoremap[py, px] corresponds to conv[tpl_h//2 + py, tpl_w//2 + px]
    # 
    # Previous logic used conv[tpl_h-1:, tpl_w-1:] which was WRONG
    # This caused an offset of approximately (tpl_h//2 - (tpl_h-1), tpl_w//2 - (tpl_w-1))
    # = (85 - 169, 84 - 167) = (-84, -83) but actually the difference is:
    # tpl_h//2 = 85, tpl_h-1 = 169, difference = -84
    # This wrong offset explains why scoremap peak is off by ~8-10 pixels!
    
    anchor_offset_y = tpl_h // 2  # Center row of kernel
    anchor_offset_x = tpl_w // 2  # Center column of kernel
    
    # Extract valid region starting from where anchor can first be placed
    out = conv[anchor_offset_y: dt.shape[0], anchor_offset_x: dt.shape[1]]
    if out.size == 0:
        raise ValueError("Template larger than ROI/image.")
    
    # DIAGNOSTIC: Check the actual distance sums (before inversion)
    # out contains distance sums - lower values mean better matches
    # Let's check what the distance sum is at the expected NCC location
    # If NCC is correct, template center should be at dt[1004, 768] (image coords)
    # In dt coords (NumPy indexing), this is dt[768, 1004]
    # This maps to conv[768, 1004]
    # Which maps to scoremap[768 - tpl_h//2, 1004 - tpl_w//2] = scoremap[683, 920]
    test_ncc_scoremap_y = 768 - tpl_h // 2  # 768 - 85 = 683
    test_ncc_scoremap_x = 1004 - tpl_w // 2  # 1004 - 84 = 920
    
    # Also check a few nearby positions to see if there's a pattern
    test_offsets = [(-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2)]
    
    if 0 <= test_ncc_scoremap_y < out.shape[0] and 0 <= test_ncc_scoremap_x < out.shape[1]:
        distance_sum_at_ncc = out[test_ncc_scoremap_y, test_ncc_scoremap_x]
        actual_min_distance_sum = out.min()
        actual_min_idx = np.unravel_index(out.argmin(), out.shape)
        distance_sum_at_actual_peak = out[actual_min_idx[0], actual_min_idx[1]]
        
        print(f'[DT_KERNEL_DIAG] Distance sum at expected NCC location scoremap[{test_ncc_scoremap_y}, {test_ncc_scoremap_x}]: {distance_sum_at_ncc:.2f}')
        print(f'[DT_KERNEL_DIAG] Actual min distance sum: {actual_min_distance_sum:.2f} at scoremap[{actual_min_idx[0]}, {actual_min_idx[1]}]')
        print(f'[DT_KERNEL_DIAG] Distance sum difference: {distance_sum_at_ncc - actual_min_distance_sum:.2f}')
        print(f'[DT_KERNEL_DIAG] Ratio: {distance_sum_at_ncc / actual_min_distance_sum:.4f} (should be close to 1.0 if correct)')
        
        # Check nearby positions to see if there's a gradient
        print(f'[DT_KERNEL_DIAG] Checking nearby positions around NCC location:')
        for dy, dx in test_offsets:
            test_y = test_ncc_scoremap_y + dy
            test_x = test_ncc_scoremap_x + dx
            if 0 <= test_y < out.shape[0] and 0 <= test_x < out.shape[1]:
                nearby_sum = out[test_y, test_x]
                print(f'[DT_KERNEL_DIAG]   Offset ({dy}, {dx}): distance_sum={nearby_sum:.2f}, ratio={nearby_sum/actual_min_distance_sum:.4f}')
    
    # convert distance sum to similarity: smaller -> larger
    # Invert so that lower distance sums become higher similarity scores
    out = out.max() - out
    # normalize
    m, M = float(out.min()), float(out.max())
    if M - m > 1e-8:
        out = (out - m) / (M - m)
    else:
        out[:] = 0.0
    
    # DIAGNOSTIC: Check similarity scores after inversion
    if 0 <= test_ncc_scoremap_y < out.shape[0] and 0 <= test_ncc_scoremap_x < out.shape[1]:
        score_at_ncc = out[test_ncc_scoremap_y, test_ncc_scoremap_x]
        actual_max_score = out.max()
        actual_max_idx = np.unravel_index(out.argmax(), out.shape)
        print(f'[DT_KERNEL_DIAG] Similarity score at NCC location: {score_at_ncc:.6f}')
        print(f'[DT_KERNEL_DIAG] Actual max similarity score: {actual_max_score:.6f} at scoremap[{actual_max_idx[0]}, {actual_max_idx[1]}]')
        print(f'[DT_KERNEL_DIAG] Score ratio: {score_at_ncc / actual_max_score:.6f}')
    
    return out.astype(np.float32), (offx, offy)

def refine_peak_subpixel(scoremap, px, py):
    """Refine peak location to subpixel accuracy using weighted centroid method.
    
    Args:
        scoremap: 2D array of scores
        px, py: integer peak location (x, y)
    
    Returns:
        (subpx_x, subpx_y): subpixel peak location (float)
    """
    h, w = scoremap.shape
    
    # Extract 3x3 neighborhood around peak
    x0 = max(0, px - 1)
    x1 = min(w - 1, px + 1)
    y0 = max(0, py - 1)
    y1 = min(h - 1, py + 1)
    
    # Extract neighborhood values
    neighborhood = scoremap[y0:y1+1, x0:x1+1].astype(np.float32)
    
    # Method 1: Weighted centroid (simple and robust)
    # Create coordinate grids relative to the extracted region
    yy_local, xx_local = np.mgrid[y0:y1+1, x0:x1+1]
    
    # Compute weighted centroid
    total_weight = neighborhood.sum()
    if total_weight > 1e-8:
        subpx_x = float((neighborhood * xx_local).sum() / total_weight)
        subpx_y = float((neighborhood * yy_local).sum() / total_weight)
    else:
        subpx_x, subpx_y = float(px), float(py)
    
    # Ensure subpixel location is within bounds
    subpx_x = float(np.clip(subpx_x, 0, w - 1))
    subpx_y = float(np.clip(subpx_y, 0, h - 1))
    
    return subpx_x, subpx_y

def topk_peaks(scoremap, K=5, min_dist=8, subpixel=True):
    """Return top-K peak locations (x,y,score) with subpixel accuracy.
    
    Args:
        scoremap: 2D array of scores
        K: number of peaks to find
        min_dist: minimum distance between peaks
        subpixel: if True, use subpixel refinement; if False, integer precision only
    
    Returns:
        List of (x, y, score) tuples with subpixel precision when subpixel=True
    """
    s = scoremap.copy()
    h, w = s.shape
    out = []
    
    for _ in range(K):
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(s)
        if maxVal <= 0:
            break
        
        px, py = maxLoc[0], maxLoc[1]  # integer peak location
        
        if subpixel:
            # Refine to subpixel accuracy
            subpx_x, subpx_y = refine_peak_subpixel(scoremap, px, py)
            out.append((subpx_x, subpx_y, float(maxVal)))
        else:
            # Integer precision (original behavior)
            out.append((float(px), float(py), float(maxVal)))
        
        # Suppress peak in integer location for next iteration
        cv2.circle(s, (px, py), min_dist, 0, -1)
    
    return out

def chamfer_score_rotated(model, image, center_x, center_y, theta, scale,
                           roi=None, use_weights=True, kernel_radius=0):
    """計算旋轉和縮放後的 Chamfer 匹配分數。
    
    對模型點進行旋轉和縮放變換，然後在距離變換圖中取樣，計算加權距離和。
    距離和越小，匹配越好。分數被轉換為 [0, 1] 範圍，越大越好。
    
    Args:
        model: 形狀模型（從 shape_model.create 獲得）
        image: 場景圖像 (gray/BGR)
        center_x, center_y: 模板中心在場景中的位置
        theta: 旋轉角度（弧度），正角度為逆時針旋轉（從用戶視角）
        scale: 縮放因子
        roi: 可選的搜索區域 (x, y, w, h)
        use_weights: 是否使用模型權重（默認 True）
        kernel_radius: kernel 半徑（目前未使用，保留以保持 API 一致性）
    
    Returns:
        score: Chamfer 分數 [0, 1]，越大越好（距離和越小，分數越高）
    """
    if image.ndim == 3:
        g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        g = image
    
    # 應用 ROI（如果提供）
    if roi is not None:
        x, y, w, h = roi
        crop = g[y:y+h, x:x+w]
        # 調整中心位置到 ROI 座標系
        center_x = center_x - x
        center_y = center_y - y
        g = crop
    
    # 計算場景圖像的邊緣和距離變換
    low, high = model.get("edge_threshold", (60, 140))
    edg = canny_edges(g, low, high)
    dt = distance_from_edges(edg, mask_size=5)
    
    # 獲取模型點和權重
    tpl_h, tpl_w = model["template"].shape[:2]
    pts = model["points"].copy()  # Nx2, 模板座標系中的點 (float32)
    weights = model["weights"] if use_weights else np.ones(len(pts), dtype=np.float32)
    
    # 將點從模板座標系轉換到以中心為原點的座標系
    pts_centered = pts - np.array([tpl_w / 2.0, tpl_h / 2.0], dtype=np.float32)
    
    # 旋轉矩陣（逆時針旋轉，與 NCC 一致）
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, -sin_t], 
                  [sin_t, cos_t]], dtype=np.float32)
    
    # 旋轉 + 縮放
    pts_rotated_scaled = (pts_centered @ R.T) * scale
    
    # 轉換回場景座標系：平移到模板中心位置
    pts_scene = pts_rotated_scaled + np.array([center_x, center_y], dtype=np.float32)
    
    # 計算加權距離和：對於每個變換後的模型點，在 DT 中取樣
    total_distance = 0.0
    total_weight = 0.0
    valid_points = 0
    
    h_dt, w_dt = dt.shape[:2]
    
    for (px, py), w in zip(pts_scene, weights):
        # 檢查是否在距離變換圖範圍內（保留邊界外的點，使用邊界值）
        # 這樣可以處理部分超出邊界的情況
        px_clipped = np.clip(px, 0, w_dt - 1)
        py_clipped = np.clip(py, 0, h_dt - 1)
        
        # 使用雙線性插值取樣 DT 值（提高精度）
        x0 = int(px_clipped)
        y0 = int(py_clipped)
        x1 = min(x0 + 1, w_dt - 1)
        y1 = min(y0 + 1, h_dt - 1)
        
        fx = px_clipped - x0
        fy = py_clipped - y0
        
        # 雙線性插值
        d00 = dt[y0, x0]
        d01 = dt[y0, x1]
        d10 = dt[y1, x0]
        d11 = dt[y1, x1]
        
        distance = (1.0 - fx) * (1.0 - fy) * d00 + \
                   fx * (1.0 - fy) * d01 + \
                   (1.0 - fx) * fy * d10 + \
                   fx * fy * d11
        
        # 如果點在邊界內，才計入權重
        if 0 <= px < w_dt and 0 <= py < h_dt:
            total_distance += distance * w
            total_weight += w
            valid_points += 1
        else:
            # 邊界外的點給予較大懲罰（可選）
            # 這裡選擇不計入，或者給予較大距離值
            pass
    
    # 如果沒有有效點，返回低分
    if total_weight == 0 or valid_points == 0:
        return 0.0
    
    # 平均距離（越小越好）
    avg_distance = total_distance / total_weight
    
    # 轉換為相似度分數（越大越好）
    # 使用指數衰減轉換：exp(-distance/scale_factor)
    # 或者使用線性歸一化（類似 chamfer_coarse）
    # 這裡使用一個合理的距離閾值來歸一化
    max_reasonable_distance = 50.0  # 可調整，基於實際測試
    score = max(0.0, 1.0 - avg_distance / max_reasonable_distance)
    
    # 使用 sigmoid 函數進行更平滑的轉換（可選）
    # score = 1.0 / (1.0 + avg_distance / 10.0)
    
    return float(np.clip(score, 0.0, 1.0))


def grid_search_theta_scale_chamfer(model, image, center_x, center_y,
                                     ang_grid, scales, topk=5, roi=None,
                                     use_weights=True, kernel_radius=0,
                                     use_parallel=False, num_workers=None):
    """在角度和縮放空間進行網格搜索，使用 Chamfer 評分。
    
    類似於 grid_search_theta_scale_ncc，但使用 Chamfer 距離。
    支持並行化計算以提高性能。
    
    Args:
        model: 形狀模型（從 shape_model.create 獲得）
        image: 場景圖像
        center_x, center_y: 初始模板中心位置
        ang_grid: 角度網格（弧度）
        scales: 縮放因子列表
        topk: 返回前 K 個最佳候選
        roi: 可選的搜索區域限制 (x, y, w, h)
        use_weights: 是否使用權重
        kernel_radius: kernel 半徑（保留以保持 API 一致性）
        use_parallel: 是否使用並行化計算（默認 False）
        num_workers: 並行化時的工作進程數（None 表示自動檢測 CPU 核心數）
    
    Returns:
        List of (score, x, y, theta, scale) tuples, sorted by score descending
    """
    import multiprocessing as mp
    from functools import partial
    
    total_search = len(ang_grid) * len(scales)
    
    # 自動判斷是否使用並行化（任務足夠大時才值得）
    if use_parallel and num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)  # 保留一個核心給主進程
    
    # 判斷是否使用並行化（任務足夠大時才值得）
    should_parallel = (use_parallel and total_search > 20 and num_workers > 1)
    
    def _compute_chamfer_score_parallel(args):
        """並行計算 Chamfer 分數的輔助函數"""
        model, image, center_x, center_y, theta, scale, roi, use_weights, kernel_radius = args
        score = chamfer_score_rotated(model, image, center_x, center_y, theta, scale,
                                      roi=roi, use_weights=use_weights, kernel_radius=kernel_radius)
        return (score, center_x, center_y, theta, scale)
    
    if should_parallel:
        # 並行計算
        args_list = [(model, image, center_x, center_y, theta, scale, roi, use_weights, kernel_radius)
                     for theta in ang_grid for scale in scales]
        
        with mp.Pool(num_workers) as pool:
            candidates = pool.map(_compute_chamfer_score_parallel, args_list)
    else:
        # 串行計算
        candidates = []
        for theta in ang_grid:
            for scale in scales:
                score = chamfer_score_rotated(
                    model, image, center_x, center_y, theta, scale,
                    roi=roi, use_weights=use_weights, kernel_radius=kernel_radius
                )
                candidates.append((score, center_x, center_y, theta, scale))
    
    # 排序並返回前 topk
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # 調整座標（如果使用了 ROI）
    result = []
    for sc, x, y, th, s in candidates[:topk]:
        if roi is not None:
            x += roi[0]
            y += roi[1]
        result.append((sc, x, y, th, s))
    
    return result
